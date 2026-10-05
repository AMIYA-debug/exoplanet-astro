from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.io_utils import extract_kic_id, ensure_dir
from process.preprocessing import preprocess_lightcurve, fold_lightcurve
from process.bls import run_bls
from process.transit_model import fit_batman_transit
from process.mcmc import run_mcmc_sampler
from process.features import extract_candidate_features, features_to_dataframe
from process.download import get_kepids_by_row_range, download_kepid_fits
from process.merge import merge_kic_directory

DEFAULT_METADATA_CSV = PROJECT_ROOT / "data/raw/metadata/cumulative_2026.10.02_06.40.58_kepid_descending.csv"
DEFAULT_FITS_ROOT = PROJECT_ROOT / "data/raw/fits"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data/processed"
DEFAULT_ML_ROOT = PROJECT_ROOT / "data/ml"


def process_fits_file(
    fits_path: Path | str,
    output_dir: Path | str | None = None,
    kic_id: str | None = None,
    run_mcmc: bool = True,
    n_periods: int = 50000,
    mcmc_steps: int = 2500,
) -> dict[str, Any]:
    fits_file = Path(fits_path).resolve()
    if not fits_file.is_file():
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    if kic_id is None:
        kic_id = extract_kic_id(fits_file, fallback=fits_file.stem)

    print(f"Processing FITS: {fits_file}")
    print(f"Target KIC:      {kic_id}")

    raw_df, cleaned_df, processed_df = preprocess_lightcurve(fits_file)

    time_arr = processed_df["TIME"].to_numpy(dtype=np.float64)
    flux_arr = processed_df["NORMALIZED_FLUX"].to_numpy(dtype=np.float64)

    bls_params, _, _ = run_bls(
        time=time_arr,
        flux=flux_arr,
        period_min=1.0,
        period_max=15.0,
        n_periods=n_periods,
        duration=0.1,
        oversample=20,
    )

    best_period = float(bls_params["BEST_PERIOD"])
    best_t0 = float(bls_params["BEST_TRANSIT_TIME"])

    _, binned_df = fold_lightcurve(
        time=time_arr,
        flux=flux_arr,
        period=best_period,
        t0=best_t0,
        bin_width=0.002,
        window=0.3,
    )

    t_fit = binned_df["phase_days"].to_numpy(dtype=np.float64)
    f_fit = binned_df["binned_flux"].to_numpy(dtype=np.float64)

    fit_params, _ = fit_batman_transit(
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
    )

    mcmc_params: dict[str, Any] = {}
    if run_mcmc:
        initial_mcmc = [
            fit_params["k_rp_rstar"],
            fit_params["a_rstar"],
            fit_params["inc_deg"],
            fit_params["t0_days"],
        ]
        mcmc_params, _ = run_mcmc_sampler(
            t_fit=t_fit,
            f_fit=f_fit,
            period=best_period,
            initial=initial_mcmc,
            nwalkers=32,
            nsteps=mcmc_steps,
            discard=500,
            thin=10,
        )

    obs_stats = {
        "raw": len(raw_df),
        "cleaned": len(cleaned_df),
        "processed": len(processed_df),
        "time_span_days": float(np.nanmax(time_arr) - np.nanmin(time_arr)),
    }

    feature_dict = extract_candidate_features(
        kic=kic_id,
        candidate_rank=1,
        bls_params=bls_params,
        fit_params=fit_params,
        mcmc_params=mcmc_params if run_mcmc else None,
        obs_stats=obs_stats,
    )

    summary = {
        "kic": kic_id,
        "fits_file": str(fits_file),
        "observations": obs_stats,
        "bls": bls_params,
        "transit_model": fit_params,
        "mcmc": mcmc_params,
        "features": feature_dict,
    }

    print(f"Results for KIC {kic_id}:")
    print(f"  BLS Period:   {best_period:.5f} days")
    print(f"  BLS t0:       {best_t0:.5f} BKJD")
    print(f"  BATMAN Rp/R*: {fit_params['k_rp_rstar']:.4f}")
    print(f"  BATMAN Depth: {fit_params['transit_depth_ppm']:.1f} ppm")
    if run_mcmc and "k" in mcmc_params:
        print(f"  MCMC Rp/R*:   {mcmc_params['k']['median']:.4f}")
        print(f"  MCMC Depth:   {mcmc_params['transit_depth_ppm']:.1f} ppm")

    return summary


def load_koi_catalog(csv_path: Path | str) -> dict[str, list[dict[str, Any]]]:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        return {}

    catalog: dict[str, list[dict[str, Any]]] = {}
    with csv_file.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kepid_str = str(row.get("kepid", "")).strip()
            if not kepid_str:
                continue
            try:
                period_val = float(row.get("koi_period", 0.0))
            except ValueError:
                period_val = 0.0

            entry = {
                "kepoi_name": row.get("kepoi_name", ""),
                "kepler_name": row.get("kepler_name", ""),
                "koi_disposition": row.get("koi_disposition", "UNKNOWN"),
                "koi_period": period_val,
                "koi_time0bk": float(row.get("koi_time0bk", 0.0) or 0.0),
                "koi_depth": float(row.get("koi_depth", 0.0) or 0.0),
                "koi_duration": float(row.get("koi_duration", 0.0) or 0.0),
            }
            catalog.setdefault(kepid_str, []).append(entry)
    return catalog


def match_candidate_to_koi(
    kic: str,
    period: float,
    koi_catalog: dict[str, list[dict[str, Any]]],
    tolerance: float = 0.05,
) -> tuple[str | None, str]:
    entries = koi_catalog.get(str(kic), [])
    if not entries:
        return None, "UNMATCHED"

    best_match: dict[str, Any] | None = None
    min_delta = float("inf")

    for entry in entries:
        cat_period = entry["koi_period"]
        if cat_period <= 0:
            continue
        rel_diff = abs(period - cat_period) / cat_period
        if rel_diff < tolerance and rel_diff < min_delta:
            min_delta = rel_diff
            best_match = entry

    if best_match is not None:
        return best_match["kepoi_name"], best_match["koi_disposition"]

    return None, "UNMATCHED"


def run_dataset_pipeline(
    start_index: int,
    end_index: int,
    metadata_csv: Path | str = DEFAULT_METADATA_CSV,
    fits_root: Path | str = DEFAULT_FITS_ROOT,
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
    ml_root: Path | str = DEFAULT_ML_ROOT,
    run_mcmc: bool = False,
    mcmc_steps: int = 1500,
    n_periods: int = 50000,
) -> pd.DataFrame:
    fits_dir = ensure_dir(fits_root)
    processed_dir = ensure_dir(processed_root)
    ml_dir = ensure_dir(ml_root)
    output_csv = ml_dir / "transit_dataset.csv"

    koi_catalog = load_koi_catalog(metadata_csv)
    target_kics = get_kepids_by_row_range(metadata_csv, start_index, end_index)

    print(f"Dataset indices {start_index}–{end_index}: {len(target_kics)} unique KICs")
    all_new_rows: list[dict[str, Any]] = []

    for kic in target_kics:
        kic_str = str(kic)
        kic_folder = fits_dir / kic_str
        combined_file = kic_folder / "combined.fits"

        if not combined_file.is_file():
            print(f"Downloading KIC {kic_str}...")
            try:
                individual_fits = download_kepid_fits(kic, fits_dir)
            except Exception as err:
                print(f"  Download failed for KIC {kic_str}: {err}")
                continue

            if not individual_fits:
                print(f"  No FITS products found for KIC {kic_str}, skipping.")
                continue

            print(f"  Merging {len(individual_fits)} files for KIC {kic_str}...")
            try:
                merge_success = merge_kic_directory(kic_folder)
                if not merge_success:
                    print(f"  Merge produced no combined.fits for KIC {kic_str}, skipping.")
                    continue
            except Exception as err:
                print(f"  Merge failed for KIC {kic_str}: {err}")
                continue
        else:
            print(f"KIC {kic_str}: combined.fits already exists, skipping download/merge.")

        try:
            summary = process_fits_file(
                fits_path=combined_file,
                output_dir=processed_dir,
                kic_id=kic_str,
                run_mcmc=run_mcmc,
                n_periods=n_periods,
                mcmc_steps=mcmc_steps,
            )

            feature_row = dict(summary.get("features", {}))
            bls_p = float(summary.get("bls", {}).get("BEST_PERIOD", 0.0))
            koi_name, disposition = match_candidate_to_koi(kic_str, bls_p, koi_catalog)
            feature_row["koi_name"] = koi_name
            feature_row["koi_disposition"] = disposition

            row_df = features_to_dataframe([feature_row])
            if output_csv.is_file():
                existing = pd.read_csv(output_csv)
                combined_df = pd.concat([existing, row_df], ignore_index=True).drop_duplicates(
                    subset=["kic", "candidate_rank"], keep="last"
                )
                combined_df.to_csv(output_csv, index=False)
            else:
                row_df.to_csv(output_csv, index=False)

            all_new_rows.append(feature_row)

            if kic_folder.exists():
                shutil.rmtree(kic_folder, ignore_errors=True)

            kic_processed_folder = processed_dir / kic_str
            if kic_processed_folder.exists():
                shutil.rmtree(kic_processed_folder, ignore_errors=True)

            for residual_csv in processed_dir.glob(f"*{kic_str}*.csv"):
                try:
                    residual_csv.unlink()
                except OSError:
                    pass

        except Exception as err:
            print(f"  Processing failed for KIC {kic_str}: {err}")
            continue

    if output_csv.is_file():
        final_df = pd.read_csv(output_csv)
        print(f"Dataset updated: {len(final_df)} rows at {output_csv}")
    else:
        final_df = features_to_dataframe(all_new_rows)

    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Astronitr dataset pipeline — downloads, processes, and extracts "
            "transit features for a range of rows from the NASA cumulative KOI CSV."
        )
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="ROW",
        help="First row index (1-based) from the NASA cumulative CSV.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        metavar="ROW",
        help="Last row index (1-based, inclusive) from the NASA cumulative CSV.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="ROW",
        help="Single row index (1-based). Equivalent to --start N --end N.",
    )
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--fits-root", type=Path, default=DEFAULT_FITS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--ml-root", type=Path, default=DEFAULT_ML_ROOT)
    parser.add_argument("--mcmc", action="store_true", help="Run MCMC sampling.")
    parser.add_argument("--mcmc-steps", type=int, default=1500)
    parser.add_argument("--n-periods", type=int, default=50000)
    args = parser.parse_args()

    if args.index is not None:
        start_index = args.index
        end_index = args.index
    elif args.start is not None and args.end is not None:
        start_index = args.start
        end_index = args.end
    else:
        parser.error("Provide --start and --end (row indices) or --index (single row).")

    run_dataset_pipeline(
        start_index=start_index,
        end_index=end_index,
        metadata_csv=args.metadata_csv,
        fits_root=args.fits_root,
        processed_root=args.processed_root,
        ml_root=args.ml_root,
        run_mcmc=args.mcmc,
        mcmc_steps=args.mcmc_steps,
        n_periods=args.n_periods,
    )


if __name__ == "__main__":
    main()
