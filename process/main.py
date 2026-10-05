from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.io_utils import extract_kic_id, save_json, ensure_dir
from process.preprocessing import preprocess_lightcurve, fold_lightcurve
from process.bls import run_bls
from process.transit_model import fit_batman_transit, evaluate_batman
from process.mcmc import run_mcmc_sampler
from process.plots import (
    plot_processed_lightcurve,
    plot_bls_detection,
    plot_phase_folded,
    plot_transit_candidate_zoom,
    plot_batman_fit,
    plot_residuals,
    plot_mcmc_corner,
    plot_mcmc_transit_fit,
)
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

    if output_dir is None:
        target_dir = ensure_dir(PROJECT_ROOT / "outputs" / str(kic_id))
    else:
        out_path = Path(output_dir)
        if out_path.name == str(kic_id):
            target_dir = ensure_dir(out_path)
        else:
            target_dir = ensure_dir(out_path / str(kic_id))

    print(f"Processing FITS: {fits_file}")
    print(f"Target KIC:      {kic_id}")
    print(f"Output folder:   {target_dir}")

    raw_df, cleaned_df, processed_df = preprocess_lightcurve(fits_file)

    raw_path = target_dir / "raw_extracted.csv"
    raw_df.to_csv(raw_path, index=False)

    cleaned_path = target_dir / "cleaned_time_flux.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    processed_path = target_dir / "processed_time_flux.csv"
    processed_df.to_csv(processed_path, index=False)

    time_arr = processed_df["TIME"].to_numpy(dtype=np.float64)
    flux_arr = processed_df["NORMALIZED_FLUX"].to_numpy(dtype=np.float64)
    segments = processed_df["SEGMENT"].to_numpy(dtype=int)

    plot_lc_path = target_dir / "processed_lightcurve.png"
    plot_processed_lightcurve(time_arr, flux_arr, segments, kic_id, plot_lc_path)

    bls_params, periodogram_df, bls_power = run_bls(
        time=time_arr,
        flux=flux_arr,
        period_min=1.0,
        period_max=15.0,
        n_periods=n_periods,
        duration=0.1,
        oversample=20,
    )

    bls_results_path = target_dir / "bls_results.json"
    save_json(bls_params, bls_results_path)

    periodogram_path = target_dir / "bls_periodogram.csv"
    periodogram_df.to_csv(periodogram_path, index=False)

    best_period = float(bls_params["BEST_PERIOD"])
    best_t0 = float(bls_params["BEST_TRANSIT_TIME"])

    bls_plot_path = target_dir / "bls_periodogram.png"
    period_grid = np.exp(np.linspace(np.log(1.0), np.log(15.0), n_periods))
    plot_bls_detection(
        bls_power=bls_power,
        period_grid=period_grid,
        best_period=best_period,
        best_t0=best_t0,
        time=time_arr,
        flux=flux_arr,
        kic_id=kic_id,
        output_path=bls_plot_path,
    )

    folded_df, binned_df = fold_lightcurve(
        time=time_arr,
        flux=flux_arr,
        period=best_period,
        t0=best_t0,
        bin_width=0.002,
        window=0.3,
    )

    folded_path = target_dir / "phase_folded.csv"
    folded_df.to_csv(folded_path, index=False)

    binned_path = target_dir / "binned_transit.csv"
    binned_df.to_csv(binned_path, index=False)

    t_fit = binned_df["phase_days"].to_numpy(dtype=np.float64)
    f_fit = binned_df["binned_flux"].to_numpy(dtype=np.float64)

    folded_plot_path = target_dir / "phase_folded.png"
    plot_phase_folded(
        x_fold=folded_df["phase_days"].to_numpy(),
        y_flux=folded_df["normalized_flux"].to_numpy(),
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
        kic_id=kic_id,
        output_path=folded_plot_path,
    )

    zoom_plot_path = target_dir / "transit_candidate_zoom.png"
    plot_transit_candidate_zoom(
        t_fit=t_fit,
        f_fit=f_fit,
        kic_id=kic_id,
        period=best_period,
        output_path=zoom_plot_path,
    )

    fit_params, model_df = fit_batman_transit(
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
    )

    transit_fit_params_path = target_dir / "transit_fit_params.json"
    save_json(fit_params, transit_fit_params_path)

    model_df_path = target_dir / "model_fit.csv"
    model_df.to_csv(model_df_path, index=False)

    model_plot_path = target_dir / "transit_model.png"
    plot_batman_fit(
        t_fit=t_fit,
        f_fit=f_fit,
        model_flux=model_df["model_flux"].to_numpy(),
        popt=fit_params,
        kic_id=kic_id,
        output_path=model_plot_path,
    )

    residual_plot_path = target_dir / "residuals.png"
    plot_residuals(
        t_fit=t_fit,
        residuals=model_df["residuals"].to_numpy(),
        kic_id=kic_id,
        output_path=residual_plot_path,
    )

    mcmc_params: dict[str, Any] = {}
    if run_mcmc:
        initial_mcmc = [
            fit_params["k_rp_rstar"],
            fit_params["a_rstar"],
            fit_params["inc_deg"],
            fit_params["t0_days"],
        ]
        mcmc_params, samples_df = run_mcmc_sampler(
            t_fit=t_fit,
            f_fit=f_fit,
            period=best_period,
            initial=initial_mcmc,
            nwalkers=32,
            nsteps=mcmc_steps,
            discard=500,
            thin=10,
        )

        mcmc_results_path = target_dir / "mcmc_results.json"
        save_json(mcmc_params, mcmc_results_path)

        samples_path = target_dir / "posterior_samples.csv"
        samples_df.to_csv(samples_path, index=False)

        corner_plot_path = target_dir / "mcmc_corner.png"
        truths = [
            mcmc_params["k"]["median"],
            mcmc_params["a_rstar"]["median"],
            mcmc_params["inc_deg"]["median"],
            mcmc_params["t0_phase_days"]["median"],
        ]
        plot_mcmc_corner(samples_df, truths, kic_id, corner_plot_path)

        mcmc_best_model = evaluate_batman(
            t_fit,
            mcmc_params["k"]["median"],
            mcmc_params["a_rstar"]["median"],
            mcmc_params["inc_deg"]["median"],
            mcmc_params["t0_phase_days"]["median"],
            best_period,
        )
        mcmc_fit_plot_path = target_dir / "mcmc_transit_fit.png"
        plot_mcmc_transit_fit(t_fit, f_fit, mcmc_best_model, kic_id, mcmc_fit_plot_path)

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
        "saved_files": {
            "raw_extracted": str(raw_path),
            "cleaned_time_flux": str(cleaned_path),
            "processed_time_flux": str(processed_path),
            "bls_results": str(bls_results_path),
            "bls_periodogram": str(periodogram_path),
            "phase_folded": str(folded_path),
            "binned_transit": str(binned_path),
            "transit_fit_params": str(transit_fit_params_path),
            "model_fit": str(model_df_path),
            "mcmc_results": str(target_dir / "mcmc_results.json") if run_mcmc else None,
            "posterior_samples": str(target_dir / "posterior_samples.csv") if run_mcmc else None,
            "plots": {
                "processed_lightcurve": str(plot_lc_path),
                "bls_periodogram": str(bls_plot_path),
                "phase_folded": str(folded_plot_path),
                "transit_candidate_zoom": str(zoom_plot_path),
                "transit_model": str(model_plot_path),
                "residuals": str(residual_plot_path),
                "mcmc_corner": str(target_dir / "mcmc_corner.png") if run_mcmc else None,
                "mcmc_transit_fit": str(target_dir / "mcmc_transit_fit.png") if run_mcmc else None,
            },
        },
    }

    save_json(summary, target_dir / "summary.json")

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
    keep_combined: bool = True,
    cleanup_individual: bool = True,
) -> pd.DataFrame:
    fits_dir = ensure_dir(fits_root)
    processed_dir = ensure_dir(processed_root)
    ml_dir = ensure_dir(ml_root)

    koi_catalog = load_koi_catalog(metadata_csv)
    target_kics = get_kepids_by_row_range(metadata_csv, start_index, end_index)

    print(f"Dataset indices {start_index}–{end_index}: {len(target_kics)} unique KICs")
    dataset_rows: list[dict[str, Any]] = []

    for kic in target_kics:
        kic_str = str(kic)
        kic_folder = fits_dir / kic_str
        combined_file = kic_folder / "combined.fits"

        individual_fits: list[Path] = []

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
            dataset_rows.append(feature_row)

        except Exception as err:
            print(f"  Processing failed for KIC {kic_str}: {err}")
            continue

        if cleanup_individual and individual_fits:
            for f in individual_fits:
                try:
                    if f.exists() and f.name != "combined.fits":
                        f.unlink()
                except OSError:
                    pass

    df_ml = features_to_dataframe(dataset_rows)
    output_csv = ml_dir / "transit_dataset.csv"

    if output_csv.is_file():
        existing = pd.read_csv(output_csv)
        combined_df = pd.concat([existing, df_ml], ignore_index=True).drop_duplicates(
            subset=["kic", "candidate_rank"], keep="last"
        )
        combined_df.to_csv(output_csv, index=False)
        print(f"Appended {len(df_ml)} rows → dataset now {len(combined_df)} rows at {output_csv}")
    else:
        df_ml.to_csv(output_csv, index=False)
        print(f"Dataset saved: {len(df_ml)} rows at {output_csv}")

    return df_ml


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
    parser.add_argument(
        "--keep-combined",
        action="store_true",
        default=True,
        help="Keep combined.fits after processing (default: True).",
    )
    parser.add_argument(
        "--delete-combined",
        action="store_true",
        default=False,
        help="Delete combined.fits after successful processing.",
    )
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
        keep_combined=not args.delete_combined,
        cleanup_individual=True,
    )


if __name__ == "__main__":
    main()
