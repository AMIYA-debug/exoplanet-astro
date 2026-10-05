from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.io_utils import save_json, ensure_dir
from process.preprocessing import clean_observations, normalize_by_segment, fold_lightcurve
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
from process.features import extract_candidate_features

INTERPRETATIONS_ROOT = PROJECT_ROOT / "interpretations"


def load_time_flux_csv(csv_path: Path | str) -> pd.DataFrame:
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    required = {"TIME", "PDCSAP_FLUX"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def prepare_time_flux(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "SAP_QUALITY" not in df.columns:
        df = df.copy()
        df["SAP_QUALITY"] = 0
    if "PDCSAP_FLUX_ERR" not in df.columns:
        df = df.copy()
        df["PDCSAP_FLUX_ERR"] = np.nan
    if "SEGMENT" not in df.columns:
        df = df.copy()
        df["SEGMENT"] = 1

    cleaned_df = clean_observations(df)
    processed_df = normalize_by_segment(cleaned_df)
    return cleaned_df, processed_df


def analyze_time_flux(
    csv_path: Path | str,
    output_name: str | None = None,
    output_dir: Path | str | None = None,
    run_mcmc: bool = True,
    n_periods: int = 50000,
    mcmc_steps: int = 2500,
) -> dict[str, Any]:
    csv_file = Path(csv_path).resolve()
    label = output_name if output_name else csv_file.stem

    if output_dir is not None:
        target_dir = ensure_dir(output_dir)
    else:
        target_dir = ensure_dir(INTERPRETATIONS_ROOT / label)

    print(f"Input CSV:     {csv_file}")
    print(f"Output folder: {target_dir}")

    raw_df = load_time_flux_csv(csv_file)
    cleaned_df, processed_df = prepare_time_flux(raw_df)

    raw_df.to_csv(target_dir / "raw_input.csv", index=False)
    cleaned_df.to_csv(target_dir / "cleaned_time_flux.csv", index=False)
    processed_df.to_csv(target_dir / "processed_time_flux.csv", index=False)

    time_arr = processed_df["TIME"].to_numpy(dtype=np.float64)
    flux_arr = processed_df["NORMALIZED_FLUX"].to_numpy(dtype=np.float64)
    segments = processed_df["SEGMENT"].to_numpy(dtype=int) if "SEGMENT" in processed_df.columns else None

    plot_processed_lightcurve(time_arr, flux_arr, segments, label, target_dir / "processed_lightcurve.png")

    bls_params, periodogram_df, bls_power = run_bls(
        time=time_arr,
        flux=flux_arr,
        period_min=1.0,
        period_max=15.0,
        n_periods=n_periods,
        duration=0.1,
        oversample=20,
    )

    save_json(bls_params, target_dir / "bls_results.json")
    periodogram_df.to_csv(target_dir / "bls_periodogram.csv", index=False)

    best_period = float(bls_params["BEST_PERIOD"])
    best_t0 = float(bls_params["BEST_TRANSIT_TIME"])

    period_grid = np.exp(np.linspace(np.log(1.0), np.log(15.0), n_periods))
    plot_bls_detection(
        bls_power=bls_power,
        period_grid=period_grid,
        best_period=best_period,
        best_t0=best_t0,
        time=time_arr,
        flux=flux_arr,
        kic_id=label,
        output_path=target_dir / "bls_periodogram.png",
    )

    folded_df, binned_df = fold_lightcurve(
        time=time_arr,
        flux=flux_arr,
        period=best_period,
        t0=best_t0,
        bin_width=0.002,
        window=0.3,
    )

    folded_df.to_csv(target_dir / "phase_folded.csv", index=False)
    binned_df.to_csv(target_dir / "binned_transit.csv", index=False)

    t_fit = binned_df["phase_days"].to_numpy(dtype=np.float64)
    f_fit = binned_df["binned_flux"].to_numpy(dtype=np.float64)

    plot_phase_folded(
        x_fold=folded_df["phase_days"].to_numpy(),
        y_flux=folded_df["normalized_flux"].to_numpy(),
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
        kic_id=label,
        output_path=target_dir / "phase_folded.png",
    )

    plot_transit_candidate_zoom(
        t_fit=t_fit,
        f_fit=f_fit,
        kic_id=label,
        period=best_period,
        output_path=target_dir / "transit_candidate_zoom.png",
    )

    fit_params, model_df = fit_batman_transit(
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
    )

    save_json(fit_params, target_dir / "transit_fit_params.json")
    model_df.to_csv(target_dir / "model_fit.csv", index=False)

    plot_batman_fit(
        t_fit=t_fit,
        f_fit=f_fit,
        model_flux=model_df["model_flux"].to_numpy(),
        popt=fit_params,
        kic_id=label,
        output_path=target_dir / "transit_model.png",
    )

    plot_residuals(
        t_fit=t_fit,
        residuals=model_df["residuals"].to_numpy(),
        kic_id=label,
        output_path=target_dir / "residuals.png",
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

        save_json(mcmc_params, target_dir / "mcmc_results.json")
        samples_df.to_csv(target_dir / "posterior_samples.csv", index=False)

        truths = [
            mcmc_params["k"]["median"],
            mcmc_params["a_rstar"]["median"],
            mcmc_params["inc_deg"]["median"],
            mcmc_params["t0_phase_days"]["median"],
        ]
        plot_mcmc_corner(samples_df, truths, label, target_dir / "mcmc_corner.png")

        mcmc_best_model = evaluate_batman(
            t_fit,
            mcmc_params["k"]["median"],
            mcmc_params["a_rstar"]["median"],
            mcmc_params["inc_deg"]["median"],
            mcmc_params["t0_phase_days"]["median"],
            best_period,
        )
        plot_mcmc_transit_fit(
            t_fit, f_fit, mcmc_best_model, label, target_dir / "mcmc_transit_fit.png"
        )

    obs_stats = {
        "raw": len(raw_df),
        "cleaned": len(cleaned_df),
        "processed": len(processed_df),
        "time_span_days": float(np.nanmax(time_arr) - np.nanmin(time_arr)),
    }

    features = extract_candidate_features(
        kic=label,
        candidate_rank=1,
        bls_params=bls_params,
        fit_params=fit_params,
        mcmc_params=mcmc_params if run_mcmc else None,
        obs_stats=obs_stats,
    )

    summary = {
        "input": str(csv_file),
        "label": label,
        "observations": obs_stats,
        "bls": bls_params,
        "transit_model": fit_params,
        "mcmc": mcmc_params,
        "features": features,
    }
    save_json(summary, target_dir / "summary.json")

    print(f"Results for {label}:")
    print(f"  BLS Period:   {best_period:.5f} days")
    print(f"  BLS t0:       {best_t0:.5f}")
    print(f"  BATMAN Rp/R*: {fit_params['k_rp_rstar']:.4f}")
    print(f"  BATMAN Depth: {fit_params['transit_depth_ppm']:.1f} ppm")
    if run_mcmc and "k" in mcmc_params:
        print(f"  MCMC Rp/R*:   {mcmc_params['k']['median']:.4f}")
        print(f"  MCMC Depth:   {mcmc_params['transit_depth_ppm']:.1f} ppm")
    print(f"Output:        {target_dir}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a TIME + PDCSAP_FLUX CSV file and generate transit analysis "
            "plots and summaries in interpretations/<name>/."
        )
    )
    parser.add_argument("csv_file", type=Path, help="Path to CSV with TIME and PDCSAP_FLUX columns.")
    parser.add_argument("--name", type=str, default=None, help="Output folder name (default: CSV stem).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory path.")
    parser.add_argument("--no-mcmc", action="store_true", help="Skip MCMC sampling.")
    parser.add_argument("--n-periods", type=int, default=50000, help="BLS period grid size.")
    parser.add_argument("--mcmc-steps", type=int, default=2500, help="MCMC production steps.")
    args = parser.parse_args()

    analyze_time_flux(
        csv_path=args.csv_file,
        output_name=args.name,
        output_dir=args.output_dir,
        run_mcmc=not args.no_mcmc,
        n_periods=args.n_periods,
        mcmc_steps=args.mcmc_steps,
    )


if __name__ == "__main__":
    main()
