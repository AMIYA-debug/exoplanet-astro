"""End-to-end exoplanet transit pipeline.

Orchestrates:
1. FITS extraction (all LIGHTCURVE HDUs)
2. Quality filtering (SAP_QUALITY == 0 & finite)
3. Quarter-by-quarter median normalization (eliminating baseline offsets)
4. Astropy BoxLeastSquares (BLS) period search (log-uniform 50k grid)
5. Best period, duration, epoch, depth, power detection
6. Phase folding & manual binning
7. BATMAN transit model fit via curve_fit
8. Residual calculation and inspection
9. MCMC parameter estimation via emcee
10. Plot generation and structured data export
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from process.io import preprocess_fits
from process.bls import run_bls
from process.fold import fold_lightcurve
from process.transit_model import fit_batman_transit, evaluate_batman
from process.mcmc import run_mcmc_sampler
from process.plotting import (
    plot_processed_lightcurve,
    plot_bls_detection,
    plot_phase_folded,
    plot_transit_candidate_zoom,
    plot_batman_fit,
    plot_residuals,
    plot_mcmc_corner,
    plot_mcmc_transit_fit,
)

NASA_CATALOG_REFERENCE: dict[str, dict[str, Any]] = {
    "12885212": {
        "kepoi_name": "K02184.01",
        "period_days": 2.057499811,
        "epoch_bkjd": 133.31318,
        "depth_ppm": 325.6,
        "duration_hours": 2.569,
        "rp_rstar": 0.01583,
        "inclination_deg": 89.96,
    },
    "12935144": {
        "kepoi_name": "K02847.01",
        "period_days": 1.099291188,
        "epoch_bkjd": 131.50242,
        "depth_ppm": 282.6,
        "duration_hours": 1.765,
        "rp_rstar": 0.014873,
        "inclination_deg": 89.09,
    },
}


def process_kic(
    fits_path: Path | str,
    output_dir: Path | str,
    kic_id: str | None = None,
    run_mcmc: bool = True,
    n_periods: int = 50000,
    mcmc_steps: int = 2500,
) -> dict[str, Any]:
    """Run full notebook workflow on a single KIC's combined.fits file.

    Args:
        fits_path: Path to input combined.fits.
        output_dir: Directory where outputs for this KIC will be saved.
        kic_id: Optional KIC ID string (inferred from path if None).
        run_mcmc: Whether to run MCMC sampling stage.
        n_periods: Number of BLS period grid points.
        mcmc_steps: Number of MCMC steps.

    Returns:
        Structured dictionary of target properties, detection, fits, and validation.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if kic_id is None:
        kic_id = fits_path.parent.name

    print(f"\n==================================================")
    print(f"Starting processing for KIC {kic_id}")
    print(f"FITS file: {fits_path}")
    print(f"Output directory: {output_dir}")
    print(f"==================================================")

    # ----------------------------------------------------
    # Stage 1: FITS reading, cleaning, quarter-normalization
    # ----------------------------------------------------
    print("[1/6] Extracting and preprocessing light curve...")
    raw_df, cleaned_df, processed_df = preprocess_fits(fits_path)

    raw_path = output_dir / "raw_extracted.csv"
    raw_df.to_csv(raw_path, index=False)

    cleaned_path = output_dir / "cleaned_time_flux.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    processed_path = output_dir / "processed_time_flux.csv"
    processed_df.to_csv(processed_path, index=False)

    time = processed_df["TIME"].to_numpy(dtype=np.float64)
    flux = processed_df["NORMALIZED_FLUX"].to_numpy(dtype=np.float64)
    segments = processed_df["SEGMENT"].to_numpy(dtype=int)

    n_raw = len(raw_df)
    n_cleaned = len(cleaned_df)
    n_processed = len(processed_df)
    print(f"      Raw observations:       {n_raw}")
    print(f"      Cleaned observations:   {n_cleaned}")
    print(f"      Processed observations: {n_processed}")
    print(f"      Baseline median:        {np.median(flux):.6f}")

    # Plot processed light curve across time
    plot_processed_lc_path = output_dir / "processed_lightcurve.png"
    plot_processed_lightcurve(time, flux, segments, kic_id, plot_processed_lc_path)

    # ----------------------------------------------------
    # Stage 2: Box Least Squares (BLS) detection
    # ----------------------------------------------------
    print("[2/6] Running Box Least Squares (BLS) periodogram...")
    bls_params, periodogram_df, bls_power = run_bls(
        time=time,
        flux=flux,
        period_min=1.0,
        period_max=15.0,
        n_periods=n_periods,
        duration=0.1,
        oversample=20,
    )

    bls_results_path = output_dir / "bls_results.json"
    with open(bls_results_path, "w", encoding="utf-8") as f:
        json.dump(bls_params, f, indent=2)

    periodogram_path = output_dir / "bls_periodogram.csv"
    periodogram_df.to_csv(periodogram_path, index=False)

    best_period = bls_params["BEST_PERIOD"]
    best_t0 = bls_params["BEST_TRANSIT_TIME"]
    best_duration = bls_params["BEST_DURATION"]
    best_power = bls_params["BEST_POWER"]
    best_depth = bls_params["BEST_DEPTH"]

    print(f"      BLS Best Period:        {best_period:.5f} days")
    print(f"      BLS Transit Epoch (t0): {best_t0:.5f} BKJD")
    print(f"      BLS Transit Duration:   {best_duration:.4f} days ({best_duration*24:.2f} h)")
    print(f"      BLS Peak Power:         {best_power:.6f}")
    print(f"      BLS Transit Depth:      {best_depth*1e6:.1f} ppm ({best_depth:.6f})")

    # Plot BLS detection
    bls_plot_path = output_dir / "bls_periodogram.png"
    period_grid = np.exp(np.linspace(np.log(1.0), np.log(15.0), n_periods))
    plot_bls_detection(
        bls_power=bls_power,
        period_grid=period_grid,
        best_period=best_period,
        best_t0=best_t0,
        time=time,
        flux=flux,
        kic_id=kic_id,
        output_path=bls_plot_path,
    )

    # ----------------------------------------------------
    # Stage 3: Phase folding & manual binning
    # ----------------------------------------------------
    print("[3/6] Phase-folding and binning light curve...")
    folded_df, binned_df = fold_lightcurve(
        time=time,
        flux=flux,
        period=best_period,
        t0=best_t0,
        bin_width=0.002,
        window=0.3,
    )

    folded_path = output_dir / "phase_folded.csv"
    folded_df.to_csv(folded_path, index=False)

    binned_path = output_dir / "binned_transit.csv"
    binned_df.to_csv(binned_path, index=False)

    t_fit = binned_df["phase_days"].to_numpy(dtype=np.float64)
    f_fit = binned_df["binned_flux"].to_numpy(dtype=np.float64)

    # Plot phase-folded and candidate zoom
    folded_plot_path = output_dir / "phase_folded.png"
    plot_phase_folded(
        x_fold=folded_df["phase_days"].to_numpy(),
        y_flux=folded_df["normalized_flux"].to_numpy(),
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
        kic_id=kic_id,
        output_path=folded_plot_path,
    )

    zoom_plot_path = output_dir / "transit_candidate_zoom.png"
    plot_transit_candidate_zoom(
        t_fit=t_fit,
        f_fit=f_fit,
        kic_id=kic_id,
        period=best_period,
        output_path=zoom_plot_path,
    )

    # ----------------------------------------------------
    # Stage 4: BATMAN transit model fit via curve_fit
    # ----------------------------------------------------
    print("[4/6] Fitting BATMAN transit model...")
    fit_params, model_df = fit_batman_transit(
        t_fit=t_fit,
        f_fit=f_fit,
        period=best_period,
    )

    transit_fit_params_path = output_dir / "transit_fit_params.json"
    with open(transit_fit_params_path, "w", encoding="utf-8") as f:
        json.dump(fit_params, f, indent=2)

    model_df_path = output_dir / "model_fit.csv"
    model_df.to_csv(model_df_path, index=False)

    print(f"      BATMAN Rp/R* (k):       {fit_params['k_rp_rstar']:.4f} +/- {fit_params['k_err']:.4f}")
    print(f"      BATMAN a/R*:            {fit_params['a_rstar']:.2f} +/- {fit_params['a_err']:.2f}")
    print(f"      BATMAN Inclination:     {fit_params['inc_deg']:.2f}° +/- {fit_params['inc_err']:.2f}°")
    print(f"      BATMAN Mid-transit t0:  {fit_params['t0_days']:.5f} d +/- {fit_params['t0_err']:.5f} d")
    print(f"      Fitted Transit Depth:   {fit_params['transit_depth_ppm']:.1f} ppm")
    print(f"      Fitted Duration:        {fit_params['transit_duration_hours']:.2f} hours")
    print(f"      Residuals RMS:          {fit_params['rms_residuals']*1e6:.1f} ppm")

    # Plot BATMAN fit and residuals
    model_plot_path = output_dir / "transit_model.png"
    plot_batman_fit(
        t_fit=t_fit,
        f_fit=f_fit,
        model_flux=model_df["model_flux"].to_numpy(),
        popt=fit_params,
        kic_id=kic_id,
        output_path=model_plot_path,
    )

    residual_plot_path = output_dir / "residuals.png"
    plot_residuals(
        t_fit=t_fit,
        residuals=model_df["residuals"].to_numpy(),
        kic_id=kic_id,
        output_path=residual_plot_path,
    )

    # ----------------------------------------------------
    # Stage 5: MCMC parameter estimation (emcee)
    # ----------------------------------------------------
    mcmc_params = {}
    if run_mcmc:
        print("[5/6] Running MCMC posterior sampling (emcee)...")
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

        mcmc_results_path = output_dir / "mcmc_results.json"
        with open(mcmc_results_path, "w", encoding="utf-8") as f:
            json.dump(mcmc_params, f, indent=2)

        samples_path = output_dir / "posterior_samples.csv"
        samples_df.to_csv(samples_path, index=False)

        print(f"      MCMC Rp/R* (k):         {mcmc_params['k']['median']:.4f} (-{mcmc_params['k']['err_minus']:.4f}, +{mcmc_params['k']['err_plus']:.4f})")
        print(f"      MCMC a/R*:              {mcmc_params['a_rstar']['median']:.2f} (-{mcmc_params['a_rstar']['err_minus']:.2f}, +{mcmc_params['a_rstar']['err_plus']:.2f})")
        print(f"      MCMC Inclination:       {mcmc_params['inc_deg']['median']:.2f}° (-{mcmc_params['inc_deg']['err_minus']:.2f}°, +{mcmc_params['inc_deg']['err_plus']:.2f}°)")
        print(f"      MCMC Mid-transit t0:    {mcmc_params['t0_phase_days']['median']:.5f} d")
        print(f"      MCMC Depth:             {mcmc_params['transit_depth_ppm']:.1f} +/- {mcmc_params['transit_depth_err_ppm']:.1f} ppm")
        print(f"      MCMC Duration:          {mcmc_params['transit_duration_hours']:.2f} hours")

        # Plot MCMC corner and best-fit
        corner_plot_path = output_dir / "mcmc_corner.png"
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
        mcmc_fit_plot_path = output_dir / "mcmc_transit_fit.png"
        plot_mcmc_transit_fit(t_fit, f_fit, mcmc_best_model, kic_id, mcmc_fit_plot_path)
    else:
        print("[5/6] Skipping MCMC as requested.")

    # ----------------------------------------------------
    # Stage 6: Validation against NASA catalog
    # ----------------------------------------------------
    print("[6/6] Independent validation against NASA Exoplanet Archive...")
    validation = {}
    if kic_id in NASA_CATALOG_REFERENCE:
        ref = NASA_CATALOG_REFERENCE[kic_id]
        period_diff_pct = abs(best_period - ref["period_days"]) / ref["period_days"] * 100.0
        depth_diff_pct = abs(fit_params["transit_depth_ppm"] - ref["depth_ppm"]) / ref["depth_ppm"] * 100.0
        ror_diff_pct = abs(fit_params["k_rp_rstar"] - ref["rp_rstar"]) / ref["rp_rstar"] * 100.0

        is_consistent = (period_diff_pct < 0.1) and (depth_diff_pct < 25.0)
        validation = {
            "catalog_reference": ref,
            "period_diff_pct": float(period_diff_pct),
            "depth_diff_pct": float(depth_diff_pct),
            "rp_rstar_diff_pct": float(ror_diff_pct),
            "consistent_with_catalog": bool(is_consistent),
        }
        print(f"      NASA KOI Name:          {ref['kepoi_name']}")
        print(f"      NASA Catalog Period:    {ref['period_days']:.5f} d (Δ = {period_diff_pct:.4f}%)")
        print(f"      NASA Catalog Depth:     {ref['depth_ppm']:.1f} ppm (Δ = {depth_diff_pct:.2f}%)")
        print(f"      Consistent:             {is_consistent}")
    else:
        validation = {"note": "No reference entry in local demo catalog."}

    summary = {
        "kic": kic_id,
        "fits_file": str(fits_path),
        "observations": {
            "raw": n_raw,
            "cleaned": n_cleaned,
            "processed": n_processed,
        },
        "bls": bls_params,
        "transit_model": fit_params,
        "mcmc": mcmc_params,
        "validation": validation,
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
            "mcmc_results": str(output_dir / "mcmc_results.json") if run_mcmc else None,
            "posterior_samples": str(output_dir / "posterior_samples.csv") if run_mcmc else None,
            "plots": {
                "processed_lightcurve": str(plot_processed_lc_path),
                "bls_periodogram": str(bls_plot_path),
                "phase_folded": str(folded_plot_path),
                "transit_candidate_zoom": str(zoom_plot_path),
                "transit_model": str(model_plot_path),
                "residuals": str(residual_plot_path),
                "mcmc_corner": str(output_dir / "mcmc_corner.png") if run_mcmc else None,
                "mcmc_transit_fit": str(output_dir / "mcmc_transit_fit.png") if run_mcmc else None,
            },
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Processing complete for KIC {kic_id}. Summary saved to {summary_path}")
    return summary
