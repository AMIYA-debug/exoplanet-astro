"""Box Least Squares (BLS) period search following the reference notebook.

Reproduces the notebook methodology:
    period_grid = np.exp(np.linspace(np.log(1), np.log(15), 50000))
    bls = BoxLeastSquares(time, flux)
    bls_power = bls.power(period_grid, 0.1, oversample=20)
    index = np.argmax(bls_power.power)
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 1.0,
    period_max: float = 15.0,
    n_periods: int = 50000,
    duration: float = 0.1,
    oversample: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    """Execute Box Least Squares periodogram matching notebook parameters.

    Args:
        time: Array of observation times (float days, e.g. BKJD).
        flux: Array of normalized flux values.
        period_min: Minimum period in days (notebook default: 1.0).
        period_max: Maximum period in days (notebook default: 15.0).
        n_periods: Number of grid points (notebook default: 50000).
        duration: Transit duration for BLS in days (notebook default: 0.1).
        oversample: Oversampling factor (notebook default: 20).

    Returns:
        (best_params, periodogram_df, bls_power_object)
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    # Replicate exact log-uniform period grid from notebook
    period_grid = np.exp(np.linspace(np.log(period_min), np.log(period_max), n_periods))

    # Astropy BLS computation
    bls = BoxLeastSquares(time, flux)
    bls_power = bls.power(period_grid, duration, oversample=oversample)

    # Best candidate selection (argmax of BLS power)
    best_idx = int(np.argmax(bls_power.power))
    best_period = float(bls_power.period[best_idx])
    best_t0 = float(bls_power.transit_time[best_idx])
    best_depth = float(bls_power.depth[best_idx])
    best_power = float(bls_power.power[best_idx])
    best_duration = float(bls_power.duration[best_idx])

    # Compute depth SNR if available
    depth_snr = (
        float(bls_power.depth_snr[best_idx])
        if hasattr(bls_power, "depth_snr")
        else float(np.nan)
    )

    # Identify secondary peaks sufficiently separated from the best period
    sorted_indices = np.argsort(bls_power.power)[::-1]
    top_candidates = []
    for idx in sorted_indices:
        p_cand = float(bls_power.period[idx])
        pow_cand = float(bls_power.power[idx])
        if not any(abs(p_cand - prev["period"]) < 0.05 * prev["period"] for prev in top_candidates):
            top_candidates.append(
                {
                    "rank": len(top_candidates) + 1,
                    "period": p_cand,
                    "power": pow_cand,
                    "transit_time": float(bls_power.transit_time[idx]),
                    "depth": float(bls_power.depth[idx]),
                    "duration": float(bls_power.duration[idx]),
                }
            )
            if len(top_candidates) >= 5:
                break

    best_params = {
        "BEST_PERIOD": best_period,
        "BEST_DURATION": best_duration,
        "BEST_TRANSIT_TIME": best_t0,
        "BEST_POWER": best_power,
        "BEST_DEPTH": best_depth,
        "BEST_DEPTH_SNR": depth_snr,
        "TOP_CANDIDATES": top_candidates,
    }

    periodogram_df = pd.DataFrame(
        {
            "period": bls_power.period,
            "power": bls_power.power,
            "transit_time": bls_power.transit_time,
            "depth": bls_power.depth,
            "duration": bls_power.duration,
        }
    )

    return best_params, periodogram_df, bls_power
