"""Phase folding and binning following the reference notebook.

Reproduces notebook Step 1 & 2 in Cell 16 and Cell 28:
    x_fold = (time - bls_t0 + 0.5*bls_period) % bls_period - 0.5*bls_period
    y_flux = flux / np.median(flux)
    bin_width = 0.002
    bins = np.arange(-0.3, 0.3 + bin_width, bin_width)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    bin_width: float = 0.002,
    window: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold and bin light curve data around mid-transit.

    Args:
        time: Array of observation times (days).
        flux: Array of flux values.
        period: Orbital period (days).
        t0: Mid-transit time / epoch (days).
        bin_width: Phase bin width in days (notebook default: 0.002).
        window: Half-window around transit center in days (notebook default: 0.3).

    Returns:
        (folded_df, binned_df)
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    # Replicate exact folding formula from notebook
    x_fold = (time - t0 + 0.5 * period) % period - 0.5 * period
    y_flux = flux / np.median(flux)

    folded_df = pd.DataFrame(
        {
            "phase_days": x_fold,
            "normalized_flux": y_flux,
        }
    )

    # Replicate exact manual binning from notebook Cell 28
    bins = np.arange(-window, window + bin_width, bin_width)
    t_centers = 0.5 * (bins[1:] + bins[:-1])

    t_binned = []
    f_binned = []
    err_binned = []
    count_binned = []

    for i in range(len(bins) - 1):
        m = (x_fold >= bins[i]) & (x_fold < bins[i + 1])
        if np.any(m):
            t_binned.append(t_centers[i])
            f_binned.append(np.median(y_flux[m]))
            err_binned.append(
                np.std(y_flux[m]) / np.sqrt(np.sum(m))
                if np.sum(m) > 1
                else np.std(y_flux)
            )
            count_binned.append(int(np.sum(m)))

    t_fit = np.array(t_binned, dtype=np.float64)
    f_fit = np.array(f_binned, dtype=np.float64)
    err_fit = np.array(err_binned, dtype=np.float64)
    counts = np.array(count_binned, dtype=int)

    # Sort for stability as in notebook
    order = np.argsort(t_fit)
    t_fit = t_fit[order]
    f_fit = f_fit[order]
    err_fit = err_fit[order]
    counts = counts[order]

    binned_df = pd.DataFrame(
        {
            "phase_days": t_fit,
            "binned_flux": f_fit,
            "binned_flux_err": err_fit,
            "n_points": counts,
        }
    )

    return folded_df, binned_df
