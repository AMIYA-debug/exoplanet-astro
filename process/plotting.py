"""Plotting routines matching the reference notebook visualizations.

Implements:
1. Processed light curve across time (demonstrating common flat baseline)
2. BLS periodogram and folded candidate (notebook Cell 16)
3. Phase-folded transit (binned + unbinned)
4. Transit candidate zoom
5. BATMAN transit fit (notebook Cell 28)
6. Residuals inspection
7. MCMC corner plot (notebook Cell 33)
8. MCMC best-fit model (notebook Cell 33)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner


def plot_processed_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    segments: np.ndarray | None,
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Plot the full processed light curve to verify the uniform baseline."""
    plt.figure(figsize=(15, 5))
    plt.plot(time, flux, "k.", ms=1.5, alpha=0.5, label="Normalized Flux")
    plt.axhline(1.0, color="red", linestyle="--", lw=1.0, label="Baseline (1.0)")

    plt.xlabel("Time [BKJD]")
    plt.ylabel("Normalized Flux")
    plt.title(f"KIC {kic_id} – Preprocessed & Quarter-Normalized Light Curve")
    plt.ylim(0.994, 1.006)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bls_detection(
    bls_power: Any,
    period_grid: np.ndarray,
    best_period: float,
    best_t0: float,
    time: np.ndarray,
    flux: np.ndarray,
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Generate two-panel BLS detection plot matching notebook Cell 16."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Periodogram
    ax = axes[0]
    ax.axvline(np.log10(best_period), color="C1", lw=4, alpha=0.8)
    ax.plot(np.log10(bls_power.period), bls_power.power, "k", lw=0.8)
    ax.annotate(
        f"period = {best_period:.5f} d",
        (0.02, 0.92),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="C1", lw=1.5),
    )
    ax.set_ylabel("BLS power")
    ax.set_yticks([])
    ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
    ax.set_xlabel("log10(period)")
    ax.set_title(f"KIC {kic_id} – BLS Periodogram")
    ax.grid(True, alpha=0.25)

    # 2. Folded Transit (notebook histogram-weighted binning)
    ax = axes[1]
    x_fold = (time - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period
    m = np.abs(x_fold) < 0.4

    bins = np.linspace(-0.41, 0.41, 64)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=flux)
    denom[num == 0] = 1.0

    norm_factor = np.median(num / denom)
    ax.plot(x_fold[m], flux[m] / norm_factor, ".k", ms=2.5, alpha=0.4, label="Observations")
    b_centers = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(
        b_centers,
        (num / denom) / norm_factor,
        color="C1",
        lw=2.5,
        label="Binned profile",
    )

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylabel("De-trended flux")
    ax.set_xlabel("Phase (days)")
    ax.set_title(f"KIC {kic_id} – Phase-Folded at P = {best_period:.5f} d")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_phase_folded(
    x_fold: np.ndarray,
    y_flux: np.ndarray,
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    period: float,
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Plot phase-folded light curve with unbinned observations and binned points."""
    plt.figure(figsize=(10, 5))
    m = np.abs(x_fold) <= 0.3
    plt.plot(x_fold[m], y_flux[m], "k.", ms=2.0, alpha=0.3, label="Unbinned data")
    plt.plot(t_fit, f_fit, "r-", lw=2.0, label="Binned curve (width=0.002 d)")

    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"KIC {kic_id} – Phase Folded Light Curve (P = {period:.5f} d)")
    plt.xlim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_transit_candidate_zoom(
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    kic_id: str,
    period: float,
    output_path: Path | str,
) -> None:
    """Plot zoomed-in view of the transit dip."""
    plt.figure(figsize=(8, 5))
    plt.plot(t_fit, f_fit, "ko-", ms=4, lw=1.2, label="Binned observations")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"KIC {kic_id} – Transit Candidate Zoom (P = {period:.5f} d)")
    plt.xlim(-0.15, 0.15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_batman_fit(
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    model_flux: np.ndarray,
    popt: dict[str, Any],
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Plot BATMAN transit model fit matching notebook Cell 28."""
    plt.figure(figsize=(10, 5))
    plt.plot(t_fit, f_fit, "ko", ms=4, label="Binned data")
    plt.plot(t_fit, model_flux, "r-", lw=2, label="Best-fit model")

    k_val = popt.get("k_rp_rstar", 0.0)
    a_val = popt.get("a_rstar", 0.0)
    inc_val = popt.get("inc_deg", 0.0)
    t0_val = popt.get("t0_days", 0.0)

    label_text = f"Rp/R* = {k_val:.4f}\na/R* = {a_val:.2f}\ninc = {inc_val:.2f}°\nt0 = {t0_val:.5f} d"
    plt.annotate(
        label_text,
        (0.03, 0.15),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=1.2),
    )

    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized flux")
    plt.title(f"KIC {kic_id} – BATMAN Transit Fit")
    plt.xlim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residuals(
    t_fit: np.ndarray,
    residuals: np.ndarray,
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Plot fit residuals to inspect for systematic errors."""
    plt.figure(figsize=(10, 4))
    plt.plot(t_fit, residuals, "ko", ms=4, alpha=0.7, label="Residuals (Data - Model)")
    plt.axhline(0.0, color="red", linestyle="--", lw=1.5)

    rms = np.sqrt(np.mean(residuals**2))
    plt.annotate(
        f"RMS = {rms * 1e6:.1f} ppm",
        (0.03, 0.85),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.0),
    )

    plt.xlabel("Phase (days)")
    plt.ylabel("Residual Flux")
    plt.title(f"KIC {kic_id} – Transit Fit Residuals")
    plt.xlim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_mcmc_corner(
    samples_df: Any,
    truths: list[float],
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Generate corner plot of MCMC posteriors matching notebook Cell 33."""
    samples = samples_df.to_numpy() if hasattr(samples_df, "to_numpy") else np.asarray(samples_df)
    labels = ["k (Rp/R*)", "a/R*", "inc [deg]", "t0 [d]"]

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 11},
    )
    fig.suptitle(f"KIC {kic_id} – MCMC Posterior Distributions", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mcmc_transit_fit(
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    best_model_flux: np.ndarray,
    kic_id: str,
    output_path: Path | str,
) -> None:
    """Plot best-fit MCMC transit model matching notebook Cell 33."""
    plt.figure(figsize=(7, 5))
    plt.plot(t_fit, f_fit, ".k", label="Binned data")
    plt.plot(t_fit, best_model_flux, "r-", lw=2, label="Best-fit model (MCMC)")

    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"KIC {kic_id} – BATMAN + MCMC Transit Fit")
    plt.xlim(-0.3, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
