"""BATMAN transit model fitting following the reference notebook.

Reproduces notebook Step 3 & 4 in Cell 28:
    params = batman.TransitParams()
    params.t0  = t0
    params.per = bls_period
    params.rp  = k
    params.a   = a
    params.inc = inc_deg
    params.ecc = 0.0
    params.w   = 0.0
    params.limb_dark = "quadratic"
    params.u = [0.1, 0.0]

    popt, pcov = curve_fit(transit_model, t_fit, f_fit, p0=p0, bounds=bounds, maxfev=20000)
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import batman
from scipy.optimize import curve_fit


def evaluate_batman(
    t: np.ndarray,
    k: float,
    a: float,
    inc_deg: float,
    t0: float,
    period: float,
    u: list[float] | None = None,
) -> np.ndarray:
    """Evaluate BATMAN transit model at times t (phase in days)."""
    if u is None:
        u = [0.1, 0.0]

    params = batman.TransitParams()
    params.t0 = float(t0)
    params.per = float(period)
    params.rp = float(k)
    params.a = float(a)
    params.inc = float(inc_deg)
    params.ecc = 0.0
    params.w = 0.0
    params.limb_dark = "quadratic"
    params.u = u

    m = batman.TransitModel(params, np.asarray(t, dtype=np.float64))
    return m.light_curve(params)


def calculate_analytical_duration(
    period: float, k: float, a: float, inc_deg: float
) -> float:
    """Calculate total transit duration T_14 in days from orbital parameters."""
    inc_rad = np.radians(inc_deg)
    b = a * np.cos(inc_rad)
    arg_sq = (1.0 + k) ** 2 - b**2
    if arg_sq <= 0:
        return 0.0
    sin_term = (1.0 / a) * np.sqrt(arg_sq) / np.sin(inc_rad)
    sin_term = np.clip(sin_term, -1.0, 1.0)
    duration_days = (period / np.pi) * np.arcsin(sin_term)
    return float(duration_days)


def fit_batman_transit(
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    period: float,
    p0: list[float] | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit BATMAN transit model to binned light curve data using curve_fit.

    Args:
        t_fit: Phase centers (days, sorted).
        f_fit: Binned normalized flux.
        period: Orbital period fixed from BLS.
        p0: Initial guess [k, a/R*, inc, t0].
        bounds: Parameter bounds ([lower], [upper]).

    Returns:
        (fit_params_dict, model_df)
    """
    t_fit = np.asarray(t_fit, dtype=np.float64)
    f_fit = np.asarray(f_fit, dtype=np.float64)

    # Initial guess: estimate k from minimum binned flux if p0 not provided
    if p0 is None:
        depth_est = max(1e-5, 1.0 - float(np.min(f_fit)))
        k_init = float(np.sqrt(depth_est))
        p0 = [k_init, 10.0, 87.0, 0.0]

    if bounds is None:
        # Replicates notebook bounds with robust lower k bound for shallow transits
        bounds_lower = [0.001, 2.0, 60.0, -0.05]
        bounds_upper = [0.50, 30.0, 90.0, 0.05]
        bounds = (bounds_lower, bounds_upper)

    def model_func(t: np.ndarray, k: float, a: float, inc_deg: float, t0: float) -> np.ndarray:
        return evaluate_batman(t, k, a, inc_deg, t0, period)

    popt, pcov = curve_fit(
        model_func,
        t_fit,
        f_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)
    k_fit, a_fit, inc_fit, t0_fit = popt
    k_err, a_err, inc_err, t0_err = perr

    model_flux = model_func(t_fit, *popt)
    residuals = f_fit - model_flux
    rms_residuals = float(np.sqrt(np.mean(residuals**2)))

    # Transit depth = (Rp/R*)^2
    transit_depth = float(k_fit**2)
    transit_depth_ppm = float(transit_depth * 1e6)

    # Transit duration:
    duration_days = calculate_analytical_duration(period, k_fit, a_fit, inc_fit)
    duration_hours = float(duration_days * 24.0)

    fit_params = {
        "k_rp_rstar": float(k_fit),
        "k_err": float(k_err),
        "a_rstar": float(a_fit),
        "a_err": float(a_err),
        "inc_deg": float(inc_fit),
        "inc_err": float(inc_err),
        "t0_days": float(t0_fit),
        "t0_err": float(t0_err),
        "transit_depth": transit_depth,
        "transit_depth_ppm": transit_depth_ppm,
        "transit_duration_days": duration_days,
        "transit_duration_hours": duration_hours,
        "rms_residuals": rms_residuals,
        "period_fixed": float(period),
    }

    model_df = pd.DataFrame(
        {
            "phase_days": t_fit,
            "observed_flux": f_fit,
            "model_flux": model_flux,
            "residuals": residuals,
        }
    )

    return fit_params, model_df
