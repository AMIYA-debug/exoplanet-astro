from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import emcee
from process.transit_model import evaluate_batman, calculate_analytical_duration


def run_mcmc_sampler(
    t_fit: np.ndarray,
    f_fit: np.ndarray,
    period: float,
    initial: list[float] | np.ndarray,
    nwalkers: int = 32,
    nsteps: int = 2500,
    discard: int = 500,
    thin: int = 10,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    np.random.seed(random_seed)
    t = np.asarray(t_fit, dtype=np.float64)
    f = np.asarray(f_fit, dtype=np.float64)

    oot_mask = np.abs(t) > 0.08
    sigma_val = float(np.std(f[oot_mask])) if np.sum(oot_mask) > 10 else float(np.std(f))
    sigma = sigma_val * np.ones_like(f)

    k_min, k_max = 0.001, 0.30
    a_min, a_max = 2.0, 30.0
    inc_min, inc_max = 80.0, 90.0
    t0_min, t0_max = -0.05, 0.05

    def log_prior(theta: np.ndarray) -> float:
        k, a, inc, t0 = theta
        if (
            k_min < k < k_max
            and a_min < a < a_max
            and inc_min < inc < inc_max
            and t0_min < t0 < t0_max
        ):
            return 0.0
        return -np.inf

    def log_likelihood(theta: np.ndarray) -> float:
        k, a, inc, t0 = theta
        model = evaluate_batman(t, k, a, inc, t0, period)
        return -0.5 * float(np.sum(((f - model) / sigma) ** 2))

    def log_prob(theta: np.ndarray) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = 4
    init_clamped = np.array(
        [
            np.clip(initial[0], k_min + 0.001, k_max - 0.01),
            np.clip(initial[1], a_min + 0.5, a_max - 0.5),
            np.clip(initial[2], inc_min + 0.5, inc_max - 0.5),
            np.clip(initial[3], t0_min + 0.005, t0_max - 0.005),
        ],
        dtype=np.float64,
    )

    p0 = init_clamped + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)

    samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    k_m, a_m, inc_m, t0_m = np.median(samples, axis=0)
    p16 = np.percentile(samples, 16, axis=0)
    p84 = np.percentile(samples, 84, axis=0)
    stds = np.std(samples, axis=0)

    transit_depth_m = float(k_m**2)
    transit_depth_err = float(2.0 * k_m * stds[0])
    duration_days_m = calculate_analytical_duration(period, k_m, a_m, inc_m)

    mcmc_results = {
        "k": {
            "median": float(k_m),
            "std": float(stds[0]),
            "p16": float(p16[0]),
            "p84": float(p84[0]),
            "err_minus": float(k_m - p16[0]),
            "err_plus": float(p84[0] - k_m),
        },
        "a_rstar": {
            "median": float(a_m),
            "std": float(stds[1]),
            "p16": float(p16[1]),
            "p84": float(p84[1]),
            "err_minus": float(a_m - p16[1]),
            "err_plus": float(p84[1] - a_m),
        },
        "inc_deg": {
            "median": float(inc_m),
            "std": float(stds[2]),
            "p16": float(p16[2]),
            "p84": float(p84[2]),
            "err_minus": float(inc_m - p16[2]),
            "err_plus": float(p84[2] - inc_m),
        },
        "t0_phase_days": {
            "median": float(t0_m),
            "std": float(stds[3]),
            "p16": float(p16[3]),
            "p84": float(p84[3]),
            "err_minus": float(t0_m - p16[3]),
            "err_plus": float(p84[3] - t0_m),
        },
        "transit_depth": transit_depth_m,
        "transit_depth_ppm": float(transit_depth_m * 1e6),
        "transit_depth_err_ppm": float(transit_depth_err * 1e6),
        "transit_duration_days": duration_days_m,
        "transit_duration_hours": float(duration_days_m * 24.0),
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "discard": discard,
        "thin": thin,
        "total_samples": len(samples),
    }

    samples_df = pd.DataFrame(
        samples,
        columns=["k", "a_rstar", "inc_deg", "t0"],
    )

    return mcmc_results, samples_df
