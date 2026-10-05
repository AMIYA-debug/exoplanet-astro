from __future__ import annotations

from typing import Any
import pandas as pd


def extract_candidate_features(
    kic: str,
    candidate_rank: int,
    bls_params: dict[str, Any],
    fit_params: dict[str, Any] | None = None,
    mcmc_params: dict[str, Any] | None = None,
    obs_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "kic": str(kic),
        "candidate_rank": int(candidate_rank),
        "bls_period_days": float(bls_params.get("BEST_PERIOD", 0.0)),
        "bls_transit_time_bkjd": float(bls_params.get("BEST_TRANSIT_TIME", 0.0)),
        "bls_duration_days": float(bls_params.get("BEST_DURATION", 0.0)),
        "bls_duration_hours": float(bls_params.get("BEST_DURATION", 0.0)) * 24.0,
        "bls_power": float(bls_params.get("BEST_POWER", 0.0)),
        "bls_depth": float(bls_params.get("BEST_DEPTH", 0.0)),
        "bls_depth_ppm": float(bls_params.get("BEST_DEPTH", 0.0)) * 1e6,
        "bls_depth_snr": float(bls_params.get("BEST_DEPTH_SNR", float("nan"))),
    }

    if fit_params:
        features.update(
            {
                "fit_rp_rstar": float(fit_params.get("k_rp_rstar", float("nan"))),
                "fit_rp_rstar_err": float(fit_params.get("k_err", float("nan"))),
                "fit_a_rstar": float(fit_params.get("a_rstar", float("nan"))),
                "fit_a_rstar_err": float(fit_params.get("a_err", float("nan"))),
                "fit_inc_deg": float(fit_params.get("inc_deg", float("nan"))),
                "fit_inc_err": float(fit_params.get("inc_err", float("nan"))),
                "fit_t0_phase_days": float(fit_params.get("t0_days", float("nan"))),
                "fit_t0_err": float(fit_params.get("t0_err", float("nan"))),
                "fit_depth_ppm": float(fit_params.get("transit_depth_ppm", float("nan"))),
                "fit_duration_hours": float(fit_params.get("transit_duration_hours", float("nan"))),
                "fit_rms_residuals_ppm": float(fit_params.get("rms_residuals", 0.0)) * 1e6,
            }
        )

    if mcmc_params and "k" in mcmc_params:
        k_dict = mcmc_params.get("k", {})
        a_dict = mcmc_params.get("a_rstar", {})
        inc_dict = mcmc_params.get("inc_deg", {})
        t0_dict = mcmc_params.get("t0_phase_days", {})
        features.update(
            {
                "mcmc_rp_rstar_median": float(k_dict.get("median", float("nan"))),
                "mcmc_rp_rstar_std": float(k_dict.get("std", float("nan"))),
                "mcmc_rp_rstar_err_minus": float(k_dict.get("err_minus", float("nan"))),
                "mcmc_rp_rstar_err_plus": float(k_dict.get("err_plus", float("nan"))),
                "mcmc_a_rstar_median": float(a_dict.get("median", float("nan"))),
                "mcmc_a_rstar_std": float(a_dict.get("std", float("nan"))),
                "mcmc_inc_deg_median": float(inc_dict.get("median", float("nan"))),
                "mcmc_inc_deg_std": float(inc_dict.get("std", float("nan"))),
                "mcmc_t0_phase_median": float(t0_dict.get("median", float("nan"))),
                "mcmc_t0_phase_std": float(t0_dict.get("std", float("nan"))),
                "mcmc_depth_ppm": float(mcmc_params.get("transit_depth_ppm", float("nan"))),
                "mcmc_depth_err_ppm": float(mcmc_params.get("transit_depth_err_ppm", float("nan"))),
                "mcmc_duration_hours": float(mcmc_params.get("transit_duration_hours", float("nan"))),
            }
        )

    if obs_stats:
        raw_cnt = int(obs_stats.get("raw", 0))
        clean_cnt = int(obs_stats.get("cleaned", 0))
        proc_cnt = int(obs_stats.get("processed", 0))
        t_span = float(obs_stats.get("time_span_days", float("nan")))
        period = float(features.get("bls_period_days", 1.0))
        duration_d = float(features.get("bls_duration_days", 0.0))

        features.update(
            {
                "obs_raw_count": raw_cnt,
                "obs_cleaned_count": clean_cnt,
                "obs_processed_count": proc_cnt,
                "obs_clean_ratio": (clean_cnt / raw_cnt) if raw_cnt > 0 else float("nan"),
                "obs_time_span_days": t_span,
                "transit_duty_cycle": (duration_d / period) if period > 0 else float("nan"),
            }
        )

    return features


def features_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
