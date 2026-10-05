from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits


def extract_raw_lightcurve(fits_path: Path | str) -> pd.DataFrame:
    fits_file = Path(fits_path)
    if not fits_file.is_file():
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    records: list[pd.DataFrame] = []
    with fits.open(fits_file, memmap=False) as hdul:
        segment_id = 0
        for hdu in hdul:
            if hdu.name != "LIGHTCURVE" or hdu.data is None:
                continue

            segment_id += 1
            data = hdu.data
            names = data.names if hasattr(data, "names") else []

            time = data["TIME"] if "TIME" in names else np.full(len(data), np.nan)
            flux = data["PDCSAP_FLUX"] if "PDCSAP_FLUX" in names else np.full(len(data), np.nan)
            flux_err = (
                data["PDCSAP_FLUX_ERR"]
                if "PDCSAP_FLUX_ERR" in names
                else np.full(len(data), np.nan)
            )
            quality = (
                data["SAP_QUALITY"]
                if "SAP_QUALITY" in names
                else np.zeros(len(data), dtype=int)
            )

            records.append(
                pd.DataFrame(
                    {
                        "TIME": time,
                        "PDCSAP_FLUX": flux,
                        "PDCSAP_FLUX_ERR": flux_err,
                        "SAP_QUALITY": quality,
                        "SEGMENT": segment_id,
                    }
                )
            )

    if not records:
        raise ValueError(f"No LIGHTCURVE HDUs found in {fits_file}")

    return pd.concat(records, ignore_index=True)


def clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = (
        np.isfinite(df["TIME"])
        & np.isfinite(df["PDCSAP_FLUX"])
        & (df["SAP_QUALITY"] == 0)
    )
    return df.loc[valid_mask].copy().reset_index(drop=True)


def normalize_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    norm_flux = np.empty(len(df_out), dtype=np.float64)
    norm_err = np.empty(len(df_out), dtype=np.float64)

    for _, group in df_out.groupby("SEGMENT"):
        idx = group.index
        seg_flux = group["PDCSAP_FLUX"].to_numpy(dtype=np.float64)
        seg_err = group["PDCSAP_FLUX_ERR"].to_numpy(dtype=np.float64)

        med = np.nanmedian(seg_flux)
        if med > 0:
            norm_flux[idx] = seg_flux / med
            norm_err[idx] = seg_err / med
        else:
            norm_flux[idx] = seg_flux
            norm_err[idx] = seg_err

    df_out["NORMALIZED_FLUX"] = norm_flux
    df_out["NORMALIZED_FLUX_ERR"] = norm_err
    return df_out.sort_values(by="TIME").reset_index(drop=True)


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    bin_width: float = 0.002,
    window: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_arr = np.asarray(time, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)

    x_fold = (time_arr - t0 + 0.5 * period) % period - 0.5 * period
    y_flux = flux_arr / np.median(flux_arr)

    folded_df = pd.DataFrame(
        {
            "phase_days": x_fold,
            "normalized_flux": y_flux,
        }
    )

    bins = np.arange(-window, window + bin_width, bin_width)
    t_centers = 0.5 * (bins[1:] + bins[:-1])

    t_binned: list[float] = []
    f_binned: list[float] = []
    err_binned: list[float] = []
    count_binned: list[int] = []

    for i in range(len(bins) - 1):
        m = (x_fold >= bins[i]) & (x_fold < bins[i + 1])
        if np.any(m):
            t_binned.append(float(t_centers[i]))
            f_binned.append(float(np.median(y_flux[m])))
            err_binned.append(
                float(np.std(y_flux[m]) / np.sqrt(np.sum(m)))
                if np.sum(m) > 1
                else float(np.std(y_flux))
            )
            count_binned.append(int(np.sum(m)))

    t_fit = np.array(t_binned, dtype=np.float64)
    f_fit = np.array(f_binned, dtype=np.float64)
    err_fit = np.array(err_binned, dtype=np.float64)
    counts = np.array(count_binned, dtype=int)

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


def preprocess_lightcurve(
    fits_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = extract_raw_lightcurve(fits_path)
    cleaned_df = clean_observations(raw_df)
    processed_df = normalize_by_segment(cleaned_df)
    return raw_df, cleaned_df, processed_df
