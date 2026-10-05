"""Input/output and preprocessing for Kepler combined FITS files.

Implements quarter-by-quarter extraction, quality masking, and median
normalization to eliminate inter-quarter baseline offsets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from astropy.io import fits


def extract_raw_lightcurve(fits_path: Path | str) -> pd.DataFrame:
    """Read all LIGHTCURVE HDUs from a combined Kepler FITS file.

    Extracts TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR, SAP_QUALITY, and records
    the segment (quarter HDU) index.
    """
    fits_path = Path(fits_path)
    if not fits_path.is_file():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    records = []
    with fits.open(fits_path, memmap=False) as hdul:
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

            seg_df = pd.DataFrame(
                {
                    "TIME": time,
                    "PDCSAP_FLUX": flux,
                    "PDCSAP_FLUX_ERR": flux_err,
                    "SAP_QUALITY": quality,
                    "SEGMENT": segment_id,
                }
            )
            records.append(seg_df)

    if not records:
        raise ValueError(f"No LIGHTCURVE HDUs found in {fits_path}")

    return pd.concat(records, ignore_index=True)


def clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality flag filtering and remove non-finite observations.

    Reproduces the notebook's quality condition:
        q = raw_lc.quality == 0
    along with requiring finite time and flux.
    """
    valid_mask = (
        np.isfinite(df["TIME"])
        & np.isfinite(df["PDCSAP_FLUX"])
        & (df["SAP_QUALITY"] == 0)
    )
    return df.loc[valid_mask].copy().reset_index(drop=True)


def normalize_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize flux by segment/quarter median to eliminate baseline jumps.

    Matches notebook methodology:
        normalized_flux = flux / np.median(flux)
    applied per Kepler quarter/observing segment so that every segment
    shares an identical common baseline centered at 1.0.
    """
    df_out = df.copy()
    norm_flux = np.empty(len(df_out), dtype=np.float64)
    norm_err = np.empty(len(df_out), dtype=np.float64)

    for seg_id, group in df_out.groupby("SEGMENT"):
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

    # Sort strictly by time
    df_out = df_out.sort_values(by="TIME").reset_index(drop=True)
    return df_out


def preprocess_fits(
    fits_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete data extraction and preprocessing.

    Returns:
        (raw_df, cleaned_df, processed_df)
    """
    raw_df = extract_raw_lightcurve(fits_path)
    cleaned_df = clean_observations(raw_df)
    processed_df = normalize_by_segment(cleaned_df)
    return raw_df, cleaned_df, processed_df
