"""Extract raw time-series photometry from combined Kepler FITS files.

Extracts TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR, SAP_QUALITY, and records the
segment index for each quarter.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.io import extract_raw_lightcurve


FITS_ROOT = Path("/home/amiya/projects/astronitr/data/raw/fits")
OUTPUT_ROOT = Path("/home/amiya/projects/astronitr/process/data")


def main() -> None:
    for id_directory in sorted(path for path in FITS_ROOT.iterdir() if path.is_dir()):
        fits_path = id_directory / "combined.fits"
        if not fits_path.is_file():
            continue

        output_directory = OUTPUT_ROOT / id_directory.name
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / "time_flux.csv"
        df = extract_raw_lightcurve(fits_path)
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
