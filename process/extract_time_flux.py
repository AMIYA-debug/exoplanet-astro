from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import pandas as pd


FITS_ROOT = Path("/home/amiya/projects/astronitr/data/raw/fits")
OUTPUT_ROOT = Path("/home/amiya/projects/astronitr/process/data")
COLUMNS = ["TIME", "PDCSAP_FLUX"]


def extract_time_flux(fits_path: Path) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []

    with fits.open(fits_path, memmap=False) as hdus:
        for hdu in hdus:
            if hdu.data is None or not hasattr(hdu.data, "names"):
                continue

            column_names = hdu.data.names
            if column_names is not None and all(column in column_names for column in COLUMNS):
                tables.append(pd.DataFrame({column: hdu.data[column] for column in COLUMNS}))

    if not tables:
        raise ValueError(f"No table HDU with {COLUMNS} found in {fits_path}")

    return pd.concat(tables, ignore_index=True)


def main() -> None:
    for id_directory in sorted(path for path in FITS_ROOT.iterdir() if path.is_dir()):
        fits_path = id_directory / "combined.fits"
        if not fits_path.is_file():
            continue

        output_directory = OUTPUT_ROOT / id_directory.name
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / "time_flux.csv"
        extract_time_flux(fits_path).to_csv(output_path, index=False, columns=COLUMNS)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
