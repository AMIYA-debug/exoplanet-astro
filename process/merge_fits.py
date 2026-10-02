from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits


def merge_directory(directory: Path, output_name: str = "combined.fits") -> bool:

    output_path = directory / output_name
    source_paths = sorted(
        path
        for path in directory.glob("*.fits")
        if path.name != output_name
    )

    if not source_paths:
        return False

    combined_hdus = []
    for index, source_path in enumerate(source_paths):
        with fits.open(source_path, memmap=False) as source_hdus:
            if index == 0:
                combined_hdus.append(source_hdus[0].copy())
            combined_hdus.extend(hdu.copy() for hdu in source_hdus[1:])

    fits.HDUList(combined_hdus).writeto(output_path, overwrite=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge FITS files within each immediate ID directory."
    )
    parser.add_argument(
        "fits_directory",
        nargs="?",
        type=Path,
        default=Path("/home/amiya/projects/astronitr/data/raw/fits"),
        help=(
            "Directory containing ID subdirectories "
            "(default: /home/amiya/projects/astronitr/data/raw/fits)"
        ),
    )
    args = parser.parse_args()

    for id_directory in sorted(path for path in args.fits_directory.iterdir() if path.is_dir()):
        if merge_directory(id_directory):
            print(f"Wrote {id_directory / 'combined.fits'}")


if __name__ == "__main__":
    main()
