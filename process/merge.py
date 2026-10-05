from __future__ import annotations

import argparse
from pathlib import Path
from astropy.io import fits


def merge_kic_directory(directory: Path | str, output_name: str = "combined.fits") -> bool:
    kic_dir = Path(directory)
    if not kic_dir.is_dir():
        return False

    output_path = kic_dir / output_name
    source_paths = sorted(
        path
        for path in kic_dir.glob("*.fit*")
        if path.name != output_name and not path.name.endswith(".tmp")
    )

    if not source_paths:
        return output_path.is_file() and output_path.stat().st_size > 0

    combined_hdus: list[fits.Header | fits.PrimaryHDU | fits.ImageHDU | fits.BinTableHDU] = []
    temp_output = kic_dir / f"{output_name}.tmp"

    try:
        for index, source_path in enumerate(source_paths):
            with fits.open(source_path, memmap=False) as source_hdus:
                if index == 0:
                    combined_hdus.append(source_hdus[0].copy())
                combined_hdus.extend(hdu.copy() for hdu in source_hdus[1:])

        fits.HDUList(combined_hdus).writeto(temp_output, overwrite=True)

        if not temp_output.is_file() or temp_output.stat().st_size == 0:
            if temp_output.exists():
                temp_output.unlink()
            return False

        temp_output.replace(output_path)

        for source_path in source_paths:
            try:
                source_path.unlink()
            except OSError:
                pass

        return True

    except Exception as error:
        if temp_output.exists():
            temp_output.unlink()
        raise RuntimeError(f"Merging failed for {kic_dir}: {error}") from error


def merge_all_kics(fits_root: Path | str, output_name: str = "combined.fits") -> dict[str, bool]:
    root_path = Path(fits_root)
    results: dict[str, bool] = {}
    for id_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        success = merge_kic_directory(id_dir, output_name=output_name)
        results[id_dir.name] = success
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge individual FITS files into combined.fits.")
    parser.add_argument(
        "fits_directory",
        nargs="?",
        type=Path,
        default=Path("data/raw/fits"),
        help="Root directory containing KIC subdirectories.",
    )
    args = parser.parse_args()

    results = merge_all_kics(args.fits_directory)
    for kic_name, success in results.items():
        status = "Merged & Cleaned" if success else "Skipped/Failed"
        print(f"KIC {kic_name}: {status}")


if __name__ == "__main__":
    main()
