from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import time

from astropy.io import fits
import lightkurve as lk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.preprocessing import extract_raw_lightcurve
from process_fits.getfitsdata import analyze_time_flux


def download_kic_fits(kic_id: str | int, target_dir: Path | str, max_retries: int = 3) -> list[Path]:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    search = lk.search_lightcurve(f"KIC {kic_id}", mission="Kepler")
    if len(search) == 0 or search.table is None:
        return []

    search_table = search.table
    is_kepler = [str(author).strip() == "Kepler" for author in search_table["author"]]
    products = search[is_kepler]

    if len(products) == 0 or products.table is None:
        return []

    product_table = products.table
    downloaded_paths: list[Path] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for index in range(len(products)):
            product = products[index]
            uri = str(product_table["dataURI"][index])
            filename = uri.split("/")[-1]
            destination = target_path / filename

            if destination.exists() and destination.is_file() and destination.stat().st_size > 0:
                downloaded_paths.append(destination)
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    product.download(download_dir=str(temp_path))
                    candidates = list(temp_path.rglob(filename))
                    if not candidates:
                        raise RuntimeError(f"Download produced no file: {filename}")
                    downloaded_file = candidates[0]
                    if downloaded_file.stat().st_size == 0:
                        raise RuntimeError(f"Download produced empty file: {filename}")
                    shutil.move(str(downloaded_file), str(destination))
                    downloaded_paths.append(destination)
                    break
                except Exception as error:
                    for residual in temp_path.rglob(filename):
                        if residual.exists():
                            residual.unlink()
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Failed to download {filename} after {max_retries} attempts: {error}"
                        ) from error
                    time.sleep(attempt * 5)

    return downloaded_paths


def merge_kic_fits(directory: Path | str, output_name: str = "combined.fits") -> bool:
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
        return True

    except Exception as error:
        if temp_output.exists():
            temp_output.unlink()
        raise RuntimeError(f"Merging failed for {kic_dir}: {error}") from error


def process_kic(kic_id: str | int) -> None:
    kic_str = str(kic_id).strip()
    if not kic_str:
        raise ValueError("Invalid KIC ID provided.")

    kic_dir = Path(__file__).resolve().parent / kic_str
    kic_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FITS products for KIC {kic_str} into {kic_dir}...")
    downloaded = download_kic_fits(kic_str, kic_dir)
    if not downloaded:
        raise RuntimeError(f"No Kepler FITS products found for KIC {kic_str}.")

    print(f"Saved {len(downloaded)} FITS files for KIC {kic_str}.")

    combined_file = kic_dir / "combined.fits"
    print(f"Merging FITS into {combined_file}...")
    merge_success = merge_kic_fits(kic_dir)
    if not merge_success or not combined_file.is_file():
        raise RuntimeError(f"FITS merge failed for KIC {kic_str}.")

    print(f"Extracting TIME and PDCSAP_FLUX from {combined_file}...")
    raw_df = extract_raw_lightcurve(combined_file)
    extracted_csv = kic_dir / "extracted_time_flux.csv"
    raw_df[["TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_QUALITY", "SEGMENT"]].to_csv(
        extracted_csv, index=False
    )

    print(f"Running analysis pipeline on {extracted_csv}...")
    analyze_time_flux(
        csv_path=extracted_csv,
        output_name=kic_str,
        output_dir=kic_dir,
    )
    print(f"Completed processing for KIC {kic_str}. All outputs saved to {kic_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and analyze Kepler FITS for a KIC ID.")
    parser.add_argument("kic_id", type=str, help="Kepler Input Catalog (KIC) ID.")
    args = parser.parse_args()

    process_kic(args.kic_id)


if __name__ == "__main__":
    main()
