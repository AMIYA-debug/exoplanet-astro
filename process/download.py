from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import tempfile
import time

import lightkurve as lk

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_METADATA_CSV = (
    PROJECT_ROOT
    / "data/raw/metadata/cumulative_2026.10.02_06.40.58_kepid_descending.csv"
)

DEFAULT_FITS_ROOT = PROJECT_ROOT / "data/raw/fits"


def get_kepids_by_row_range(
    csv_path: Path | str,
    start_row: int,
    end_row: int,
) -> list[int]:
    if start_row < 2 or end_row < 2:
        raise ValueError("Row numbers must be at least 2")

    if start_row > end_row:
        raise ValueError(
            "Start row must be less than or equal to end row"
        )

    csv_file = Path(csv_path)

    if not csv_file.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_file}"
        )

    kepids: list[int] = []
    seen: set[int] = set()

    with csv_file.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for csv_row, row in enumerate(reader, start=2):
            if csv_row < start_row:
                continue

            if csv_row > end_row:
                break

            raw = row.get("kepid", "").strip()

            if not raw:
                continue

            kepid = int(raw)

            if kepid not in seen:
                seen.add(kepid)
                kepids.append(kepid)

    return kepids


def download_kepid_fits(
    kepid: int,
    fits_root: Path | str,
    max_retries: int = 3,
) -> list[Path]:
    target_dir = Path(fits_root) / str(kepid)
    target_dir.mkdir(parents=True, exist_ok=True)

    search = lk.search_lightcurve(
        f"KIC {kepid}",
        mission="Kepler",
    )

    if len(search) == 0 or search.table is None:
        return []

    search_table = search.table

    is_kepler = [
        str(author).strip() == "Kepler"
        for author in search_table["author"]
    ]

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
            destination = target_dir / filename

            if (
                destination.exists()
                and destination.is_file()
                and destination.stat().st_size > 0
            ):
                downloaded_paths.append(destination)
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    product.download(
                        download_dir=str(temp_path)
                    )

                    candidates = list(
                        temp_path.rglob(filename)
                    )

                    if not candidates:
                        raise RuntimeError(
                            f"Download produced no file: {filename}"
                        )

                    downloaded_file = candidates[0]

                    if downloaded_file.stat().st_size == 0:
                        raise RuntimeError(
                            f"Download produced empty file: {filename}"
                        )

                    shutil.move(
                        str(downloaded_file),
                        str(destination),
                    )

                    downloaded_paths.append(destination)
                    break

                except Exception as error:
                    for residual in temp_path.rglob(filename):
                        if residual.exists():
                            residual.unlink()

                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Failed to download {filename} after "
                            f"{max_retries} attempts: {error}"
                        ) from error

                    time.sleep(attempt * 5)

    return downloaded_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Kepler FITS products using inclusive "
            "metadata CSV row numbers."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="First CSV row to process, including the header as row 1.",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Last CSV row to process, inclusive.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
    )

    parser.add_argument(
        "--fits-root",
        type=Path,
        default=DEFAULT_FITS_ROOT,
    )

    args = parser.parse_args()

    if args.start < 2:
        parser.error("--start must be 2 or greater.")

    if args.end < 2:
        parser.error("--end must be 2 or greater.")

    if args.start > args.end:
        parser.error(
            "--start must be less than or equal to --end."
        )

    try:
        target_kepids = get_kepids_by_row_range(
            args.csv,
            args.start,
            args.end,
        )
    except (FileNotFoundError, ValueError, csv.Error) as error:
        parser.error(str(error))

    print(
        f"CSV rows: {args.start} to {args.end}"
    )
    print(
        f"Target KICs: {len(target_kepids)}"
    )

    for kepid in target_kepids:
        print(
            f"Downloading KIC {kepid}..."
        )

        try:
            downloaded = download_kepid_fits(
                kepid,
                args.fits_root,
            )

            print(
                f"  Saved {len(downloaded)} files"
            )

        except Exception as error:
            print(
                f"  Error: {error}"
            )


if __name__ == "__main__":
    main()
