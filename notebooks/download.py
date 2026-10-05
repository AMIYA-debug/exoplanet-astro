import csv
from pathlib import Path
import shutil
import tempfile
import time
import lightkurve as lk 

root = Path(__file__).resolve().parent.parent
csv_file = root / "data/raw/metadata/cumulative_2026.10.02_06.40.58_kepid_descending.csv"
fits_folder = root / "data/raw/fits"


def get_kepids(start_kepid, end_kepid):
    with csv_file.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return list(
            dict.fromkeys(
                int(row["kepid"])
                for row in rows
                if start_kepid <= int(row["kepid"]) <= end_kepid
            )
        )


def download_kepid(kepid, temp_folder):
    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")
    products = search[[str(author) == "Kepler" for author in search.table["author"]]]

    if len(products) == 0:
        print(f"No Kepler FITS found for {kepid}")
        return

    for index in range(len(products)):
        product = products[index]
        filename = str(products.table["dataURI"][index]).split("/")[-1]
        destination = fits_folder / filename

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already exists: {filename}")
            continue

        for attempt in range(1, 4):
            try:
                product.download(download_dir=str(temp_folder))
                downloaded = list(temp_folder.rglob(filename))

                if not downloaded or downloaded[0].stat().st_size == 0:
                    raise RuntimeError(f"Could not download {filename}")

                shutil.move(str(downloaded[0]), str(destination))
                print(f"Downloaded: {filename}")
                break
            except Exception as error:
                print(f"Failed {filename}, attempt {attempt}: {error}")
                for file in temp_folder.rglob(filename):
                    file.unlink()
                if attempt < 3:
                    time.sleep(attempt * 5)


start_kepid = int(input("Start KEPID: "))
end_kepid = int(input("End KEPID: "))

if start_kepid > end_kepid:
    raise ValueError("Start KEPID must be smaller than or equal to end KEPID")

fits_folder.mkdir(parents=True, exist_ok=True)
kepids = get_kepids(start_kepid, end_kepid)

if not kepids:
    print("No KEPIDs found in this range")
else:
    print(f"Downloading FITS for {len(kepids)} KEPIDs")
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_folder = Path(temp_folder)
        for kepid in kepids:
            print(f"KIC {kepid}")
            try:
                download_kepid(kepid, temp_folder)
            except Exception as error:
                print(f"Failed KIC {kepid}: {error}")
