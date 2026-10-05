from pathlib import Path
import re
import shutil


fits_folder = Path(__file__).resolve().parent.parent / "data/raw/fits"

for file in fits_folder.iterdir():
    if not file.is_file() or file.suffix.lower() not in {".fits", ".fit"}:
        continue
    
    match = re.match(r"kplr0*(\d+)-", file.name, re.IGNORECASE)
    if match is None:
        continue

    kepid = str(int(match.group(1)))
    target_folder = fits_folder / kepid
    target_file = target_folder / file.name

    if target_file.exists():
        print(f"Skipped: {file.name}")
        continue

    target_folder.mkdir(exist_ok=True)
    shutil.move(str(file), str(target_file))
    print(f"Moved: {file.name} -> {kepid}/")
