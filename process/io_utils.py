from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from astropy.io import fits


def extract_kic_id(fits_path: Path | str, fallback: str = "unknown") -> str:
    target_path = Path(fits_path)
    try:
        with fits.open(target_path, memmap=False) as hdul:
            for hdu in hdul:
                header = hdu.header
                if "KEPLERID" in header and header["KEPLERID"]:
                    return str(int(header["KEPLERID"]))
                if "OBJECT" in header and header["OBJECT"]:
                    obj_str = str(header["OBJECT"])
                    match = re.search(r"KIC\s*(\d+)", obj_str, re.IGNORECASE)
                    if match:
                        return str(int(match.group(1)))
    except Exception:
        pass

    match = re.search(r"kplr0*(\d+)", target_path.name, re.IGNORECASE)
    if match:
        return str(int(match.group(1)))

    parent_name = target_path.parent.name
    if parent_name.isdigit():
        return parent_name

    return fallback


def save_json(data: dict[str, Any], filepath: Path | str, indent: int = 2) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Path | str) -> dict[str, Any]:
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path | str) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
