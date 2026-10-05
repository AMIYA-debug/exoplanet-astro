"""Clean and normalize raw observations.

Applies:
1. Quality filtering: SAP_QUALITY == 0 (notebook methodology)
2. Non-finite removal: finite TIME and PDCSAP_FLUX
3. Quarter-by-quarter median normalization: eliminates inter-quarter baseline offsets
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.io import clean_observations, normalize_by_segment


INPUT_ROOT = Path("/home/amiya/projects/astronitr/process/data")


def clean_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean observations and apply quarter-by-quarter median normalization."""
    cleaned = clean_observations(df)
    normalized = normalize_by_segment(cleaned)
    return cleaned, normalized


def main() -> None:
    for id_directory in sorted(path for path in INPUT_ROOT.iterdir() if path.is_dir()):
        input_path = id_directory / "time_flux.csv"
        if not input_path.is_file():
            continue

        print(f"Processing ID: {id_directory.name}")
        raw_df = pd.read_csv(input_path)
        cleaned_df, normalized_df = clean_and_normalize(raw_df)

        output_path = id_directory / "cleaned_time_flux.csv"
        normalized_df.to_csv(output_path, index=False)

        print(f"Original observations: {len(raw_df)}")
        print(f"Valid observations:    {len(normalized_df)}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
