from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process.preprocessing import clean_observations, normalize_by_segment

OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def clean_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = clean_observations(df)
    normalized = normalize_by_segment(cleaned)
    return cleaned, normalized


def main() -> None:
    for id_directory in sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_dir()):
        input_path = id_directory / "raw_extracted.csv"
        if not input_path.is_file():
            continue

        print(f"Processing ID: {id_directory.name}")
        raw_df = pd.read_csv(input_path)
        cleaned_df, normalized_df = clean_and_normalize(raw_df)

        output_path = id_directory / "cleaned_time_flux.csv"
        normalized_df.to_csv(output_path, index=False)

        print(f"  Original: {len(raw_df)}  Valid: {len(normalized_df)}  Saved: {output_path}")


if __name__ == "__main__":
    main()
