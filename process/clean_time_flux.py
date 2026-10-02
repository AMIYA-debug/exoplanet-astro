from pathlib import Path

import numpy as np
import pandas as pd


INPUT_ROOT = Path("/home/amiya/projects/astronitr/process/data")
COLUMNS = ["TIME", "PDCSAP_FLUX"]


def clean_file(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the input observations and the subset with finite required values."""
    observations = pd.read_csv(input_path, usecols=COLUMNS)
    valid_rows = np.isfinite(observations["TIME"]) & np.isfinite(
        observations["PDCSAP_FLUX"]
    )
    return observations, observations.loc[valid_rows, COLUMNS]


def main() -> None:
    for id_directory in sorted(path for path in INPUT_ROOT.iterdir() if path.is_dir()):
        input_path = id_directory / "time_flux.csv"
        if not input_path.is_file():
            continue

        print(f"Processing ID: {id_directory.name}")
        observations, cleaned_observations = clean_file(input_path)

        output_path = id_directory / "cleaned_time_flux.csv"
        cleaned_observations.to_csv(output_path, index=False)

        print(f"Original observations: {len(observations)}")
        print(f"Valid observations: {len(cleaned_observations)}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
