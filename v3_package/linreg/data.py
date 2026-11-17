"""Loading the data - and checking it is what we think it is."""
from pathlib import Path

import numpy as np

# The default dataset sits in <project root>/data/. Building the path from
# the location of THIS file means the program works no matter which folder
# you run it from (v1 and v2 only work from the project root).
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "study_hours.csv"


def load_data(path=DEFAULT_DATA):
    """Read a CSV with a header row and two columns: hours, score.

    Returns two 1-D arrays, x (hours) and y (score).
    Raises a clear error if the file is missing or looks wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no data file at {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)

    if data.shape[1] != 2:
        raise ValueError(
            f"{path.name}: expected 2 columns (hours, score), "
            f"found {data.shape[1]}"
        )
    if len(data) < 2:
        raise ValueError(f"{path.name}: need at least 2 rows to fit a line")
    if np.isnan(data).any():
        raise ValueError(f"{path.name}: contains missing values")

    return data[:, 0], data[:, 1]
