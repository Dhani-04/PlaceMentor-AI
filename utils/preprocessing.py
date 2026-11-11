"""Data preparation utilities for PlaceMentor AI.

Everything that touches raw data lives here so that training and prediction
share exactly the same feature list, ranges and validation rules.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Feature definitions (single source of truth for the whole project)
# --------------------------------------------------------------------------
FEATURES = [
    "CGPA",
    "Aptitude",
    "Coding",
    "Communication",
    "Projects",
    "Internships",
    "Attendance",
]

# Valid (min, max) for every feature.
FEATURE_RANGES = {
    "CGPA": (0.0, 10.0),
    "Aptitude": (0.0, 100.0),
    "Coding": (0.0, 100.0),
    "Communication": (0.0, 100.0),
    "Projects": (0, 10),
    "Internships": (0, 5),
    "Attendance": (0.0, 100.0),
}

INTEGER_FEATURES = ("Projects", "Internships")

FEATURE_LABELS = {
    "CGPA": "CGPA (0 to 10)",
    "Aptitude": "Aptitude score (0 to 100)",
    "Coding": "Coding score (0 to 100)",
    "Communication": "Communication score (0 to 100)",
    "Projects": "Completed projects (0 to 10)",
    "Internships": "Internships (0 to 5)",
    "Attendance": "Attendance percentage (0 to 100)",
}

# --------------------------------------------------------------------------
# Simulated dataset generation
# --------------------------------------------------------------------------
# Three loosely defined student archetypes. The generator only creates
# overlapping, noisy groups; K-Means is never told which student belongs where.
_GROUP_SPECS = [
    {
        "n": 62,
        "CGPA": (8.7, 0.45), "Aptitude": (80, 7.5), "Coding": (81, 8.0),
        "Communication": (76, 8.0), "Projects": (5.4, 1.4),
        "Internships": (2.0, 0.8), "Attendance": (91, 4.5),
    },
    {
        "n": 68,
        "CGPA": (7.5, 0.50), "Aptitude": (63, 8.5), "Coding": (60, 9.5),
        "Communication": (62, 8.5), "Projects": (3.0, 1.1),
        "Internships": (1.0, 0.75), "Attendance": (82, 6.0),
    },
    {
        "n": 50,
        "CGPA": (6.3, 0.55), "Aptitude": (46, 9.0), "Coding": (38, 10.0),
        "Communication": (47, 9.0), "Projects": (1.0, 0.9),
        "Internships": (0.25, 0.5), "Attendance": (71, 8.0),
    },
]


def generate_simulated_dataset(seed: int = 42) -> pd.DataFrame:
    """Create a reproducible SIMULATED student dataset (180 rows).

    The data is artificial and is NOT real institutional placement data.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for spec in _GROUP_SPECS:
        n = spec["n"]
        # A shared "ability" factor makes academic and skill scores partly
        # correlated, as they usually are in real cohorts.
        shared = rng.normal(0.0, 1.0, n)
        data = {}
        for feat in ("CGPA", "Aptitude", "Coding", "Communication"):
            mean, sd = spec[feat]
            noise = 0.5 * shared + np.sqrt(0.75) * rng.normal(0.0, 1.0, n)
            data[feat] = mean + sd * noise
        for feat in ("Projects", "Internships", "Attendance"):
            mean, sd = spec[feat]
            data[feat] = rng.normal(mean, sd, n)
        frames.append(pd.DataFrame(data))

    df = pd.concat(frames, ignore_index=True)

    df["CGPA"] = df["CGPA"].clip(4.5, 10.0).round(2)
    for feat in ("Aptitude", "Coding", "Communication"):
        df[feat] = df[feat].clip(10, 100).round(1)
    df["Projects"] = np.rint(df["Projects"]).clip(0, 10).astype(int)
    df["Internships"] = np.rint(df["Internships"]).clip(0, 5).astype(int)
    df["Attendance"] = df["Attendance"].clip(45, 100).round(1)

    # Shuffle so the file does not reveal the generating groups.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df[FEATURES]


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Loading and validating the dataset
# --------------------------------------------------------------------------
def load_and_validate(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the CSV, check columns, fill missing values, clip to valid ranges.

    Returns the cleaned feature table and a small report dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    raw = pd.read_csv(path)
    missing_cols = [c for c in FEATURES if c not in raw.columns]
    if missing_cols:
        raise ValueError(
            "The dataset is missing required column(s): " + ", ".join(missing_cols)
        )

    df = raw[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)

    missing_filled = int(df.isna().sum().sum())
    if missing_filled:
        df = df.fillna(df.median(numeric_only=True))

    clipped = 0
    for feat, (lo, hi) in FEATURE_RANGES.items():
        out_of_range = ((df[feat] < lo) | (df[feat] > hi)).sum()
        clipped += int(out_of_range)
        df[feat] = df[feat].clip(lo, hi)

    for feat in INTEGER_FEATURES:
        df[feat] = df[feat].round().astype(int)

    if len(df) < 10:
        raise ValueError("The dataset needs at least 10 student records.")

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values_filled": missing_filled,
        "values_clipped": clipped,
    }
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Validating user input from the assessment form
# --------------------------------------------------------------------------
def validate_user_input(values: dict) -> list[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    errors: list[str] = []
    for feat in FEATURES:
        lo, hi = FEATURE_RANGES[feat]
        raw = values.get(feat)
        if raw is None or raw == "":
            errors.append(f"{feat} is required.")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{feat} must be a number.")
            continue
        if math.isnan(number) or math.isinf(number):
            errors.append(f"{feat} must be a valid number.")
            continue
        if not lo <= number <= hi:
            errors.append(f"{feat} must be between {lo:g} and {hi:g}.")
            continue
        if feat in INTEGER_FEATURES and not float(number).is_integer():
            errors.append(f"{feat} must be a whole number.")
    return errors


def build_input_frame(values: dict) -> pd.DataFrame:
    """Turn a dictionary of user inputs into a one-row DataFrame.

    Columns are ordered exactly like the training data.
    """
    row = {feat: float(values[feat]) for feat in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)
