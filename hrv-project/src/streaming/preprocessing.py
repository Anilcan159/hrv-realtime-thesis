# src/streaming/preprocessing.py
# Converts raw PhysioNet RR txt files into per-subject clean CSV files.

from pathlib import Path
import pandas as pd


# Root paths
# __file__ = .../hrv-project/src/streaming/preprocessing.py
BASE_DIR = Path(__file__).resolve().parents[2]  # -> .../hrv-project
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "rr-interval-healthy-subjects-1.0.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "rr_clean"


def load_rr_series(subject_id: str) -> pd.DataFrame:
    """
    Loads raw RR series (ms) for a given subject as a DataFrame.
    Expects a file like '001.txt' in RAW_DATA_DIR.
    """
    file_path = RAW_DATA_DIR / f"{subject_id}.txt"
    df = pd.read_csv(file_path, header=None, names=["rr_ms"])
    return df


def clean_rr_values(
    df: pd.DataFrame,
    rr_min: int = 300,
    rr_max: int = 2000,
) -> pd.DataFrame:
    """
    Basic RR cleaning: keep only values in [rr_min, rr_max] ms.
    """
    df = df.copy()
    df = df[(df["rr_ms"] >= rr_min) & (df["rr_ms"] <= rr_max)]
    df = df.dropna().reset_index(drop=True)
    return df


def add_time_and_hr(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    Adds cumulative time (s), heart rate (bpm) and subject_id columns.
    """
    df = df.copy()
    # cumulative time in seconds
    df["time_s"] = df["rr_ms"].cumsum() / 1000.0
    # heart rate in bpm (optional, can be useful later)
    df["hr_bpm"] = 60000.0 / df["rr_ms"]
    # subject id
    df["subject_id"] = subject_id
    return df


def process_single_subject(subject_id: str) -> None:
    """
    Full pipeline for a single subject: txt -> cleaned CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_rr_series(subject_id)
    df_clean = clean_rr_values(df_raw)
    df_final = add_time_and_hr(df_clean, subject_id)

    out_path = PROCESSED_DIR / f"{subject_id}_clean.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def process_all_subjects() -> None:
    """
    Loops over all txt files in RAW_DATA_DIR and processes each one.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} raw txt files.")

    for f in txt_files:
        subject_id = f.stem  # "001", "002", ...
        print(f"Processing subject {subject_id} ...")
        process_single_subject(subject_id)


if __name__ == "__main__":
    # When you run this file as a script, process all subjects
    process_all_subjects()
