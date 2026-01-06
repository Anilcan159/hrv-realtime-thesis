from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent
TFM_PATH = ROOT / "Datas" / "raw" / "TFM.csv"
OUT_DIR = ROOT / "data" / "processed" / "rr_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = ROOT / "data" / "processed" / "patient-anon-map.csv"


def find_hr_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that look like HR (bpm) signals."""
    hr_cols = []
    for col in df.columns:
        col_str = str(col)
        if "HeartRate" in col_str or "HRM" in col_str:
            hr_cols.append(col)
    return hr_cols


def find_time_column(df: pd.DataFrame) -> str:
    """Find the TIME-like column (case-insensitive, strip spaces)."""
    for col in df.columns:
        if "TIME" in str(col).upper().strip():
            return col
    raise ValueError(f"No TIME-like column found. Columns: {list(df.columns)}")


def parse_time_to_seconds(s: str) -> float:
    """Parse TIME like '06:29:23 132' -> seconds from midnight."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    parts = s.split()
    hms = parts[0]
    ms = 0.0
    if len(parts) > 1:
        # '132' -> 0.132 s varsayımı
        try:
            ms = float(parts[1]) / 1000.0
        except ValueError:
            ms = 0.0

    try:
        h, m, sec = hms.split(":")
        h = float(h)
        m = float(m)
        sec = float(sec.replace(",", "."))
    except Exception:
        return np.nan

    return h * 3600.0 + m * 60.0 + sec + ms


def get_time_axis(df: pd.DataFrame) -> np.ndarray:
    """Extract time axis in seconds (0-based) from TIME column."""
    time_col = find_time_column(df)
    t = df[time_col].apply(parse_time_to_seconds).to_numpy(dtype=float)

    mask = np.isfinite(t)
    if not np.any(mask):
        raise ValueError("No valid TIME values.")
    t = t[mask]
    t = t - t[0]  # start at 0
    return t


def clean_hr_series(hr: pd.Series) -> np.ndarray:
    """Basic HR cleaning, return hr_bpm array."""
    s = hr.astype(str).str.replace(",", ".", regex=False)
    h = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    # Physiologic HR range
    mask = (h >= 30.0) & (h <= 220.0)
    h = h[mask]
    return h


def reconstruct_rr_from_hr(time_s: np.ndarray, hr_bpm: np.ndarray) -> pd.Series:
    """
    Reconstruct RR intervals (seconds) from HR time series.

    Idea:
      - Instantaneous frequency f(t) = HR(t) / 60  [beats/s]
      - Cumulative beats N(t) = integral f(t) dt
      - Beat n occurs when N(t) crosses integer n
      - RR_k = t_k - t_{k-1}
    """
    t = np.asarray(time_s, dtype=float)
    h = np.asarray(hr_bpm, dtype=float)

    # joint finite mask
    mask = np.isfinite(t) & np.isfinite(h)
    t = t[mask]
    h = h[mask]

    if t.size < 2:
        return pd.Series(dtype=float, name="rr_sec")

    # Instantaneous frequency in Hz
    f = h / 60.0

    # dt between samples
    dt = np.diff(t)
    valid_dt = dt > 0
    if not np.all(valid_dt):
        t = t[np.concatenate(([True], valid_dt))]
        f = f[np.concatenate(([True], valid_dt))]
        dt = np.diff(t)

    if t.size < 2:
        return pd.Series(dtype=float, name="rr_sec")

    # cumulative beats via trapezoidal rule
    beats_seg = 0.5 * (f[:-1] + f[1:]) * dt
    N = np.concatenate(([0.0], np.cumsum(beats_seg)))

    total_beats = int(np.floor(N[-1]))
    if total_beats < 2:
        return pd.Series(dtype=float, name="rr_sec")

    beat_times = np.empty(total_beats, dtype=float)
    k = 0
    for n in range(1, total_beats + 1):
        while k + 1 < N.size and N[k + 1] < n:
            k += 1
        if k + 1 >= N.size:
            beat_times[n - 1] = t[-1]
            continue

        if N[k + 1] == N[k]:
            tb = t[k + 1]
        else:
            frac = (n - N[k]) / (N[k + 1] - N[k])
            tb = t[k] + frac * (t[k + 1] - t[k])
        beat_times[n - 1] = tb

    rr = np.diff(beat_times)

    # Physiologic RR filter
    mask_rr = (rr >= 0.25) & (rr <= 2.0)
    rr = rr[mask_rr]

    return pd.Series(rr, name="rr_sec").reset_index(drop=True)


def main() -> None:
    print(f"[INFO] Reading {TFM_PATH}")
    # TFM.csv noktalı virgülle ayrılmış ve ondalık virgül kullanıyor
    df = pd.read_csv(TFM_PATH, sep=";", engine="python", decimal=",")

    time_s = get_time_axis(df)

    hr_cols = find_hr_columns(df)
    if not hr_cols:
        print("[WARN] No HR columns with 'HeartRate' or 'HRM' in name.")
        return

    print(f"[INFO] Found {len(hr_cols)} HR columns.")

    mapping_rows = []
    subject_idx = 1

    for col in hr_cols:
        hr_bpm = clean_hr_series(df[col])
        rr_sec_series = reconstruct_rr_from_hr(time_s, hr_bpm)

        if rr_sec_series.empty:
            print(f"[WARN] Empty RR after reconstruction for column '{col}'")
            continue

        subject_code = f"S{subject_idx:02d}"
        out_path = OUT_DIR / f"{subject_code}_clean.csv"
        rr_sec_series.to_csv(out_path, index=False)

        print(f"[INFO] Saved {out_path.name} (n={len(rr_sec_series)}) from column '{col}'")

        mapping_rows.append(
            {
                "subject_code": subject_code,
                "source_column": str(col),
            }
        )

        subject_idx += 1

    if mapping_rows:
        map_df = pd.DataFrame(mapping_rows)
        map_df.to_csv(MAP_PATH, index=False)
        print(f"[INFO] Anon mapping saved to {MAP_PATH.name}")
    else:
        print("[WARN] No subjects generated.")


if __name__ == "__main__":
    main()