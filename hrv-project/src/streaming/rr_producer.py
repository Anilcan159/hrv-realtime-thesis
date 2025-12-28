# Multi-subject RR producer (round-robin streaming)
"""
Streams cleaned RR series for multiple subjects to Kafka in a round-robin fashion.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from kafka import KafkaProducer

# --- ensure project root is on sys.path ---
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../hrv-project
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.settings import settings



# Config-driven paths and Kafka parameters
RR_DIR: Path = settings.paths.rr_processed_dir
KAFKA_BOOTSTRAP: str = settings.kafka.bootstrap_servers
KAFKA_TOPIC: str = settings.kafka.rr_topic

# Ust limit for concurrently streamed subjects (demo-friendly)
DEFAULT_SUBJECT_LIMIT: int = 8


def load_rr_ms(csv_path: Path) -> np.ndarray:
    """
    Load RR intervals (ms) from a per-subject cleaned CSV file.

    Expected columns:
        - 'rr'     : seconds  -> converted to ms
        - 'rr_ms'  : already in milliseconds

    Returns:
        1D numpy array (float64) of RR intervals in milliseconds,
        with non-finite values removed.
    """
    df = pd.read_csv(csv_path)

    if "rr" in df.columns:
        rr_ms = df["rr"].to_numpy(dtype=float) * 1000.0
    elif "rr_ms" in df.columns:
        rr_ms = df["rr_ms"].to_numpy(dtype=float)
    else:
        raise ValueError(f"RR column not found in {csv_path}. Columns={list(df.columns)}")

    rr_ms = rr_ms[np.isfinite(rr_ms)]
    return rr_ms


def find_subjects(limit: int = DEFAULT_SUBJECT_LIMIT) -> List[str]:
    """
    Discover available subjects from the cleaned RR directory.

    Files:
        {code}_clean.csv  -> subject code is the numeric prefix.

    Returns:
        Sorted list of subject codes (as strings), limited by `limit`.
    """
    codes: List[str] = []

    for p in RR_DIR.glob("*_clean.csv"):
        code = p.stem.replace("_clean", "")
        codes.append(code)

    # numeric sort when possible, non-numeric sent to the end
    def _sort_key(x: str) -> int:
        try:
            return int(x)
        except ValueError:
            return 999_999

    codes = sorted(codes, key=_sort_key)
    return codes[:limit]


def main(
    subjects: List[str] | None = None,
    loop: bool = True,
    min_sleep_s: float = 0.05,
) -> None:
    """
    Main producer loop.

    Args:
        subjects:
            List of subject codes to stream. If None, uses `find_subjects()`.
        loop:
            If True, wraps around when the end of a subject's RR series is reached.
            If False, stops streaming that subject at the end of its series.
        min_sleep_s:
            Minimum sleep between consecutive sends to avoid overwhelming the broker
            and to keep the demo visually reasonable.
    """
    if subjects is None:
        subjects = find_subjects(limit=DEFAULT_SUBJECT_LIMIT)

    if not subjects:
        print("[rr_producer] no subjects found in rr_clean directory.")
        return

    series: List[Tuple[str, np.ndarray, int]] = []
    for s in subjects:
        csv_path = RR_DIR / f"{s}_clean.csv"
        if not csv_path.exists():
            print(f"[rr_producer] warning: file not found for subject {s}: {csv_path}")
            continue

        rr_ms = load_rr_ms(csv_path)
        if rr_ms.size == 0:
            print(f"[rr_producer] warning: empty RR series for subject {s}")
            continue

        # (subject_code, rr_ms_array, current_index)
        series.append((s, rr_ms, 0))

    if not series:
        print("[rr_producer] no valid RR series to stream, exiting.")
        return

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )

    print(
        f"[rr_producer] Producing subjects={ [s for s, _, _ in series] } "
        f"to topic='{KAFKA_TOPIC}' (loop={loop}) ..."
    )

    try:
        while True:
            any_active = False
            new_series: List[Tuple[str, np.ndarray, int]] = []

            # round-robin: send one RR sample per subject per cycle
            for subject, rr_ms_arr, idx in series:
                if idx >= rr_ms_arr.size:
                    if loop:
                        idx = 0
                    else:
                        new_series.append((subject, rr_ms_arr, idx))
                        continue

                rr = float(rr_ms_arr[idx])
                msg = {"subject": subject, "ts": time.time(), "rr_ms": rr}
                producer.send(KAFKA_TOPIC, msg)
                any_active = True

                # Birden fazla subject aynı anda akarken "gerçek zaman" birebir değildir;
                # ama dashboard demo için yeterli: her subject düzenli update alır.
                effective_sleep = max(rr / 1000.0 / max(len(series), 1), min_sleep_s)
                time.sleep(effective_sleep)

                new_series.append((subject, rr_ms_arr, idx + 1))

            series = new_series
            producer.flush()

            if not any_active:
                # All subjects exhausted and loop=False
                break

    finally:
        try:
            producer.flush()
        except Exception:
            pass
        producer.close()
        print("[rr_producer] Done.")


if __name__ == "__main__":
    main(subjects=None, loop=True)
