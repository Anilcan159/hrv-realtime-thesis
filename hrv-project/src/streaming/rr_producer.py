# Multi-subject RR producer (round-robin streaming)
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = "localhost:9092"
TOPIC = "rr-stream"

ROOT_DIR = Path(__file__).resolve().parents[2]
RR_DIR = ROOT_DIR / "data" / "processed" / "rr_clean"

def load_rr_ms(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "rr" in df.columns:
        rr_ms = df["rr"].to_numpy(dtype=float) * 1000.0
    elif "rr_ms" in df.columns:
        rr_ms = df["rr_ms"].to_numpy(dtype=float)
    else:
        raise ValueError(f"RR column not found. Columns={list(df.columns)}")
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    return rr_ms

def find_subjects(limit: int = 8) -> list[str]:
    codes = []
    for p in RR_DIR.glob("*_clean.csv"):
        code = p.stem.replace("_clean", "")
        codes.append(code)
    codes = sorted(codes, key=lambda x: int(x) if x.isdigit() else 999999)
    return codes[:limit]

def main(subjects: list[str] | None = None, loop: bool = True, min_sleep_s: float = 0.05):
    if subjects is None:
        subjects = find_subjects(limit=8)

    series = []
    for s in subjects:
        csv_path = RR_DIR / f"{s}_clean.csv"
        series.append((s, load_rr_ms(csv_path), 0))  # (subject, rr_ms_array, idx)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )

    print(f"Producing subjects={subjects} to {TOPIC} (loop={loop}) ...")

    try:
        while True:
            # Round-robin: her subject'ten bir sample
            any_active = False

            new_series = []
            for subject, rr_ms_arr, idx in series:
                if idx >= rr_ms_arr.size:
                    if loop:
                        idx = 0
                    else:
                        new_series.append((subject, rr_ms_arr, idx))
                        continue

                rr = float(rr_ms_arr[idx])
                msg = {"subject": subject, "ts": time.time(), "rr_ms": rr}
                producer.send(TOPIC, msg)
                any_active = True

                # aynı anda 8 subject akarken "gerçek zaman" birebir olmaz;
                # ama dashboard demo için yeter: her subject düzenli update alır
                time.sleep(max(rr / 1000.0 / max(len(series), 1), min_sleep_s))

                new_series.append((subject, rr_ms_arr, idx + 1))

            series = new_series
            producer.flush()

            if not any_active:
                break

    finally:
        try:
            producer.flush()
        except Exception:
            pass
        producer.close()
        print("Done.")

if __name__ == "__main__":
    main(subjects=None, loop=True)
