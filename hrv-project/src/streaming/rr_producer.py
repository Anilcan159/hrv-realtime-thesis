# Sends RR intervals to Kafka topic rr-stream (CSV replay as real-time)
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = "localhost:9092"
TOPIC = "rr-stream"

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

def main(subject: str = "000", loop: bool = True):
    root_dir = Path(__file__).resolve().parents[2]  # hrv-project
    csv_path = root_dir / "data" / "processed" / "rr_clean" / f"{subject}_clean.csv"

    rr_ms = load_rr_ms(csv_path)

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
    )

    print(f"Producing {rr_ms.size} RR samples to {TOPIC} (subject={subject}, loop={loop}) ...")

    sent = 0
    try:
        while True:
            for rr in rr_ms:
                msg = {"subject": subject, "ts": time.time(), "rr_ms": float(rr)}
                producer.send(TOPIC, msg)
                sent += 1

                # flush'u her mesajda yapma (yavaşlatır)
                if sent % 200 == 0:
                    producer.flush()

                time.sleep(max(float(rr) / 1000.0, 0.05))

            producer.flush()
            if not loop:
                break

            print("Looping CSV from start...")

    finally:
        try:
            producer.flush()
        except Exception:
            pass
        producer.close()
        print("Done.")

if __name__ == "__main__":
    # basit kullanım: istersen burada subject'i değiştir
    main(subject="000", loop=True)
