# Consumes RR samples from Kafka topic and fills the global memory buffer
import json
import threading
import time
from kafka import KafkaConsumer

from src.streaming.rr_buffer import GLOBAL_RR_BUFFER

BOOTSTRAP = "localhost:9092"
TOPIC = "rr-stream"
GROUP_ID = "hrv-consumer"

_consumer_thread = None

def _run_consumer_forever():
    while True:
        consumer = None
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )

            print(f"Listening {TOPIC} ...")
            for msg in consumer:
                data = msg.value
                subject = str(data.get("subject", "000"))
                ts = float(data.get("ts", time.time()))
                rr_ms = float(data.get("rr_ms"))
                GLOBAL_RR_BUFFER.add(subject=subject, rr_ms=rr_ms, ts=ts)

        except Exception as e:
            print(f"[rr_consumer] error: {e} (retrying in 2s)")
            time.sleep(2.0)

        finally:
            try:
                if consumer is not None:
                    consumer.close()
            except Exception:
                pass


def start_consumer_background():
    global _consumer_thread
    if _consumer_thread and _consumer_thread.is_alive():
        return

    t = threading.Thread(target=_run_consumer_forever, daemon=True)
    t.start()
    _consumer_thread = t


if __name__ == "__main__":
    _run_consumer_forever()
        