# Consumes RR samples from Kafka topic and fills the global in-memory buffer.
"""
Kafka RR consumer.

Responsibilities:
    - Subscribe to the configured RR topic.
    - Deserialize JSON messages coming from rr_producer.
    - Route (subject, ts, rr_ms) into GLOBAL_RR_BUFFER.
    - Run in a resilient infinite loop with automatic reconnect.

Configuration:
    - Kafka connection and topic: src.config.settings.settings.kafka
    - Buffer implementation: src.streaming.rr_buffer.GLOBAL_RR_BUFFER
"""

import json
import threading
import time
from typing import Optional

from kafka import KafkaConsumer

from src.config.settings import settings
from src.streaming.rr_buffer import GLOBAL_RR_BUFFER


# Centralized config
KAFKA_BOOTSTRAP: str = settings.kafka.bootstrap_servers
KAFKA_TOPIC: str = settings.kafka.rr_topic
KAFKA_GROUP_ID: str = settings.kafka.group_id

# Single background consumer thread handle
_consumer_thread: Optional[threading.Thread] = None


def _run_consumer_forever() -> None:
    """
    Blocking loop that continuously consumes RR messages from Kafka
    and feeds them into the global RR buffer.

    Behaviour:
        - Creates a KafkaConsumer inside a retry loop.
        - On any exception (network, broker restart, etc.), waits briefly
          and then reconnects.
        - Each valid message is expected to have:
              {
                  "subject": <str>,
                  "ts": <float, epoch seconds>,
                  "rr_ms": <float, milliseconds>
              }
    """
    while True:
        consumer: Optional[KafkaConsumer] = None

        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=KAFKA_GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )

            print(f"[rr_consumer] listening on topic='{KAFKA_TOPIC}' "
                  f"(bootstrap={KAFKA_BOOTSTRAP}, group_id={KAFKA_GROUP_ID})")

            for msg in consumer:
                data = msg.value

                # Basic schema validation / robustness
                if not isinstance(data, dict):
                    print(f"[rr_consumer] skipped non-dict message: {data!r}")
                    continue

                if "rr_ms" not in data:
                    print(f"[rr_consumer] skipped message without 'rr_ms': {data!r}")
                    continue

                subject = str(data.get("subject", "000"))
                ts = float(data.get("ts", time.time()))
                rr_ms = float(data["rr_ms"])

                GLOBAL_RR_BUFFER.add(subject=subject, rr_ms=rr_ms, ts=ts)

        except Exception as e:
            # Any error (broker down, network, etc.): log and retry
            print(f"[rr_consumer] error: {e!r} (retrying in 2s)")
            time.sleep(2.0)

        finally:
            if consumer is not None:
                try:
                    consumer.close()
                except Exception:
                    # If close() itself fails, there is nothing more to do.
                    pass


def start_consumer_background() -> None:
    """
    Starts the Kafka RR consumer in a daemon thread, if not already running.

    This is intended to be called once at Dash startup:
        from src.streaming.rr_consumer import start_consumer_background
        start_consumer_background()
    """
    global _consumer_thread

    if _consumer_thread is not None and _consumer_thread.is_alive():
        # Already running, nothing to do.
        return

    t = threading.Thread(target=_run_consumer_forever, daemon=True)
    t.start()
    _consumer_thread = t
    print(f"[rr_consumer] background consumer thread started (daemon={t.daemon})")


if __name__ == "__main__":
    # If run as a standalone script, block in the consumer loop.
    _run_consumer_forever()
