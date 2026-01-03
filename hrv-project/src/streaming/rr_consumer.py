# src/streaming/rr_consumer.py
# Consumes RR samples from Kafka topic and fills the global memory buffer

import json
import threading
import time
import sys
from pathlib import Path
from typing import Optional

from kafka import KafkaConsumer

# Proje kökünü sys.path'e ekle (script olarak çalıştırıldığında da src.* import edilsin)
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../hrv-project
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.settings import settings
from src.streaming.rr_buffer import GLOBAL_RR_BUFFER
from src.utils.logging_utils import get_logger

# Config üzerinden Kafka parametreleri
BOOTSTRAP = settings.kafka.bootstrap_servers
TOPIC = settings.kafka.rr_topic
GROUP_ID = settings.kafka.group_id

# Module-level logger (logs/consumer.log)
logger = get_logger(module_name="rr_consumer", logfile_name="consumer.log")

_consumer_thread: Optional[threading.Thread] = None


def _run_consumer_forever() -> None:
    """
    Kafka'dan RR örneklerini sürekli tüketir ve GLOBAL_RR_BUFFER'a yazar.

    Hata olduğunda:
        - Hata loglanır (ERROR)
        - 2 saniye beklenir
        - Yeniden bağlanmayı dener
    """
    logger.info(
        "Background consumer loop starting (topic='%s', bootstrap='%s', group_id='%s')",
        TOPIC,
        BOOTSTRAP,
        GROUP_ID,
    )

    while True:
        consumer: Optional[KafkaConsumer] = None

        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )

            logger.info("Connected to Kafka, listening on topic='%s'", TOPIC)

            for msg in consumer:
                try:
                    data = msg.value
                    subject = str(data.get("subject", "000"))
                    ts = float(data.get("ts", time.time()))
                    rr_ms = float(data.get("rr_ms"))

                    GLOBAL_RR_BUFFER.add(subject=subject, rr_ms=rr_ms, ts=ts)
                    # Her mesajı loglamıyoruz; dosyayı şişirir.
                except Exception as e:
                    logger.warning(
                        "Error while processing message from topic='%s': %r",
                        TOPIC,
                        e,
                        exc_info=True,
                    )

        except Exception as e:
            logger.error(
                "Kafka consumer error (bootstrap='%s', topic='%s'), retrying in 2s: %r",
                BOOTSTRAP,
                TOPIC,
                e,
                exc_info=True,
            )
            time.sleep(2.0)

        finally:
            if consumer is not None:
                try:
                    consumer.close()
                    logger.info("KafkaConsumer closed.")
                except Exception:
                    logger.warning("Error while closing KafkaConsumer.", exc_info=True)


def start_consumer_background() -> None:
    """
    Kafka consumer'ı arka planda (daemon thread) başlatır.

    FastAPI startup event'inde veya başka bir servis tarafında çağrılmak üzere tasarlanmıştır.
    """
    global _consumer_thread

    if _consumer_thread is not None and _consumer_thread.is_alive():
        logger.info("Background consumer thread already running: %r", _consumer_thread)
        return

    t = threading.Thread(target=_run_consumer_forever, daemon=True)
    t.start()
    _consumer_thread = t

    logger.info("Background consumer thread started (daemon=True): %r", t)


if __name__ == "__main__":
    # Script direkt çalıştırılırsa foreground consumer olarak çalışır
    logger.info("Starting foreground consumer loop (standalone).")
    _run_consumer_forever()
