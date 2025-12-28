# src/utils/logging_utils.py
"""
Central logging utilities for the project.

Amaç:
    - Her modül için tutarlı log formatı sağlamak.
    - Log dosyalarını proje kökündeki "logs" klasörüne yazmak.
    - INFO / WARNING / ERROR seviyelerinde günlükleme yapmak.

Kullanım:
    from src.utils.logging_utils import get_logger

    logger = get_logger(module_name="rr_producer", logfile_name="producer.log")
    logger.info("Producer started.")
    logger.error("Unexpected error", exc_info=True)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Proje kökü: .../hrv-project
ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(
    module_name: str,
    logfile_name: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Belirli bir modül için dosya tabanlı logger döner.

    Args:
        module_name:
            Logger adı (örn. "rr_producer", "rr_consumer", "hrv_service").
        logfile_name:
            Log dosyası adı (örn. "producer.log").
            Dosya, proje kökündeki "logs" klasörüne yazılır.
        level:
            Log seviyesi (logging.INFO, logging.WARNING, logging.ERROR, ...).
        max_bytes:
            Rotating log için maksimum dosya boyutu (default: 5 MB).
        backup_count:
            Maksimum yedek log dosyası sayısı (örn. producer.log.1, producer.log.2, ...).

    Returns:
        logging.Logger objesi.
    """
    logger = logging.getLogger(module_name)

    # Eğer logger daha önce konfigüre edildiyse tekrar handler eklemeyelim
    if logger.handlers:
        return logger

    logger.setLevel(level)

    log_path = LOG_DIR / logfile_name

    # Rotating file handler: dosya belli boyutu geçince döner
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Üst loggers'a propagate etme (çift log olmasın)
    logger.propagate = False

    return logger
