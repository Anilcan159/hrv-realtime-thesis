# src/config/settings.py
"""
Central application configuration for the HRV realtime project.

Tüm sabitler ve parametreler:
    - Kafka ayarları
    - Dosya yolları
    - HRV preprocessing ve analiz parametreleri
    - Dashboard ile ilgili sınırlar

Buradan okunur:
    from src.config.settings import settings
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# Proje kökü: .../hrv-project
BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathSettings:
    """
    Dosya ve klasör yolları.
    """

    base_dir: Path = BASE_DIR
    raw_data_dir: Path = (
        BASE_DIR
        / "Datas"
        / "raw"
        / "physionet.org"
        / "files"
        / "rr-interval-healthy-subjects"
        / "1.0.0"
    )
    rr_processed_dir: Path = BASE_DIR / "data" / "processed" / "rr_clean"
    patient_info_path: Path = BASE_DIR / "Datas" / "processed" / "patient-info.csv"


@dataclass(frozen=True)
class KafkaSettings:
    """
    Kafka bağlantı ayarları.

    Ortam değişkenleri:
        KAFKA_BOOTSTRAP
        KAFKA_RR_TOPIC
        KAFKA_GROUP_ID
    """

    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    rr_topic: str = os.getenv("KAFKA_RR_TOPIC", "rr-stream")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "hrv-consumer")


@dataclass(frozen=True)
class HRVSettings:
    """
    HRV preprocessing + analiz parametreleri.
    """

    # Fizyolojik RR sınırları (ms)
    rr_min_ms: int = 300
    rr_max_ms: int = 2000

    # Kısa boşluk (beat sayısı) interpolasyon sınırı
    max_gap_beats: int = 3

    # Bellekte tutulacak maksimum RR geçmişi (saniye)
    rr_max_age_s: float = 10 * 60.0

    # Frekans-domeni yeniden örnekleme frekansı (Hz)
    fs_resample: float = 4.0

    # Spektral band sınırları (Hz)
    vlf_band: Tuple[float, float] = (0.0033, 0.04)
    lf_band: Tuple[float, float] = (0.04, 0.15)
    hf_band: Tuple[float, float] = (0.15, 0.40)


@dataclass(frozen=True)
class DashboardSettings:
    """
    Dashboard ile ilgili varsayılan ayarlar.
    """

    # "Last 5 minutes" penceresi
    default_window_s: float = 5 * 60.0

    # HR time-series grafiğinde tutulacak maksimum nokta
    max_hr_points: int = 500

    # Poincaré scatter için maksimum nokta
    max_poincare_points: int = 1000



@dataclass(frozen=True)
class ApiSettings:
    """
    HRV FastAPI servisi ayarlari.
    """
    base_url: str = os.getenv("HRV_API_BASE_URL", "http://127.0.0.1:8000")


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings = PathSettings()
    kafka: KafkaSettings = KafkaSettings()
    hrv: HRVSettings = HRVSettings()
    dashboard: DashboardSettings = DashboardSettings()
    api: ApiSettings = ApiSettings()  # <--- BURAYI EKLE




# Tek global config objesi
settings = AppSettings()
