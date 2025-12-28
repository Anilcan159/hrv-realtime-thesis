# src/config/settings.py
from dataclasses import dataclass
from pathlib import Path
import os

# Proje kökü
BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class KafkaSettings:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    rr_topic: str = os.getenv("KAFKA_RR_TOPIC", "rr-stream")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "hrv-consumer")


@dataclass(frozen=True)
class PathSettings:
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
class HRVSettings:
    # streaming / buffer
    rr_max_age_s: float = float(os.getenv("RR_MAX_AGE_S", 10 * 60.0))

    # time-domain temizleme parametreleri
    rr_min_ms: int = int(os.getenv("HRV_RR_MIN_MS", 300))
    rr_max_ms: int = int(os.getenv("HRV_RR_MAX_MS", 2000))
    max_gap_beats: int = int(os.getenv("HRV_MAX_GAP_BEATS", 3))

    # freq-domain
    fs_resample: float = float(os.getenv("HRV_FS_RESAMPLE", 4.0))
    vlf_band: tuple[float, float] = (0.0033, 0.04)
    lf_band: tuple[float, float] = (0.04, 0.15)
    hf_band: tuple[float, float] = (0.15, 0.40)


@dataclass(frozen=True)
class DashboardSettings:
    default_window_s: float = 5 * 60.0  # “Last 5 minutes”
    max_hr_points: int = 500
    max_poincare_points: int = 1000


@dataclass(frozen=True)
class AppSettings:
    kafka: KafkaSettings = KafkaSettings()
    paths: PathSettings = PathSettings()
    hrv: HRVSettings = HRVSettings()
    dashboard: DashboardSettings = DashboardSettings()


# Uygulama genelinde kullanılacak tek entry point
settings = AppSettings()
