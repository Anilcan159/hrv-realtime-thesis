# src/api/fastapi_app.py
"""
FastAPI-based HRV metrics service.

Bu servis:
    - Kafka -> rr_consumer -> GLOBAL_RR_BUFFER üzerinden gelen RR serisini kullanır.
    - service_hrv.py içindeki fonksiyonlarla time / freq / poincare metriklerini hesaplar.
    - Sonuçları JSON olarak REST endpoint'leri üzerinden dışarı açar.

Endpointler:
    - GET /health
    - GET /metrics/time
    - GET /metrics/freq
    - GET /metrics/poincare
    - GET /metrics/status
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Proje kökünü sys.path'e ekle (script olarak çalıştırıldığında da src.* import edilsin)
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../hrv-project
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.settings import settings
from src.streaming.rr_consumer import start_consumer_background
from src.hrv_metrics.service_hrv import (
    get_time_domain_metrics,
    get_freq_domain_metrics,
    get_poincare_data,
    get_signal_status,
)
from src.utils.logging_utils import get_logger


# FastAPI app instance
app = FastAPI(
    title="HRV Realtime Metrics API",
    version="1.0.0",
    description="Provides time, frequency and non-linear HRV metrics as REST endpoints.",
)

# CORS (simdilik gelistirme icin herkese acik)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # gelistirme icin serbest; production'da daraltirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = get_logger(module_name="hrv_api", logfile_name="api.log")


# --- Helper: window paramini saniyeye cevir --- #

def _resolve_window_s(
    window: Optional[str],
    window_s: Optional[float],
) -> Optional[float]:
    """
    window / window_s parametrelerini tek bir window_length_s degerine indirger.

    Oncelik sirasi:
        1) window_s (saniye cinsinden dogrudan)
        2) window == "last_5min" -> settings.dashboard.default_window_s
        3) window == "full" veya None -> None (tum kayit)
    """
    if window_s is not None:
        try:
            ws = float(window_s)
            if ws > 0:
                return ws
        except (TypeError, ValueError):
            pass

    if window == "last_5min":
        return float(settings.dashboard.default_window_s)

    # "full" veya None: tum kayit
    return None


# --- Startup: Kafka consumer thread --- #

@app.on_event("startup")
def startup_event() -> None:
    """
    Uygulama baslarken Kafka consumer'ı arka planda baslatir.
    """
    logger.info(
        "FastAPI HRV service starting up. Kafka bootstrap='%s', topic='%s', group_id='%s'",
        settings.kafka.bootstrap_servers,
        settings.kafka.rr_topic,
        settings.kafka.group_id,
    )
    start_consumer_background()
    logger.info("Background Kafka consumer thread requested from FastAPI startup.")


# --- Health check --- #

@app.get("/health")
def health() -> Dict[str, str]:
    """
    Basit saglik kontrolu endpoint'i.
    """
    return {"status": "ok"}


# --- Time-domain metrics endpoint --- #

@app.get("/metrics/time")
def time_metrics(
    subject: str = Query(..., description="Subject code, e.g. '000'"),
    window: Optional[str] = Query(
        None,
        description="Named window: 'full' or 'last_5min'. If provided, overrides window_s.",
    ),
    window_s: Optional[float] = Query(
        None,
        description="Window length in seconds. If not provided, uses named window or full recording.",
    ),
) -> Dict[str, Any]:
    """
    Time-domain HRV metrikleri (SDNN, RMSSD, pNN50, mean HR, HR min/max, HTI, TINN, vb.)
    """
    window_length_s = _resolve_window_s(window, window_s)
    logger.info(
        "GET /metrics/time subject=%s window=%s window_s=%s -> resolved window_length_s=%.3f",
        subject,
        window,
        window_s,
        float(window_length_s) if window_length_s is not None else -1.0,
    )

    metrics = get_time_domain_metrics(subject_code=subject, window_length_s=window_length_s)
    # FastAPI dict'i otomatik JSON'a cevirir
    return metrics


# --- Frequency-domain metrics endpoint --- #

@app.get("/metrics/freq")
def freq_metrics(
    subject: str = Query(..., description="Subject code, e.g. '000'"),
    window: Optional[str] = Query(
        None,
        description="Named window: 'full' or 'last_5min'. If provided, overrides window_s.",
    ),
    window_s: Optional[float] = Query(
        None,
        description="Window length in seconds. If not provided, uses named window or full recording.",
    ),
    fs_resample: Optional[float] = Query(
        None,
        description="Resampling frequency for RR series (Hz). Defaults to settings.hrv.fs_resample.",
    ),
) -> Dict[str, Any]:
    """
    Frekans-domeni HRV metrikleri (VLF/LF/HF bant gucleri, LF/HF oranı, PSD spektrumu).
    """
    window_length_s = _resolve_window_s(window, window_s)
    fs = float(fs_resample) if fs_resample is not None else float(settings.hrv.fs_resample)

    logger.info(
        "GET /metrics/freq subject=%s window=%s window_s=%s fs_resample=%.3f -> window_length_s=%.3f",
        subject,
        window,
        window_s,
        fs,
        float(window_length_s) if window_length_s is not None else -1.0,
    )

    fd = get_freq_domain_metrics(
        subject_code=subject,
        window_length_s=window_length_s,
        fs_resample=fs,
    )
    return fd


# --- Poincare / non-linear metrics endpoint --- #

@app.get("/metrics/poincare")
def poincare_metrics(
    subject: str = Query(..., description="Subject code, e.g. '000'"),
    window: Optional[str] = Query(
        None,
        description="Named window: 'full' or 'last_5min'. If provided, overrides window_s.",
    ),
    window_s: Optional[float] = Query(
        None,
        description="Window length in seconds. If not provided, uses named window or full recording.",
    ),
    max_points: Optional[int] = Query(
        None,
        description="Maximum number of RR points for the Poincare scatter.",
    ),
) -> Dict[str, Any]:
    """
    Poincare tabanlı non-linear HRV metrikleri:
        - SD1, SD2, SD1/SD2
        - stress_index
        - opsiyonel scatter verisi (x, y)
    """
    window_length_s = _resolve_window_s(window, window_s)
    mp = max_points if max_points is not None else settings.dashboard.max_poincare_points

    logger.info(
        "GET /metrics/poincare subject=%s window=%s window_s=%s max_points=%s -> window_length_s=%.3f",
        subject,
        window,
        window_s,
        mp,
        float(window_length_s) if window_length_s is not None else -1.0,
    )

    data = get_poincare_data(
        subject_code=subject,
        max_points=mp,
        window_length_s=window_length_s,
    )
    return data


# --- Signal quality / status endpoint --- #

@app.get("/metrics/status")
def status_metrics(
    subject: str = Query(..., description="Subject code, e.g. '000'"),
    window: Optional[str] = Query(
        None,
        description="Named window: 'full' or 'last_5min'. If provided, overrides window_s.",
    ),
    window_s: Optional[float] = Query(
        None,
        description="Window length in seconds. If not provided, uses named window or full recording.",
    ),
) -> Dict[str, Any]:
    """
    Sinyal kalitesi ozeti:
        - quality_label (OK / Moderate / Poor / Short recording / No data)
        - status_text
        - outlier_percent
        - n_total, n_outliers
    """
    window_length_s = _resolve_window_s(window, window_s)

    logger.info(
        "GET /metrics/status subject=%s window=%s window_s=%s -> window_length_s=%.3f",
        subject,
        window,
        window_s,
        float(window_length_s) if window_length_s is not None else -1.0,
    )

    status = get_signal_status(subject_code=subject, window_length_s=window_length_s)
    return status


# --- Local run entrypoint (uvicorn) --- #

if __name__ == "__main__":
    # Gelistirme icin:
    #   python -m uvicorn src.api.fastapi_app:app --reload
    # yerine:
    #   python src/api/fastapi_app.py
    # ile de calistirabilmek icin ufak bir convenience.
    import uvicorn

    uvicorn.run(
        "src.api.fastapi_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
