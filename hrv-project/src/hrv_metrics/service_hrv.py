# src/hrv_metrics/service_hrv.py
"""
HRV service layer.

Sorumluluklar:
    - Canlı RR verisini GLOBAL_RR_BUFFER üzerinden okumak.
    - Gerekirse (LIVE_ONLY kapalıysa) CSV fallback ile offline RR verisini yüklemek.
    - Time-domain, frequency-domain ve Poincaré temelli HRV metriklerini hesaplamak.
    - Subject listesi ve demografik bilgileri sağlamak.
    - Sinyal kalitesi (teknik) özeti üretmek.

Konfigürasyon:
    - Veri yolları:    settings.paths
    - HRV parametreleri (fs_resample, bant sınırları): settings.hrv
    - Dashboard ile ilgili limitler (ör. max_points):  settings.dashboard
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.logging_utils import get_logger
from scipy.interpolate import CubicSpline
from scipy.signal import welch

import numpy as np
import pandas as pd
logger = get_logger(module_name="hrv_service", logfile_name="hrv_service.log")
from typing import Dict, Any  

from scipy.interpolate import CubicSpline
from scipy.signal import welch

from src.config.settings import settings
from src.streaming.rr_buffer import GLOBAL_RR_BUFFER
from vmdpy import VMD

# Canlı mod (True) iken, RR kaynağı sadece GLOBAL_RR_BUFFER'dır.
# False yapılırsa, RR verisi yoksa CSV fallback devreye girer (offline analiz).
LIVE_ONLY: bool = True

# Konfigürasyondan gelen yollar
PATIENT_INFO_PATH: Path = settings.paths.patient_info_path
RR_DIR: Path = settings.paths.rr_processed_dir

# VMDON parametreleri (offline script ile uyumlu)
VMD_ALPHA: float = 2000.0
VMD_TAU: float = 0.0
VMD_INIT: int = 1
VMD_TOL: float = 1e-7

VMDON_WINDOW_S: Dict[str, float] = {
    "HF": 6.5,
    "LF": 25.0,
    "VLF": 303.0,
    "ULF": 1800.0,
}

def get_available_subject_codes():
    """
    rr_clean klasöründeki *_clean.csv dosyalarından subject_code listesini üretir.
    Örn: 000_clean.csv -> '000'
    """
    codes = []
    for path in RR_DIR.glob("*_clean.csv"):
        code = path.stem.replace("_clean", "")
        codes.append(code)

    # sayısal sıraya göre sırala (000,002,003,005,401,...)
    def _sort_key(c):
        try:
            return int(c)
        except ValueError:
            return 999999  # numara olmayanları sona at

    codes = sorted(codes, key=_sort_key)

    if not codes:
        logger.warning("No *_clean.csv files found in RR_DIR=%s", RR_DIR)

    return codes



# -------------------- RR KAYNAĞI -------------------- #

def load_rr(subject_code: str, window_length_s: Optional[float] = None) -> np.ndarray:
    """
    Verilen subject için RR serisini saniye cinsinden döner.

    Öncelik:
        1) GLOBAL_RR_BUFFER içindeki canlı veri
        2) (LIVE_ONLY False ise) CSV fallback:
           data/processed/rr_clean/{subject_code}_clean.csv

    Args:
        subject_code:
            Örn. "000", "002", "401".
        window_length_s:
            Eğer None ise:
                Tam buffer veya tam kayıt kullanılır.
            Eğer > 0 ise:
                Sadece son window_length_s saniyeye düşen RR intervalleri alınır.

    Returns:
        1D numpy array (float64), saniye cinsinden RR intervalleri.
    """
    rr_live = GLOBAL_RR_BUFFER.get_rr_sec(subject_code, window_s=window_length_s)
    if rr_live and len(rr_live) >= 2:
        return np.asarray(rr_live, dtype=float)

    if LIVE_ONLY:
        # Dashboard demo modunda CSV'ye düşme
        return np.asarray([], dtype=float)

    rr = load_rr_from_csv(subject_code)
    return _apply_time_window(rr, window_length_s)


def load_rr_from_csv(subject_code: str) -> np.ndarray:
    """
    Belirli bir denek için RR serisini CSV'den okur.
    subject_code: '000', '002', '401' gibi.
    Beklenen dosya adı: 000_clean.csv, 002_clean.csv, 401_clean.csv ...
    """
    csv_path = RR_DIR / f"{subject_code}_clean.csv"

    if not csv_path.exists():
        logger.error("RR file not found for subject=%s: %s", subject_code, csv_path)
        raise FileNotFoundError(f"RR file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # BURADA BİR VARSAYIM YAPIYORUM:
    #   - Eğer kolon adı 'rr' ise direkt alıyoruz
    #   - Eğer 'rr_ms' ise saniyeye çeviriyoruz (ms / 1000)
    # Eğer sende farklı isimse (örneğin 'RR_interval_ms') sadece aşağıdaki kısmı değiştirmen yeter.
    if "rr" in df.columns:
        rr_sec = df["rr"].to_numpy(dtype=float)
    elif "rr_ms" in df.columns:
        rr_sec = df["rr_ms"].to_numpy(dtype=float) / 1000.0
    else:
        logger.error(
            "RR column not found in %s for subject=%s. Expected 'rr' or 'rr_ms'. Columns=%s",
            csv_path,
            subject_code,
            list(df.columns),
        )
        raise ValueError(
            f"RR column not found in {csv_path}. "
            f"Expected one of: 'rr', 'rr_ms'. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    return rr_sec



# -------------------- TIME-DOMAIN HESAP -------------------- #

def _compute_time_domain_from_rr(rr: np.ndarray) -> Dict[str, float]:
    """
    Tek boyutlu RR serisinden temel ve ilerletilmiş time-domain HRV metriklerini hesaplar.

    Girdi:
        rr:
            Saniye cinsinden RR intervalleri (ör: 0.8 = 800 ms)

    Dönen değerler (ms veya bpm):
        - sdnn, sdrr, rmssd
        - nn50, pnn50
        - mean_hr, hr_max, hr_min, hr_range
        - hti, tinn (histogram tabanlı metrikler)
        - n_beats, duration_s
    """
    rr = np.asarray(rr, dtype=float)

    # Çok kısa dizi için NaN dön (dashboard'ı patlatma)
    if rr.ndim != 1 or rr.size < 2:
        return {
            "sdnn": float("nan"),
            "sdrr": float("nan"),
            "rmssd": float("nan"),
            "nn50": 0.0,
            "pnn50": float("nan"),
            "mean_hr": float("nan"),
            "hr_max": float("nan"),
            "hr_min": float("nan"),
            "hr_range": float("nan"),
            "hti": float("nan"),
            "tinn": float("nan"),
            "n_beats": float(rr.size),
            "duration_s": float(np.sum(rr)) if rr.size > 0 else 0.0,
        }

    # ms cinsine çevir
    rr_ms = rr * 1000.0

    # Temel istatistikler
    n_beats = int(rr_ms.size)
    duration_s = float(np.sum(rr))  # saniye

    # SDNN / SDRR (ms)
    sdnn = float(np.std(rr_ms, ddof=1))
    sdrr = sdnn

    # RMSSD (ms)
    diff_ms = np.diff(rr_ms)
    if diff_ms.size > 0:
        rmssd = float(np.sqrt(np.mean(diff_ms ** 2)))
    else:
        rmssd = float("nan")

    # NN50 & pNN50 (%)
    nn50 = float(np.sum(np.abs(diff_ms) > 50.0))
    pnn50 = float(nn50 / diff_ms.size * 100.0) if diff_ms.size > 0 else float("nan")

    # Mean RR, HR min/max (bpm)
    mean_rr = float(np.mean(rr))          # saniye
    min_rr = float(np.min(rr))
    max_rr = float(np.max(rr))

    mean_hr = float(60.0 / mean_rr) if mean_rr > 0 else float("nan")
    hr_max = float(60.0 / min_rr) if min_rr > 0 else float("nan")  # en kısa RR -> en yüksek HR
    hr_min = float(60.0 / max_rr) if max_rr > 0 else float("nan")  # en uzun RR -> en düşük HR
    hr_range = (
        float(hr_max - hr_min)
        if np.isfinite(hr_max) and np.isfinite(hr_min)
        else float("nan")
    )

    # Histogram tabanlı HTI ve TINN
    try:
        counts, bins = np.histogram(rr_ms, bins="auto")
        if counts.size == 0 or counts.max() == 0:
            hti = float("nan")
            tinn = float("nan")
        else:
            # HTI: toplam beat sayısı / en yüksek histogram barı
            hti = float(n_beats / counts.max())

            peak_idx = int(np.argmax(counts))
            left_idx = np.where(counts[:peak_idx] < 0.05 * counts.max())[0]
            right_idx = np.where(counts[peak_idx:] < 0.05 * counts.max())[0]

            if left_idx.size > 0:
                left = bins[left_idx[-1]]
            else:
                left = bins[0]

            if right_idx.size > 0:
                right = bins[peak_idx + right_idx[0]]
            else:
                right = bins[-1]

            tinn = float(right - left)
    except Exception:
        hti = float("nan")
        tinn = float("nan")

    return {
        "sdnn": sdnn,
        "sdrr": sdrr,
        "rmssd": rmssd,
        "nn50": nn50,
        "pnn50": pnn50,
        "mean_hr": mean_hr,
        "hr_max": hr_max,
        "hr_min": hr_min,
        "hr_range": hr_range,
        "hti": hti,
        "tinn": tinn,
        "n_beats": float(n_beats),
        "duration_s": duration_s,
    }


def _apply_time_window(rr: np.ndarray, window_length_s: Optional[float]) -> np.ndarray:
    """
    RR serisini (saniye cinsinden) verilen pencere süresine göre kırpar.

    - window_length_s None ise:
        Tüm kayıt kullanılır.
    - window_length_s > 0 ise:
        Sadece SON window_length_s saniyeyi kapsayan RR intervalleri döner.

    Örnek:
        total_duration = 1200 s ve window_length_s = 300 s ise,
        son 300 saniyeye düşen RR'ler alınır.
    """
    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return rr
    if window_length_s is None or window_length_s <= 0:
        return rr

    t_cum = np.cumsum(rr)  # kümülatif zaman (s)
    total_duration = float(t_cum[-1])

    # Kayıt zaten window_length_s'ten kısaysa -> dokunma
    if total_duration <= window_length_s:
        return rr

    start_time = total_duration - window_length_s
    mask = t_cum >= start_time
    return rr[mask]


# -------------------- FREQUENCY-DOMAIN HESAP -------------------- #

def _compute_freq_domain_from_rr(
    rr: np.ndarray,
    fs_resample: float,
    vlf_band: Tuple[float, float],
    lf_band: Tuple[float, float],
    hf_band: Tuple[float, float],
) -> Dict[str, object]:
    """
    RR (saniye) serisinden frekans-domeni HRV metriklerini hesaplar.

    Adımlar:
      1) Kümülatif zamana göre RR(t) elde et.
      2) Sabit örneklem frekansına (fs_resample) göre yeniden örnekle.
      3) Ortalama çıkar ve Welch yöntemi ile PSD hesapla.
      4) VLF / LF / HF band güçlerini integre et.
    """
    rr = np.asarray(rr, dtype=float)
    if rr.size < 4:
        return {
            "freq": [],
            "psd": [],
            "band_powers": {"VLF": 0.0, "LF": 0.0, "HF": 0.0},
            "lf_hf_ratio": float("nan"),
        }

    # Zaman ekseni (saniye)
    t = np.cumsum(rr)
    total_duration = float(t[-1])
    if total_duration <= 0:
        return {
            "freq": [],
            "psd": [],
            "band_powers": {"VLF": 0.0, "LF": 0.0, "HF": 0.0},
            "lf_hf_ratio": float("nan"),
        }

    # Sabit örneklem zaman ekseni
    dt = 1.0 / fs_resample
    t_uniform = np.arange(0.0, total_duration, dt)
    if t_uniform.size < 4:
        return {
            "freq": [],
            "psd": [],
            "band_powers": {"VLF": 0.0, "LF": 0.0, "HF": 0.0},
            "lf_hf_ratio": float("nan"),
        }

    # RR(t) sinyalini yeniden örnekle (cubic spline ile, olmazsa lineer)
    try:
        cs = CubicSpline(t, rr, bc_type="natural")
        rr_interp = cs(t_uniform)
    except Exception:
        rr_interp = np.interp(t_uniform, t, rr)

    # Detrend (ortalama çıkar)
    rr_detrended = rr_interp - np.mean(rr_interp)

    # Welch PSD
    nperseg = min(256, t_uniform.size)
    freq, psd = welch(
        rr_detrended,
        fs=fs_resample,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )

    # Bant güçlerini hesapla
    def _band_power(f_low: float, f_high: float) -> float:
        mask = (freq >= f_low) & (freq < f_high)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freq[mask]))

    vlf = _band_power(*vlf_band)
    lf = _band_power(*lf_band)
    hf = _band_power(*hf_band)

    lf_hf_ratio = float(lf / hf) if hf > 0 else float("nan")

    return {
        "freq": freq.tolist(),
        "psd": psd.tolist(),
        "band_powers": {"VLF": vlf, "LF": lf, "HF": hf},
        "lf_hf_ratio": lf_hf_ratio,
    }

def _rr_to_uniform_hrv(
    rr: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RR intervals in seconds to a uniform HRV(t) signal at fs Hz."""
    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return np.asarray([]), np.asarray([])

    t_rr = np.cumsum(rr)
    t_rr = t_rr - t_rr[0]

    total_dur = float(t_rr[-1])
    dt = 1.0 / fs
    t_uniform = np.arange(0.0, total_dur, dt)
    if t_uniform.size < 4:
        hrv = np.interp(t_uniform, t_rr, rr)
        return t_uniform, hrv

    try:
        cs = CubicSpline(t_rr, rr)
        hrv = cs(t_uniform)
    except Exception:
        hrv = np.interp(t_uniform, t_rr, rr)

    return t_uniform, hrv


def _run_vmd_1mode(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Run single-mode VMD and return the decomposed component."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 10:
        return np.zeros_like(x)

    u, u_hat, omega = VMD(
        x,
        alpha,
        VMD_TAU,
        1,         # K=1
        0,         # dc=0
        VMD_INIT,
        VMD_TOL,
    )
    # u shape: (K, N) -> (1, N)
    return u[0]


def _sliding_component_vmdon(
    signal: np.ndarray,
    fs: float,
    window_s: float,
    use_vmd: bool,
    alpha: float,
    stride: int,
) -> np.ndarray:
    """Compute one sliding-window VMDON component (HF/LF/VLF/ULF)."""
    x = np.asarray(signal, dtype=float)
    N = x.size
    W = int(round(window_s * fs))
    if W < 4 or W >= N:
        return np.zeros_like(x)

    acc = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    t_local = np.arange(W)

    for start in range(0, N - W + 1, stride):
        w = x[start:start + W]
        p = np.polyfit(t_local, w, 1)
        trend = np.polyval(p, t_local)
        w_detr = w - trend

        if use_vmd:
            comp_win = _run_vmd_1mode(w_detr, alpha=alpha)
        else:
            comp_win = trend

        acc[start:start + W] += comp_win
        cnt[start:start + W] += 1.0

    out = np.zeros(N, dtype=float)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out


def _vmdon_decompose(
    hrv: np.ndarray,
    fs: float,
    stride_hf: int = 1,
    stride_lf: int = 2,
    stride_vlf: int = 10,
    stride_ulf: int = 60,
) -> Dict[str, np.ndarray]:
    """VMDon-like offline decomposition with sliding windows."""
    x = np.asarray(hrv, dtype=float)
    if x.size < 10:
        return {
            "HF": np.zeros_like(x),
            "LF": np.zeros_like(x),
            "VLF": np.zeros_like(x),
            "ULF": np.zeros_like(x),
        }

    hf = _sliding_component_vmdon(
        x, fs,
        VMDON_WINDOW_S["HF"],
        use_vmd=True,
        alpha=VMD_ALPHA,
        stride=stride_hf,
    )
    r = x - hf

    lf = _sliding_component_vmdon(
        r, fs,
        VMDON_WINDOW_S["LF"],
        use_vmd=True,
        alpha=VMD_ALPHA,
        stride=stride_lf,
    )
    r = r - lf

    vlf = _sliding_component_vmdon(
        r, fs,
        VMDON_WINDOW_S["VLF"],
        use_vmd=True,
        alpha=VMD_ALPHA,
        stride=stride_vlf,
    )
    r = r - vlf

    ulf = _sliding_component_vmdon(
        r, fs,
        VMDON_WINDOW_S["ULF"],
        use_vmd=False,
        alpha=VMD_ALPHA,
        stride=stride_ulf,
    )

    return {"HF": hf, "LF": lf, "VLF": vlf, "ULF": ulf}


def get_freq_domain_metrics(
    subject_code: str,
    window_length_s: Optional[float] = None,
    fs_resample: Optional[float] = None,
) -> Dict[str, object]:
    """
    Verilen subject için frekans-domeni HRV metriklerini döner.

    Parametreler:
        window_length_s:
            None -> tüm buffer/kayıt
            >0  -> son window_length_s saniye
        fs_resample:
            Yeniden örnekleme frekansı (Hz). None ise settings.hrv.fs_resample kullanılır.
    """
    if fs_resample is None:
        fs_resample = settings.hrv.fs_resample

    vlf_band = settings.hrv.vlf_band
    lf_band = settings.hrv.lf_band
    hf_band = settings.hrv.hf_band

    rr = load_rr(subject_code, window_length_s=window_length_s)
    fd = _compute_freq_domain_from_rr(
        rr,
        fs_resample=fs_resample,
        vlf_band=vlf_band,
        lf_band=lf_band,
        hf_band=hf_band,
    )
    fd["window_length_s"] = window_length_s
    return fd

def get_vmdon_components(
    subject_code: str,
    window_length_s: Optional[float] = None,
    fs_resample: Optional[float] = None,
) -> Dict[str, object]:
    """Compute VMDON-based HF/LF/VLF/ULF components for a given subject/window."""
    if fs_resample is None:
        fs_resample = settings.hrv.fs_resample

    rr = load_rr(subject_code, window_length_s=window_length_s)
    if rr.size < 4:
        return {
            "t_sec": [],
            "hf": [],
            "lf": [],
            "vlf": [],
            "ulf": [],
            "window_length_s": window_length_s,
        }

    # RR -> uniform HRV(t)
    t_uniform, hrv = _rr_to_uniform_hrv(rr, fs=fs_resample)
    if hrv.size < 10:
        return {
            "t_sec": t_uniform.tolist(),
            "hf": [],
            "lf": [],
            "vlf": [],
            "ulf": [],
            "window_length_s": window_length_s,
        }

    # Basit detrend (mean)
    hrv_detrended = hrv - np.mean(hrv)

    comps = _vmdon_decompose(hrv_detrended, fs=fs_resample)

    return {
        "t_sec": t_uniform.tolist(),
        "hf": comps["HF"].tolist(),
        "lf": comps["LF"].tolist(),
        "vlf": comps["VLF"].tolist(),
        "ulf": comps["ULF"].tolist(),
        "window_length_s": window_length_s,
    }

# -------------------- HR TIME-SERIES -------------------- #

def get_hr_timeseries(
    subject_code: str,
    max_points: Optional[int] = None,
    window_length_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verilen subject için HR zaman serisini döner.

    Dönüş:
        t_sec  : kümülatif zaman (saniye)
        hr_bpm : kalp hızı (bpm)

    max_points None ise, settings.dashboard.max_hr_points kullanılır.
    """
    rr = load_rr(subject_code, window_length_s=window_length_s)

    if rr.size < 1:
        return np.asarray([]), np.asarray([])

    if max_points is None:
        max_points = settings.dashboard.max_hr_points

    t_sec = np.cumsum(rr)
    hr_bpm = 60.0 / rr

    if t_sec.size > max_points:
        t_sec = t_sec[-max_points:]
        hr_bpm = hr_bpm[-max_points:]

    return t_sec, hr_bpm


def get_time_domain_metrics(
    subject_code: str = "000",
    window_length_s: Optional[float] = None,
) -> Dict[str, float]:
    """
    Time-domain HRV metriklerini döner (sdnn, rmssd, pnn50, hr_min, hr_max, ...).
    """
    rr = load_rr(subject_code, window_length_s=window_length_s)
    td = _compute_time_domain_from_rr(rr)
    td["window_length_s"] = window_length_s
    return td


def get_poincare_data(
    subject_code: str,
    max_points: Optional[int] = None,
    window_length_s: Optional[float] = None,
) -> Dict[str, object]:
    """
    Poincaré plot verisini ve non-linear metrikleri döner.

    Dönüş:
        x, y:
            RRₙ ve RRₙ₊₁ (ms cinsinden), scatter plot için
        sd1, sd2:
            Kısa ve uzun eksen standart sapmaları
        sd1_sd2_ratio:
            SD1 / SD2 oranı
        stress_index:
            SD2 / SD1 oranına dayalı basit bir stres indeksi
    """
    rr = load_rr(subject_code, window_length_s=window_length_s)

    if rr.size < 3:
        return {
            "x": [],
            "y": [],
            "sd1": float("nan"),
            "sd2": float("nan"),
            "sd1_sd2_ratio": float("nan"),
            "stress_index": float("nan"),
        }

    if max_points is None:
        max_points = settings.dashboard.max_poincare_points

    rr_ms = rr * 1000.0
    x = rr_ms[:-1]
    y = rr_ms[1:]

    if x.size > max_points:
        x = x[-max_points:]
        y = y[-max_points:]

    sdnn = float(np.std(rr_ms, ddof=1))
    diff_ms = np.diff(rr_ms)
    var_diff = float(np.var(diff_ms, ddof=1))

    sd1 = float(np.sqrt(0.5 * var_diff))
    sd2_inside = 2.0 * (sdnn ** 2) - 0.5 * var_diff
    sd2 = float(np.sqrt(max(sd2_inside, 0.0)))

    sd1_sd2_ratio = float(sd1 / sd2) if sd2 > 0 else float("nan")
    stress_index = float(sd2 / sd1) if sd1 > 0 else float("nan")

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1_sd2_ratio,
        "stress_index": stress_index,
    }


from typing import Dict, Any  # dosyanın üstünde varsa tekrar ekleme

def get_subject_info(subject_code: str) -> Dict[str, Any]:
    """
    Demografik bilgiler: age / sex / group (child/adult/older).

    CSV formatını esnek okur:
    - İlk satır "code;age;sex" veya "File;age;sex" olabilir (header).
    - Ya da hiç header olmayıp direkt "0;53;M" ile başlayabilir.
    - İlk kolon bazen "File" bazen "code" olabilir.
    """
    try:
        df = pd.read_csv(
            PATIENT_INFO_PATH,
            sep=";",              # ; ile ayrılmış
            header=None,          # header olduğunu varsaymıyoruz
            names=["code", "age", "sex"],
        )
    except FileNotFoundError:
        logger.warning(
            "Patient info CSV not found at %s. Returning empty demographics for subject=%s",
            PATIENT_INFO_PATH,
            subject_code,
        )
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    # Olası header satırlarını (code/File) at
    df["code"] = df["code"].astype(str)
    mask_header = df["code"].str.lower().isin(["code", "file"])
    df = df[~mask_header]

    if df.empty:
        logger.warning(
            "Patient info CSV at %s is empty after header filtering. subject=%s",
            PATIENT_INFO_PATH,
            subject_code,
        )
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    # code kolonunu sayıya çevir (0, 2, 401 vs.)
    df["code"] = pd.to_numeric(df["code"], errors="coerce")

    # subject_code -> int (örn. "000" -> 0)
    try:
        code_int = int(subject_code)
    except ValueError:
        code_int = None

    if code_int is not None:
        row = df[df["code"] == code_int]
    else:
        # Sayıya çevrilemiyorsa fallback string karşılaştırma
        row = df[df["code"].astype(str) == str(subject_code)]

    if row.empty:
        logger.warning(
            "No matching patient info row for subject_code=%s in %s",
            subject_code,
            PATIENT_INFO_PATH,
        )
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    age = row.iloc[0]["age"]
    sex = row.iloc[0]["sex"]

    # Yaşa göre grup (sadece gerçekten sayıysa)
    try:
        age_val = float(age)
    except (TypeError, ValueError):
        age_val = None

    if age_val is None:
        group = None
    elif age_val < 18:
        group = "Child / adolescent"
    elif age_val < 65:
        group = "Adult"
        # buradaki eşikler istersen settings.hrv içinde parametrelenebilir
    else:
        group = "Older adult"

    return {"code": subject_code, "age": age, "sex": sex, "group": group}



def _compute_signal_quality(rr: np.ndarray) -> Dict[str, object]:
    """
    RR serisinden basit bir sinyal kalite özeti üretir.

    Not:
        Bu sadece teknik bir kalite değerlendirmesidir, klinik/medikal yorum değildir.
    """
    rr = np.asarray(rr, dtype=float)
    n_total = rr.size

    if n_total == 0:
        return {
            "quality_label": "No data",
            "status_text": "No RR intervals available",
            "outlier_percent": 100.0,
            "n_total": 0,
            "n_outliers": 0,
        }

    # Basit aralık kontrolü (saniye cinsinden)
    lower, upper = 0.3, 2.0
    mask_range = (rr < lower) | (rr > upper)

    # Ardışık farklar
    diff = np.abs(np.diff(rr))
    jump_threshold = 0.3  # 300 ms
    jump_mask = diff > jump_threshold

    n_out_of_range = int(mask_range.sum())
    n_big_jumps = int(jump_mask.sum())
    n_outliers = n_out_of_range + n_big_jumps

    outlier_percent = (n_outliers / n_total) * 100.0 if n_total > 0 else 0.0

    # Kısa kayıt kontrolü
    if n_total < 10:
        quality_label = "Short recording"
        status_text = "Recording too short for stable HRV estimation"
    else:
        # Çok kabaca 3 seviye
        if outlier_percent < 1.0:
            quality_label = "OK"
            status_text = "Normal (no alerts detected in the last window)"
        elif outlier_percent < 5.0:
            quality_label = "Moderate"
            status_text = f"Check signal (≈{outlier_percent:.1f}% irregular RR intervals)"
        else:
            quality_label = "Poor"
            status_text = f"Low signal quality (≈{outlier_percent:.1f}% irregular RR intervals)"

    return {
        "quality_label": quality_label,
        "status_text": status_text,
        "outlier_percent": float(outlier_percent),
        "n_total": int(n_total),
        "n_outliers": int(n_outliers),
    }


def get_signal_status(subject_code: str, window_length_s: Optional[float] = None) -> Dict[str, object]:
    """
    Belirli bir subject ve pencere için sinyal kalitesi özetini döner.
    """
    rr = load_rr(subject_code, window_length_s=window_length_s)
    return _compute_signal_quality(rr)


