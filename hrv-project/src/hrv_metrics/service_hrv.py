# src/hrv_metrics/service_hrv.py

from pathlib import Path
import numpy as np
import pandas as pd


# Proje kökü: .../hrv-project
ROOT_DIR = Path(__file__).parents[2]

# RR dosyalarının olduğu klasör:
# hrv-project/data/processed/rr_clean
RR_DIR = ROOT_DIR / "data" / "processed" / "rr_clean"

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
    return codes


# -------------------- RR KAYNAĞI -------------------- #

def load_rr_from_csv(subject_code: str) -> np.ndarray:
    """
    Belirli bir denek için RR serisini CSV'den okur.
    subject_code: '000', '002', '401' gibi.
    Beklenen dosya adı: 000_clean.csv, 002_clean.csv, 401_clean.csv ...
    """
    csv_path = RR_DIR / f"{subject_code}_clean.csv"

    if not csv_path.exists():
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
        raise ValueError(
            f"RR column not found in {csv_path}. "
            f"Expected one of: 'rr', 'rr_ms'. "
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    return rr_sec


# -------------------- TIME-DOMAIN HESAP -------------------- #

def _compute_time_domain_from_rr(rr: np.ndarray) -> dict:
    """
    Tek boyutlu RR serisinden temel time-domain HRV metriklerini hesaplar.
    rr: saniye cinsinden RR intervalleri (ör: 0.8 = 800 ms)
    Dönen değerler dict: sdnn, rmssd, pnn50, mean_hr, hr_max, hr_min
    """
    rr = np.asarray(rr, dtype=float)

    if rr.ndim != 1 or rr.size < 2:
        raise ValueError("RR series must be 1D and contain at least 2 samples")

    # ms cinsine çevir
    rr_ms = rr * 1000.0

    # SDNN (ms)
    sdnn = float(np.std(rr_ms, ddof=1))

    # RMSSD (ms)
    diff_ms = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_ms ** 2)))

    # NN50 & pNN50 (%)
    nn50 = int(np.sum(np.abs(diff_ms) > 50.0))
    if diff_ms.size > 0:
        pnn50 = float(nn50 / diff_ms.size * 100.0)
    else:
        pnn50 = 0.0

    # Mean RR, HR min/max (bpm)
    mean_rr = float(np.mean(rr))
    min_rr = float(np.min(rr))
    max_rr = float(np.max(rr))

    mean_hr = float(60.0 / mean_rr) if mean_rr > 0 else float("nan")
    hr_max = float(60.0 / min_rr) if min_rr > 0 else float("nan")
    hr_min = float(60.0 / max_rr) if max_rr > 0 else float("nan")

    return {
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "mean_hr": mean_hr,
        "hr_max": hr_max,
        "hr_min": hr_min,
    }

def get_hr_timeseries(subject_code: str, max_points: int = 500):
    """
    Belirli bir denek için RR serisinden HR (bpm) zaman serisi üretir.
    - RR saniye cinsinden.
    - Zaman ekseni: kümülatif RR (gerçek geçen süre).
    """
    rr = load_rr_from_csv(subject_code)  # zaten var olan fonksiyon

    if rr.size < 1:
        return [], []

    # Zaman ekseni: rr'lerin kümülatif toplamı (saniye)
    t_sec = np.cumsum(rr)

    # Kalp hızı: bpm
    hr_bpm = 60.0 / rr

    # Çok uzun serileri ekran için kısalt (son max_points örnek)
    if t_sec.size > max_points:
        t_sec = t_sec[-max_points:]
        hr_bpm = hr_bpm[-max_points:]

    return t_sec, hr_bpm



def get_time_domain_metrics(subject_code: str = "000") -> dict:
    """
    Dashboard tarafından kullanılan ana fonksiyon.
    - Belirli bir subject_code için (örn: '000') RR verisini CSV'den okur,
    - Time-domain metrikleri hesaplar,
    - Dashboard'un beklediği sade dict'i döner.
    """
    rr = load_rr_from_csv(subject_code)
    td = _compute_time_domain_from_rr(rr)
    return td

def get_poincare_data(subject_code: str, max_points: int = 1000) -> dict:
    """
    Poincaré diyagramı için:
    - RR_n ve RR_{n+1} (ms cinsinden)
    - SD1, SD2, SD1/SD2 oranı ve basit bir stress index döner.
    """
    rr = load_rr_from_csv(subject_code)  # saniye cinsinden RR

    if rr.size < 3:
        return {
            "x": [],
            "y": [],
            "sd1": float("nan"),
            "sd2": float("nan"),
            "sd1_sd2_ratio": float("nan"),
            "stress_index": float("nan"),
        }

    # ms'e çevir
    rr_ms = rr * 1000.0

    # Poincaré noktaları: RR_n (x), RR_{n+1} (y)
    x = rr_ms[:-1]
    y = rr_ms[1:]

    # Çok uzun serilerde son max_points noktayı al
    if x.size > max_points:
        x = x[-max_points:]
        y = y[-max_points:]

    # SD1 / SD2 hesapları (Task Force formülleri)
    sdnn = float(np.std(rr_ms, ddof=1))              # tüm RR'nin std'si (ms)
    diff_ms = np.diff(rr_ms)                         # ardışık farklar
    var_diff = float(np.var(diff_ms, ddof=1))

    sd1 = float(np.sqrt(0.5 * var_diff))            # sd1
    # içi negatif olmasın diye max(..., 0.0)
    sd2_inside = 2.0 * (sdnn ** 2) - 0.5 * var_diff
    sd2 = float(np.sqrt(max(sd2_inside, 0.0)))      # sd2

    sd1_sd2_ratio = float(sd1 / sd2) if sd2 > 0 else float("nan")
    # Şimdilik basit bir numerik stress index: SD2/SD1
    stress_index = float(sd2 / sd1) if sd1 > 0 else float("nan")

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1_sd2_ratio,
        "stress_index": stress_index,
    }
