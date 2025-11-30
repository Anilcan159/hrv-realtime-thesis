# src/hrv_metrics/service_hrv.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import welch

# Proje kökü: .../hrv-project
ROOT_DIR = Path(__file__).parents[2]
PATIENT_INFO_PATH = ROOT_DIR / "Datas" / "processed" / "patient-info.csv"

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

# -------------------- FREQUENCY-DOMAIN HESAP -------------------- #

# Computes VLF / LF / HF band powers from RR series (seconds)
def _compute_freq_domain_from_rr(rr: np.ndarray, fs_resample: float = 4.0) -> dict:
    """
    RR serisini (saniye cinsinden) zaman eksenine yerleştirip
    sabit örneklemeli bir sinyale (4 Hz) yeniden örnekler,
    ardından Welch yöntemi ile PSD hesaplayıp VLF / LF / HF güçlerini döner.
    """
    rr = np.asarray(rr, dtype=float)

    if rr.ndim != 1 or rr.size < 4:
        raise ValueError("RR series must be 1D and contain at least 4 samples")

    # 1) Zaman ekseni: kümülatif RR (saniye)
    t = np.cumsum(rr)
    t = t - t[0]  # 0'dan başlat

    # 2) Uniform zaman ekseni (örneğin 4 Hz)
    if t[-1] <= 0:
        raise ValueError("Total duration must be positive")

    dt = 1.0 / fs_resample
    t_uniform = np.arange(0.0, t[-1], dt)

    if t_uniform.size < 8:
        raise ValueError("Not enough duration for frequency analysis")

    # 3) RR sinyalini uniform eksene spline ile yeniden örnekle
    spline = CubicSpline(t, rr)
    rr_interp = spline(t_uniform)

    # 4) Trend'i kaldır (ortalama çıkar)
    rr_detrended = rr_interp - np.mean(rr_interp)

    # 5) Welch ile PSD
    nperseg = min(256, rr_detrended.size)
    f, Pxx = welch(rr_detrended, fs=fs_resample, nperseg=nperseg)

    # 6) Bant güçleri
    bands = {
        "VLF": (0.0033, 0.04),
        "LF":  (0.04, 0.15),
        "HF":  (0.15, 0.40),
    }

    band_powers = {}
    for name, (low, high) in bands.items():
        mask = (f >= low) & (f < high)
        if np.any(mask):
            band_powers[name] = float(np.trapz(Pxx[mask], f[mask]))
        else:
            band_powers[name] = 0.0

    lf_power = band_powers.get("LF", 0.0)
    hf_power = band_powers.get("HF", 0.0)
    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else float("nan")

    # İleride Dash grafikleri için lazım olabilecek her şeyi döndürüyoruz
    return {
        "freq": f.tolist(),          # frekans ekseni (Hz)
        "psd": Pxx.tolist(),         # PSD değerleri
        "band_powers": band_powers,  # dict: VLF/LF/HF
        "lf_hf_ratio": lf_hf_ratio,
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

def get_time_domain_metrics(subject_code: str = "000") -> dict:
    """
    Dashboard tarafından kullanılan ana fonksiyon.
    ...
    """
    rr = load_rr_from_csv(subject_code)
    td = _compute_time_domain_from_rr(rr)
    return td


def get_freq_domain_metrics(subject_code: str = "000", fs_resample: float = 4.0) -> dict:
    """
    Dashboard tarafından kullanılan frekans domeni fonksiyonu.
    - Belirli bir subject_code için RR verisini okur,
    - VLF / LF / HF band güçlerini ve LF/HF oranını hesaplar.
    """
    rr = load_rr_from_csv(subject_code)
    fd = _compute_freq_domain_from_rr(rr, fs_resample=fs_resample)
    return fd


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


def get_subject_info(subject_code: str) -> dict:
    """
    Demografik bilgiler: age / sex / group (child/adult/older).

    CSV formatını esnek okur:
    - İlk satır "code;age;sex" veya "File;age;sex" olabilir (header).
    - Ya da hiç header olmayıp direkt "0;53;M" ile başlayabilir.
    - İlk kolon bazen "File" bazen "code" olabilir.
    """

    # 1) CSV'yi oku (header varmış/yokmuş gibi düşünmeden)
    try:
        df = pd.read_csv(
            PATIENT_INFO_PATH,
            sep=";",              # ; ile ayrılmış
            header=None,          # header olduğunu varsaymıyoruz
            names=["code", "age", "sex"],
        )
    except FileNotFoundError:
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    # 2) Eğer ilk satır aslında header ise (code/File yazıyorsa) onu at
    df["code"] = df["code"].astype(str)
    mask_header = df["code"].str.lower().isin(["code", "file"])
    df = df[~mask_header]

    if df.empty:
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    # 3) code kolonunu sayıya çevir (0, 2, 401 vs.)
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
        return {"code": subject_code, "age": None, "sex": None, "group": None}

    age = row.iloc[0]["age"]
    sex = row.iloc[0]["sex"]

    # 4) Yaşa göre grup (sadece gerçekten sayıysa)
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
    else:
        group = "Older adult"

    return {"code": subject_code, "age": age, "sex": sex, "group": group}



