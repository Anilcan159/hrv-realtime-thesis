# src/streaming/preprocessing.py
# Converts raw PhysioNet RR txt files into per-subject CLEAN CSV files.
# Bilimsel arka plan:
# - RR < 300 ms veya > 2000 ms genellikle artefakt / ektopik olarak kabul edilir. [Vest et al., 2018]
# - |RRn - RRn-1| > ~200 ms veya >%20 fark içeren atımlar sık kullanılan eşiklerden biridir.
#   (Clifford 2006, Vest 2018; özet: PhysioNet toolbox artefact rules)

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# Root paths
# __file__ = .../hrv-project/src/streaming/preprocessing.py
BASE_DIR = Path(__file__).resolve().parents[2]  # -> .../hrv-project
RAW_DATA_DIR = (
    BASE_DIR
    / "Datas"
    / "raw"
    / "physionet.org"
    / "files"
    / "rr-interval-healthy-subjects"
    / "1.0.0"
)
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "rr_clean"


# ---------------------------------------------------------------------
# 1) RR SERISINI YUKLE
# ---------------------------------------------------------------------
def load_rr_series(subject_id: str) -> pd.DataFrame:
    """
    Ham RR serisini (ms) DataFrame olarak yükler.
    RAW_DATA_DIR içinde '001.txt' gibi dosya beklenir.

    Çıktı:
        df_raw: tek kolonlu DataFrame, kolon adı 'rr_ms'
    """
    file_path = RAW_DATA_DIR / f"{subject_id}.txt"
    df = pd.read_csv(file_path, header=None, names=["rr_ms"])
    return df


# ---------------------------------------------------------------------
# 2) ARTEFAKT TESPITI
# ---------------------------------------------------------------------
def detect_rr_artifacts(
    rr_ms: np.ndarray,
    rr_min_ms: int = 300,
    rr_max_ms: int = 2000,
    max_abs_diff_ms: int = 200,
    max_rel_diff: float = 0.20,
) -> np.ndarray:
    """
    RR serisinde artefakt / şüpheli atımları işaretler.

    Kullanılan kurallar (literatürde yaygın pratiklere dayanır):
      1) Fiziksel sınırlar:
         - RR < rr_min_ms  (≈ HR > 200 bpm)
         - RR > rr_max_ms  (≈ HR < 30 bpm)

      2) Ardışık mutlak fark:
         - |RR_n - RR_{n-1}| > max_abs_diff_ms

      3) Göreli fark:
         - |RR_n - RR_{n-1}| / RR_{n-1} > max_rel_diff  (örn. > %20 değişim)

    Parametreler:
        rr_ms          : RR intervalleri (ms)
        rr_min_ms      : minimum kabul edilebilir RR (ms)
        rr_max_ms      : maksimum kabul edilebilir RR (ms)
        max_abs_diff_ms: ardışık RR farkı için mutlak eşik (ms)
        max_rel_diff   : ardışık RR farkı için göreli eşik (0.20 = %20)

    Döner:
        is_artifact: bool array, True olan indexler artefakt / şüpheli atım.
    """
    rr = np.asarray(rr_ms, dtype=float)
    n = rr.size

    if n == 0:
        return np.array([], dtype=bool)

    # Başlangıç: NaN / inf ve fiziksel aralık dışında kalanlar
    is_artifact = ~np.isfinite(rr)
    is_artifact |= (rr < rr_min_ms) | (rr > rr_max_ms)

    if n > 1:
        diff = np.abs(np.diff(rr))
        rel_diff = diff / np.maximum(rr[:-1], 1e-6)

        big_abs_jump = diff > max_abs_diff_ms
        big_rel_jump = rel_diff > max_rel_diff

        # Bu sıçramayı hem yeni beat'e hem gerekirse önceki beat'e atfedebiliriz.
        # Burada sade olması için sadece "yeni" atımı işaretliyoruz.
        jump_mask = big_abs_jump | big_rel_jump
        is_artifact[1:] |= jump_mask

    return is_artifact


# ---------------------------------------------------------------------
# 3) ARTEFAKT DUZELTME (REMOVE / INTERPOLATE)
# ---------------------------------------------------------------------
def correct_rr_artifacts(
    rr_ms: np.ndarray,
    is_artifact: np.ndarray,
    method: str = "interpolate",
) -> np.ndarray:
    """
    Artefakt olarak işaretlenen RR değerlerini düzeltir.

    method:
        - "remove"      : artefakt beat'leri tamamen çıkarır
        - "interpolate" : artefakt beat'lerin değerini komşu düzgün beat'ler
                          arasında lineer interpolasyon ile doldurur
        - "none"        : hiçbir düzeltme yapmaz, rr_ms'i aynen döner

    Döner:
        rr_clean_ms: temizlenmiş RR serisi (ms)
    """
    rr = np.asarray(rr_ms, dtype=float)
    is_artifact = np.asarray(is_artifact, dtype=bool)

    if rr.size == 0:
        return rr

    if method == "none":
        return rr

    # Eğer hiç artefakt yoksa direkt dönebiliriz
    if not np.any(is_artifact):
        return rr

    if method == "remove":
        rr_clean = rr[~is_artifact]
        return rr_clean

    if method == "interpolate":
        rr_clean = rr.copy()
        n = rr.size
        good_idx = np.where(~is_artifact)[0]

        # Eğer yeterli sağlam beat yoksa (<=2), dokunmadan döndür
        if good_idx.size < 2:
            return rr

        artifact_idx = np.where(is_artifact)[0]

        for i in artifact_idx:
            # Soldaki ve sağdaki sağlam indexleri bul
            left_candidates = good_idx[good_idx < i]
            right_candidates = good_idx[good_idx > i]

            if left_candidates.size == 0 and right_candidates.size == 0:
                # Her yer artefakt ise yapacak pek bir şey yok
                continue
            elif left_candidates.size == 0:
                # Sadece sağ komşu var => en yakın sağlam değeri kopyala
                rr_clean[i] = rr[right_candidates[0]]
            elif right_candidates.size == 0:
                # Sadece sol komşu var
                rr_clean[i] = rr[left_candidates[-1]]
            else:
                i0 = left_candidates[-1]
                i1 = right_candidates[0]
                # index alanında lineer interpolasyon
                rr_clean[i] = np.interp(i, [i0, i1], [rr[i0], rr[i1]])

        return rr_clean

    raise ValueError(f"Unknown artifact correction method: {method}")


# ---------------------------------------------------------------------
# 4) TEMIZLEME + RAPOR
# ---------------------------------------------------------------------
def clean_rr_values(
    df: pd.DataFrame,
    rr_min: int = 300,
    rr_max: int = 2000,
    max_abs_diff_ms: int = 200,
    max_rel_diff: float = 0.20,
    correction_method: str = "interpolate",
    return_info: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    RR serisi için tam temizlik pipeline'ı.

    Adımlar:
      1) rr_ms kolonunu numerik yap, NaN'leri at.
      2) detect_rr_artifacts ile artefakt beat'leri işaretle.
      3) correct_rr_artifacts ile seçilen metoda göre düzelt.
      4) Sadece 'rr_ms' kolonunu içeren yeni bir DataFrame döndür.

    Parametreler:
        rr_min, rr_max      : fiziksel aralık (ms)
        max_abs_diff_ms     : ardışık fark için mutlak eşik (ms)
        max_rel_diff        : ardışık fark için göreli eşik
        correction_method   : "interpolate" / "remove" / "none"
        return_info         : True ise temizlik istatistiklerini de döner.

    Döner:
        df_clean: tek kolonlu DataFrame ("rr_ms")
        info    : (opsiyonel) temizlik istatistikleri dict
    """
    df = df.copy()

    # 1) rr_ms'i sayıya çevir ve NaN/boş satırları at
    df["rr_ms"] = pd.to_numeric(df["rr_ms"], errors="coerce")
    df = df.dropna(subset=["rr_ms"])

    rr = df["rr_ms"].to_numpy(dtype=float)
    n_total = int(rr.size)

    if n_total == 0:
        df_clean = pd.DataFrame(columns=["rr_ms"])
        info = {
            "n_total": 0,
            "n_artifacts": 0,
            "artifact_percent": 0.0,
            "correction_method": correction_method,
        }
        if return_info:
            return df_clean, info
        return df_clean, {}

    # 2) Artefaktları tespit et
    is_artifact = detect_rr_artifacts(
        rr_ms=rr,
        rr_min_ms=rr_min,
        rr_max_ms=rr_max,
        max_abs_diff_ms=max_abs_diff_ms,
        max_rel_diff=max_rel_diff,
    )

    n_artifacts = int(is_artifact.sum())
    artifact_percent = (n_artifacts / n_total) * 100.0

    # 3) Düzelt
    rr_clean = correct_rr_artifacts(rr, is_artifact, method=correction_method)

    # 4) DataFrame oluştur ve index sıfırla
    df_clean = pd.DataFrame({"rr_ms": rr_clean})
    df_clean = df_clean.reset_index(drop=True)

    info = {
        "n_total": n_total,
        "n_artifacts": n_artifacts,
        "artifact_percent": float(artifact_percent),
        "correction_method": correction_method,
    }

    if return_info:
        return df_clean, info
    return df_clean, {}


# ---------------------------------------------------------------------
# 5) ZAMAN VE HR EKLE
# ---------------------------------------------------------------------
def add_time_and_hr(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    Kümülatif zaman (s), kalp hızı (bpm) ve subject_id kolonlarını ekler.

    Varsayım:
        df["rr_ms"] temizlenmiş RR intervallerini içerir.
    """
    df = df.copy()

    # Kümülatif zaman (saniye)
    df["time_s"] = df["rr_ms"].cumsum() / 1000.0

    # Kalp hızı (bpm) -> 60000 / rr_ms
    df["hr_bpm"] = 60000.0 / df["rr_ms"]

    # Subject ID
    df["subject_id"] = subject_id

    return df


# ---------------------------------------------------------------------
# 6) TEK BIR SUBJECT ICIN PIPELINE
# ---------------------------------------------------------------------
def process_single_subject(subject_id: str) -> None:
    """
    Tek bir subject için tam pipeline:
        raw txt -> temizlenmiş RR -> zaman & HR ekle -> *_clean.csv

    Çıktı CSV kolonları:
        - rr_ms     : temiz RR intervalleri (ms)
        - time_s    : kümülatif zaman (s)
        - hr_bpm    : anlık kalp hızı (bpm)
        - subject_id
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_rr_series(subject_id)
    df_clean, info = clean_rr_values(
        df_raw,
        rr_min=300,
        rr_max=2000,
        max_abs_diff_ms=200,
        max_rel_diff=0.20,
        correction_method="interpolate",  # istersen "remove" yapabilirsin
        return_info=True,
    )
    df_final = add_time_and_hr(df_clean, subject_id)

    out_path = PROCESSED_DIR / f"{subject_id}_clean.csv"
    df_final.to_csv(out_path, index=False)

    n_total = info["n_total"]
    n_artifacts = info["n_artifacts"]
    perc = info["artifact_percent"]

    print(
        f"Saved: {out_path} | beats: {n_total} | "
        f"artifacts: {n_artifacts} ({perc:.2f}%) | "
        f"method: {info['correction_method']}"
    )


# ---------------------------------------------------------------------
# 7) TUM SUBJECT'LER ICIN PIPELINE
# ---------------------------------------------------------------------
def process_all_subjects() -> None:
    """
    RAW_DATA_DIR altındaki tüm txt dosyalarını dönerek işler.
    Sadece ismi tamamen sayılardan oluşan dosyalar (000, 401, ...) subject kabul edilir.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} raw txt files.")

    for f in txt_files:
        subject_id = f.stem  # "000", "401", "LICENSE", ...

        if not subject_id.isdigit():
            print(f"Skipping non-subject file: {f.name}")
            continue

        print(f"Processing subject {subject_id} ...")
        process_single_subject(subject_id)


if __name__ == "__main__":
    # Script olarak çalıştırıldığında tüm subject'leri işle
    process_all_subjects()
