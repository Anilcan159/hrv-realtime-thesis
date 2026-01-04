# src/streaming/preprocessing.py
# Offline preprocessing:
#   - Raw PhysioNet RR txt -> cleaned per-subject CSV
#   - Physiological filter + artifact correction + safe interpolation + time / HR

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd

from src.config.settings import settings


# Root paths (config-driven)
RAW_DATA_DIR: Path = settings.paths.raw_data_dir
PROCESSED_DIR: Path = settings.paths.rr_processed_dir


# Load raw RR series (ms) for a single subject.
def load_rr_series(subject_id: str) -> pd.DataFrame:
    """
    Raw RR serisini (ms) tek sütunlu DataFrame olarak yükler.

    Örnek:
        RAW_DATA_DIR / '000.txt'  ->  rr_ms kolonu

    Her satır bir RR interval (ms) temsil eder.
    """
    file_path = RAW_DATA_DIR / f"{subject_id}.txt"
    df = pd.read_csv(file_path, header=None, names=["rr_ms"])
    return df


# Mark RR values outside physiological bounds as NaN.
def _mark_outliers_physio(
    rr_ms: pd.Series,
    rr_min: Optional[int] = None,
    rr_max: Optional[int] = None,
) -> pd.Series:
    """
    Fizyolojik sınırlar dışındaki RR değerlerini NaN yapar.

    Parametreler:
        rr_min:
            Alt sınır (ms). None ise settings.hrv.rr_min_ms kullanılır.
        rr_max:
            Üst sınır (ms). None ise settings.hrv.rr_max_ms kullanılır.

    Not:
        300–2000 ms bandı, kısa süreli HRV çalışmalarında sık kullanılan
        kaba bir fizyolojik aralıktır (yaklaşık 30–200 bpm).
    """
    if rr_min is None:
        rr_min = settings.hrv.rr_min_ms
    if rr_max is None:
        rr_max = settings.hrv.rr_max_ms

    rr = pd.to_numeric(rr_ms, errors="coerce").astype(float)

    # Fizyolojik aralık dışında kalanları NaN yap
    mask_out = (rr < rr_min) | (rr > rr_max)
    rr[mask_out] = np.nan

    return rr


# Mark beats with abnormal beat-to-beat changes as NaN.
def _mark_outliers_diff(
    rr_ms: pd.Series,
    rel_threshold: float = 0.20,
    abs_threshold_ms: int = 200,
) -> pd.Series:
    """
    Ardışık iki RR arasındaki farkın çok büyük olduğu beat'leri NaN yapar.

    Kriterler (beat i için):
        |RR_i - RR_{i-1}| > abs_threshold_ms  VEYA
        |RR_i - RR_{i-1}| > rel_threshold * RR_{i-1}

    HRV literatüründe ~%20–25 civarı relatif değişim ektopik / artefakt için
    sık kullanılan bir eşiktir.
    """
    rr = pd.to_numeric(rr_ms, errors="coerce").astype(float).copy()
    if rr.size < 2:
        return rr

    diff = np.abs(np.diff(rr))
    prev = rr[:-1]

    bad_from_diff = (diff > abs_threshold_ms) | (diff > rel_threshold * prev)

    bad = np.zeros_like(rr, dtype=bool)
    bad[1:] = bad_from_diff

    rr[bad] = np.nan
    return rr


# Mark beats with large deviation from local median as NaN.
def _mark_outliers_local_median(
    rr_ms: pd.Series,
    window_beats: int = 11,
    rel_threshold: float = 0.20,
) -> pd.Series:
    """
    Lokal medyan etrafında büyük sapma gösteren beat'leri NaN yapar.

    Her beat için:
        |RR_i - median_local| / median_local > rel_threshold  -> NaN

    Bu yaklaşım, pek çok HRV yazılımında (Kubios benzeri) kullanılan
    "local median filter" mantığını taklit eder.
    """
    rr = pd.to_numeric(rr_ms, errors="coerce").astype(float).copy()
    if rr.size == 0:
        return rr

    s = pd.Series(rr)
    med = s.rolling(window=window_beats, center=True, min_periods=1).median()

    med_safe = med.replace(0.0, np.nan)
    rel_dev = (np.abs(s - med_safe) / med_safe).fillna(0.0)

    bad = rel_dev > rel_threshold
    rr[bad.to_numpy()] = np.nan
    return rr


# Interpolate short NaN gaps and drop long unreliable segments.
def _interpolate_short_gaps(
    rr_ms: np.ndarray,
    max_gap_beats: int = settings.hrv.max_gap_beats,
) -> np.ndarray:
    """
    RR (ms) dizisinde NaN olan KISA boşlukları güvenli şekilde doldurur.

    Mantık:
      - max_gap_beats uzunluğa kadar olan NaN blokları:
          * Mümkünse sol ve sağ komşu değerler arasında LINEER interpolasyon.
          * Sadece tek tarafta değer varsa forward / backward fill.
      - max_gap_beats'ten UZUN bloklar:
          * Olduğu gibi NaN bırakılır (güvenilmez segmentler).
      - En sonda kalan NaN'lar (uzun bloklar) tamamen DROPlanır.

    Bu, "kısa boşlukları düzelt, uzun segmentleri dışla" fikrinin
    sade ve deterministik bir implementasyonudur.
    """
    rr = np.asarray(rr_ms, dtype=float).copy()
    n = rr.size

    if n == 0:
        return rr

    isnan = np.isnan(rr)

    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue

        # Bir NaN bloğuna girdik
        start = i
        while i < n and isnan[i]:
            i += 1
        end = i  # [start, end-1] NaN

        length = end - start

        if length <= max_gap_beats:
            # Kısa boşluk: interpolasyon / doldurma
            left_idx = start - 1
            right_idx = end if end < n else None

            left_val = rr[left_idx] if left_idx >= 0 else np.nan
            right_val = rr[right_idx] if right_idx is not None else np.nan

            if not np.isnan(left_val) and not np.isnan(right_val):
                # İki taraf da mevcut -> lineer interpolasyon
                interp_vals = np.linspace(left_val, right_val, length + 2)[1:-1]
                rr[start:end] = interp_vals
            elif not np.isnan(left_val):
                # Sadece sol taraf var -> ileri doldurma
                rr[start:end] = left_val
            elif not np.isnan(right_val):
                # Sadece sağ taraf var -> geri doldurma
                rr[start:end] = right_val
            else:
                # İki taraf da NaN ise bir şey yapma; NaN kalacak, sonra droplanır
                pass
        else:
            # Uzun boşluk: NaN bırak, daha sonra droplayacağız
            continue

    # Uzun boşluklara karşı kalan NaN'ları tamamen at
    rr_clean = rr[~np.isnan(rr)]

    return rr_clean


# Clean RR series using multi-stage artifact correction.
def clean_rr_values(
    df: pd.DataFrame,
    rr_min: Optional[int] = None,
    rr_max: Optional[int] = None,
    max_gap_beats: Optional[int] = None,
    rel_diff_threshold: float = 0.20,
    abs_diff_threshold_ms: int = 200,
    local_median_window: int = 11,
    local_median_rel_threshold: float = 0.20,
    return_stats: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Raw RR DataFrame'ini çok aşamalı artefakt düzeltmeyle temizler.

    Adımlar:
      1) 'rr_ms' kolonunu sayısal tipe çevir.
      2) Fizyolojik aralık (rr_min–rr_max, ms) dışını NaN yap (hard limits).
      3) Ardışık RR fark filtresi (ectopic / missed beat / false detection).
      4) Lokal medyan filtresi (sliding window).
      5) Kısa NaN bloklarını (<= max_gap_beats) interpolasyonla düzelt.
      6) Uzun blokları tamamen at.
      7) Temiz RR serisini 'rr_ms' kolonu olarak döner;
         istenirse temizlik istatistiklerini de döner.

    Tipik parametreler (HRV literatürü):
      - rr_min_ms, rr_max_ms: ~300–2000 ms (≈30–200 bpm).
      - rel_diff_threshold: ~0.20 (ardışık beat'te %20 değişim).
      - abs_diff_threshold_ms: 200 ms (yavaş HR için ekstra limit).
      - local_median_window: 11 beat (her beat etrafında ~10–12 beat).
      - local_median_rel_threshold: ~0.20 (lokal medyan etrafında %20 sapma).
    """
    if rr_min is None:
        rr_min = settings.hrv.rr_min_ms
    if rr_max is None:
        rr_max = settings.hrv.rr_max_ms
    if max_gap_beats is None:
        max_gap_beats = settings.hrv.max_gap_beats

    df = df.copy()
    rr_raw = pd.to_numeric(df["rr_ms"], errors="coerce").astype(float)

    # 1) Hard physiological limits
    rr_step1 = _mark_outliers_physio(rr_raw, rr_min=rr_min, rr_max=rr_max)

    # 2) Beat-to-beat difference filter
    rr_step2 = _mark_outliers_diff(
        rr_step1,
        rel_threshold=rel_diff_threshold,
        abs_threshold_ms=abs_diff_threshold_ms,
    )

    # 3) Local-median filter
    rr_step3 = _mark_outliers_local_median(
        rr_step2,
        window_beats=local_median_window,
        rel_threshold=local_median_rel_threshold,
    )

    # 4) Interpolate short gaps, drop long gaps
    rr_array = rr_step3.to_numpy()
    rr_clean = _interpolate_short_gaps(rr_array, max_gap_beats=max_gap_beats)

    df_clean = pd.DataFrame({"rr_ms": rr_clean}).reset_index(drop=True)

    if not return_stats:
        return df_clean

    # Basit temizlik istatistikleri
    n_raw = int(rr_raw.size)
    n_after_physio = int(np.isfinite(rr_step1).sum())
    n_after_diff = int(np.isfinite(rr_step2).sum())
    n_after_local = int(np.isfinite(rr_step3).sum())
    n_final = int(rr_clean.size)

    stats: Dict[str, Any] = {
        "n_raw": n_raw,
        "n_after_physio": n_after_physio,
        "n_after_diff": n_after_diff,
        "n_after_local": n_after_local,
        "n_final": n_final,
        "removed_pct_total": 100.0 * (n_raw - n_final) / max(n_raw, 1),
    }

    return df_clean, stats


# Add cumulative time (s) and heart rate (bpm) columns.
def add_time_and_hr(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    Temiz RR serisine:
      - kümülatif zaman (time_s, saniye),
      - kalp hızı (hr_bpm),
      - subject_id
    kolonlarını ekler.
    """
    df = df.copy()

    # time_s: ms -> s, kümülatif toplam
    df["time_s"] = df["rr_ms"].cumsum() / 1000.0

    # hr_bpm: 60000 ms / RR (ms)
    df["hr_bpm"] = 60000.0 / df["rr_ms"]

    df["subject_id"] = subject_id

    return df


# Run full preprocessing pipeline for a single subject.
def process_single_subject(
    subject_id: str,
    rr_min: Optional[int] = None,
    rr_max: Optional[int] = None,
    max_gap_beats: Optional[int] = None,
    log_stats: bool = True,
) -> None:
    """
    Tek bir subject için tam pipeline:
      raw txt -> temizlenmiş rr_ms + time_s + hr_bpm -> CSV.

    Varsayılan temizleme parametreleri settings.hrv üzerinden gelir.
    İstenirse, temizlik istatistikleri log'a yazılır.
    """
    if rr_min is None:
        rr_min = settings.hrv.rr_min_ms
    if rr_max is None:
        rr_max = settings.hrv.rr_max_ms
    if max_gap_beats is None:
        max_gap_beats = settings.hrv.max_gap_beats

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_rr_series(subject_id)
    df_clean, stats = clean_rr_values(
        df_raw,
        rr_min=rr_min,
        rr_max=rr_max,
        max_gap_beats=max_gap_beats,
        return_stats=True,
    )

    if log_stats:
        print(
            f"[preprocessing] subject={subject_id} "
            f"raw={stats['n_raw']} -> final={stats['n_final']} "
            f"removed={stats['removed_pct_total']:.1f}%"
        )

    df_final = add_time_and_hr(df_clean, subject_id)

    out_path = PROCESSED_DIR / f"{subject_id}_clean.csv"
    df_final.to_csv(out_path, index=False)
    print(f"[preprocessing] Saved: {out_path}")


# Run preprocessing for all subjects in RAW_DATA_DIR.
def process_all_subjects() -> None:
    """
    RAW_DATA_DIR içindeki tüm *.txt dosyalarını dolaşır,
    sayısal ID'lere sahip olanları işler.

    Not:
      - LICENSE vb. metadata dosyaları atlanır.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RAW_DATA_DIR.glob("*.txt"))
    print(f"[preprocessing] Found {len(txt_files)} raw txt files in {RAW_DATA_DIR}")

    for f in txt_files:
        subject_id = f.stem  # "000", "401", "LICENSE", ...

        if not subject_id.isdigit():
            print(f"[preprocessing] Skipping non-subject file: {f.name}")
            continue

        print(f"[preprocessing] Processing subject {subject_id} ...")
        process_single_subject(subject_id)


if __name__ == "__main__":
    # Script olarak çalıştırıldığında tüm subject'leri yeniden üretir.
    process_all_subjects()
