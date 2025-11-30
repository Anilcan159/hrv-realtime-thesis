# src/streaming/preprocessing.py
# Offline preprocessing:
#   - Raw PhysioNet RR txt -> cleaned per-subject CSV
#   - Fizyolojik filtre + güvenli interpolasyon + zaman / HR hesapları

from pathlib import Path
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


def load_rr_series(subject_id: str) -> pd.DataFrame:
    """
    Raw RR serisini (ms) tek sütunlu DataFrame olarak yükler.
    Örn: RAW_DATA_DIR / '000.txt' -> rr_ms kolonu.

    Her satır bir RR interval (ms).
    """
    file_path = RAW_DATA_DIR / f"{subject_id}.txt"
    df = pd.read_csv(file_path, header=None, names=["rr_ms"])
    return df


# -------------------- YARDIMCI: FİZYOLOJİK FİLTRE -------------------- #

def _mark_outliers_physio(
    rr_ms: pd.Series,
    rr_min: int = 300,
    rr_max: int = 2000,
) -> pd.Series:
    """
    Fizyolojik sınırlar dışındaki RR değerlerini NaN yapar.

    - rr_min, rr_max ms cinsindendir.
    - 300–2000 ms bandı Task Force ve pratik HRV çalışmalarında yaygın kullanılan
      bir kaba aralıktır (çok hızlı / çok yavaş veya artefakt atımlar elenir).
    """
    rr = pd.to_numeric(rr_ms, errors="coerce").astype(float)

    # Fiziksel aralık dışında kalanları NaN yap
    mask_out = (rr < rr_min) | (rr > rr_max)
    rr[mask_out] = np.nan

    return rr


# -------------------- YARDIMCI: KISA BOŞLUKLARI DOLDUR -------------------- #

def _interpolate_short_gaps(
    rr_ms: np.ndarray,
    max_gap_beats: int = 3,
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

    Bu, eski defterde kullandığın "kısa boşlukları düzelt, uzunları kullanma"
    fikrinin sade ve deterministik bir implementasyonudur.
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


# -------------------- ANA TEMİZLİK FONKSİYONU -------------------- #

def clean_rr_values(
    df: pd.DataFrame,
    rr_min: int = 300,
    rr_max: int = 2000,
    max_gap_beats: int = 3,
) -> pd.DataFrame:
    """
    Raw RR DataFrame'ini temizler:

    1) 'rr_ms' kolonunu sayısal tipe çevirir.
    2) 300–2000 ms fizyolojik aralığı dışını NaN yapar.
    3) Kısa NaN bloklarını (<= max_gap_beats) interpolasyonla düzeltir.
    4) Uzun blokları tamamen atar.
    5) Temiz RR serisini 'rr_ms' kolonu olarak döner.

    Bu adımlar, eski Colab defterindeki:
      - fizyolojik filtre
      - güvenli interpolasyon (kısa boşluk)
      - problemli segmentleri dışlama
    mantığıyla uyumludur.
    """
    df = df.copy()

    # 1) Fizyolojik outlier'ları NaN yap
    rr_marked = _mark_outliers_physio(df["rr_ms"], rr_min=rr_min, rr_max=rr_max)

    # 2) Kısa boşlukları doldur, uzun boşlukları dışla
    rr_array = rr_marked.to_numpy()
    rr_clean = _interpolate_short_gaps(rr_array, max_gap_beats=max_gap_beats)

    # 3) Son DataFrame
    df_clean = pd.DataFrame({"rr_ms": rr_clean})
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


# -------------------- ZAMAN VE HR EKLE -------------------- #

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


# -------------------- TEK BİR SUBJECT İŞLE -------------------- #

def process_single_subject(
    subject_id: str,
    rr_min: int = 300,
    rr_max: int = 2000,
    max_gap_beats: int = 3,
) -> None:
    """
    Tek bir subject için tam pipeline:
      raw txt -> temizlenmiş rr_ms + time_s + hr_bpm -> CSV.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_rr_series(subject_id)
    df_clean = clean_rr_values(
        df_raw,
        rr_min=rr_min,
        rr_max=rr_max,
        max_gap_beats=max_gap_beats,
    )
    df_final = add_time_and_hr(df_clean, subject_id)

    out_path = PROCESSED_DIR / f"{subject_id}_clean.csv"
    df_final.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# -------------------- TÜM SUBJECT'LERİ İŞLE -------------------- #

def process_all_subjects() -> None:
    """
    RAW_DATA_DIR içindeki tüm *.txt dosyalarını dolaşır,
    sayısal ID'lere sahip olanları işler.

    Not:
      - LICENSE vs. gibi dosyalar atlanır.
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
    # Script olarak çalıştırıldığında tüm subject'leri yeniden üretir.
    process_all_subjects()
