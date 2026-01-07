"""
PySpark job for offline HRV decomposition (AVMD / VMDon).

- Reads all *_clean.csv files from rr_clean directory
- For each subject (file):
    * load RR intervals
    * RR -> 2 Hz uniform HRV(t)
    * detrend
    * decompose with:
        - AVMD (adaptive VMD with HRV-aware K selection), or
        - VMDon-like sliding-window decomposition
    * compute simple band summaries per subject and band

Output:
    One Parquet file with rows:
        subject, method, band, mean, var, var_demean, n_samples, duration_min

Example:
    spark-submit src/spark/hrv_vmd_spark.py \
        --method avmd \
        --max-minutes 30 \
        --output data/processed/hrv_bands_avmd.parquet
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

# ----------------- PROJECT ROOT & IMPORTS ----------------- #

CURRENT_DIR = Path(__file__).resolve().parent           # .../src/spark
PROJECT_ROOT = CURRENT_DIR.parent.parent                # .../hrv-project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ayarları ve offline VMD/AVMD/VMDon fonksiyonlarını içeri al
from src.config.settings import settings

# Buradaki import yolu:
#   vmd_hrv_offline.py dosyan senin projende büyük ihtimalle
#   src/hrv_vmd/vmd_hrv_offline.py altına koyulacak şekilde ayarla.
from vmd_hrv_offline import (
    load_rr_sec_from_clean,
    rr_to_uniform_hrv,
    detrend_signal,
    run_avmd_hrv,
    vmdon_offline,
    FS_RESAMPLE,
    VMD_ALPHA,
    REQUIRED_BANDS,
)
import numpy as np
from typing import Sequence

# ... mevcut importlar burada ...

def reconstruct_components(modes: np.ndarray, band) -> np.ndarray:
    """
    Seçili modları toplayarak o banda ait yeniden oluşturulmuş zaman serisini döndürür.
    modes: shape (K, N)  -> K tane IMF, her biri uzunluk N
    band:  HRVBandResult gibi bir obje, içinde seçilen mod indeksleri var.
    """
    # band içindeki index alanının ismini kendi offline koduna göre ayarla:
    # örnek: band.mode_indices veya band.indices veya band.imf_idxs
    idxs: Sequence[int] = getattr(band, "mode_indices", None) or getattr(band, "indices", None)

    if idxs is None:
        # hiç alan yoksa: tüm modları topla (en garanti fallback)
        return np.sum(modes, axis=0)

    if len(idxs) == 0:
        return np.zeros_like(modes[0])

    return np.sum(modes[list(idxs), :], axis=0)


# ----------------- HELPERS ----------------- #

def _load_rr_sec_from_csv(csv_path: Path) -> np.ndarray:
    """
    Load RR series (seconds) from a *_clean.csv file.

    Tries common RR column names:
        rr_sec, rr_s, rr, rr_ms
    If 'ms' suffix is found, converts to seconds.
    """
    df = pd.read_csv(csv_path)

    rr_col_candidates = ["rr_sec", "rr_s", "rr", "rr_ms"]
    rr_col = None
    for col in rr_col_candidates:
        if col in df.columns:
            rr_col = col
            break

    if rr_col is None:
        raise ValueError(
            f"RR column not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    rr = df[rr_col].to_numpy(dtype=float)

    if rr_col.lower().endswith("ms"):
        rr_sec = rr / 1000.0
    else:
        rr_sec = rr

    rr_sec = rr_sec[np.isfinite(rr_sec)]
    rr_sec = rr_sec[rr_sec > 0.1]
    return rr_sec


def _compute_band_summaries(
    subject: str,
    method: str,
    t: np.ndarray,
    comps: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Compute simple statistics per band and return as list of dicts.
    """
    rows: List[Dict[str, Any]] = []

    duration_min = float(t[-1] / 60.0) if t.size > 0 else 0.0

    for band_name, x in comps.items():
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            mean = var = var_demean = np.nan
            n_samples = 0
        else:
            mean = float(np.mean(x))
            var = float(np.var(x))
            var_demean = float(np.var(x - mean))
            n_samples = int(x.size)

        rows.append(
            {
                "subject": subject,
                "method": method,
                "band": band_name,
                "mean": mean,
                "var": var,
                "var_demean": var_demean,
                "n_samples": n_samples,
                "duration_min": duration_min,
            }
        )

    return rows


def _process_one_file(
    path_str: str,
    method: str,
    max_minutes: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Tek bir *_clean.csv dosyasını AVMD veya VMDon ile işleyip
    frekans bandı özetlerini döndürür.
    Bu fonksiyon Spark worker içinde (RDD map) çalışır.
    """
    csv_path = Path(path_str)
    subject = csv_path.stem.replace("_clean", "")

    # 1) RR yükle
    rr_sec = _load_rr_sec_from_csv(csv_path)
    if rr_sec.size == 0:
        return []

    # 2) RR -> uniform HRV(t)
    t, hrv = rr_to_uniform_hrv(rr_sec, fs=FS_RESAMPLE, max_minutes=max_minutes)
    if hrv.size == 0:
        return []

    # 3) Detrend (offline pipeline ile aynı ayar)
    hrv_p = detrend_signal(hrv, mode="mean")

    # 4) Bileşen ayırma
    if method == "avmd":
        # HRV-aware AVMD: K seçimi + band assignment
        modes, omega, K, band = run_avmd_hrv(
            hrv_p,
            FS_RESAMPLE,
            alpha=VMD_ALPHA,
            kmin=4,
            kmax=12,
            energy_loss=0.01,
            required_bands=REQUIRED_BANDS,
            use_omega=True,
            dc=0,
        )
        comps = reconstruct_components(modes, band)

    elif method == "vmdon":
        # VMDon benzeri sliding-window ayrıştırma
        comps = vmdon_offline(hrv_p, FS_RESAMPLE)

    else:
        raise ValueError(f"Unknown method: {method}")

    # --- BURADAN SONRASI: GÜVENLİ NORMALİZASYON & UZUNLUK EŞLEME ---

    import numpy as np

    # comps mutlaka dict olsun (band -> np.ndarray)
    if not isinstance(comps, dict):
        # Eğer tek bir array geldiyse, bunu HF gibi düşün ve sar
        if isinstance(comps, np.ndarray):
            comps = {"HF": comps}
        elif isinstance(comps, (list, tuple)):
            # Liste/tuple ise mode_0, mode_1 ... diye sar
            comps = {f"mode_{i}": np.asarray(c) for i, c in enumerate(comps)}
        else:
            comps = {"HF": np.asarray(comps)}

    # Tüm komponentlerin ortak uzunluğunu bul (en kısa olan)
    lengths = [np.asarray(v).size for v in comps.values() if v is not None]
    if len(lengths) == 0:
        return []

    n = min(lengths)

    # t ve tüm komponentleri aynı uzunluğa kes
    t_aligned = t[:n]
    comps_aligned = {k: np.asarray(v)[:n] for k, v in comps.items() if v is not None}

    # 5) Band özetlerini çıkar
    rows = _compute_band_summaries(subject, method, t_aligned, comps_aligned)
    return rows



# ----------------- MAIN SPARK JOB ----------------- #

def run_spark_job(
    method: str,
    max_minutes: Optional[float],
    output_path: Path,
    num_partitions: Optional[int] = None,
) -> None:
    """
    Run PySpark job over all *_clean.csv files in RR_DIR and write band summaries as Parquet.
    """
    rr_dir: Path = settings.paths.rr_processed_dir
    csv_files = sorted(rr_dir.glob("*_clean.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No *_clean.csv files found in {rr_dir}")

    paths = [str(p) for p in csv_files]

    spark = (
        SparkSession.builder
        .appName(f"HRV_{method.upper()}_Spark")
        .getOrCreate()
    )
    sc = spark.sparkContext

    # Eğer num_partitions verilmemişse, dosya sayısı kadar partition kullan
    if num_partitions is None or num_partitions <= 0:
        num_partitions = len(paths)

    rdd = sc.parallelize(paths, numSlices=num_partitions)

    def _map(path_str: str) -> List[Dict[str, Any]]:
        # Worker tarafında işlem
        return _process_one_file(path_str, method=method, max_minutes=max_minutes)

    # Her dosyadan gelen satır listelerini tek RDD'ye düzleştir
    rows_rdd = rdd.flatMap(_map)

    # RDD[dict] -> DataFrame
    df = spark.createDataFrame(rows_rdd)  # schema inference

    # Parquet'e yaz
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.write.mode("overwrite").parquet(str(output_path))

    print(f"[HRV-SPARK] Wrote {df.count()} rows to {output_path}")

    spark.stop()


# ----------------- CLI ----------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline HRV decomposition (AVMD / VMDon) with PySpark."
    )
    p.add_argument(
        "--method",
        choices=["avmd", "vmdon"],
        default="avmd",
        help="Decomposition method to use.",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=30.0,
        help="Max duration (minutes) per subject to analyze (None = full).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output Parquet path, e.g. data/processed/hrv_bands_avmd.parquet",
    )
    p.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="Number of Spark partitions (default: len(files)).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    max_minutes: Optional[float]
    if args.max_minutes is None or args.max_minutes <= 0:
        max_minutes = None
    else:
        max_minutes = float(args.max_minutes)

    out_path = Path(args.output)

    run_spark_job(
        method=args.method,
        max_minutes=max_minutes,
        output_path=out_path,
        num_partitions=args.num_partitions,
    )
