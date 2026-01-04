# convert_tfm_to_rr_clean.py
#
# TFM.csv -> data/processed/rr_clean/<subject_id>_clean.csv
# Format: rr_ms,time_s,hr_bpm,subject_id

from pathlib import Path
import pandas as pd
import numpy as np
import re


def find_tfm_csv(project_root: Path) -> Path:
    """
    Farklı olası yerlerde TFM.csv arar.
    Bulursa path'i döner, bulamazsa hata fırlatır.
    """
    candidates = [
        project_root / "Datas" / "raw" / "TFM.csv",
        project_root / "Datas" / "TFM.csv",
        project_root / "TFM.csv",
    ]

    print("[INFO] TFM.csv aranıyor...")
    for p in candidates:
        print(f"  - {p}")
        if p.exists():
            print(f"[INFO] Kullanılacak TFM.csv: {p}")
            return p

    raise FileNotFoundError("TFM.csv hiçbir aday path'te bulunamadı.")


def build_time_s(df: pd.DataFrame) -> pd.Series:
    """
    TIME kolonu: '06:29:23 132' -> göreli saniye (time_s)
    """
    if "TIME" not in df.columns:
        raise ValueError("TFM.csv içinde 'TIME' kolonu yok.")

    time_str = df["TIME"].astype(str).str.strip()
    dt = pd.to_datetime(time_str, format="%H:%M:%S %f", errors="coerce")

    if dt.dropna().empty:
        raise ValueError("TIME kolonu datetime'a çevrilemedi, formatı kontrol et.")

    t0 = dt.dropna().iloc[0]
    time_s = (dt - t0).dt.total_seconds()
    return time_s


def make_subject_id_from_column(col_name: str) -> str:
    """
    'WIMU_01_Marta_Baños   HRM HeartRate(bpm)(SYNC)'
      -> 'WIMU_01_Marta_Banos'
    """
    base = col_name.split("HRM")[0].strip()
    base = base.replace(" ", "_")
    subject_id = re.sub(r"[^A-Za-z0-9_]+", "", base)

    if not subject_id:
        subject_id = "TFM_subject"

    return subject_id


def convert_tfm():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "data" / "processed" / "rr_clean"

    # 1) TFM.csv yolunu bul
    tfm_path = find_tfm_csv(project_root)

    # 2) CSV'yi oku
    print(f"[INFO] Reading TFM.csv: {tfm_path}")
    df = pd.read_csv(tfm_path, sep=";", decimal=",", engine="python")

    # Kolon isimlerinden boşlukları temizle
    df.columns = [c.strip() for c in df.columns]

    # 3) time_s kolonunu oluştur
    print("[INFO] TIME -> time_s hesaplanıyor...")
    time_s = build_time_s(df)

    # 4) HR kolonlarını bul
    hr_columns = [c for c in df.columns if "HRM HeartRate" in c]
    if not hr_columns:
        raise ValueError("Herhangi bir 'HRM HeartRate' içeren kolon bulunamadı.")

    print("[INFO] Bulunan HR kolonları:")
    for c in hr_columns:
        print(f"  - {c}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 5) Her kolon için ayrı subject dosyası üret
    for col_name in hr_columns:
        subject_id = make_subject_id_from_column(col_name)
        print(f"\n[INFO] Converting column -> subject_id: '{col_name}' -> '{subject_id}'")

        # HR değerlerini float'a çevir
        hr_bpm = pd.to_numeric(df[col_name], errors="coerce")

        # 0 veya negatifleri at
        hr_bpm = hr_bpm.where(hr_bpm > 0)

        # RR(ms) = 60000 / HR(bpm)
        rr_ms = 60000.0 / hr_bpm

        out = pd.DataFrame(
            {
                "rr_ms": rr_ms,
                "time_s": time_s,
                "hr_bpm": hr_bpm,
                "subject_id": subject_id,
            }
        )

        out = out.dropna(subset=["rr_ms", "time_s"])

        out_file = output_dir / f"{subject_id}_clean.csv"
        out.to_csv(out_file, index=False)
        print(f"[OK] Saved: {out_file}  (rows={len(out)})")

    print("\n[DONE] TFM.csv dönüştürme tamamlandı.")


if __name__ == "__main__":
    print("[DEBUG] convert_tfm_to_rr_clean main başladı")
    try:
        convert_tfm()
    except Exception as e:
        print("[ERROR] Script hata verdi:")
        print(e)
# End of convert_tfm_to_rr_clean.py