# src/tools/export_rr_sessions.py

from pathlib import Path
import json
import pandas as pd

# Proje kökü: .../hrv-project
ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "data"

# Giriş/çıkış dosyaları
INPUT_CSV = DATA_DIR / "all_rr_records.csv"     # <-- senin CSV'nin adı
OUTPUT_JSON = DATA_DIR / "rr_sessions.json"     # oluşturacağımız dosya


def main():
    # CSV'yi oku
    df = pd.read_csv(INPUT_CSV)

    # Gerekirse kolon isimlerini burada uyarlarsın
    # Örn: 'Subject', 'Session', 'RR_ms' vs. ise:
    # df = df.rename(columns={"Subject": "subject_id", "Session": "session", "RR_ms": "rr"})

    sessions = {}

    # subject_id + session'a göre grupla
    for (subject_id, session), group in df.groupby(["subject_id", "session"]):
        key = f"{subject_id}_{session}"  # örn: sub001_rest
        rr_list = group["rr"].astype(float).tolist()
        sessions[key] = {
            "subject_id": subject_id,
            "session": session,
            "rr": rr_list,
        }

    # JSON olarak kaydet
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sessions)} sessions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
