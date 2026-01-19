# Real-time HRV Streaming & Dashboard

This repository contains a real-time Heart Rate Variability (HRV) monitoring pipeline.

The system:

- replays RR-interval time series from a **public HRV database** (PhysioNet-style text/CSV),
- streams cleaned RR intervals through **Kafka**,
- computes time-, frequency- and non-linear HRV metrics in real time,
- estimates HF / LF / VLF / ULF components with a VMDon-like online method,
- exposes metrics via a **FastAPI** service,
- visualises them in a **Dash** web dashboard.

---

## 1. Project structure

Main components (paths as in the source tree):

- `run_all.py`  
  Orchestration script to start the full system (Spark job, API, dashboard, Kafka producer).

- `src/config/settings.py`  
  Central configuration:
  - data directories (raw RR, cleaned RR, plots),
  - Kafka settings (bootstrap servers, topic),
  - HRV parameters (resampling rate, RR buffer horizon, etc.).

- `src/streaming/preprocessing.py`  
  Offline preprocessing of public RR files into per-subject cleaned CSVs:
  - artifact handling and basic filtering,
  - interpolation and time axis generation,
  - instantaneous heart rate.

- `src/streaming/rr_producer.py`  
  Kafka RR producer:
  - reads `{subject}_clean.csv`,
  - replays RR intervals for one or multiple subjects.

- `src/streaming/rr_consumer.py` & `src/streaming/rr_buffer.py`  
  - consumer that reads from Kafka and updates a global in-memory RR buffer,
  - thread-safe per-subject buffers with a limited time window.

- `src/hrv_metrics/service_hrv.py`  
  Core HRV logic:
  - loading RR from the buffer or CSV,
  - preprocessing and resampling (2 Hz),
  - time-domain and frequency-domain metrics,
  - Poincaré metrics,
  - VMDon-like decomposition into HF/LF/VLF/ULF components.

- `src/api/fastapi_app.py`  
  FastAPI service exposing HRV metrics and subject information:
  - `/health`, `/subjects`, `/metrics/time`, `/metrics/freq`,
  - `/metrics/poincare`, `/metrics/status`, `/metrics/vmdon`, etc.

- `src/dashboard/app.py`  
  Dash dashboard:
  - subject selection,
  - time-domain metrics and HR trace,
  - VMDon components and band-power visualisation,
  - Poincaré plot,
  - status bar for buffer / signal quality.

- `src/hrv_metrics/vmd_hrv_offline.py`, `hrv_vmd_spark.py` (optional)  
  Offline VMD / AVMD experiments and Spark-based AVMD job (for reference analyses and plots).

---

## 2. Installation

Requirements:

- Python 3.8+
- A running **Kafka** broker
- (Optional) **Apache Spark** for the AVMD job

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
