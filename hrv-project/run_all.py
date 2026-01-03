# run_all.py
# Runs FastAPI HRV service + producer (optional) + Dash app in one command

import os
import sys
import time
import signal
import socket
import subprocess
from pathlib import Path
from typing import List, Tuple

from src.config.settings import settings

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# Kafka ayarları config'ten
KAFKA_BOOTSTRAP = settings.kafka.bootstrap_servers  # örn: "localhost:9092" veya "kafka:9092"


def _popen(cmd: List[str], name: str) -> subprocess.Popen:
    print(f"[run_all] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=os.environ.copy(),
    )


def _parse_host_port(bootstrap: str) -> Tuple[str, int]:
    """
    settings.kafka.bootstrap_servers string'inden host ve port'u çıkarır.

    Örnek:
        "localhost:9092"         -> ("localhost", 9092)
        "kafka:9092"             -> ("kafka", 9092)
        "localhost"              -> ("localhost", 9092)  # port yoksa default 9092
        "host1:9092,host2:9093"  -> ilk tanımı kullanır ("host1", 9092)
    """
    first = bootstrap.split(",")[0].strip()
    if ":" in first:
        host_str, port_str = first.split(":", 1)
        host = host_str.strip() or "localhost"
        try:
            port = int(port_str)
        except ValueError:
            port = 9092
    else:
        host = first or "localhost"
        port = 9092

    return host, port


def _wait_for_kafka(
    host: str,
    port: int,
    timeout_s: float = 30.0,
    interval_s: float = 1.0,
) -> bool:
    """Wait until Kafka port is reachable (simple TCP check)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print(f"[run_all] Kafka is reachable at {host}:{port}")
                return True
        except OSError:
            print(f"[run_all] waiting for Kafka at {host}:{port} ...")
            time.sleep(interval_s)

    print(f"[run_all] Kafka not reachable after {timeout_s:.0f}s, continuing anyway...")
    return False


def main() -> None:
    # ---- CONFIG ----
    start_producer = os.environ.get("START_PRODUCER", "1") == "1"
    dash_module = os.environ.get("DASH_APP", "src/dashboard/app.py")
    producer_module = os.environ.get("PRODUCER_APP", "src/streaming/rr_producer.py")

    kafka_host, kafka_port = _parse_host_port(KAFKA_BOOTSTRAP)

    procs: List[Tuple[str, subprocess.Popen]] = []

    # Komutlar
    api_cmd = [
        PY,
        "-m",
        "uvicorn",
        "src.api.fastapi_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    dash_cmd = [PY, str(ROOT / dash_module)]
    producer_cmd = [PY, str(ROOT / producer_module)]

    try:
        # 1) FastAPI HRV servisini başlat (rr_consumer burada startup'ta açılıyor)
        api_proc = _popen(api_cmd, "api")
        procs.append(("api", api_proc))

        # Küçük bir nefes payı (API ayağa kalksın)
        time.sleep(2.0)

        # 2) Dash'i başlat (UI)
        dash_proc = _popen(dash_cmd, "dash")
        procs.append(("dash", dash_proc))

        # 3) Producer (opsiyonel)
        if start_producer:
            _wait_for_kafka(host=kafka_host, port=kafka_port, timeout_s=30.0)
            producer_proc = _popen(producer_cmd, "producer")
            procs.append(("producer", producer_proc))

        print("\n[run_all] running. stop with CTRL+C\n")

        # Supervisor döngüsü
        while True:
            time.sleep(0.5)

            for idx, (name, proc) in enumerate(list(procs)):
                ret = proc.poll()
                if ret is None:
                    continue  # hâlâ çalışıyor

                # Process exit etmiş
                if name in ("api", "dash"):
                    # API veya Dash kritik: ölürse her şeyi kapat
                    print(f"[run_all] {name} exited (code={ret}). Shutting down...")
                    raise RuntimeError(f"{name} exited unexpectedly (code={ret})")

                if name == "producer":
                    # Producer ölürse yeniden başlatmayı dene
                    print(f"[run_all] producer exited (code={ret}). Restarting in 2s...")
                    time.sleep(2.0)
                    try:
                        new_p = _popen(producer_cmd, "producer")
                        procs[idx] = ("producer", new_p)
                    except Exception as e:
                        print(f"[run_all] failed to restart producer: {e}")
                        # Producer'ı listeden çıkar, diğerleri çalışmaya devam etsin
                        procs.pop(idx)

    except KeyboardInterrupt:
        print("\n[run_all] CTRL+C received, shutting down...")

    finally:
        # Tüm child processleri kapat
        for name, p in procs[::-1]:
            try:
                print(f"[run_all] sending SIGINT to {name} (pid={p.pid})")
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

        time.sleep(1.0)

        for name, p in procs[::-1]:
            try:
                if p.poll() is None:
                    print(f"[run_all] terminating {name} (pid={p.pid})")
                    p.terminate()
            except Exception:
                pass

        print("[run_all] done.")


if __name__ == "__main__":
    main()
