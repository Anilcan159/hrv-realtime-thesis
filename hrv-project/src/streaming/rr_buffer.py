# Keeps last N seconds of RR samples in memory (per subject)
from collections import deque
from dataclasses import dataclass
from threading import Lock
import time

@dataclass
class RRSample:
    ts: float
    rr_ms: float

class RRBuffer:
    def __init__(self, max_age_s: float = 10 * 60.0):
        self.max_age_s = float(max_age_s)
        self._data: dict[str, deque] = {}
        self._lock = Lock()

    def add(self, subject: str, rr_ms: float, ts: float | None = None) -> None:
        if ts is None:
            ts = time.time()
        with self._lock:
            dq = self._data.setdefault(str(subject), deque())
            dq.append(RRSample(ts=float(ts), rr_ms=float(rr_ms)))
            self._trim_locked(dq)

    def get_rr_sec(self, subject: str, window_s: float | None = None) -> list[float]:
        now = time.time()
        with self._lock:
            dq = self._data.get(str(subject))
            if dq is None:
                return []

            self._trim_locked(dq)

            if window_s is None or window_s <= 0:
                samples = list(dq)
            else:
                start = now - float(window_s)
                samples = [s for s in dq if s.ts >= start]

        return [s.rr_ms / 1000.0 for s in samples]

    def _trim_locked(self, dq: deque) -> None:
        cutoff = time.time() - self.max_age_s
        while dq and dq[0].ts < cutoff:
            dq.popleft()

GLOBAL_RR_BUFFER = RRBuffer(max_age_s=10 * 60.0)
