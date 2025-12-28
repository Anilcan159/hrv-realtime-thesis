# Keeps last N seconds of RR samples in memory (per subject)
"""
In-memory RR buffer for real-time HRV analysis.

Responsibilities:
    - Store the last `max_age_s` seconds of RR samples per subject.
    - Provide windowed RR series in seconds for downstream HRV metrics.
    - Enforce time-based eviction in a thread-safe way.

Configuration:
    - Buffer horizon (max_age_s) is driven by settings.hrv.rr_max_age_s
"""

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, List, Optional

from src.config.settings import settings


@dataclass
class RRSample:
    """
    Single RR sample in milliseconds with its capture timestamp.

    Attributes:
        ts:
            Epoch timestamp in seconds (float, time.time()).
        rr_ms:
            RR interval in milliseconds.
    """
    ts: float
    rr_ms: float


class RRBuffer:
    """
    Thread-safe, time-bounded buffer for RR intervals per subject.

    The buffer keeps, for each subject, only the samples whose timestamps are
    within the last `max_age_s` seconds relative to "now". Older samples are
    automatically discarded on insert and on read.

    This design is intentionally minimal but robust:
        - O(1) append and pop-left via deque.
        - Global lock for simplicity (per-subject locks would be overkill here).
        - Time-based eviction, no index bookkeeping.
    """

    def __init__(self, max_age_s: float) -> None:
        """
        Args:
            max_age_s:
                Maximum age (in seconds) of samples to retain in the buffer.
                Samples older than (now - max_age_s) are dropped.
        """
        self.max_age_s: float = float(max_age_s)
        self._data: Dict[str, Deque[RRSample]] = {}
        self._lock: Lock = Lock()

    def add(self, subject: str, rr_ms: float, ts: Optional[float] = None) -> None:
        """
        Append a new RR sample to the buffer for a given subject.

        Args:
            subject:
                Subject identifier (string, e.g. "000", "401").
            rr_ms:
                RR interval in milliseconds.
            ts:
                Optional timestamp in seconds. If None, uses time.time().
        """
        if ts is None:
            ts = time.time()

        sample = RRSample(ts=float(ts), rr_ms=float(rr_ms))

        with self._lock:
            dq = self._data.setdefault(str(subject), deque())
            dq.append(sample)
            self._trim_locked(dq)

    def get_rr_sec(self, subject: str, window_s: Optional[float] = None) -> List[float]:
        """
        Retrieve RR series (in seconds) for a subject, optionally restricted
        to the last `window_s` seconds.

        Args:
            subject:
                Subject identifier.
            window_s:
                If None or <= 0:
                    Returns all samples currently stored for this subject
                    (bounded by max_age_s).
                If > 0:
                    Returns only samples with ts >= (now - window_s).

        Returns:
            List of RR intervals in seconds, ordered by time.
        """
        now = time.time()

        with self._lock:
            dq = self._data.get(str(subject))
            if dq is None:
                return []

            # First enforce global horizon (max_age_s)
            self._trim_locked(dq)

            # Then apply optional window filter
            if window_s is None or window_s <= 0:
                samples = list(dq)
            else:
                start_ts = now - float(window_s)
                samples = [s for s in dq if s.ts >= start_ts]

        return [s.rr_ms / 1000.0 for s in samples]

    def _trim_locked(self, dq: Deque[RRSample]) -> None:
        """
        Remove samples from the left of the deque that are older
        than the global horizon (now - max_age_s).

        Assumes the caller holds self._lock.
        """
        cutoff = time.time() - self.max_age_s
        while dq and dq[0].ts < cutoff:
            dq.popleft()


# Global buffer instance used by the streaming / HRV service layers.
# The horizon is centralized in the configuration:
#   settings.hrv.rr_max_age_s  (e.g. 600s = 10 minutes)
GLOBAL_RR_BUFFER = RRBuffer(max_age_s=settings.hrv.rr_max_age_s)
