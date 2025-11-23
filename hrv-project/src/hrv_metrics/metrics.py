# src/hrv_metrics/metrics.py
# Pure HRV metric computations from RR intervals (ms).

from typing import Union, Sequence, Dict
import numpy as np
import pandas as pd


ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _to_numpy(rr_ms: ArrayLike) -> np.ndarray:
    """Converts input RR series to a clean 1D numpy array (ms)."""
    arr = np.asarray(rr_ms, dtype=np.float64)
    # drop NaN values if any
    arr = arr[~np.isnan(arr)]
    return arr


def sdnn(rr_ms: ArrayLike) -> float:
    """Calculates SDNN (standard deviation of NN intervals, ms)."""
    rr = _to_numpy(rr_ms)
    if rr.size < 2:
        return np.nan
    return float(np.std(rr, ddof=1))


def rmssd(rr_ms: ArrayLike) -> float:
    """Calculates RMSSD (root mean square of successive differences, ms)."""
    rr = _to_numpy(rr_ms)
    if rr.size < 2:
        return np.nan
    diff = np.diff(rr)
    return float(np.sqrt(np.mean(diff ** 2)))


def nn50(rr_ms: ArrayLike, threshold_ms: float = 50.0) -> int:
    """Counts successive RR differences greater than threshold_ms (default 50 ms)."""
    rr = _to_numpy(rr_ms)
    if rr.size < 2:
        return 0
    diff = np.abs(np.diff(rr))
    return int(np.sum(diff > threshold_ms))


def pnn50(rr_ms: ArrayLike, threshold_ms: float = 50.0) -> float:
    """Calculates pNN50 (% of successive RR differences > threshold_ms)."""
    rr = _to_numpy(rr_ms)
    n = rr.size
    if n < 2:
        return np.nan
    nn50_count = nn50(rr, threshold_ms=threshold_ms)
    # number of successive pairs is (n - 1)
    return float(100.0 * nn50_count / (n - 1))


def hr_min(rr_ms: ArrayLike) -> float:
    """Calculates minimum heart rate (bpm) from RR intervals in ms."""
    rr = _to_numpy(rr_ms)
    if rr.size == 0:
        return np.nan
    hr = 60000.0 / rr
    return float(np.min(hr))


def hr_max(rr_ms: ArrayLike) -> float:
    """Calculates maximum heart rate (bpm) from RR intervals in ms."""
    rr = _to_numpy(rr_ms)
    if rr.size == 0:
        return np.nan
    hr = 60000.0 / rr
    return float(np.max(hr))


def hr_range(rr_ms: ArrayLike) -> float:
    """Calculates HR range (HR_max - HR_min, bpm)."""
    return hr_max(rr_ms) - hr_min(rr_ms)


def compute_time_domain_metrics(rr_ms: ArrayLike) -> Dict[str, float]:
    """
    Computes a basic set of time-domain HRV metrics from RR intervals (ms).
    Returns a dict for easier use in dashboard/analysis.
    """
    return {
        "SDNN_ms": sdnn(rr_ms),
        "RMSSD_ms": rmssd(rr_ms),
        "NN50_count": nn50(rr_ms),
        "pNN50_percent": pnn50(rr_ms),
        "HR_min_bpm": hr_min(rr_ms),
        "HR_max_bpm": hr_max(rr_ms),
        "HR_range_bpm": hr_range(rr_ms),
    }
