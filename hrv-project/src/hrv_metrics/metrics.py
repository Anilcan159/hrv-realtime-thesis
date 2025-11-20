"""Heart rate variability metric calculations."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _to_sequence(values: Iterable[float]) -> Sequence[float]:
    """Convert iterable values to a list to allow multiple passes."""
    return list(values)


def sdnn(rr_intervals: Iterable[float]) -> float:
    """Calculate the standard deviation of NN intervals (SDNN).

    Args:
        rr_intervals: Iterable of RR intervals in milliseconds.

    Returns:
        Standard deviation of the intervals. Returns 0.0 when fewer than
        two intervals are provided.
    """

    intervals = _to_sequence(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    mean_rr = sum(intervals) / len(intervals)
    squared_diffs = [(rr - mean_rr) ** 2 for rr in intervals]
    return math.sqrt(sum(squared_diffs) / (len(intervals) - 1))


def rmssd(rr_intervals: Iterable[float]) -> float:
    """Calculate the root mean square of successive differences (RMSSD)."""

    intervals = _to_sequence(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    diffs = [intervals[i] - intervals[i - 1] for i in range(1, len(intervals))]
    squared_diffs = [diff ** 2 for diff in diffs]
    return math.sqrt(sum(squared_diffs) / len(squared_diffs))


def nn50(rr_intervals: Iterable[float]) -> int:
    """Count successive RR interval differences greater than 50 ms."""

    intervals = _to_sequence(rr_intervals)
    if len(intervals) < 2:
        return 0

    return sum(1 for i in range(1, len(intervals)) if abs(intervals[i] - intervals[i - 1]) > 50)


def pnn50(rr_intervals: Iterable[float]) -> float:
    """Calculate the percentage of NN50 events (pNN50)."""

    intervals = _to_sequence(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    count = nn50(intervals)
    return (count / (len(intervals) - 1)) * 100


def _heart_rates(rr_intervals: Sequence[float]) -> list[float]:
    """Convert RR intervals in milliseconds to heart rates in bpm."""

    return [60000.0 / rr for rr in rr_intervals if rr > 0]


def heart_rate_min(rr_intervals: Iterable[float]) -> float:
    """Minimum heart rate derived from RR intervals."""

    intervals = _to_sequence(rr_intervals)
    rates = _heart_rates(intervals)
    return min(rates) if rates else 0.0


def heart_rate_max(rr_intervals: Iterable[float]) -> float:
    """Maximum heart rate derived from RR intervals."""

    intervals = _to_sequence(rr_intervals)
    rates = _heart_rates(intervals)
    return max(rates) if rates else 0.0


def heart_rate_range(rr_intervals: Iterable[float]) -> float:
    """Range of heart rate values derived from RR intervals."""

    intervals = _to_sequence(rr_intervals)
    rates = _heart_rates(intervals)
    if not rates:
        return 0.0

    return max(rates) - min(rates)


def heart_rate_triangular_index(rr_intervals: Iterable[float], bin_size_ms: float = 7.8125) -> float:
    """Compute the heart rate triangular index (HTI).

    HTI is defined as the total number of NN intervals divided by the height
    of the histogram of all NN intervals.
    """

    intervals = _to_sequence(rr_intervals)
    if not intervals:
        return 0.0

    counts, _ = _histogram(intervals, bin_size_ms)
    if not counts:
        return 0.0

    peak = max(counts)
    return len(intervals) / peak if peak else 0.0


def tinn(rr_intervals: Iterable[float], bin_size_ms: float = 7.8125) -> float:
    """Compute the triangular interpolation of NN interval histogram (TINN)."""

    intervals = _to_sequence(rr_intervals)
    if len(intervals) < 3:
        return 0.0

    counts, bin_edges = _histogram(intervals, bin_size_ms)
    if not counts:
        return 0.0

    mode_index = max(range(len(counts)), key=counts.__getitem__)

    left_index = 0
    for i in range(mode_index, -1, -1):
        if counts[i] == 0:
            left_index = i
            break

    right_index = len(counts) - 1
    for i in range(mode_index, len(counts)):
        if counts[i] == 0:
            right_index = i
            break

    left_edge = bin_edges[left_index]
    right_edge = bin_edges[right_index + 1] if right_index + 1 < len(bin_edges) else bin_edges[-1]

    return max(0.0, right_edge - left_edge)


def _histogram(values: Sequence[float], bin_size_ms: float) -> tuple[list[int], list[float]]:
    """Create a histogram for the given values using a fixed bin size."""

    min_val = min(values)
    max_val = max(values)
    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")

    bin_count = int(math.ceil((max_val - min_val) / bin_size_ms)) or 1
    counts = [0 for _ in range(bin_count)]

    for value in values:
        index = min(int((value - min_val) / bin_size_ms), bin_count - 1)
        counts[index] += 1

    bin_edges = [min_val + i * bin_size_ms for i in range(bin_count + 1)]
    return counts, bin_edges
