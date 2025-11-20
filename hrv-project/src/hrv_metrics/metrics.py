"""Heart rate variability metric calculations."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _clean_intervals(values: Iterable[float]) -> list[float]:
    """Convert RR intervals to a reusable list and discard non-positive data."""

    return [float(value) for value in values if value is not None and value > 0]


def sdnn(rr_intervals: Iterable[float]) -> float:
    """Calculate the standard deviation of NN intervals (SDNN).

    Args:
        rr_intervals: Iterable of RR intervals in milliseconds.

    Returns:
        Standard deviation of the intervals. Returns 0.0 when fewer than
        two intervals are provided.
    """

    intervals = _clean_intervals(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    mean_rr = sum(intervals) / len(intervals)
    squared_diffs = [(rr - mean_rr) ** 2 for rr in intervals]
    return math.sqrt(sum(squared_diffs) / (len(intervals) - 1))


def rmssd(rr_intervals: Iterable[float]) -> float:
    """Calculate the root mean square of successive differences (RMSSD)."""

    intervals = _clean_intervals(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    diffs = [intervals[i] - intervals[i - 1] for i in range(1, len(intervals))]
    squared_diffs = [diff ** 2 for diff in diffs]
    return math.sqrt(sum(squared_diffs) / len(squared_diffs))


def nn50(rr_intervals: Iterable[float]) -> int:
    """Count successive RR interval differences greater than 50 ms."""

    intervals = _clean_intervals(rr_intervals)
    if len(intervals) < 2:
        return 0

    return sum(1 for i in range(1, len(intervals)) if abs(intervals[i] - intervals[i - 1]) > 50)


def pnn50(rr_intervals: Iterable[float]) -> float:
    """Calculate the percentage of NN50 events (pNN50)."""

    intervals = _clean_intervals(rr_intervals)
    if len(intervals) < 2:
        return 0.0

    count = nn50(intervals)
    return (count / (len(intervals) - 1)) * 100


def _heart_rates(rr_intervals: Sequence[float]) -> list[float]:
    """Convert RR intervals in milliseconds to heart rates in bpm."""

    return [60000.0 / rr for rr in rr_intervals if rr > 0]


def heart_rate_min(rr_intervals: Iterable[float]) -> float:
    """Minimum heart rate derived from RR intervals."""

    intervals = _clean_intervals(rr_intervals)
    rates = _heart_rates(intervals)
    return min(rates) if rates else 0.0


def heart_rate_max(rr_intervals: Iterable[float]) -> float:
    """Maximum heart rate derived from RR intervals."""

    intervals = _clean_intervals(rr_intervals)
    rates = _heart_rates(intervals)
    return max(rates) if rates else 0.0


def heart_rate_range(rr_intervals: Iterable[float]) -> float:
    """Range of heart rate values derived from RR intervals."""

    intervals = _clean_intervals(rr_intervals)
    rates = _heart_rates(intervals)
    if not rates:
        return 0.0

    return max(rates) - min(rates)


def heart_rate_triangular_index(rr_intervals: Iterable[float], bin_size_ms: float = 7.8125) -> float:
    """Compute the heart rate triangular index (HTI).

    HTI is defined as the total number of NN intervals divided by the height
    of the histogram of all NN intervals.
    """

    intervals = _clean_intervals(rr_intervals)
    if not intervals:
        return 0.0

    counts, _ = _histogram(intervals, bin_size_ms)
    if not counts:
        return 0.0

    peak = max(counts)
    return len(intervals) / peak if peak else 0.0


def tinn(rr_intervals: Iterable[float], bin_size_ms: float = 7.8125) -> float:
    """Compute the triangular interpolation of NN interval histogram (TINN)."""

    intervals = _clean_intervals(rr_intervals)
    if len(intervals) < 3:
        return 0.0

    counts, bin_edges = _histogram(intervals, bin_size_ms)
    if not counts:
        return 0.0

    mode_index = max(range(len(counts)), key=counts.__getitem__)
    peak = counts[mode_index]
    if peak <= 0:
        return 0.0

    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]

    left_zero = _find_zero(counts, mode_index, direction=-1)
    right_zero = _find_zero(counts, mode_index, direction=1)

    left_support = _find_support(counts, mode_index, direction=-1)
    right_support = _find_support(counts, mode_index, direction=1)

    left_x = _intercept_at_zero(
        centers[left_support],
        counts[left_support],
        centers[mode_index],
        peak,
        fallback=bin_edges[0] if left_zero is None else centers[left_zero],
    )

    right_x = _intercept_at_zero(
        centers[right_support],
        counts[right_support],
        centers[mode_index],
        peak,
        fallback=bin_edges[-1] if right_zero is None else centers[right_zero],
    )

    width = max(0.0, right_x - left_x)
    return width


def _find_zero(counts: Sequence[int], start: int, direction: int) -> int | None:
    """Locate the nearest bin with zero counts from a starting index."""

    step = -1 if direction < 0 else 1
    for idx in range(start, -1 if direction < 0 else len(counts), step):
        if counts[idx] == 0:
            return idx
    return None


def _find_support(counts: Sequence[int], start: int, direction: int) -> int:
    """Find the closest non-zero bin when walking left/right from the mode."""

    step = -1 if direction < 0 else 1
    idx = start + step
    while 0 <= idx < len(counts) and counts[idx] == 0:
        idx += step
    if not (0 <= idx < len(counts)):
        return start
    return idx


def _intercept_at_zero(x1: float, y1: float, x2: float, y2: float, fallback: float) -> float:
    """Find the x-axis intercept of the line through two points.

    If the slope is zero (no change), return the provided fallback location.
    """

    if x1 == x2 or y1 == y2:
        return fallback

    slope = (y2 - y1) / (x2 - x1)
    if slope == 0:
        return fallback

    intercept = x1 - (y1 / slope)
    return intercept


def _histogram(values: Sequence[float], bin_size_ms: float) -> tuple[list[int], list[float]]:
    """Create a histogram for the given values using a fixed bin size."""

    if not values:
        return [], []

    if bin_size_ms <= 0:
        raise ValueError("bin_size_ms must be positive")

    min_val = min(values)
    max_val = max(values)
    span = max_val - min_val
    bin_count = int(math.ceil(span / bin_size_ms)) or 1
    counts = [0 for _ in range(bin_count)]

    for value in values:
        index = min(int((value - min_val) / bin_size_ms), bin_count - 1)
        counts[index] += 1

    bin_edges = [min_val + i * bin_size_ms for i in range(bin_count + 1)]
    return counts, bin_edges
