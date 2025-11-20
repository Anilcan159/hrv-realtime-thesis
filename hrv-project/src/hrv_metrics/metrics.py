def calculate_rmssd(rr_intervals):
    """Calculate the Root Mean Square of Successive Differences (RMSSD) from RR intervals."""
    if len(rr_intervals) < 2:
        return 0.0
    differences = [rr_intervals[i] - rr_intervals[i - 1] for i in range(1, len(rr_intervals))]
    squared_differences = [diff ** 2 for diff in differences]
    return (sum(squared_differences) / len(squared_differences)) ** 0.5

def calculate_sdnn(rr_intervals):
    """Calculate the Standard Deviation of NN intervals (SDNN) from RR intervals."""
    if len(rr_intervals) < 2:
        return 0.0
    mean_rr = sum(rr_intervals) / len(rr_intervals)
    squared_differences = [(rr - mean_rr) ** 2 for rr in rr_intervals]
    return (sum(squared_differences) / (len(rr_intervals) - 1)) ** 0.5

def calculate_pnn50(rr_intervals):
    """Calculate the percentage of successive RR intervals that differ by more than 50 ms (PNN50)."""
    if len(rr_intervals) < 2:
        return 0.0
    count = sum(1 for i in range(1, len(rr_intervals)) if abs(rr_intervals[i] - rr_intervals[i - 1]) > 50)
    return (count / (len(rr_intervals) - 1)) * 100

# Additional HRV metric functions can be added here as needed.