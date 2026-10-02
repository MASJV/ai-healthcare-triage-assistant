"""
latency.py

Prints a latency summary for an evaluation run, so a run before and
after a change (e.g. JeV) can be compared on speed as well as accuracy.
"""


def print_latency(latencies):
    """
    latencies is a list of seconds, one per call.
    Prints the average, median, 95th percentile and slowest call.
    """
    if not latencies:
        return

    ordered = sorted(latencies)
    count = len(ordered)

    total = 0
    for seconds in ordered:
        total += seconds
    average = total / count

    median = ordered[count // 2]
    p95 = ordered[int(0.95 * (count - 1))]

    print("\nLatency per call (seconds):")
    print(f"  average {average:.2f}   median {median:.2f}   p95 {p95:.2f}   max {ordered[-1]:.2f}")
