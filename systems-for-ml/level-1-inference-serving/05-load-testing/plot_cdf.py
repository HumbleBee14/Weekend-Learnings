"""
Plot the latency CDF (G2) from Locust's CSV output.

After running:
    locust -f locustfile.py --host http://localhost:8000 \
        --headless --users 16 --spawn-rate 4 --run-time 5m --csv g2

You'll get g2_stats_history.csv. This script reads the per-request latency
from the *_stats.csv file (Locust's per-endpoint summary).

Run:
    python plot_cdf.py g2
"""

import csv
import sys

import matplotlib.pyplot as plt


def main(prefix: str):
    # Locust writes <prefix>_stats.csv with summary stats and <prefix>_stats_history.csv with per-time-step.
    # For per-request latencies use the "*_stats.csv" file's percentile columns (Locust 2.x).
    stats_file = f"{prefix}_stats.csv"

    rows = []
    with open(stats_file) as f:
        for r in csv.DictReader(f):
            if r["Name"] == "Aggregated":
                continue
            rows.append(r)

    fig, ax = plt.subplots(figsize=(8, 5))

    for row in rows:
        # Locust reports percentile columns named like "50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "100%"
        percentiles = [50, 66, 75, 80, 90, 95, 98, 99, 99.9]
        latencies = [float(row[f"{p}%"]) for p in percentiles]
        ax.plot(latencies, [p / 100 for p in percentiles], "o-", label=row["Name"])

    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("G2: latency CDF")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale("log")  # latency tends to span orders of magnitude; log makes the tail visible

    fig.tight_layout()
    out = f"{prefix}_cdf.png"
    plt.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_cdf.py <csv_prefix>")
        sys.exit(1)
    main(sys.argv[1])
