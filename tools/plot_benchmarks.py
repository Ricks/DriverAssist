#!/usr/bin/env python3
"""
Combines multiple sessions' benchmark.py results into one chart — e.g. every
nano/off drive across several days, to see accuracy and latency pooled
across sessions rather than one video at a time.

Usage:
    python3 plot_benchmarks.py <session1> <session2> ... [--output combined.png]

Each <session> is a packaged session directory (see package_session.py) --
this finds that directory's single <video>-benchmark.json itself, so a
session with no benchmark results yet (only benchmark.py's --benchmark run
writes one) is a clear error rather than a silent skip.

Combining is NOT just averaging each session's numbers — TP/FP/FN are summed
per (config, class) and precision/recall/F1 recomputed from the pooled
totals, and latency lists are concatenated before recomputing mean/median/
p95. That's the statistically correct way to pool sessions of different
lengths; see benchmark_common.py.

The chart's caption lists every source video's filename so it's clear which
drives went into it.
"""
import argparse
import sys
from pathlib import Path

from benchmark_common import load_results, make_chart, merge_results, print_summary


def find_benchmark_json(session_dir: Path) -> Path:
    if not session_dir.is_dir():
        sys.exit(f"{session_dir} isn't a directory.")
    matches = sorted(session_dir.glob("*-benchmark.json"))
    if not matches:
        sys.exit(
            f"No *-benchmark.json found in {session_dir} -- run benchmark.py (or "
            "package_session.py --benchmark) on it first."
        )
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        sys.exit(f"Found more than one *-benchmark.json in {session_dir}: {names} -- expected just one.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", type=Path, nargs="+", help="One or more packaged session directories")
    parser.add_argument("--output", type=Path, default=None, help="Chart PNG path; defaults to combined-benchmark.png")
    args = parser.parse_args()

    output = args.output or Path("combined-benchmark.png")

    loaded = []
    for session_dir in args.sessions:
        loaded.append(load_results(find_benchmark_json(session_dir)))

    stats, latencies, sessions, iou_thresholds = merge_results(loaded)

    if len(iou_thresholds) > 1:
        print(
            f"WARNING: combining results computed at different IoU thresholds ({sorted(iou_thresholds)}) — "
            "accuracy numbers below aren't directly comparable across sessions. Consider re-running "
            "benchmark.py with a consistent --iou-threshold.",
            file=sys.stderr,
        )
    iou_threshold = sorted(iou_thresholds)[0] if iou_thresholds else 0.5

    print(f"Combined {len(sessions)} session(s): {', '.join(sessions)}\n")
    print_summary(stats, latencies, iou_threshold)
    make_chart(stats, latencies, output, iou_threshold, sessions=sessions)
    print(f"\nChart written to {output}")


if __name__ == "__main__":
    main()
