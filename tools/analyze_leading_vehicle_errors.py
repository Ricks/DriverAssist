#!/usr/bin/env python3
"""
Breaks down classify_leading/hysteresis's errors (against label_leading_
vehicle.py ground truth) into two kinds:

  - "timing" errors: the frame is near a ground-truth segment boundary (a
    start_t or end_t -- a moment where the *correct* followed vehicle is
    changing), where the hysteresis lock's confirm_frames/grace_frames
    delay is expected to lag briefly before catching up. Not a real
    misclassification -- it eventually gets there.
  - "genuine" errors: the frame is NOT near any boundary (steady-state
    middle of a labeled segment or an unlabeled stretch), where there's
    been plenty of time to settle on the right answer. These are the
    errors actually worth caring about.

Reuses tune_leading_vehicle.py's precompute/classify machinery directly
rather than duplicating it. Runs over the whole labeled video (not just a
train/held-out split) -- this is pure diagnosis of a fixed, already-decided
parameter set, not a search, so there's no overfitting risk in using all
the data here.

Usage:
    python3 analyze_leading_vehicle_errors.py <session_dir> [--windows 0.5,1.0,2.0]
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from package_session import find_session_video
from tune_leading_vehicle import DEFAULT_PARAMS, classify_all, ground_truth_at, precompute_frames


def segment_boundaries(segments: list) -> list:
    boundaries = set()
    for s in segments:
        boundaries.add(s["start_t"])
        if s["end_t"] is not None:
            boundaries.add(s["end_t"])
    return sorted(boundaries)


def nearest_boundary_distance(t: float, boundaries: list) -> float:
    i = bisect.bisect_left(boundaries, t)
    dist = float("inf")
    if i < len(boundaries):
        dist = min(dist, abs(boundaries[i] - t))
    if i > 0:
        dist = min(dist, abs(boundaries[i - 1] - t))
    return dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--debug-log", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--windows", type=str, default="0.5,1.0,2.0", help="Comma-separated timing windows in seconds")
    parser.add_argument("--examples", type=int, default=15, help="How many genuine-error examples to print (at the largest window)")
    parser.add_argument(
        "--reviewed-start", type=float, default=None,
        help="Frames before this time are excluded (unreviewed, not genuine 'nothing to follow' ground truth). "
             "Defaults to the earliest labeled segment's start_t.",
    )
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    gt_path = args.ground_truth or (args.session_dir / "ground_truth.json")
    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")
    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)

    entries = load_detections(detections_path)
    segments = json.loads(gt_path.read_text())
    frames = precompute_frames(entries, start_epoch)
    predictions = classify_all(frames, DEFAULT_PARAMS)
    boundaries = segment_boundaries(segments)

    reviewed_start = args.reviewed_start
    if reviewed_start is None:
        reviewed_start = min(s["start_t"] for s in segments)
    excluded = sum(1 for f in frames if f["t"] < reviewed_start)
    print(f"Excluding {excluded} frames before t={reviewed_start:.1f}s (unreviewed -- before your first label, "
          f"not genuine 'nothing to follow' ground truth).\n")

    windows = [float(w) for w in args.windows.split(",")]

    # Classify every frame once (correct / error + boundary distance), then
    # bucket by each window size -- avoids re-running classify_all per window.
    rows = []
    for f, pred in zip(frames, predictions):
        if f["t"] < reviewed_start:
            continue
        gt = ground_truth_at(segments, f["t"])
        correct = pred == gt
        dist = None if correct else nearest_boundary_distance(f["t"], boundaries)
        rows.append((f["t"], gt, pred, correct, dist))

    total = len(rows)
    total_correct = sum(1 for r in rows if r[3])
    total_errors = total - total_correct
    print(f"Total frames: {total}   correct: {total_correct} ({total_correct/total:.1%})   "
          f"errors: {total_errors} ({total_errors/total:.1%})\n")

    print(f"{'window (s)':>10}  {'timing errors':>15}  {'genuine errors':>15}  {'adjusted accuracy':>18}")
    for w in windows:
        timing = sum(1 for r in rows if not r[3] and r[4] <= w)
        genuine = total_errors - timing
        adjusted = (total_correct + timing) / total
        print(f"{w:>10.2f}  {timing:>15}  {genuine:>15}  {adjusted:>17.1%}")

    largest = max(windows)
    genuine_rows = [r for r in rows if not r[3] and r[4] > largest]
    n = min(args.examples, len(genuine_rows))
    stride = max(1, len(genuine_rows) // n) if n else 1
    sampled = genuine_rows[::stride][:n]
    print(f"\n{n} genuine errors (window={largest}s), spread across the whole video, for manual spot-checking:")
    for t, gt, pred, _, dist in sampled:
        print(f"  t={t:8.2f}  ground_truth={str(gt):>6}  predicted={str(pred):>6}  nearest_boundary={dist:.2f}s")


if __name__ == "__main__":
    main()
