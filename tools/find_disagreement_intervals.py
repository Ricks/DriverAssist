#!/usr/bin/env python3
"""
Finds time intervals where the algorithm's hysteresis-smoothed pick
disagrees with human-labeled ground truth -- for cutting a short "diff reel"
out of a full annotated review video instead of watching the whole thing.

Three kinds of disagreement, all just "pred != gt" per frame, no special-
casing needed to detect them (the review video's tinting already tells them
apart visually: an orange-only box is "algorithm picked, nothing labeled";
a magenta-only box is "labeled, algorithm found nothing"; both colors on
different boxes is "picked the wrong vehicle"):
  - gt is a trackID, pred is a different trackID (wrong vehicle)
  - gt is a trackID, pred is None (missed a real close-following case)
  - gt is None, pred is a trackID (false positive)

Excludes disagreement frames near a ground-truth segment boundary (see
analyze_leading_vehicle_errors.py) -- those are expected hysteresis lag
(confirm_frames/grace_frames catching up to a label change that just
happened), not a real classifier mistake, and would otherwise flood the
reel with boring, expected blips at every label transition.

Nearby genuine-disagreement stretches within --merge-gap of each other are
merged into one interval (avoids a flurry of near-adjacent tiny clips), and
padded by --pad seconds on each side for viewing context.

Usage:
    python3 find_disagreement_intervals.py <session_dir> \\
        --ground-truth ground_truth_close_range.json --tuning-results leading-vehicle-tuning-close-range.json
"""
import argparse
import json
import sys
from pathlib import Path

from analyze_leading_vehicle_errors import nearest_boundary_distance, segment_boundaries
from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from package_session import find_session_video
from tune_leading_vehicle import classify_all, ground_truth_at, precompute_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--debug-log", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None, help="Defaults to <session_dir>/ground_truth_close_range.json")
    parser.add_argument("--symmetry-cache", type=Path, default=None, help="Defaults to <session_dir>/symmetry-cache.json")
    parser.add_argument("--tuning-results", type=Path, default=None, help="leading-vehicle-tuning*.json; uses its best_params")
    parser.add_argument("--boundary-window", type=float, default=1.0, help="Seconds from a label boundary to treat as expected timing lag, not genuine disagreement")
    parser.add_argument("--merge-gap", type=float, default=3.0, help="Merge genuine-disagreement stretches within this many seconds of each other")
    parser.add_argument("--pad", type=float, default=1.5, help="Extra seconds of context on each side of a kept interval")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <session_dir>/disagreement-intervals.json")
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    gt_path = args.ground_truth or (args.session_dir / "ground_truth_close_range.json")
    symmetry_path = args.symmetry_cache or (args.session_dir / "symmetry-cache.json")
    tuning_path = args.tuning_results or (args.session_dir / "leading-vehicle-tuning-close-range.json")
    output_path = args.output or (args.session_dir / "disagreement-intervals.json")
    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")

    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
    entries = load_detections(detections_path)
    symmetry_scores = json.loads(symmetry_path.read_text()) if symmetry_path.exists() else None
    frames = precompute_frames(entries, start_epoch, symmetry_scores)

    segments = json.loads(gt_path.read_text())
    params = json.loads(tuning_path.read_text())["best_params"]
    predictions = classify_all(frames, params)
    boundaries = segment_boundaries(segments)

    video_end = frames[-1]["t"]

    # Walk frames in order, tracking a genuine-disagreement run.
    raw_intervals = []
    run_start = None
    last_bad_t = None
    disagreement_kinds_in_run = []

    def close_run(end_t):
        raw_intervals.append({"start_t": run_start, "end_t": end_t, "kinds": sorted(set(disagreement_kinds_in_run))})

    for f, pred in zip(frames, predictions):
        gt = ground_truth_at(segments, f["t"])
        if pred == gt:
            if run_start is not None:
                close_run(last_bad_t)
                run_start = None
                disagreement_kinds_in_run = []
            continue

        if nearest_boundary_distance(f["t"], boundaries) <= args.boundary_window:
            continue  # expected hysteresis lag, not genuine

        if gt is None:
            kind = "false_positive"
        elif pred is None:
            kind = "missed"
        else:
            kind = "wrong_vehicle"

        if run_start is None:
            run_start = f["t"]
        disagreement_kinds_in_run.append(kind)
        last_bad_t = f["t"]

    if run_start is not None:
        close_run(last_bad_t)

    print(f"{len(raw_intervals)} genuine-disagreement stretches found (boundary_window={args.boundary_window}s excluded).")

    # Pad, then merge anything within merge_gap of the next one.
    padded = []
    for iv in raw_intervals:
        padded.append({
            "start_t": max(0.0, iv["start_t"] - args.pad),
            "end_t": min(video_end, iv["end_t"] + args.pad),
            "kinds": iv["kinds"],
        })

    merged = []
    for iv in padded:
        if merged and iv["start_t"] - merged[-1]["end_t"] <= args.merge_gap:
            merged[-1]["end_t"] = max(merged[-1]["end_t"], iv["end_t"])
            merged[-1]["kinds"] = sorted(set(merged[-1]["kinds"]) | set(iv["kinds"]))
        else:
            merged.append(iv)

    total_kept = sum(iv["end_t"] - iv["start_t"] for iv in merged)
    print(f"Merged (gap<={args.merge_gap}s) + padded ({args.pad}s) into {len(merged)} clips, "
          f"{total_kept:.1f}s total (of {video_end:.1f}s full video, {total_kept/video_end:.1%}).")

    for iv in merged:
        print(f"  {iv['start_t']:8.2f}s - {iv['end_t']:8.2f}s  ({iv['end_t']-iv['start_t']:5.1f}s)  {'/'.join(iv['kinds'])}")

    output_path.write_text(json.dumps(merged, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
