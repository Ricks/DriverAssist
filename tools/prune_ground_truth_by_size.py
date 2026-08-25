#!/usr/bin/env python3
"""
Pares down super_tool.py ground truth to only the sub-ranges of
each labeled segment where the followed vehicle's box is at least
`--min-width` wide -- i.e. only the "close enough that a following-too-
closely warning would matter" portion of each segment, not the full
at-any-distance range a human labeled.

Motivation: accuracy at close range matters more than accuracy at distance
for the actual product goal (a too-close warning), so tuning/evaluating
against the full-range ground truth mixes in a lot of frames that don't
matter for that goal. This produces a second ground-truth file scoped to
just the range that does, without touching the original.

Threshold is a box-width fraction (matches classify_leading's own
min_width gate's units), picked by eye once at a specific frame ("does this
look close enough to matter") and meant to be tweaked -- not a calibrated
distance. Real distance calibration (see DistanceEstimator.swift) will
eventually let this be expressed in real units instead.

A segment can produce zero, one, or multiple output sub-segments: if the
followed vehicle's box dips below the threshold in the middle (got farther
away, then closer again) rather than monotonically shrinking, this emits
separate sub-segments for each qualifying stretch rather than merging them
-- collapsing two genuinely separate close approaches into one span would
misrepresent the far stretch between them as "close."

A frame where the segment's trackID isn't detected at all counts as
not-qualifying (same as too-small) -- there's no evidence it was close.

Usage:
    python3 prune_ground_truth_by_size.py <session_dir> --min-width 0.0238 \\
        --ground-truth ground_truth.json --output ground_truth_close_range.json
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from package_session import find_session_video
from tracker import ByteTracker


def precompute_frames(entries: list, start_epoch: float) -> list:
    """Same trackID recompute as tune_leading_vehicle.py's precompute_frames
    -- kept independent here rather than imported, since this only needs
    trackID + width, not velocity/hysteresis."""
    tracker = ByteTracker()
    frames = []
    for entry in entries:
        dets = entry["detections"]
        track_ids = tracker.update(dets)
        for det, tid in zip(dets, track_ids):
            det["trackID"] = tid
        frames.append({"t": entry["t"] - start_epoch, "detections": dets})
    return frames


def prune_segment(segment: dict, frames: list, min_width: float) -> list:
    """Returns zero or more {start_t, end_t, trackID} sub-segments covering
    only the maximal contiguous stretches within `segment` where its
    trackID's box width >= min_width."""
    track_id = segment["trackID"]
    start_t, end_t = segment["start_t"], segment["end_t"]

    out = []
    run_start = None
    last_qualifying_t = None

    for f in frames:
        if f["t"] < start_t:
            continue
        if end_t is not None and f["t"] >= end_t:
            break

        det = next((d for d in f["detections"] if d.get("trackID") == track_id), None)
        qualifies = det is not None and det["w"] >= min_width

        if qualifies:
            if run_start is None:
                run_start = f["t"]
            last_qualifying_t = f["t"]
        else:
            if run_start is not None:
                out.append({"start_t": run_start, "end_t": f["t"], "trackID": track_id})
                run_start = None

    if run_start is not None:
        # Ran qualifying right up to the segment's own end (or video end for
        # an open segment) -- close it there rather than dropping it.
        out.append({"start_t": run_start, "end_t": end_t if end_t is not None else last_qualifying_t, "trackID": track_id})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--debug-log", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None, help="Defaults to <session_dir>/ground_truth.json")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <session_dir>/ground_truth_close_range.json")
    parser.add_argument("--min-width", type=float, default=0.0238)
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    gt_path = args.ground_truth or (args.session_dir / "ground_truth.json")
    output_path = args.output or (args.session_dir / "ground_truth_close_range.json")
    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")

    if not gt_path.exists():
        sys.exit(f"{gt_path} doesn't exist.")

    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
    entries = load_detections(detections_path)
    frames = precompute_frames(entries, start_epoch)

    segments = json.loads(gt_path.read_text())
    print(f"Loaded {len(segments)} segments from {gt_path}, min_width={args.min_width}")

    pruned = []
    fully_dropped = 0
    for seg in segments:
        sub_segments = prune_segment(seg, frames, args.min_width)
        if not sub_segments:
            fully_dropped += 1
        pruned.extend(sub_segments)

    pruned.sort(key=lambda s: s["start_t"])

    print(f"{len(pruned)} close-range sub-segments kept (from {len(segments)} original segments, "
          f"{fully_dropped} fully dropped).")

    if output_path.exists():
        backup_path = output_path.with_name(
            f"{output_path.stem}.pre_prune_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}"
        )
        shutil.copy(output_path, backup_path)
        print(f"Backed up existing {output_path} -> {backup_path}")

    output_path.write_text(json.dumps(pruned, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
