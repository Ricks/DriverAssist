#!/usr/bin/env python3
"""
Checks whether the on-device app's *live* ByteTracker output (the trackID
DetectionLogger.swift now logs per detection) matches what tools/tracker.py's
from-scratch Python port computes on the exact same raw detection boxes.

This is NOT an accuracy comparison the way track_benchmark.py's is -- there,
the on-device and reference sides see different boxes from different models,
so a reference model can meaningfully stand in as ground truth. Here, both
sides start from the *identical* boxes (the same logged detections.jsonl);
only the association logic differs (live Swift vs. offline Python). So this
is a consistency/regression check: does the Swift port actually behave like
the validated-offline Python reference it was ported from? Because the boxes
are identical and in the same order, each detection matches itself by index
-- no IoU-based cross-matching is needed the way benchmark.py/
track_benchmark.py need it.

Usage:
    python3 track_consistency.py <session_dir>

Needs a session recorded with a build that logs trackID/trackingLevel (see
DetectionLogger.swift) -- sessions recorded before that change have nothing
to compare against and this exits with a clear error.

Known, expected sources of PARTIAL divergence -- not automatically bugs:
  - GMC: the on-device tracker estimates camera motion via Vision's optical
    flow (GMC.swift); this recomputes it via tools/gmc.py's OpenCV-based
    sparse-flow estimator. Different algorithms estimating the same thing --
    some disagreement here is expected, not proof of a Swift-side defect.
  - Appearance: as of this writing, AppearanceEmbedder always returns nil
    on-device (no ReID model integrated yet), so both sides are effectively
    geometry+GMC-only regardless of which TrackingLevel was logged for a
    given entry -- the offline recomputation never uses appearance either,
    to match. Once a real on-device embedder exists, this script's offline
    side would need one too, or divergence here would be expected and
    uninformative.

A LOT of divergence, especially concentrated right after a genuine gap in
the underlying boxes (not an artifact of GMC/appearance), is the signal
actually worth investigating -- that points at a real difference in the
Kalman filter math, the Hungarian assignment, or track lifecycle logic
between the two implementations.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from gmc import GMC
from package_session import find_session_video
from tracker import ByteTracker

METRICS = [
    "num_frames", "mota", "motp", "idf1", "precision", "recall",
    "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost",
]


def recompute_offline(entries: list, video: Path, start_epoch: float, use_gmc: bool) -> list:
    """Sequential single decode pass -- mirrors reconstruct_annotated.py's
    fixed (non-seek-based) approach. A naive per-entry cap.set(...) seek
    measured 30-40x slower than sequential reads and isn't worth
    reintroducing here (see reconstruct_annotated.py's history)."""
    tracker = ByteTracker()
    gmc = GMC() if use_gmc else None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    all_ids: list = [None] * len(entries)
    next_idx = 0

    while next_idx < len(entries):
        ok, frame = cap.read()
        if not ok:
            break
        frame_epoch = start_epoch + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        while next_idx < len(entries) and entries[next_idx]["t"] <= frame_epoch:
            entry = entries[next_idx]
            gmc_kwargs = {}
            if gmc is not None:
                warp = gmc.apply(frame)
                gmc_kwargs = {"gmc_warp": warp, "frame_shape": frame.shape[:2]}
            all_ids[next_idx] = tracker.update(entry["detections"], **gmc_kwargs)
            next_idx += 1

    cap.release()
    return all_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Packaged session directory (see package_session.py)")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to <session_dir>/detections.jsonl")
    parser.add_argument("--video", type=Path, default=None, help="Defaults to the one raw recording in <session_dir>")
    parser.add_argument("--debug-log", type=Path, default=None, help="Defaults to <session_dir>/overlay-debug.log")
    parser.add_argument(
        "--use-gmc", dest="use_gmc", action="store_true", default=True,
        help="Recompute with GMC too, matching the app (which always runs it) (default: on)",
    )
    parser.add_argument("--no-gmc", dest="use_gmc", action="store_false")
    parser.add_argument("--results", type=Path, default=None, help="JSON path; defaults to <session_dir>/<video-stem>-track-consistency.json")
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist.")

    entries = load_detections(detections_path)

    has_on_device_ids = any("trackID" in det for entry in entries for det in entry["detections"])
    if not has_on_device_ids:
        sys.exit(
            f"{detections_path} has no on-device trackID field on any detection -- this session predates "
            "the tracking-aware DetectionLogger. Record a fresh session with the updated app first."
        )

    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")
    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)

    results_path = args.results or args.session_dir / f"{video.stem}-track-consistency.json"

    print("Recomputing offline (Python reference)...", file=sys.stderr)
    offline_ids = recompute_offline(entries, video, start_epoch, args.use_gmc)

    accumulators = defaultdict(lambda: mm.MOTAccumulator(auto_id=True))
    evaluated = 0
    for entry, offline_entry_ids in zip(entries, offline_ids):
        if offline_entry_ids is None:
            continue  # video ended before this entry's timestamp
        on_device_entry_ids = [det.get("trackID") for det in entry["detections"]]

        n = len(entry["detections"])
        costs = np.full((n, n), np.nan)
        np.fill_diagonal(costs, 0.0)  # each detection matches itself by index -- identical boxes, both sides

        level = entry.get("trackingLevel", "unknown")
        accumulators[level].update(on_device_entry_ids, offline_entry_ids, costs)
        evaluated += 1

    print(f"Done: {evaluated} entries compared.")
    if not accumulators:
        sys.exit("Nothing evaluated -- video may have ended before any logged entry's timestamp.")

    mh = mm.metrics.create()
    summaries = {level: mh.compute(acc, metrics=METRICS, name=level) for level, acc in accumulators.items()}

    full = pd.concat(summaries.values())
    print("\nOn-device (live Swift) vs. offline (Python reference) tracking consistency -- "
          "1.0 idf1 means the Swift port produced exactly the same track-ID partition as the Python "
          "reference; see module docstring for expected (non-bug) sources of divergence:")
    print(full.to_string(float_format=lambda v: f"{v:.3f}"))

    results = {level: df.iloc[0].to_dict() for level, df in summaries.items()}
    with open(results_path, "w") as f:
        json.dump({"video": video.name, "use_gmc": args.use_gmc, "results": results}, f, indent=2)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
