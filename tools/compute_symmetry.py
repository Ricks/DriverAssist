#!/usr/bin/env python3
"""
Precomputes a left-right symmetry score for every vehicle detection in a
session, by cropping each box from the actual video frame and correlating
its left half against its mirrored right half. Rear (and front) views of a
vehicle are close to bilaterally symmetric (tail lights, bumper, plate);
oblique/side views (e.g. cross-traffic, or a car in an adjacent lane) are
not -- see the design discussion on using this as a complementary signal to
classify_leading's velocity-based cross-traffic gate.

Deliberately a separate precompute pass, not computed inline inside
classify_leading -- that function is called hundreds of times per frame
during tune_leading_vehicle.py's random search, and decoding video is far
more expensive than re-running pure arithmetic over already-computed
detections (reconstruct_annotated.py separately measured per-entry re-
seeking at 30-40x slower than one sequential decode pass). Computing this
once and caching it mirrors track_benchmark.py's cache-what's-expensive
split -- the search sweeps free arithmetic over a fixed score, same as
tracking/velocity already do.

Cache format: a JSON list, one entry per detections.jsonl entry (same
order, via driverassist_sync.load_detections), each itself a list of
per-detection scores (float 0-1, higher = more symmetric) or null
(non-vehicle label, or box too small/degenerate to crop meaningfully) --
aligned by list index to that entry's own `detections` list, NOT by
trackID, so it's immune to any future change in tracker/ByteTracker
behavior or ordering.

Usage:
    python3 compute_symmetry.py <session_dir> [--video ...] [--min-width 0.01] [--output ...]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from leading_vehicle import VEHICLE_LABELS
from package_session import find_session_video

# Small + fixed -- symmetry doesn't need real resolution, just enough
# structure (tail lights, plate, bumper edges) to correlate against its
# mirror. Even width so the left/right split has no off-by-one.
CROP_SIZE = 32


def symmetry_score(gray_crop: np.ndarray) -> float:
    """1.0 = perfectly left-right symmetric, 0.0 = maximally different."""
    resized = cv2.resize(gray_crop, (CROP_SIZE, CROP_SIZE)).astype(np.float32)
    left = resized[:, : CROP_SIZE // 2]
    right = resized[:, CROP_SIZE // 2 :]
    right_flipped = np.fliplr(right)
    diff = np.abs(left - right_flipped).mean()
    return float(max(0.0, 1.0 - diff / 255.0))


def crop_box(frame: np.ndarray, det: dict):
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, int(det["x"] * frame_w))
    y1 = max(0, int(det["y"] * frame_h))
    x2 = min(frame_w, int((det["x"] + det["w"]) * frame_w))
    y2 = min(frame_h, int((det["y"] + det["h"]) * frame_h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--debug-log", type=Path, default=None)
    parser.add_argument("--min-width", type=float, default=0.01, help="Skip (null) boxes narrower than this -- too small to crop meaningfully")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <session_dir>/symmetry-cache.json")
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")
    output_path = args.output or (args.session_dir / "symmetry-cache.json")

    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
    entries = load_detections(detections_path)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    results = [None] * len(entries)
    next_idx = 0
    frames_decoded = 0

    print(f"Decoding {video.name} sequentially, scoring symmetry for {len(entries)} detection entries...", file=sys.stderr)

    # Same sequential-decode-in-lockstep pattern as reconstruct_annotated.py
    # -- whichever already-decoded frame is current when an entry's own
    # timestamp is first reached is used for that entry, rather than
    # re-seeking per entry (measured much slower there).
    while next_idx < len(entries):
        ok, frame = cap.read()
        if not ok:
            break
        frames_decoded += 1
        frame_epoch = start_epoch + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        while next_idx < len(entries) and entries[next_idx]["t"] <= frame_epoch:
            entry = entries[next_idx]
            scores = []
            for det in entry["detections"]:
                if det["label"] not in VEHICLE_LABELS or det["w"] < args.min_width:
                    scores.append(None)
                    continue
                crop = crop_box(frame, det)
                if crop is None:
                    scores.append(None)
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                scores.append(symmetry_score(gray))
            results[next_idx] = scores
            next_idx += 1

        if frames_decoded % 2000 == 0:
            print(f"  {frames_decoded} frames decoded, {next_idx}/{len(entries)} entries scored...", file=sys.stderr)

    if next_idx < len(entries):
        print(f"Warning: video ended after {frames_decoded} frames with {len(entries) - next_idx} "
              f"detection entries unmatched -- leaving those null.", file=sys.stderr)
        for i in range(next_idx, len(entries)):
            results[i] = [None for _ in entries[i]["detections"]]

    output_path.write_text(json.dumps(results))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
