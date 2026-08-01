#!/usr/bin/env python3
"""
Benchmarks the app's on-device tracking (nano/small/medium x two-pass, each
run through tracker.py's ByteTracker) against the same ByteTracker applied to
the yolo26x reference model's own detections on the same frames -- i.e. not
"are the boxes right" (that's benchmark.py), but "does the tracked identity
hold up": does an on-device track ID stay attached to the same real object
over time as well as a track ID computed from the stronger reference model
does?

Usage:
    python3 track_benchmark.py <clean.mov>

Uses the standard CLEAR MOT / IDF1 metrics (via the `motmetrics` package),
treating the reference model's tracks as ground truth and the on-device
tracks as the hypothesis being scored -- the same "reference model as ground
truth" convention benchmark.py already uses for box accuracy. Key columns:

    idf1            identity F1 -- the single best "did IDs stay right"
                    number; 1.0 is perfect
    num_switches    times an on-device track's matched reference identity
                    changed frame-to-frame (the on-device tracker confused
                    two real objects, or lost and respawned one)
    mostly_tracked  reference tracks that were correctly identified for
                    >=80% of their lifetime by a single on-device track ID
    mota            CLEAR MOT accuracy -- folds in false positives/negatives
                    too, not just identity switches

One continuous ByteTracker runs across the *whole* session for each side
(on-device, reference), regardless of which model/two-pass config was active
at a given moment -- object identity is a property of the real world, not of
which config happened to be logging at the time. Metrics are still bucketed
per config afterward, same as benchmark.py, by which config each frame's
on-device entry says was active.

CACHING: the reference model + ReID encoder pass (--reference-model,
--reid-model, --imgsz) is the expensive part and doesn't depend on any of the
--track-* tuning flags below -- those only change how the *already-computed*
boxes/embeddings get associated into tracks. So that expensive pass is cached
to <video>-track-cache.pkl (override with --cache) after the first run;
sweeping --track-max-age, --track-appearance-thresh, etc. across many values
replays cheaply against the cache instead of re-running the reference model
and encoder every time. Pass --rebuild-cache to force a fresh pass (e.g.
after changing --reference-model/--reid-model/--imgsz).

This cache is separate from benchmark.py's --benchmark step's
<video>-benchmark.json, which only saves aggregate TP/FP/FN counts, not the
raw per-frame boxes tracking needs.

See benchmark.py's docstring for how --detections/--debug-log/frame seeking
work; this shares that same machinery.
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from ultralytics import YOLO

from benchmark import DEFAULT_REFERENCE_MODEL, config_label, iou, run_reference_model
from benchmark_common import TARGET_CLASSES
from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from gmc import GMC
from tracker import (
    DEFAULT_APPEARANCE_THRESHOLD,
    DEFAULT_HIGH_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_LOW_CONF_THRESHOLD,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    DEFAULT_REID_MODEL,
    ByteTracker,
    _dets_to_pixel_xywh,
    build_reid_encoder,
)

METRICS = [
    "num_frames", "mota", "motp", "idf1", "precision", "recall",
    "num_switches", "num_fragmentations", "mostly_tracked", "mostly_lost",
    "num_false_positives", "num_misses",
]


def build_cost_matrix(objs: list, hyps: list, iou_threshold: float):
    costs = np.full((len(objs), len(hyps)), np.nan)
    for i, o in enumerate(objs):
        for j, h in enumerate(hyps):
            if o["label"] != h["label"]:
                continue
            score = iou(o, h)
            if score >= iou_threshold:
                costs[i, j] = 1.0 - score
    return costs


def build_cache(detections: list, video: Path, start_epoch: float, model, reid_encoder, args) -> tuple:
    """Runs the reference model + ReID encoder once per entry, returning
    (records, frame_shape). records is parallel to `detections` (None for a
    skipped entry, otherwise a dict with reference_boxes/
    reference_embeddings/on_device_embeddings/gmc_warp) that a later run can
    replay against without needing the video, the reference model, the
    encoder, or GMC's optical-flow estimation at all -- see module docstring.
    gmc_warp (see gmc.py) is always computed here (cheap, and needs the same
    sequential frame access this pass already has) regardless of whether
    --use-gmc is requested for this particular run."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    def embed(boxes: list, frame) -> list:
        if not boxes:
            return []
        raw = reid_encoder(frame, _dets_to_pixel_xywh(boxes, frame.shape))
        return [None if e is None else np.asarray(e, dtype=float).tolist() for e in raw]

    gmc = GMC()
    frame_shape = None
    records = []
    evaluated, skipped = 0, 0
    for entry in detections:
        capture_time = entry["t"] - entry["elapsedMs"] / 1000.0
        offset_ms = (capture_time - start_epoch) * 1000.0
        if offset_ms < 0:
            records.append(None)
            skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        ok, frame = cap.read()
        if not ok:
            records.append(None)
            skipped += 1
            continue

        frame_shape = frame_shape or frame.shape[:2]
        on_device_boxes = [d for d in entry["detections"] if d["label"] in TARGET_CLASSES]
        reference_boxes = run_reference_model(model, frame, args.device, args.imgsz)
        records.append({
            "reference_boxes": reference_boxes,
            "reference_embeddings": embed(reference_boxes, frame),
            "on_device_embeddings": embed(on_device_boxes, frame),
            "gmc_warp": gmc.apply(frame).tolist(),
        })

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"  building cache: {evaluated} entries evaluated, {skipped} skipped so far...", file=sys.stderr)

    cap.release()
    print(f"Cache built: {evaluated} entries evaluated, {skipped} skipped.")
    return records, frame_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to benchmark")
    parser.add_argument("--detections", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--debug-log", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU to count as a match (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--results", type=Path, default=None, help="JSON path; defaults to <video>-track-benchmark.json")
    parser.add_argument("--cache", type=Path, default=None, help="Defaults to <video>-track-cache.pkl -- see module docstring")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore any existing cache and rebuild it")
    parser.add_argument(
        "--reid", dest="reid", action="store_true", default=True,
        help="Use appearance/ReID matching in both trackers, in addition to motion/IoU (default: on)",
    )
    parser.add_argument("--no-reid", dest="reid", action="store_false", help="Geometry-only tracking on both sides")
    parser.add_argument("--reid-model", default=DEFAULT_REID_MODEL)

    # On-device tracker tuning (see tracker.py's module docstring for what each
    # controls) -- deliberately only tunable on the on-device side. The
    # reference tracker stays fixed at tracker.py's defaults across a sweep so
    # it remains a stable ground truth to compare against; sweeping both sides
    # at once would let matching parameters alone inflate the score.
    parser.add_argument("--track-iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--track-max-age", type=int, default=DEFAULT_MAX_AGE)
    parser.add_argument("--track-min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument("--track-high-conf", type=float, default=DEFAULT_HIGH_CONF_THRESHOLD)
    parser.add_argument("--track-low-conf", type=float, default=DEFAULT_LOW_CONF_THRESHOLD)
    parser.add_argument("--track-appearance-thresh", type=float, default=DEFAULT_APPEARANCE_THRESHOLD)
    parser.add_argument(
        "--use-gmc", action="store_true",
        help="Compensate track predictions for estimated camera motion (see gmc.py). Applied to both "
             "trackers uniformly (unlike the on-device-only --track-* flags) since it's a general motion-"
             "model correction, not an on-device-specific tuning knob. Off by default -- experimental.",
    )
    args = parser.parse_args()

    results_path = args.results or args.video.with_name(args.video.stem + "-track-benchmark.json")
    cache_path = args.cache or args.video.with_name(args.video.stem + "-track-cache.pkl")

    start_epoch, _ = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)
    detections = [e for e in detections if e["t"] >= start_epoch]
    if not detections:
        sys.exit(f"No detections at or after {args.video.name}'s start time were found in {args.detections}.")
    if args.max_entries:
        detections = detections[: args.max_entries]

    signature = {
        "video": args.video.name,
        "reference_model": str(args.reference_model),
        "imgsz": args.imgsz,
        "reid_model": args.reid_model,
        "num_detections": len(detections),
    }

    records, frame_shape = None, None
    if cache_path.exists() and not args.rebuild_cache:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        has_gmc_data = "frame_shape" in cached
        if cached.get("signature") == signature and (not args.use_gmc or has_gmc_data):
            records, frame_shape = cached["records"], cached.get("frame_shape")
            print(f"Using cached reference boxes/embeddings from {cache_path} ({len(records)} entries)")
        elif args.use_gmc and not has_gmc_data:
            print(f"Cache at {cache_path} predates GMC support -- rebuilding.", file=sys.stderr)
        else:
            print(f"Cache at {cache_path} doesn't match this run's parameters -- rebuilding.", file=sys.stderr)

    if records is None:
        print(f"Loading reference model {args.reference_model}...", file=sys.stderr)
        if not args.reference_model.exists():
            sys.exit(f"{args.reference_model} doesn't exist.")
        model = YOLO(str(args.reference_model))
        reid_encoder = build_reid_encoder(args.reid_model, device=args.device)
        records, frame_shape = build_cache(detections, args.video, start_epoch, model, reid_encoder, args)
        with open(cache_path, "wb") as f:
            pickle.dump({"signature": signature, "frame_shape": frame_shape, "records": records}, f)
        print(f"Cache written to {cache_path}")

    on_device_tracker = ByteTracker(
        iou_threshold=args.track_iou_threshold,
        max_age=args.track_max_age,
        min_hits=args.track_min_hits,
        high_conf_threshold=args.track_high_conf,
        low_conf_threshold=args.track_low_conf,
        appearance_thresh=args.track_appearance_thresh,
    )
    reference_tracker = ByteTracker()
    accumulators = defaultdict(lambda: mm.MOTAccumulator(auto_id=True))

    evaluated, skipped = 0, 0
    for entry, record in zip(detections, records):
        if record is None:
            skipped += 1
            continue

        on_device_boxes = [d for d in entry["detections"] if d["label"] in TARGET_CLASSES]
        reference_boxes = record["reference_boxes"]

        if args.reid:
            od_embeds = [None if e is None else np.array(e) for e in record["on_device_embeddings"]]
            ref_embeds = [None if e is None else np.array(e) for e in record["reference_embeddings"]]
        else:
            od_embeds = [None] * len(on_device_boxes)
            ref_embeds = [None] * len(reference_boxes)

        gmc_kwargs = {}
        if args.use_gmc:
            gmc_kwargs = {"gmc_warp": np.array(record["gmc_warp"]), "frame_shape": frame_shape}

        on_device_ids = on_device_tracker.update(on_device_boxes, embeddings=od_embeds, **gmc_kwargs)
        reference_ids = reference_tracker.update(reference_boxes, embeddings=ref_embeds, **gmc_kwargs)

        costs = build_cost_matrix(reference_boxes, on_device_boxes, args.iou_threshold)
        accumulators[config_label(entry)].update(reference_ids, on_device_ids, costs)
        evaluated += 1

    print(f"\nDone: {evaluated} entries evaluated, {skipped} skipped (out of video range).")

    if not accumulators:
        sys.exit("No entries were evaluated -- nothing to report.")

    mh = mm.metrics.create()
    summaries = {}
    for config, acc in accumulators.items():
        summaries[config] = mh.compute(acc, metrics=METRICS, name=config)

    full = pd.concat(summaries.values())
    print("\nTracking consistency vs. yolo26x-tracked reference (higher idf1/mota/mostly_tracked is better; "
          "lower num_switches/num_fragmentations is better):")
    print(full.to_string(float_format=lambda v: f"{v:.3f}"))

    results = {config: df.iloc[0].to_dict() for config, df in summaries.items()}
    tracker_params = {
        "reid": args.reid,
        "reid_model": args.reid_model if args.reid else None,
        "iou_threshold": args.track_iou_threshold,
        "max_age": args.track_max_age,
        "min_hits": args.track_min_hits,
        "high_conf_threshold": args.track_high_conf,
        "low_conf_threshold": args.track_low_conf,
        "appearance_thresh": args.track_appearance_thresh,
        "use_gmc": args.use_gmc,
    }
    with open(results_path, "w") as f:
        json.dump(
            {
                "video": args.video.name,
                "iou_threshold": args.iou_threshold,
                "tracker_params": tracker_params,
                "results": results,
            },
            f, indent=2,
        )
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
