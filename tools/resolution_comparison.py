#!/usr/bin/env python3
"""
Tests whether increasing the on-device model's input resolution closer to
the camera's own capture resolution (1152x640 today vs. 1920x1080 captured
-- see InferenceEngine.swift/YOLODecoder) would meaningfully improve
detection quality, specifically for small/distant objects -- entirely
offline, against video already recorded, before committing to a model
re-export or spending any real-drive/on-device time on it (no export
pipeline is checked into this repo -- see the design discussion this script
came out of).

Three arms, all matched against the same yolo26x reference model
(benchmark.py's own ground truth) on the exact same frames:
  1. on-device       the real logged detections.jsonl entries -- real CoreML
                      execution at the shipped 1152x640 resolution.
  2. offline-current  yolo26n.pt run offline (PyTorch/MPS) at the same
                      1152x640-equivalent resolution -- a control that
                      isolates the CoreML-vs-PyTorch runtime/precision gap
                      from the resolution question below.
  3. offline-larger   yolo26n.pt run offline at a larger imgsz (default
                      1920x1088 -- the nearest 32-stride-aligned match to
                      the camera's real 1920x1080 capture). Comparing this
                      to #2 (same runtime, only imgsz differs) is the actual
                      resolution-effect signal; comparing #1 to #2 tells you
                      how much to trust that signal given the runtime gap.

Recall is broken out by the *reference* model's own box width (small/
distant vs. larger/closer) since the hypothesis being tested is specifically
about recovering detail for small objects, not aggregate accuracy -- a
resolution bump that only helps already-easy large/close boxes wouldn't be
worth its latency/thermal cost.

Usage:
    python3 resolution_comparison.py <clean.mov> --weights ../yolo26n.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from benchmark import CONFIDENCE_THRESHOLD, DEFAULT_REFERENCE_MODEL, iou, run_reference_model
from benchmark_common import TARGET_CLASSES, precision_recall_f1
from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch

# Reference-box width fraction below which a box counts as "small/distant"
# for stratification -- roughly matches the range real marginal cases in
# this project have sat at (e.g. trackID #428, #746: 0.013-0.025).
DEFAULT_SMALL_WIDTH_THRESHOLD = 0.05

# Current shipped model input (InferenceEngine.swift/YOLODecoder) -- height
# first, matching ultralytics' imgsz=[h, w] convention.
CURRENT_IMGSZ = [640, 1152]
# Nearest 32-stride-aligned match to the camera's real 1920x1080 capture.
LARGER_IMGSZ = [1088, 1920]


def match_boxes_stratified(predicted: list, reference: list, iou_threshold: float, small_width: float) -> dict:
    """Like benchmark.match_boxes, but splits TP/FN by the matched/unmatched
    *reference* box's width (small vs large) -- aggregate P/R/F1 alone can't
    confirm or refute a hypothesis that's specifically about small objects.
    FP has no natural reference size, so it's bucketed by the predicted
    box's own width instead."""
    by_class_pred = defaultdict(list)
    by_class_ref = defaultdict(list)
    for d in predicted:
        if d["label"] in TARGET_CLASSES:
            by_class_pred[d["label"]].append(d)
    for d in reference:
        by_class_ref[d["label"]].append(d)

    result: dict = defaultdict(lambda: [0, 0, 0])  # (class, bucket) -> [tp, fp, fn]
    for cls in TARGET_CLASSES:
        preds = by_class_pred.get(cls, [])
        refs = by_class_ref.get(cls, [])
        pairs = []
        for pi, p in enumerate(preds):
            for ri, r in enumerate(refs):
                score = iou(p, r)
                if score >= iou_threshold:
                    pairs.append((score, pi, ri))
        pairs.sort(key=lambda x: -x[0])
        matched_pred, matched_ref = set(), set()
        for score, pi, ri in pairs:
            if pi in matched_pred or ri in matched_ref:
                continue
            matched_pred.add(pi)
            matched_ref.add(ri)

        for ri, r in enumerate(refs):
            bucket = "small" if r["w"] < small_width else "large"
            result[(cls, bucket)][0 if ri in matched_ref else 2] += 1
        for pi, p in enumerate(preds):
            if pi not in matched_pred:
                bucket = "small" if p["w"] < small_width else "large"
                result[(cls, bucket)][1] += 1
    return result


def print_stratified(name: str, stats: dict) -> None:
    print(f"\n[{name}]")
    totals = defaultdict(lambda: [0, 0, 0])
    for (cls, bucket), (tp, fp, fn) in stats.items():
        for i in range(3):
            totals[bucket][i] += (tp, fp, fn)[i]
    for bucket in ("small", "large"):
        tp, fp, fn = totals[bucket]
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"
        print(f"  {bucket:6s}  P={fmt(p)}  R={fmt(r)}  F1={fmt(f1)}  (tp={tp} fp={fp} fn={fn})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--detections", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--debug-log", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--weights", type=Path, required=True, help="yolo26n.pt (offline PyTorch weights)")
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--reference-imgsz", type=int, default=1280)
    parser.add_argument("--larger-imgsz", type=int, nargs=2, default=LARGER_IMGSZ, metavar=("H", "W"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--small-width", type=float, default=DEFAULT_SMALL_WIDTH_THRESHOLD)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-entries", type=int, default=None)
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"{args.weights} doesn't exist.")
    if not args.reference_model.exists():
        sys.exit(f"{args.reference_model} doesn't exist.")

    start_epoch, _ = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)
    detections = [e for e in detections if e["t"] >= start_epoch]
    if not detections:
        sys.exit(f"No detections at or after {args.video.name}'s start time found in {args.detections}.")
    if args.max_entries:
        detections = detections[: args.max_entries]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {args.video}")

    print(f"Loading reference model {args.reference_model} (imgsz={args.reference_imgsz})...", file=sys.stderr)
    reference_model = YOLO(str(args.reference_model))
    print(f"Loading offline nano weights {args.weights}...", file=sys.stderr)
    nano_model = YOLO(str(args.weights))

    stats_on_device: dict = defaultdict(lambda: [0, 0, 0])
    stats_offline_current: dict = defaultdict(lambda: [0, 0, 0])
    stats_offline_larger: dict = defaultdict(lambda: [0, 0, 0])

    evaluated = skipped = 0
    for entry in detections:
        capture_time = entry["t"] - entry["elapsedMs"] / 1000.0
        offset_ms = (capture_time - start_epoch) * 1000.0
        if offset_ms < 0:
            skipped += 1
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        ok, frame = cap.read()
        if not ok:
            skipped += 1
            continue

        reference_boxes = run_reference_model(reference_model, frame, args.device, args.reference_imgsz)
        current_boxes = run_reference_model(nano_model, frame, args.device, CURRENT_IMGSZ)
        larger_boxes = run_reference_model(nano_model, frame, args.device, list(args.larger_imgsz))

        for name, preds, stats in (
            ("on-device", entry["detections"], stats_on_device),
            ("offline-current", current_boxes, stats_offline_current),
            ("offline-larger", larger_boxes, stats_offline_larger),
        ):
            per_bucket = match_boxes_stratified(preds, reference_boxes, args.iou_threshold, args.small_width)
            for key, (tp, fp, fn) in per_bucket.items():
                stats[key][0] += tp
                stats[key][1] += fp
                stats[key][2] += fn

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"  {evaluated} entries evaluated, {skipped} skipped so far...", file=sys.stderr)

    cap.release()
    print(f"\nDone: {evaluated} entries evaluated, {skipped} skipped (out of video range).")
    if not evaluated:
        sys.exit("Nothing evaluated -- nothing to report.")

    print(f"\nAll three arms matched against yolo26x @ imgsz={args.reference_imgsz} (IoU>={args.iou_threshold}).")
    print(f"Small/distant = reference box width < {args.small_width}; large/close = >= {args.small_width}.")
    print_stratified(f"1. on-device (real CoreML, {CURRENT_IMGSZ[1]}x{CURRENT_IMGSZ[0]})", stats_on_device)
    print_stratified(f"2. offline-current (PyTorch/MPS, {CURRENT_IMGSZ[1]}x{CURRENT_IMGSZ[0]}, control)", stats_offline_current)
    print_stratified(f"3. offline-larger (PyTorch/MPS, {args.larger_imgsz[1]}x{args.larger_imgsz[0]})", stats_offline_larger)
    print(
        "\nRead this as: (3 vs 2) isolates the resolution effect (same runtime/weights, only imgsz differs) -- "
        "that's the actual answer to \"does more resolution help\". (1 vs 2) tells you how much the CoreML-vs-"
        "PyTorch runtime gap alone matters, i.e. how much to trust arm 2 as representative of on-device reality."
    )


if __name__ == "__main__":
    main()
