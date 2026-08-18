#!/usr/bin/env python3
"""
Benchmarks the app's real-time on-device detections against a stronger
reference model (yolo26x, run offline) on the same recording — real
precision/recall/F1 per object class and per on-device config (model tier x
two-pass), plus latency (the app's own logged per-frame compute time) per
config. Produces a chart, a console summary, and a JSON results file
(<video>-benchmark.json) — the JSON is what plot_benchmarks.py reads to
combine multiple sessions into one chart later, without re-running the
(slow) reference-model inference.

Usage:
    python3 benchmark.py <clean.mov>

--detections/--debug-log both default to ~/DriverAssist/logs/ (see
driverassist_sync.py for how frame sync and log matching work — a single
drive can contain multiple configs if you swiped/spoke a change mid-drive,
and each is broken out separately here, matched by exactly what
detections.jsonl logged for that frame).

The reference model runs only on frames a detection was actually logged for
— not every raw video frame — which is what keeps this from taking as long
as the drive itself. It's seeked to each entry's estimated *capture* time
(logged completion time minus the logged elapsedMs), not the completion time
itself, so it evaluates the same visual content the on-device model actually
saw, not a frame captured a fraction of a second later.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from benchmark_common import DEFAULT_REFERENCE_IMGSZ, TARGET_CLASSES, make_chart, parse_imgsz, print_summary, save_results
from driverassist_sync import DEFAULT_LOGS_DIR, MODEL_DISPLAY_NAMES, load_detections, resolve_start_epoch

CONFIDENCE_THRESHOLD = 0.25  # matches YOLODecoder.confidenceThreshold in the app
DEFAULT_REFERENCE_MODEL = Path.home() / "DriverAssist" / "yolo26x.pt"


def config_label(entry: dict) -> str:
    model = MODEL_DISPLAY_NAMES.get(entry["model"], entry["model"])
    two_pass = "on" if entry["twoPass"] else "off"
    return f"{model}, two-pass {two_pass}"


def to_xyxy(box: dict) -> tuple:
    return box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(predicted: list, reference: list, iou_threshold: float) -> dict:
    """Greedy best-IoU-first matching, independently per class. Returns
    {class: (tp, fp, fn)} for every target class (0s if neither side had it
    in this frame)."""
    by_class_pred = defaultdict(list)
    by_class_ref = defaultdict(list)
    for d in predicted:
        if d["label"] in TARGET_CLASSES:
            by_class_pred[d["label"]].append(d)
    for d in reference:
        by_class_ref[d["label"]].append(d)

    result = {}
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
        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(refs) - len(matched_ref)
        result[cls] = (tp, fp, fn)
    return result


def run_reference_model(model, frame, device: str, imgsz) -> list:
    results = model.predict(frame, verbose=False, device=device, conf=CONFIDENCE_THRESHOLD, imgsz=imgsz)[0]
    h, w = frame.shape[:2]
    names = model.names
    boxes = []
    for box in results.boxes:
        label = names[int(box.cls[0])]
        if label not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "x": x1 / w, "y": y1 / h,
            "w": (x2 - x1) / w, "h": (y2 - y1) / h,
        })
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to benchmark, named however you like")
    parser.add_argument(
        "--detections", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"detections.jsonl, or a directory of them (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--debug-log", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"overlay-debug.log, or a directory of them, for frame sync (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL,
        help=f"Reference weights (default: {DEFAULT_REFERENCE_MODEL})",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU to count as a match (default: 0.5)")
    parser.add_argument("--imgsz", type=parse_imgsz, default=DEFAULT_REFERENCE_IMGSZ, help="Reference model inference resolution, 'HxW' or a bare int (default: near-native, see benchmark_common.DEFAULT_REFERENCE_IMGSZ)")
    parser.add_argument("--device", default="mps", help="torch device for the reference model (default: mps)")
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Cap how many logged detections to evaluate, for a quick test run",
    )
    parser.add_argument("--output", type=Path, default=None, help="Chart PNG path; defaults to <video>-benchmark.png")
    parser.add_argument(
        "--results", type=Path, default=None,
        help="JSON results path (for plot_benchmarks.py later); defaults to <video>-benchmark.json",
    )
    args = parser.parse_args()

    output = args.output or args.video.with_name(args.video.stem + "-benchmark.png")
    results_path = args.results or args.video.with_name(args.video.stem + "-benchmark.json")

    start_epoch, _ = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)
    # --detections may be (and by default is) a whole directory merging many
    # unrelated sessions — drop anything from before this video started so
    # --max-entries samples from *this* drive, not whatever chronologically
    # came first across every drive ever pulled. Entries from a later,
    # unrelated session are already handled harmlessly below (the seek simply
    # fails once it's past this video's actual length).
    detections = [e for e in detections if e["t"] >= start_epoch]
    if not detections:
        sys.exit(f"No detections at or after {args.video.name}'s start time were found in {args.detections}.")
    if args.max_entries:
        detections = detections[: args.max_entries]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {args.video}")

    print(f"Loading reference model {args.reference_model}...", file=sys.stderr)
    if not args.reference_model.exists():
        sys.exit(f"{args.reference_model} doesn't exist — fetch it first (e.g. YOLO('yolo26x.pt') to auto-download).")
    model = YOLO(str(args.reference_model))

    # stats[config][class] = [tp, fp, fn]; latencies[config] = [elapsedMs, ...]
    stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    latencies: dict = defaultdict(list)

    evaluated = 0
    skipped = 0
    for entry in detections:
        # The frame that actually produced this detection was captured
        # elapsedMs before the logged completion time, not at it — seeking to
        # the completion time would evaluate a slightly later (and possibly
        # visually different) frame than the one the on-device model saw.
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

        reference_boxes = run_reference_model(model, frame, args.device, args.imgsz)
        per_class = match_boxes(entry["detections"], reference_boxes, args.iou_threshold)

        label = config_label(entry)
        for cls, (tp, fp, fn) in per_class.items():
            counts = stats[label][cls]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
        latencies[label].append(entry["elapsedMs"])

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"  {evaluated} entries evaluated, {skipped} skipped so far...", file=sys.stderr)

    cap.release()
    print(f"\nDone: {evaluated} entries evaluated against the reference model, {skipped} skipped (out of video range).")

    if not stats:
        sys.exit("No entries were evaluated — nothing to report.")

    print_summary(stats, latencies, args.iou_threshold)
    make_chart(stats, latencies, output, args.iou_threshold, sessions=[args.video.name])
    save_results(results_path, args.video.name, args.iou_threshold, stats, latencies)
    print(f"\nChart written to {output}")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
