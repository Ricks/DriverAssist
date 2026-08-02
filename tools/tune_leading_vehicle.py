#!/usr/bin/env python3
"""
Tunes leading_vehicle.py's classifier parameters against human-labeled
ground truth (see label_leading_vehicle.py), instead of the one-parameter-
at-a-time spot-checking used to get the gates to their current defaults.

Overfitting guard: we currently have ground truth for exactly one video.
Searching for parameters that maximize accuracy on the *whole* video and
then reporting that same number would just measure how well the search
memorized this one video, not whether it generalizes -- so this splits the
video's own timeline at TRAIN_FRACTION: everything before the cutoff is
used for the search, everything at/after it is held out and only ever
scored once, after the search is done, using whatever parameters won on
the train portion. A large gap between train and held-out accuracy is the
signal that we're overtraining, not the held-out number itself.

Metric: unlabeled stretches of the video (no ground-truth segment covering
that frame) vastly outnumber labeled ones, so raw per-frame accuracy would
reward a trivially over-conservative classifier that just about never
predicts anything -- it would "match" all the unlabeled time by predicting
None almost everywhere. Using balanced accuracy instead -- the average of
(a) accuracy on frames that should show a specific followed vehicle and
(b) accuracy on frames that should show nothing -- avoids that degenerate
solution, matching the same reasoning behind avoiding class-imbalance
pitfalls this project has hit before (see min_confidence/min_width's
original derivation in leading_vehicle.py).

Search: random search over the whole parameter space rather than a grid --
9 tunable parameters makes an exhaustive grid combinatorially large, and
evaluation is cheap enough (pure arithmetic over already-computed
detections, no model inference) that a few hundred random samples covers
the space well without needing per-parameter grid density decisions.

Usage:
    python3 tune_leading_vehicle.py <session_dir> [--trials 400] [--train-fraction 0.75]
"""
import argparse
import json
import random
import sys
from pathlib import Path

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from package_session import find_session_video
from leading_vehicle import (
    DEFAULT_BAND_HALF_WIDTH,
    DEFAULT_CENTER_X,
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_GRACE_FRAMES,
    DEFAULT_MAX_ABS_VELOCITY,
    DEFAULT_MAX_BOTTOM_Y,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_WIDTH,
    DEFAULT_THRESHOLD,
    VelocityEstimator,
    apply_hysteresis,
    classify_leading,
)
from tracker import ByteTracker

CLASSIFY_KEYS = [
    "center_x", "band_half_width", "threshold",
    "min_confidence", "min_width", "max_bottom_y", "max_abs_velocity",
]
HYSTERESIS_KEYS = ["confirm_frames", "grace_frames"]

DEFAULT_PARAMS = {
    "center_x": DEFAULT_CENTER_X,
    "band_half_width": DEFAULT_BAND_HALF_WIDTH,
    "threshold": DEFAULT_THRESHOLD,
    "min_confidence": DEFAULT_MIN_CONFIDENCE,
    "min_width": DEFAULT_MIN_WIDTH,
    "max_bottom_y": DEFAULT_MAX_BOTTOM_Y,
    "max_abs_velocity": DEFAULT_MAX_ABS_VELOCITY,
    "confirm_frames": DEFAULT_CONFIRM_FRAMES,
    "grace_frames": DEFAULT_GRACE_FRAMES,
}

# (low, high) for floats, or (low, high, "int") for integer params. center_x
# is deliberately NOT searched -- already found, with real data, that every
# tested departure from 0.5 makes things worse (see design discussion), so
# resampling it would just waste trials.
SEARCH_RANGES = {
    "band_half_width": (0.01, 0.15),
    "threshold": (0.1, 0.6),
    "min_confidence": (0.2, 0.8),
    "min_width": (0.005, 0.05),
    "max_bottom_y": (0.85, 0.99),
    "max_abs_velocity": (0.05, 0.5),
    "confirm_frames": (1, 8, "int"),
    "grace_frames": (1, 15, "int"),
}


def ground_truth_at(segments: list, t: float):
    """The trackID ground truth says should be followed at time t, or None
    if nothing should be (uncovered by any labeled segment)."""
    for s in segments:
        if s["start_t"] <= t and (s["end_t"] is None or t < s["end_t"]):
            return s["trackID"]
    return None


def precompute_frames(entries: list, start_epoch: float) -> list:
    """Tracker + velocity estimation don't depend on the classifier
    parameters being swept -- only classify_leading/hysteresis do -- so this
    runs once, mirroring track_benchmark.py's cache-what's-expensive split.
    Frame times are converted to video-relative seconds (t - start_epoch),
    matching ground_truth.json's start_t/end_t convention (see
    label_leading_vehicle.py, which stamps the same conversion) -- comparing
    against raw epoch timestamps would never match anything."""
    tracker = ByteTracker()
    velocity = VelocityEstimator()
    frames = []
    for entry in entries:
        dets = entry["detections"]
        track_ids = tracker.update(dets)
        for det, tid in zip(dets, track_ids):
            det["trackID"] = tid
        t_rel = entry["t"] - start_epoch
        velocity.update(dets, t_rel)
        frames.append({"t": t_rel, "detections": dets})
    return frames


def classify_all(frames: list, params: dict) -> list:
    classify_kwargs = {k: params[k] for k in CLASSIFY_KEYS}
    raw_ids = []
    for f in frames:
        leading = classify_leading(f["detections"], **classify_kwargs)
        raw_ids.append(leading["trackID"] if leading is not None else None)
    return apply_hysteresis(raw_ids, params["confirm_frames"], params["grace_frames"])


def balanced_accuracy(frames: list, predictions: list, segments: list, t_min: float, t_max: float) -> dict:
    pos_correct = pos_total = neg_correct = neg_total = 0
    for f, pred in zip(frames, predictions):
        if f["t"] < t_min or f["t"] >= t_max:
            continue
        gt = ground_truth_at(segments, f["t"])
        if gt is None:
            neg_total += 1
            neg_correct += pred is None
        else:
            pos_total += 1
            pos_correct += pred == gt
    pos_acc = pos_correct / pos_total if pos_total else None
    neg_acc = neg_correct / neg_total if neg_total else None
    balanced = (pos_acc + neg_acc) / 2 if pos_acc is not None and neg_acc is not None else None
    return {
        "pos_accuracy": pos_acc, "pos_frames": pos_total,
        "neg_accuracy": neg_acc, "neg_frames": neg_total,
        "balanced_accuracy": balanced,
    }


def sample_params(rng: random.Random) -> dict:
    params = dict(DEFAULT_PARAMS)
    for key, spec in SEARCH_RANGES.items():
        if len(spec) == 3:
            lo, hi, _ = spec
            params[key] = rng.randint(lo, hi)
        else:
            lo, hi = spec
            params[key] = rng.uniform(lo, hi)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", type=Path, default=None, help="Defaults to the one raw recording in <session_dir>")
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--debug-log", type=Path, default=None, help="Defaults to <session_dir>/overlay-debug.log")
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    gt_path = args.ground_truth or (args.session_dir / "ground_truth.json")
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist.")
    if not gt_path.exists():
        sys.exit(f"{gt_path} doesn't exist -- label this session with label_leading_vehicle.py first.")

    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")
    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)

    entries = load_detections(detections_path)
    segments = json.loads(gt_path.read_text())
    frames = precompute_frames(entries, start_epoch)

    t0, t1 = frames[0]["t"], frames[-1]["t"]
    cutoff = t0 + (t1 - t0) * args.train_fraction
    print(f"Video span: {t0:.1f}s - {t1:.1f}s. Train: [{t0:.1f}, {cutoff:.1f}). Held-out: [{cutoff:.1f}, {t1:.1f}].")
    print(f"Ground truth: {len(segments)} labeled segments.\n")

    def score(params: dict, t_min: float, t_max: float) -> dict:
        predictions = classify_all(frames, params)
        return balanced_accuracy(frames, predictions, segments, t_min, t_max)

    baseline_train = score(DEFAULT_PARAMS, t0, cutoff)
    baseline_held = score(DEFAULT_PARAMS, cutoff, t1)
    print("[Baseline: current leading_vehicle.py defaults]")
    print(f"  train:     balanced={baseline_train['balanced_accuracy']:.3f}  "
          f"pos={baseline_train['pos_accuracy']:.3f} ({baseline_train['pos_frames']} frames)  "
          f"neg={baseline_train['neg_accuracy']:.3f} ({baseline_train['neg_frames']} frames)")
    print(f"  held-out:  balanced={baseline_held['balanced_accuracy']:.3f}  "
          f"pos={baseline_held['pos_accuracy']:.3f} ({baseline_held['pos_frames']} frames)  "
          f"neg={baseline_held['neg_accuracy']:.3f} ({baseline_held['neg_frames']} frames)\n")

    rng = random.Random(args.seed)
    best_params, best_train_score = None, -1.0
    print(f"Searching {args.trials} random parameter combinations (train portion only)...", file=sys.stderr)
    for i in range(args.trials):
        params = sample_params(rng)
        result = score(params, t0, cutoff)
        if result["balanced_accuracy"] is not None and result["balanced_accuracy"] > best_train_score:
            best_train_score = result["balanced_accuracy"]
            best_params = params
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.trials}  best train balanced_accuracy so far: {best_train_score:.3f}", file=sys.stderr)

    best_train = score(best_params, t0, cutoff)
    best_held = score(best_params, cutoff, t1)
    print("\n[Best found: selected on train portion only]")
    print("  params:", json.dumps(best_params, indent=2))
    print(f"  train:     balanced={best_train['balanced_accuracy']:.3f}  "
          f"pos={best_train['pos_accuracy']:.3f} ({best_train['pos_frames']} frames)  "
          f"neg={best_train['neg_accuracy']:.3f} ({best_train['neg_frames']} frames)")
    print(f"  held-out:  balanced={best_held['balanced_accuracy']:.3f}  "
          f"pos={best_held['pos_accuracy']:.3f} ({best_held['pos_frames']} frames)  "
          f"neg={best_held['neg_accuracy']:.3f} ({best_held['neg_frames']} frames)")

    gap = best_train["balanced_accuracy"] - best_held["balanced_accuracy"]
    print(f"\n  train-vs-held-out gap: {gap:+.3f} {'(overfitting warning)' if gap > 0.08 else '(looks reasonable)'}")

    results_path = args.results or args.session_dir / "leading-vehicle-tuning.json"
    results_path.write_text(json.dumps({
        "baseline_params": DEFAULT_PARAMS, "baseline_train": baseline_train, "baseline_held_out": baseline_held,
        "best_params": best_params, "best_train": best_train, "best_held_out": best_held,
        "train_fraction": args.train_fraction, "trials": args.trials, "seed": args.seed,
    }, indent=2))
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
