#!/usr/bin/env python3
"""
Combines latency and tracking accuracy into a single pass/fail-then-rank
verdict for comparing on-device configs (nano vs small, resolution, two-pass,
etc.), instead of eyeballing an F1-vs-latency chart and guessing whether the
slower option's accuracy gain is "worth it".

Two-gate framework, applied per config found in the session:

  1. FEASIBILITY gate -- can this config keep up with the camera at all?
     Measured as achieved fps (from real inter-frame timestamps in the log,
     not assumed) against --nominal-fps * --fps-tolerance. A config that
     can't sustain the camera's capture rate is disqualified outright,
     regardless of how accurate it is -- it will drop frames in practice.

  2. ACCEPTABILITY gate -- is it fast enough to matter to a driver? Measured
     as P(WARNED IN TIME) (see below) at --lag-budget-ms, which must clear
     --warned-in-time-threshold. --lag-budget-ms is deliberately not anchored
     to the ~100-200ms just-noticeable-difference for perceived simultaneity
     -- that threshold is about whether a delay is detectable, not whether
     it's safe. The relevant budget is a driver's total perception-reaction
     time (commonly modeled in the 1-2.5s range in traffic engineering),
     against which even 300ms of extra model latency is a small tax.
     --lag-budget-ms defaults to 300 for that reason: a round, comfortably-
     conservative fraction of even the low end of that range, not a tuned/
     precise figure.

Configs clearing both gates are ranked by reliability -- miss rate (lower is
better), then idf1/mota/mostly_tracked -- not by raw speed, since among
configs that are both fast enough and keep up with the camera, catching more
real objects matters more than shaving further latency.

DETECTION LAG is the key new metric here, empirically measured rather than
derived from a formula: for every object the reference model (yolo26x) tracks
continuously for at least --min-track-frames frames, find the first frame at
which the on-device tracker's own track stream is matched to it (the same
per-frame IoU-based correspondence track_benchmark.py already computes for
IDF1/MOTA, extracted from motmetrics' own event log). The gap between the
object's first appearance and that first match is how long a real object was
present before the app could have warned about it -- it folds together both
fps-driven sampling delay AND recall misses (an object present for several
frames before the tracker catches it) into one number that's directly
comparable across configs with different fps.

P(WARNED IN TIME): objects the on-device tracker never matches at all are not
averaged into detection lag as an infinite delay (that would hide a real
distinction -- "never warned" is a different failure mode than "warned late")
nor dropped from the analysis (that would silently discard the worst cases).
Instead this is treated as a right-censored time-to-event problem, the same
technique survival analysis uses for "the patient left the study before the
event occurred": a detected object contributes an observed lag; an object the
tracker never catches contributes a censoring time (when it left the scene,
per the reference track's own last-seen frame -- not the session's end,
since once a real object is gone there's no more chance to detect it). A
Kaplan-Meier estimator over these gives a survival curve S(t) = P(still
undetected at time t), and 1-S(t) at a given reaction-time budget is
P(warned in time) -- a single, honest probability that already incorporates
both how fast detections happen AND how often they happen at all, without
an arbitrary weighting between the two (see the module-level REACTION_
BUDGET_CHECKPOINTS_MS for the reported checkpoints). The acceptability gate
below is now built on this rather than a raw p95-vs-threshold comparison.
Caveat: Kaplan-Meier assumes censoring is uninformative (that a
never-detected object isn't systematically different -- smaller, farther,
more occluded -- in ways that bias the curve); that's probably not fully
true here, but it's the same limitation every real censored dataset has.

REFERENCE SAMPLING CLOCK: the reference model runs on a FIXED, config-
independent clock (every --dense-interval-s seconds, default matching the
camera's nominal 15fps) spanning the session, NOT on whichever frames a
given on-device config happened to log. Piggybacking the reference on the
on-device config's own (possibly sparse) logging rate -- the original,
simpler design -- silently coarsens the "ground truth" object-first-
appearance clock to match that config's own slowness: an object that
appeared and left again between two widely-spaced log entries is invisible
to the reference too, so a slow config's own P(warned in time) numbers came
out optimistically biased, and more so the slower the config. Sampling
independently removes that asymmetry -- every config is now scored against
the same dense, real timeline of when objects actually appeared, regardless
of how often that config itself managed to look. Each on-device log entry is
matched (for both IDF1/MOTA scoring and detection-lag lookups) against
whichever dense reference sample is nearest in time to its own capture
moment (bounded by half the sampling interval).

Usage:
    python3 efficacy_score.py <clean.mov>

Caches to <video>-dense-reference-cache.pkl (NOT track_benchmark.py's
<video>-track-cache.pkl -- that tool's sparse, per-on-device-frame reference
is a valid design for ITS purpose, scoring tracking algorithms at matched
instants, so it's left alone; the dense cache here is specific to this
tool's cross-config lag comparison). Rerunning after the first pass is fast
-- only the (cheap) tracker matching and Kaplan-Meier computation redo, not
the reference-model inference.
"""
import argparse
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from ultralytics import YOLO

from benchmark import DEFAULT_REFERENCE_MODEL, config_label, run_reference_model
from benchmark_common import DEFAULT_REFERENCE_IMGSZ, TARGET_CLASSES, parse_imgsz
from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from gmc import GMC
from track_benchmark import METRICS, build_cost_matrix
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

DEFAULT_NOMINAL_FPS = 15.0
DEFAULT_FPS_TOLERANCE = 0.9
DEFAULT_LAG_BUDGET_MS = 300.0
DEFAULT_MIN_TRACK_FRAMES = 5
DEFAULT_WARNED_IN_TIME_THRESHOLD = 0.9
DEFAULT_DENSE_INTERVAL_S = 1.0 / 15.0

# Checkpoints P(warned in time) is reported at, for comparing configs across
# the full plausible driver perception-reaction-time range -- 300 is the
# tool's own conservative default acceptability budget; 1000/1500/2500 span
# the range traffic engineering commonly models for total PRT.
REACTION_BUDGET_CHECKPOINTS_MS = [300.0, 1000.0, 1500.0, 2500.0]


def kaplan_meier(event_times: list, censored_times: list) -> tuple:
    """Standard right-censored Kaplan-Meier estimator. event_times: observed
    times-to-detection. censored_times: times an object left the scene
    (per its reference track's own last-seen frame) without ever being
    detected -- "survived" (undetected) at least that long, true value
    unknown. Returns (times, survival), a right-continuous step function
    S(t) = P(still undetected at time t); times[0]/survival[0] = (0.0, 1.0)."""
    at_risk = len(event_times) + len(censored_times)
    if at_risk == 0:
        return [0.0], [1.0]

    combined = sorted([(t, "event") for t in event_times] + [(t, "censor") for t in censored_times])
    times, survival = [0.0], [1.0]
    s = 1.0
    i = 0
    while i < len(combined):
        t = combined[i][0]
        d = c = 0
        while i < len(combined) and combined[i][0] == t:
            if combined[i][1] == "event":
                d += 1
            else:
                c += 1
            i += 1
        if d > 0:
            s *= 1.0 - d / at_risk
            times.append(t)
            survival.append(s)
        at_risk -= d + c
    return times, survival


def survival_at(times: list, survival: list, t_query: float) -> float:
    """Step-function lookup: the survival value in effect at t_query."""
    s = 1.0
    for t, val in zip(times, survival):
        if t <= t_query:
            s = val
        else:
            break
    return s


def _embed(reid_encoder, boxes: list, frame) -> list:
    if not boxes:
        return []
    raw = reid_encoder(frame, _dets_to_pixel_xywh(boxes, frame.shape))
    return [None if e is None else np.asarray(e, dtype=float).tolist() for e in raw]


def build_dense_reference_cache(
    video: Path, video_start_epoch: float, session_start: float, session_end: float,
    model, reid_encoder, args, interval_s: float,
) -> tuple:
    """Runs the reference model on a fixed, config-independent clock (every
    interval_s seconds from session_start to session_end) instead of
    piggybacking on any particular on-device config's own logging rate --
    see module docstring. Returns (times, records, frame_shape); times[i]
    is the real capture time records[i] corresponds to (no on-device
    `detections` entries back these samples, unlike track_benchmark.py's
    build_cache)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    gmc = GMC()
    frame_shape = None
    times, records = [], []
    evaluated, skipped = 0, 0
    t = session_start
    while t <= session_end:
        offset_ms = (t - video_start_epoch) * 1000.0
        if offset_ms < 0:
            t += interval_s
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        ok, frame = cap.read()
        if not ok:
            skipped += 1
            t += interval_s
            continue

        frame_shape = frame_shape or frame.shape[:2]
        reference_boxes = run_reference_model(model, frame, args.device, args.imgsz)
        records.append({
            "reference_boxes": reference_boxes,
            "reference_embeddings": _embed(reid_encoder, reference_boxes, frame),
            "gmc_warp": gmc.apply(frame).tolist(),
        })
        times.append(t)

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"  dense reference pass: {evaluated} samples evaluated, {skipped} skipped so far...", file=sys.stderr)
        t += interval_s

    cap.release()
    print(f"Dense reference cache built: {evaluated} samples evaluated, {skipped} skipped.")
    return times, records, frame_shape


def build_on_device_side_cache(detections: list, video: Path, video_start_epoch: float, reid_encoder) -> list:
    """Per-on-device-entry embeddings + GMC warp (for the on-device
    tracker's own appearance matching / motion compensation) -- independent
    of the reference sampling clock, computed at each entry's own capture
    time. Returns a list parallel to `detections` (None for an
    out-of-range/unreadable entry)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    gmc = GMC()
    records = []
    evaluated, skipped = 0, 0
    for entry in detections:
        capture_time = entry["t"] - entry["elapsedMs"] / 1000.0
        offset_ms = (capture_time - video_start_epoch) * 1000.0
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

        on_device_boxes = [d for d in entry["detections"] if d["label"] in TARGET_CLASSES]
        records.append({
            "on_device_embeddings": _embed(reid_encoder, on_device_boxes, frame),
            "gmc_warp": gmc.apply(frame).tolist(),
        })
        evaluated += 1
        if evaluated % 100 == 0:
            print(f"  on-device-side pass: {evaluated} entries evaluated, {skipped} skipped so far...", file=sys.stderr)

    cap.release()
    print(f"On-device-side cache built: {evaluated} entries evaluated, {skipped} skipped.")
    return records


def latency_stats(entries: list) -> dict:
    elapsed = np.array([e["elapsedMs"] for e in entries if e.get("elapsedMs") is not None])
    times = np.array(sorted(e["t"] for e in entries))
    dt = np.diff(times)
    achieved_fps = 1.0 / np.median(dt) if len(dt) else 0.0
    return {
        "mean_ms": float(elapsed.mean()) if len(elapsed) else None,
        "p95_ms": float(np.percentile(elapsed, 95)) if len(elapsed) else None,
        "achieved_fps": float(achieved_fps),
        "n": len(entries),
    }


def detection_lag_stats(
    dense_reference_ids: list, dense_times: list, on_device_entry_times: list,
    hyp_events: pd.DataFrame, frame_indices: list, min_track_frames: int, gate_budget_ms: float,
) -> dict:
    """`dense_reference_ids`/`dense_times` are the fixed-clock reference
    tracker's output (see build_dense_reference_cache) -- this is where each
    track's first/last-seen time comes from, independent of any on-device
    config's own logging rate. `on_device_entry_times` is where a MATCH's
    real timestamp comes from (when the on-device app actually produced that
    detection) -- a genuinely different index space, since a match is a
    property of the on-device stream, not the dense one. `hyp_events` is one
    config's slice of an mm.MOTAccumulator's mot_events (Type/OId/HId/
    FrameId), where FrameId indexes positionally into `frame_indices` (the
    on-device entry-list positions fed to .update(), in call order)."""
    track_first_seen: dict = {}
    track_last_seen: dict = {}
    track_frame_count: dict = defaultdict(int)
    for pos, ids in enumerate(dense_reference_ids):
        if ids is None:
            continue
        for tid in ids:
            track_frame_count[tid] += 1
            if tid not in track_first_seen:
                track_first_seen[tid] = dense_times[pos]
            track_last_seen[tid] = dense_times[pos]

    eligible = {tid for tid, n in track_frame_count.items() if n >= min_track_frames}

    events = hyp_events.reset_index()  # FrameId/Event are index levels, not columns
    matches = events[events["Type"] == "MATCH"]
    first_match_frame: dict = {}
    for oid, group in matches.groupby("OId"):
        tid = int(oid)
        if tid not in eligible:
            continue
        earliest_call_idx = int(group["FrameId"].min())
        first_match_frame[tid] = frame_indices[earliest_call_idx]

    lags = []
    event_times_ms = []
    censored_times_ms = []
    never_detected = 0
    for tid in eligible:
        if tid in first_match_frame:
            lag_ms = max(0.0, (on_device_entry_times[first_match_frame[tid]] - track_first_seen[tid]) * 1000.0)
            lags.append(lag_ms)
            event_times_ms.append(lag_ms)
        else:
            never_detected += 1
            censor_ms = max(0.0, (track_last_seen[tid] - track_first_seen[tid]) * 1000.0)
            censored_times_ms.append(censor_ms)

    km_times, km_survival = kaplan_meier(event_times_ms, censored_times_ms)
    checkpoints = sorted(set(REACTION_BUDGET_CHECKPOINTS_MS) | {gate_budget_ms})
    p_warned_in_time = {
        budget: 1.0 - survival_at(km_times, km_survival, budget)
        for budget in checkpoints
    }

    lags_arr = np.array(lags)
    return {
        "eligible_tracks": len(eligible),
        "never_detected": never_detected,
        "miss_rate": never_detected / len(eligible) if eligible else None,
        "median_lag_ms": float(np.median(lags_arr)) if len(lags_arr) else None,
        "p95_lag_ms": float(np.percentile(lags_arr, 95)) if len(lags_arr) else None,
        "p_warned_in_time": p_warned_in_time,
        "survival_curve": {"times_ms": km_times, "survival": km_survival},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to score")
    parser.add_argument("--detections", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--debug-log", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--reference-model", type=Path, default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--imgsz", type=parse_imgsz, default=DEFAULT_REFERENCE_IMGSZ)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--results", type=Path, default=None, help="JSON path; defaults to <video>-efficacy-score.json")
    parser.add_argument("--cache", type=Path, default=None, help="Defaults to <video>-dense-reference-cache.pkl")
    parser.add_argument("--dense-interval-s", type=float, default=DEFAULT_DENSE_INTERVAL_S, help="Fixed reference-sampling interval, seconds (default: 1/15, matching nominal camera fps -- see module docstring)")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--reid", dest="reid", action="store_true", default=True)
    parser.add_argument("--no-reid", dest="reid", action="store_false")
    parser.add_argument("--reid-model", default=DEFAULT_REID_MODEL)
    parser.add_argument("--track-iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--track-max-age", type=int, default=DEFAULT_MAX_AGE)
    parser.add_argument("--track-min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument("--track-high-conf", type=float, default=DEFAULT_HIGH_CONF_THRESHOLD)
    parser.add_argument("--track-low-conf", type=float, default=DEFAULT_LOW_CONF_THRESHOLD)
    parser.add_argument("--track-appearance-thresh", type=float, default=DEFAULT_APPEARANCE_THRESHOLD)
    parser.add_argument("--use-gmc", action="store_true")
    parser.add_argument("--nominal-fps", type=float, default=DEFAULT_NOMINAL_FPS, help="Camera's nominal capture rate (default: 15)")
    parser.add_argument("--fps-tolerance", type=float, default=DEFAULT_FPS_TOLERANCE, help="Feasibility gate: achieved fps must be >= nominal*tolerance (default: 0.9)")
    parser.add_argument("--lag-budget-ms", type=float, default=DEFAULT_LAG_BUDGET_MS, help="Acceptability gate: P(warned in time) is evaluated at this reaction budget (default: 300ms, see module docstring)")
    parser.add_argument("--warned-in-time-threshold", type=float, default=DEFAULT_WARNED_IN_TIME_THRESHOLD, help="Acceptability gate: P(warned in time) at --lag-budget-ms must be >= this (default: 0.9)")
    parser.add_argument("--min-track-frames", type=int, default=DEFAULT_MIN_TRACK_FRAMES, help="Minimum reference-track lifetime (frames) to count toward detection lag (default: 5)")
    args = parser.parse_args()

    results_path = args.results or args.video.with_name(args.video.stem + "-efficacy-score.json")
    cache_path = args.cache or args.video.with_name(args.video.stem + "-track-cache.pkl")

    start_epoch, _ = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)
    detections = [e for e in detections if e["t"] >= start_epoch]
    if not detections:
        sys.exit(f"No detections at or after {args.video.name}'s start time were found in {args.detections}.")
    if args.max_entries:
        detections = detections[: args.max_entries]

    on_device_capture_times = [e["t"] - e["elapsedMs"] / 1000.0 for e in detections]
    session_start, session_end = min(on_device_capture_times), max(on_device_capture_times)

    dense_times, dense_records, on_device_side, frame_shape = None, None, None, None
    if cache_path.exists() and not args.rebuild_cache:
        import pickle
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        signature = {
            "video": args.video.name, "reference_model": str(args.reference_model),
            "imgsz": args.imgsz, "reid_model": args.reid_model, "num_detections": len(detections),
            "dense_interval_s": args.dense_interval_s,
        }
        if cached.get("signature") == signature:
            dense_times, dense_records = cached["dense_times"], cached["dense_records"]
            on_device_side, frame_shape = cached["on_device_side"], cached["frame_shape"]
            print(f"Using cached dense reference from {cache_path} ({len(dense_records)} samples, "
                  f"{len(on_device_side)} on-device entries)")

    if dense_records is None:
        import pickle
        print(f"Loading reference model {args.reference_model}...", file=sys.stderr)
        if not args.reference_model.exists():
            sys.exit(f"{args.reference_model} doesn't exist.")
        model = YOLO(str(args.reference_model))
        reid_encoder = build_reid_encoder(args.reid_model, device=args.device)
        dense_times, dense_records, frame_shape = build_dense_reference_cache(
            args.video, start_epoch, session_start, session_end, model, reid_encoder, args, args.dense_interval_s,
        )
        on_device_side = build_on_device_side_cache(detections, args.video, start_epoch, reid_encoder)
        signature = {
            "video": args.video.name, "reference_model": str(args.reference_model),
            "imgsz": args.imgsz, "reid_model": args.reid_model, "num_detections": len(detections),
            "dense_interval_s": args.dense_interval_s,
        }
        with open(cache_path, "wb") as f:
            pickle.dump({
                "signature": signature, "frame_shape": frame_shape,
                "dense_times": dense_times, "dense_records": dense_records, "on_device_side": on_device_side,
            }, f)
        print(f"Cache written to {cache_path}")

    # Ground-truth tracks, one continuous ByteTracker over the fixed-clock
    # dense reference sequence -- independent of any on-device config's own
    # logging rate (see module docstring). Also records each track's
    # first/last-seen time for the detection-lag computation.
    reference_tracker = ByteTracker()
    dense_reference_ids = []
    for record in dense_records:
        ref_boxes = record["reference_boxes"]
        ref_embeds = [None if e is None else np.array(e) for e in record["reference_embeddings"]]
        gmc_kwargs = {"gmc_warp": np.array(record["gmc_warp"]), "frame_shape": frame_shape}
        ids = reference_tracker.update(ref_boxes, embeddings=ref_embeds, **gmc_kwargs)
        dense_reference_ids.append(ids)

    on_device_tracker = ByteTracker(
        iou_threshold=args.track_iou_threshold, max_age=args.track_max_age, min_hits=args.track_min_hits,
        high_conf_threshold=args.track_high_conf, low_conf_threshold=args.track_low_conf,
        appearance_thresh=args.track_appearance_thresh,
    )
    accumulators = defaultdict(lambda: mm.MOTAccumulator(auto_id=True))
    config_frame_indices = defaultdict(list)  # config -> [entry-list position, in .update() call order]

    evaluated, skipped = 0, 0
    for pos, (entry, od_record) in enumerate(zip(detections, on_device_side)):
        if od_record is None:
            skipped += 1
            continue

        # Nearest dense reference sample to this on-device entry's own
        # capture time -- both sides of the comparison (boxes and track
        # IDs) come from that one dense sample, so identities stay
        # consistent with dense_reference_ids above.
        capture_time = on_device_capture_times[pos]
        i = bisect_left(dense_times, capture_time)
        if i > 0 and (i == len(dense_times) or abs(dense_times[i - 1] - capture_time) <= abs(dense_times[i] - capture_time)):
            i -= 1
        reference_boxes = dense_records[i]["reference_boxes"]
        reference_ids = dense_reference_ids[i]

        on_device_boxes = [d for d in entry["detections"] if d["label"] in TARGET_CLASSES]

        if args.reid:
            od_embeds = [None if e is None else np.array(e) for e in od_record["on_device_embeddings"]]
        else:
            od_embeds = [None] * len(on_device_boxes)

        gmc_kwargs = {}
        if args.use_gmc:
            gmc_kwargs = {"gmc_warp": np.array(od_record["gmc_warp"]), "frame_shape": frame_shape}

        on_device_ids = on_device_tracker.update(on_device_boxes, embeddings=od_embeds, **gmc_kwargs)

        costs = build_cost_matrix(reference_boxes, on_device_boxes, args.iou_threshold)
        label = config_label(entry)
        accumulators[label].update(reference_ids, on_device_ids, costs)
        config_frame_indices[label].append(pos)
        evaluated += 1

    print(f"\nDone: {evaluated} entries evaluated, {skipped} skipped.")
    if not accumulators:
        sys.exit("No entries were evaluated -- nothing to report.")

    mh = mm.metrics.create()
    verdicts = {}
    for label, acc in accumulators.items():
        entries_for_config = [detections[i] for i in config_frame_indices[label]]
        lat = latency_stats(entries_for_config)

        summary = mh.compute(acc, metrics=METRICS, name=label).iloc[0].to_dict()
        lag = detection_lag_stats(
            dense_reference_ids, dense_times, on_device_capture_times, acc.mot_events,
            config_frame_indices[label], args.min_track_frames, args.lag_budget_ms,
        )

        feasible = lat["achieved_fps"] >= args.nominal_fps * args.fps_tolerance
        p_warned = lag["p_warned_in_time"].get(args.lag_budget_ms)
        acceptable = p_warned is not None and p_warned >= args.warned_in_time_threshold
        if not feasible:
            verdict = "FAIL (can't sustain camera fps)"
        elif not acceptable:
            verdict = "FAIL (P(warned in time) below threshold)"
        else:
            verdict = "PASS"

        verdicts[label] = {
            "latency": lat, "tracking": summary, "detection_lag": lag,
            "feasible": feasible, "acceptable": acceptable, "verdict": verdict,
        }

    print("\nEfficacy scorecard:")
    for label, v in verdicts.items():
        lat, lag = v["latency"], v["detection_lag"]
        print(f"\n  {label}: {v['verdict']}")
        print(f"    latency: {lat['mean_ms']:.0f}ms mean, {lat['p95_ms']:.0f}ms p95, {lat['achieved_fps']:.1f} achieved fps")
        print(f"    reliability: idf1={v['tracking']['idf1']:.3f} mota={v['tracking']['mota']:.3f} "
              f"mostly_tracked={v['tracking']['mostly_tracked']:.0f} mostly_lost={v['tracking']['mostly_lost']:.0f}")
        if lag["median_lag_ms"] is not None:
            print(f"    detection lag: {lag['median_lag_ms']:.0f}ms median, {lag['p95_lag_ms']:.0f}ms p95 "
                  f"(n={lag['eligible_tracks']} tracks, {lag['never_detected']} never detected, "
                  f"miss rate {lag['miss_rate']:.1%})")
            checkpoints_str = ", ".join(
                f"{budget:.0f}ms={lag['p_warned_in_time'][budget]:.1%}"
                for budget in REACTION_BUDGET_CHECKPOINTS_MS
            )
            print(f"    P(warned in time): {checkpoints_str}")
        else:
            print(f"    detection lag: no eligible reference tracks (n={lag['eligible_tracks']})")

    passing = {label: v for label, v in verdicts.items() if v["verdict"] == "PASS"}
    if passing:
        ranked = sorted(
            passing.items(),
            key=lambda kv: (-kv[1]["detection_lag"]["p_warned_in_time"][args.lag_budget_ms], -kv[1]["tracking"]["idf1"]),
        )
        print(f"\nRanked (passing configs only, by P(warned in time) at {args.lag_budget_ms:.0f}ms then idf1):")
        for i, (label, v) in enumerate(ranked, 1):
            p = v["detection_lag"]["p_warned_in_time"][args.lag_budget_ms]
            print(f"  {i}. {label} (P(warned in time)={p:.1%}, idf1 {v['tracking']['idf1']:.3f})")

    with open(results_path, "w") as f:
        json.dump(
            {
                "video": args.video.name,
                "gates": {
                    "nominal_fps": args.nominal_fps, "fps_tolerance": args.fps_tolerance,
                    "lag_budget_ms": args.lag_budget_ms, "min_track_frames": args.min_track_frames,
                },
                "configs": verdicts,
            },
            f, indent=2,
        )
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
