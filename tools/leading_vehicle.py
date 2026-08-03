#!/usr/bin/env python3
"""
Prototype for "which vehicle am I following": classifies, per frame, which
tracked vehicle (if any) is the forward-leading vehicle -- the one directly
ahead in the ego lane, as opposed to any other nearby car.

Ports the geometric classifier from Wen, "Vision-based Forward Collision
Warning System Using a Single Camera on a Mobile Device" (U Ottawa, 2021),
Ch. 6.2 / Algorithm 1. That work tried a learned classifier first (AdaBoost
on [w, h, cx, cy]: 6-9% error) and found a much simpler fixed central-band
geometric test outperformed it: 1.48% error at T=0.3, near-zero at T<=0.2.
Deliberately NOT using IPM/lane detection -- see the design discussion this
prototype came out of: IPM needs camera calibration and mount geometry
(height/pitch/roll) this app doesn't establish anywhere, and a miscalibrated
projection is worse than no projection. The thesis's method needs neither.

Ported from the thesis's fixed 640x480 pixel space to *normalized* [0, 1]
top-left-origin coordinates (matching Detection.boundingBox's convention --
see DetectionLogger.swift), so it's resolution-independent.

On top of classify_leading's raw per-frame pick, LeadingVehicleLock adds the
ACC-literature hysteresis layer: a candidate must win `confirm_frames`
consecutive raw picks before it's actually adopted, and the current lock
tolerates up to `grace_frames` frames without being the raw pick before
releasing. Added after the confidence/size gate above still left real
churn on busy footage: post-gate, a day session had 902 switches, 71.7% of
them flip-flops (the raw pick bouncing straight back to the ID it had just
abandoned) -- genuine simultaneous near-ties between real candidates in
heavy traffic, not noise the gate could filter, which is exactly the
scenario this kind of hysteresis is built for.

Usage:
    python3 leading_vehicle.py <session_dir> [--center-x 0.5] [--band-half-width 0.025] [--threshold 0.3]

Needs only detections.jsonl -- no video decode, no GMC (tracker.py's
ByteTracker defaults to use_gmc=False), since this is validating the
classifier logic itself, not re-validating tracking/GMC (already validated
separately). trackIDs are recomputed offline via tracker.py on whatever raw
detections are in the log, same as track_consistency.py's approach --
sessions recorded before on-device trackID logging existed still work fine
here, since this doesn't need on-device IDs at all.
"""
import argparse
import json
import sys
from pathlib import Path

from benchmark import iou
from driverassist_sync import load_detections
from path_awareness import DEFAULT_MAX_YAW_SHIFT, DEFAULT_YAW_SHIFT_PER_DEG_S, curve_adjusted_center_x
from tracker import ByteTracker

# A switch is classified "likely churn" (same physical vehicle, new trackID
# from a brief tracking gap) rather than "likely real change" (an actual
# different vehicle) when the new selection lands close to where the old one
# last was, soon after. Deliberately coarse thresholds for a first pass --
# see analyze_switches's docstring.
CHURN_IOU_THRESHOLD = 0.3
CHURN_MAX_GAP_FRAMES = 3

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}

# Mirrors the thesis's s1=305/s2=335 on 640px-wide frames (~2.3% each side of
# center), expressed as a resolution-independent fraction.
DEFAULT_CENTER_X = 0.5
DEFAULT_BAND_HALF_WIDTH = 0.025
DEFAULT_THRESHOLD = 0.3  # T in Algorithm 1 -- higher is stricter


# Minimum-evidence gate, added after digging into real switch data: with no
# floor at all, tiny/uncertain detections near the vanishing point in busy
# scenes (intersections, multi-lane traffic) qualified as "leading" just as
# readily as a large, confident, close vehicle -- and since those are
# inherently noisy frame-to-frame, they drove most observed switching (only
# ~15% of switches were same-spot trackID churn; the rest tracked with weak
# candidates: on a real night session, 38.3% of all leading selections had
# box width <0.03 and 48.2% had confidence <0.5). These thresholds are set to
# exactly that measured population, not tuned further yet.
DEFAULT_MIN_CONFIDENCE = 0.5
# Halved from an initial 0.03 after finding it rejected a real, genuinely
# centered, plausibly-confident (up to 0.67) distant vehicle (trackID #428,
# session 26_07_30_Day_Hosp_nano_off, ~w=0.015-0.017) -- the original 0.03
# was tuned only against the night session's noise population and turned out
# to be blunter than needed: min_confidence alone does most of the real
# filtering work (dropping min_width to 0 entirely still keeps the night
# session's flip-flops at 0), so 0.015 recovers real distant vehicles while
# only moderately increasing day-session instability (flip-flops 8->17,
# still far below the pre-gate/pre-hysteresis baseline of 647).
DEFAULT_MIN_WIDTH = 0.015

# Own-hood/dashboard rejection, added after finding a real failure on a
# separate day session: bright glare on the ego vehicle's own dashboard got
# persistently misdetected as a "car" by the nano model (8192 such
# detections in one session, 0 in a night session with the same code -- a
# daylight-glare-specific failure the night data never exercised). Every one
# of those false positives had its box bottom edge at or past ~1.0 (touching
# the frame's bottom edge) -- a real forward vehicle's wheels sit on visible
# road above that line; nothing else reliably does. A pure size cap isn't
# specific enough (some of these boxes were only moderately sized), but
# "touches the very bottom edge" was true 100% of the time in the sample
# checked, so that's the gate.
DEFAULT_MAX_BOTTOM_Y = 0.95

# Cross-traffic rejection. A car crossing your path (e.g. at a T-
# intersection) has a fundamentally different motion signature than a
# followed vehicle, and this turned out to be a much cleaner signal than
# shape: aspect ratio alone doesn't separate them (a real close-following
# vehicle naturally gets a wide, vertically-cropped box too -- one real
# 28.5s-long legitimate track sat at aspect ratio ~1.9-2.0 for its whole
# steady-state duration). Horizontal velocity does separate them cleanly:
# on the same session, two confirmed cross-traffic detections (visually
# verified -- cars driving across a cross street) swept 0.40-0.46 frame-
# widths/sec, while the legitimate long-lived track's velocity never
# exceeded ~0.05 frame-widths/sec even during its initial "coming into
# view" transient. 0.2 sits with comfortable margin on both sides.
DEFAULT_MAX_ABS_VELOCITY = 0.2

# Shape (aspect ratio) gate -- worth being upfront that this file already
# looked at aspect ratio once and found it doesn't cleanly separate
# cross-traffic from real close-following vehicles on its own: one real
# 28.5s-long legitimate track sat at aspect ratio ~1.9-2.0 for its whole
# steady-state duration (see DEFAULT_MAX_ABS_VELOCITY's comment above),
# which is the same range an oblique/side view would land in. This default
# is set loose (only rejects genuinely extreme, clearly-not-rear-on boxes)
# for exactly that reason -- validate against real held-out labels (see
# tune_leading_vehicle.py) before tightening it, don't assume it helps.
DEFAULT_MAX_ASPECT_RATIO = 3.5

# Symmetry gate -- rear (and front) views of a vehicle are close to
# bilaterally symmetric (tail lights, bumper, plate); oblique/side views
# aren't. Score is precomputed per detection from the actual video frame
# (see compute_symmetry.py) since it needs pixel access this module
# otherwise deliberately avoids (see module docstring) -- 0 disables the
# gate (every detection passes) since a session without a symmetry cache
# has no `sym` field on its detections at all (see below, same
# missing-data-gets-the-benefit-of-the-doubt handling as `vcx`).
DEFAULT_MIN_SYMMETRY = 0.0

# ~15fps capture (see CameraManager.swift) -- 3 frames is ~0.2s of sustained
# evidence before switching, 5 is ~0.33s tolerance for a brief missed
# detection before releasing the lock. Not deeply tuned, just plausible
# human-perceptible responsiveness; exposed as CLI flags to retune later.
DEFAULT_CONFIRM_FRAMES = 3
DEFAULT_GRACE_FRAMES = 5


class VelocityEstimator:
    """Stamps a `vcx` field (horizontal center velocity, frame-widths/sec,
    or None on a track's first sighting) onto each detection in place, from
    simple frame-to-frame position deltas -- see DEFAULT_MAX_ABS_VELOCITY's
    comment for why this is a cleaner cross-traffic signal than shape.

    Deliberately a plain position-delta estimate, not the tracker's own
    Kalman-filtered velocity state -- self-contained here rather than
    threading velocity out of ByteTracker/tracker.py, and the signal
    separation found in real data (10x) is clean enough that Kalman
    smoothing isn't needed to make the gate work.
    """

    def __init__(self):
        self._last: dict = {}  # trackID -> (t, cx)

    def update(self, detections: list, t: float) -> None:
        for det in detections:
            tid = det.get("trackID")
            if tid is None:
                det["vcx"] = None
                continue
            cx = det["x"] + det["w"] / 2
            prev = self._last.get(tid)
            if prev is not None:
                prev_t, prev_cx = prev
                dt = t - prev_t
                det["vcx"] = (cx - prev_cx) / dt if dt > 0 else None
            else:
                det["vcx"] = None
            self._last[tid] = (t, cx)


def classify_leading(
    detections: list, center_x: float = DEFAULT_CENTER_X,
    band_half_width: float = DEFAULT_BAND_HALF_WIDTH, threshold: float = DEFAULT_THRESHOLD,
    labels: set = VEHICLE_LABELS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE, min_width: float = DEFAULT_MIN_WIDTH,
    max_bottom_y: float = DEFAULT_MAX_BOTTOM_Y, max_abs_velocity: float = DEFAULT_MAX_ABS_VELOCITY,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO, min_symmetry: float = DEFAULT_MIN_SYMMETRY,
):
    """Returns the detection dict (must include trackID) classified as the
    forward-leading vehicle for this frame, or None. Ports Algorithm 1: a
    candidate qualifies either by sitting fully inside the central band
    [s1, s2] (small/far vehicles, precisely centered) or by overlapping it
    with a margin of at least `threshold` * band_width on both sides
    (large/near vehicles, which trivially span a band this narrow as long as
    they're roughly centered at all) -- AND must clear `min_confidence`/
    `min_width` (see those constants' comment), not have its box bottom past
    `max_bottom_y` (rejects the ego vehicle's own hood/dashboard), AND not
    have `vcx` (see VelocityEstimator) exceeding `max_abs_velocity` in
    magnitude (rejects cross-traffic -- a candidate with no velocity history
    yet, `vcx is None`, is allowed through rather than rejected, since a
    brand-new track deserves the benefit of the doubt on its first sighting),
    AND not exceed `max_aspect_ratio` (w/h -- rejects extreme side/oblique
    views; see DEFAULT_MAX_ASPECT_RATIO's comment on why this is set loose),
    AND not fall below `min_symmetry` on its precomputed `sym` score if it
    has one (missing `sym`, like missing `vcx`, gets the benefit of the
    doubt rather than being rejected). If multiple candidates qualify, picks
    the tallest box as a rough closeness proxy -- rare in practice for a
    single-lane-ahead scenario, and not the thesis's distance metric (that
    used width specifically, which needs a calibrated camera we don't have
    here -- see the design discussion), just a placeholder tie-break.
    """
    s1 = center_x - band_half_width
    s2 = center_x + band_half_width
    band_width = s2 - s1

    candidates = []
    for det in detections:
        if det["label"] not in labels:
            continue
        if det["confidence"] < min_confidence or det["w"] < min_width:
            continue
        if det["y"] + det["h"] > max_bottom_y:
            continue
        vcx = det.get("vcx")
        if vcx is not None and abs(vcx) > max_abs_velocity:
            continue
        if det["h"] > 0 and (det["w"] / det["h"]) > max_aspect_ratio:
            continue
        sym = det.get("sym")
        if sym is not None and sym < min_symmetry:
            continue
        x1, x2 = det["x"], det["x"] + det["w"]
        fully_contained = x1 > s1 and x2 < s2
        straddles_with_margin = (x2 - s1) > threshold * band_width and (s2 - x1) > threshold * band_width
        if fully_contained or straddles_with_margin:
            candidates.append(det)

    if not candidates:
        return None
    return max(candidates, key=lambda d: d["h"])


class LeadingVehicleLock:
    """Temporal hysteresis over classify_leading's raw per-frame trackID
    output -- ports the ACC target-vehicle-selection literature's core idea:
    don't switch the locked target just because one frame's raw pick
    differs. Real footage shows raw-pick switching is dominated by brief
    flip-flops between two simultaneously-plausible candidates (busy-traffic
    near-ties, not noise -- see module docstring), not genuine vehicle
    changes, so debouncing the *switch decision* is the right layer, not
    filtering the candidates further.

    A new candidate must win `confirm_frames` consecutive raw picks before
    it's adopted as the lock. The current lock is sticky: it's kept even on
    a frame where a different candidate momentarily raw-wins, as long as
    that candidate hasn't yet reached `confirm_frames` in a row. If the
    locked ID stops being the raw pick at all (out of frame, or fails the
    gate) it stays locked for up to `grace_frames` frames before releasing
    to None, so one missed detection doesn't drop the lock.

    Stateful and single-pass by design (`update` is called once per frame,
    in order) so the same logic drives both this file's offline batch
    analysis (via `apply_hysteresis`) and reconstruct_annotated.py's live
    sequential frame loop, without duplicating the state machine.
    """

    def __init__(self, confirm_frames: int = DEFAULT_CONFIRM_FRAMES, grace_frames: int = DEFAULT_GRACE_FRAMES):
        self.confirm_frames = confirm_frames
        self.grace_frames = grace_frames
        self.locked_id = None
        self._frames_since_seen = 0
        self._pending_id = None
        self._pending_streak = 0

    def update(self, raw_id):
        """Call once per frame with this frame's raw classify_leading()
        trackID (or None). Returns the current locked trackID, or None."""
        if raw_id is not None and raw_id == self.locked_id:
            self._frames_since_seen = 0
            self._pending_id, self._pending_streak = None, 0
        elif raw_id is not None:
            if raw_id == self._pending_id:
                self._pending_streak += 1
            else:
                self._pending_id, self._pending_streak = raw_id, 1
            if self._pending_streak >= self.confirm_frames:
                self.locked_id = self._pending_id
                self._frames_since_seen = 0
                self._pending_id, self._pending_streak = None, 0
            else:
                self._frames_since_seen += 1
        else:
            # No raw pick this frame -- deliberately doesn't reset a pending
            # streak, so one gap frame in an otherwise-confirming sequence
            # doesn't restart confirmation from zero.
            self._frames_since_seen += 1

        if self.locked_id is not None and self._frames_since_seen > self.grace_frames:
            self.locked_id = None

        return self.locked_id


def apply_hysteresis(raw_ids: list, confirm_frames: int = DEFAULT_CONFIRM_FRAMES, grace_frames: int = DEFAULT_GRACE_FRAMES) -> list:
    """Batch convenience over LeadingVehicleLock for a whole session's raw ID
    sequence -- see that class's docstring."""
    lock = LeadingVehicleLock(confirm_frames, grace_frames)
    return [lock.update(raw_id) for raw_id in raw_ids]


def compute_switch_stats(ids: list, box_by_frame: list, entries: list) -> dict:
    """Switch/churn/flip-flop/coverage stats for a per-frame trackID
    sequence -- factored out so raw and hysteresis-smoothed sequences can be
    computed and reported side by side. `box_by_frame[i]` is the raw
    classify_leading() box for frame i (or None), used to look up each ID's
    most-recently-seen real position for switch-event IoU -- needed because
    under hysteresis the locked ID at a switch may not be the box actually
    detected in that exact frame (grace-period stickiness)."""
    switches = 0
    frames_with_leading = 0
    selected_id_counts: dict = {}
    last_real_id = None
    last_real_frame_idx = None
    ever_selected_ids: set = set()
    switch_events: list = []
    last_seen_box: dict = {}

    for frame_idx, (selected_id, raw_box) in enumerate(zip(ids, box_by_frame)):
        if raw_box is not None:
            last_seen_box[raw_box["trackID"]] = raw_box

        if selected_id is None:
            continue

        frames_with_leading += 1
        selected_id_counts[selected_id] = selected_id_counts.get(selected_id, 0) + 1

        if last_real_id is not None and selected_id != last_real_id:
            switches += 1
            prev_box = last_seen_box.get(last_real_id)
            new_box = last_seen_box.get(selected_id)
            overlap = iou(prev_box, new_box) if prev_box is not None and new_box is not None else 0.0
            gap = frame_idx - last_real_frame_idx
            switch_events.append({
                "frame_idx": frame_idx,
                "t": entries[frame_idx]["t"],
                "gap_frames": gap,
                "prev_id": last_real_id,
                "new_id": selected_id,
                "iou_prev_to_new_box": overlap,
                "is_flip_flop": selected_id in ever_selected_ids,
                "likely_churn": overlap >= CHURN_IOU_THRESHOLD and gap <= CHURN_MAX_GAP_FRAMES,
            })

        ever_selected_ids.add(selected_id)
        last_real_id = selected_id
        last_real_frame_idx = frame_idx

    likely_churn = sum(1 for e in switch_events if e["likely_churn"])
    flip_flops = sum(1 for e in switch_events if e["is_flip_flop"])

    return {
        "frames_with_leading": frames_with_leading,
        "distinct_leading_track_ids": len(selected_id_counts),
        "switches": switches,
        "switches_likely_churn": likely_churn,
        "switches_likely_real_change": switches - likely_churn,
        "switches_that_are_flip_flops": flip_flops,
        "top_track_ids_by_frame_count": sorted(
            selected_id_counts.items(), key=lambda kv: -kv[1]
        )[:5],
        "switch_events": switch_events,
    }


def run_session(
    entries: list, center_x: float, band_half_width: float, threshold: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE, min_width: float = DEFAULT_MIN_WIDTH,
    max_bottom_y: float = DEFAULT_MAX_BOTTOM_Y, max_abs_velocity: float = DEFAULT_MAX_ABS_VELOCITY,
    confirm_frames: int = DEFAULT_CONFIRM_FRAMES, grace_frames: int = DEFAULT_GRACE_FRAMES,
    yaw_rate_key: str = None, yaw_sign: float = 1.0,
    yaw_shift_per_deg_s: float = DEFAULT_YAW_SHIFT_PER_DEG_S, max_yaw_shift: float = DEFAULT_MAX_YAW_SHIFT,
) -> dict:
    """`yaw_rate_key`, if given, names the DetectionLogEntry field to read a
    signed yaw rate from (e.g. "rotationRateZDegreesPerSecond") -- which axis
    is actually yaw isn't confirmed yet (see curve_adjusted_center_x), so this
    is left as a choice for the caller rather than hardcoded. None (the
    default) disables curve adjustment entirely, matching every session
    recorded so far, none of which have rotation rate logged at all."""
    tracker = ByteTracker()  # geometry-only (use_gmc defaults False) -- see module docstring
    velocity = VelocityEstimator()
    frames_with_vehicles = 0
    raw_ids: list = []
    box_by_frame: list = []  # classify_leading's raw box (or None) per frame

    for entry in entries:
        dets = entry["detections"]
        track_ids = tracker.update(dets)
        for det, tid in zip(dets, track_ids):
            det["trackID"] = tid
        velocity.update(dets, entry["t"])

        vehicle_dets = [d for d in dets if d["label"] in VEHICLE_LABELS]
        if vehicle_dets:
            frames_with_vehicles += 1

        frame_center_x = center_x
        raw_yaw = entry.get(yaw_rate_key) if yaw_rate_key else None
        if raw_yaw is not None:
            frame_center_x = curve_adjusted_center_x(
                center_x, raw_yaw * yaw_sign, yaw_shift_per_deg_s, max_yaw_shift
            )

        leading = classify_leading(
            dets, frame_center_x, band_half_width, threshold,
            min_confidence=min_confidence, min_width=min_width,
            max_bottom_y=max_bottom_y, max_abs_velocity=max_abs_velocity,
        )
        raw_ids.append(leading["trackID"] if leading is not None else None)
        box_by_frame.append(leading)

    smoothed_ids = apply_hysteresis(raw_ids, confirm_frames, grace_frames)

    raw_stats = compute_switch_stats(raw_ids, box_by_frame, entries)
    hysteresis_stats = compute_switch_stats(smoothed_ids, box_by_frame, entries)

    def with_coverage(stats: dict) -> dict:
        return {
            **stats,
            "leading_fraction_of_vehicle_frames": (
                stats["frames_with_leading"] / frames_with_vehicles if frames_with_vehicles else None
            ),
        }

    return {
        "total_frames": len(entries),
        "frames_with_vehicles": frames_with_vehicles,
        "raw": with_coverage(raw_stats),
        "hysteresis": {
            **with_coverage(hysteresis_stats),
            "confirm_frames": confirm_frames,
            "grace_frames": grace_frames,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Packaged session directory (see package_session.py)")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to <session_dir>/detections.jsonl")
    parser.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    parser.add_argument("--band-half-width", type=float, default=DEFAULT_BAND_HALF_WIDTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-width", type=float, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--max-bottom-y", type=float, default=DEFAULT_MAX_BOTTOM_Y)
    parser.add_argument("--max-abs-velocity", type=float, default=DEFAULT_MAX_ABS_VELOCITY)
    parser.add_argument("--confirm-frames", type=int, default=DEFAULT_CONFIRM_FRAMES)
    parser.add_argument("--grace-frames", type=int, default=DEFAULT_GRACE_FRAMES)
    parser.add_argument(
        "--yaw-rate-key", type=str, default=None,
        help="DetectionLogEntry field to read signed yaw rate from (e.g. rotationRateZDegreesPerSecond) "
             "-- disabled by default, since no recorded session has this logged yet and which axis is "
             "actually yaw isn't confirmed (see path_awareness.py).",
    )
    parser.add_argument("--yaw-sign", type=float, default=1.0, help="Multiplier to flip sign if the raw axis reads backwards")
    parser.add_argument("--yaw-shift-per-deg-s", type=float, default=DEFAULT_YAW_SHIFT_PER_DEG_S)
    parser.add_argument("--max-yaw-shift", type=float, default=DEFAULT_MAX_YAW_SHIFT)
    parser.add_argument("--results", type=Path, default=None, help="JSON path; defaults to <session_dir>/leading-vehicle.json")
    parser.add_argument(
        "--dump-switches", type=int, default=10,
        help="Print this many post-hysteresis switch events (frame/time/gap/IoU/flip-flop) for manual inspection. 0 to disable.",
    )
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist.")

    entries = load_detections(detections_path)
    stats = run_session(
        entries, args.center_x, args.band_half_width, args.threshold,
        min_confidence=args.min_confidence, min_width=args.min_width,
        max_bottom_y=args.max_bottom_y, max_abs_velocity=args.max_abs_velocity,
        confirm_frames=args.confirm_frames, grace_frames=args.grace_frames,
        yaw_rate_key=args.yaw_rate_key, yaw_sign=args.yaw_sign,
        yaw_shift_per_deg_s=args.yaw_shift_per_deg_s, max_yaw_shift=args.max_yaw_shift,
    )

    def print_bucket(name: str, s: dict) -> None:
        print(f"  [{name}]")
        print(f"    frames with a leading:   {s['frames_with_leading']}", end="")
        if s["leading_fraction_of_vehicle_frames"] is not None:
            print(f"  ({s['leading_fraction_of_vehicle_frames']:.1%} of vehicle frames)")
        else:
            print()
        print(f"    distinct leading trackIDs: {s['distinct_leading_track_ids']}")
        print(f"    switches:                {s['switches']}")
        if s["switches"]:
            print(
                f"      likely churn (same-spot handoff, IoU>={CHURN_IOU_THRESHOLD}, "
                f"gap<={CHURN_MAX_GAP_FRAMES}f): {s['switches_likely_churn']} "
                f"({s['switches_likely_churn'] / s['switches']:.1%})"
            )
            print(f"      flip-flops (returned to a previously-selected ID): {s['switches_that_are_flip_flops']}")
        print(f"    top trackIDs by frames selected: {s['top_track_ids_by_frame_count']}")

    print(f"Session: {args.session_dir.name}")
    print(f"  center_x={args.center_x} band_half_width={args.band_half_width} threshold={args.threshold}")
    print(
        f"  min_confidence={args.min_confidence} min_width={args.min_width} "
        f"max_bottom_y={args.max_bottom_y} max_abs_velocity={args.max_abs_velocity}"
    )
    print(f"  confirm_frames={args.confirm_frames} grace_frames={args.grace_frames}")
    print(f"  total frames:            {stats['total_frames']}")
    print(f"  frames with a vehicle:   {stats['frames_with_vehicles']}")
    print()
    print_bucket("raw (gate only, no hysteresis)", stats["raw"])
    print()
    print_bucket("hysteresis-smoothed", stats["hysteresis"])

    if args.dump_switches and stats["hysteresis"]["switch_events"]:
        events = stats["hysteresis"]["switch_events"]
        print(f"\n  First {min(args.dump_switches, len(events))} post-hysteresis switch events:")
        for e in events[: args.dump_switches]:
            tag = "CHURN?" if e["likely_churn"] else "real? "
            flip = " [flip-flop]" if e["is_flip_flop"] else ""
            print(
                f"    frame {e['frame_idx']:>6} t={e['t']:.2f}  #{e['prev_id']:>4} -> #{e['new_id']:<4} "
                f"gap={e['gap_frames']:>3}f  iou={e['iou_prev_to_new_box']:.2f}  {tag}{flip}"
            )

    results_path = args.results or args.session_dir / "leading-vehicle.json"
    with open(results_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
