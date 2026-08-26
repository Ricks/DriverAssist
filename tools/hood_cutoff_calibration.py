#!/usr/bin/env python3
"""
Builds a growing dataset of hood-truncation samples (angle below horizontal
at which the ego vehicle's hood clips a detection's box) binned by bearing
angle (yaw from the camera's principal axis) -- lets
DistanceEstimator.hoodCutoffAngleDegrees evolve from a single scalar (fit
2026-08-15 from one walkaround session, effectively "hood cutoff averaged
across whatever bearing angles that session happened to cover") into a real
angle-dependent mapping, since a real hood's silhouette isn't a perfect arc
equidistant from the camera at every bearing.

Two commands:

  extract  -- pulls hood-clipped samples from ONE session into the shared
              dataset (calibration/hood_cutoff_samples.jsonl by default), from
              either or both of:
                - `--tethered start:end [start:end ...]`: time windows (video-
                  relative seconds) where the tester was confirmed below any
                  plausible cutoff distance (e.g. a 3m/5m tethered
                  walkaround) -- every detection in the window is hood-
                  clipped by construction, no ambiguity.
                - `ground_truth_hood_truncation.json` in the session
                  directory, if present -- the labeling tool's own segments
                  (see super_tool.py's hood_truncation mode),
                  each {start_t, end_t, trackID} interval treated the same
                  way (every frame in the segment is "currently truncated"
                  by that label's own definition, not just its start).
              Both sources are pooled into the same dataset, tagged with
              which one each sample came from (`source` field) so a later
              analysis can weight or filter them differently if the two
              turn out to disagree.

  analyze  -- reads the accumulated dataset and reports mean/std cutoff
              angle (alpha, primarily -- see below) binned by bearing
              angle, plus the overall mean.

CRITICAL: samples are computed directly from each detection's raw row/column
geometry (the exact same formula as DistanceEstimator.distanceMeters, just
solved for alpha/phi instead of distance) -- NEVER from the logged
`distanceMeters` field. That field can now be the WIDTH-based override's
value (see WidthDistanceOverride.swift) for exactly the near-cutoff
detections this tool cares about most, which would be circular: using an
already-corrected distance to re-derive the cutoff that correction depends
on.

ALPHA vs PHI -- CHANGED 2026-08-26, real request after the 26_08_26_Day_Walk
garage session settled at a real, non-trivial +0.926deg pitch after a mid-
session recalibration, meaningfully different from the 26_08_25_Day_Cones
session's near-0deg pitch. phi (angle below TRUE HORIZONTAL) bakes the
session's own referencePitchDegrees into the number; alpha (angle below the
camera's own optical axis) doesn't, and alpha is what's physically fixed
by the camera-to-hood relationship regardless of the car's momentary pitch.
Every sample now stores BOTH (plus the referencePitchDegrees it was
computed from, for full traceability), but `analyze` bins/reports alpha as
the primary, poolable-across-sessions quantity -- phi is kept per-sample
only for reference back to the original 9.899deg-mean scalar this dataset
is meant to eventually replace. Samples extracted before this date (the
original 2026-08-15 walkaround, 107 samples, OLD mount) have neither field
at all and are excluded from analysis rather than silently mixed in --
even setting the alpha/phi question aside, the mount has since moved, so
those samples reflect a physically different camera-to-hood relationship
anyway."""
import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_SAMPLES_PATH = Path(__file__).parent.parent / "calibration" / "hood_cutoff_samples.jsonl"
"""Moved from data/ to calibration/ 2026-08-26, same reasoning as
cone_calibration_points.json's own move (see cone_pinpoint_tool.py's doc
comment): this is small, hand-derived ground truth, not large regenerable
recording output, so it belongs in the directory that's actually tracked
by git instead of needing a --force add against /data/'s blanket
gitignore rule every time it grows."""

# DistanceEstimator.calibrated -- keep in sync with DistanceEstimator.swift.
CAMERA_HEIGHT_METERS = 1.02
PRINCIPAL_ROW_NORMALIZED = 0.481681
FOCAL_LENGTH_NORMALIZED = 1.322673
ASSUMED_PRINCIPAL_COLUMN_NORMALIZED = 0.5


def compute_alpha_phi_and_bearing(row: float, col: float, aspect_ratio: float, reference_pitch_deg: float, reference_roll_deg: float):
    """Same geometry as DistanceEstimator.distanceMeters, solved for alpha
    and phi (angle below the camera's own optical axis, and angle below
    TRUE HORIZONTAL, respectively) instead of distance -- and separately,
    the raw bearing angle (horizontal viewing angle from the camera's
    principal axis), which distanceMeters computes internally but doesn't
    expose.

    ALPHA, NOT PHI, IS THE PHYSICALLY-CORRECT QUANTITY TO CALIBRATE --
    CONFIRMED 2026-08-26. The hood is a rigid structure mounted to the same
    car as the camera, so wherever it blocks the view sits at a FIXED
    alpha (fixed image row, relative to the camera's own optical axis)
    regardless of the car's momentary pitch. phi = alpha + theta bakes the
    session's referencePitchDegrees (theta) into the reported number --
    fine as long as every session shares roughly the same pitch (true
    enough for the original 2026-08-15 walkaround, apparently NOT true in
    general: the 26_08_25_Day_Cones session sat near 0deg pitch, the
    26_08_26_Day_Walk garage session settled at +0.926deg after a real
    mid-session recalibration -- a difference comparable to this whole
    measurement's own claimed 0.771deg std dev). Reporting alpha instead
    makes samples from sessions at DIFFERENT pitches directly poolable,
    which phi alone can't give you -- phi is still returned too, purely
    for continuity with the original 9.899deg-mean scalar it's meant to
    eventually replace."""
    theta = math.radians(reference_pitch_deg)
    psi = math.radians(reference_roll_deg)
    focal_col = FOCAL_LENGTH_NORMALIZED / aspect_ratio
    x = (col - ASSUMED_PRINCIPAL_COLUMN_NORMALIZED) / focal_col
    y = (row - PRINCIPAL_ROW_NORMALIZED) / FOCAL_LENGTH_NORMALIZED
    derolled_y = -x * math.sin(psi) + y * math.cos(psi)
    alpha = math.atan(derolled_y)
    phi = alpha + theta
    bearing_deg = math.degrees(math.atan(x))
    return math.degrees(alpha), math.degrees(phi), bearing_deg


def parse_window(spec: str) -> tuple:
    start, end = spec.split(":")
    return float(start), float(end)


def load_detections(detections_path: Path) -> list:
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist -- run package_session.py on this session first.")
    entries = []
    with open(detections_path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def recording_start_epoch(debug_log_path: Path) -> float:
    if debug_log_path.exists():
        with open(debug_log_path) as f:
            for line in f:
                if "recording-start:" in line and "epoch=" in line:
                    return float(line.split("epoch=")[1].strip())
    sys.exit(f"Couldn't find a 'recording-start:' line in {debug_log_path} -- can't align window times.")


def cmd_extract(args) -> None:
    session_dir = Path(args.session_dir)
    session_name = session_dir.name
    # --detections/--debug-log override the plain "detections.jsonl"/
    # "overlay-debug.log" default -- needed for a multi-recording session
    # directory (package_session.py names per-video files <video>-
    # detections.jsonl/<video>-overlay-debug.log instead when more than
    # one recording shares a directory), same override pattern
    # reconstruct_annotated.py/super_tool.py already use.
    detections_path = args.detections or (session_dir / "detections.jsonl")
    debug_log_path = args.debug_log or (session_dir / "overlay-debug.log")
    start_epoch = recording_start_epoch(debug_log_path)
    entries = load_detections(detections_path)

    windows = []  # (start_t, end_t, source, track_filter_or_None, label_filter_or_None)
    for spec in args.tethered or []:
        start, end = parse_window(spec)
        # CONFIRMED bug 2026-08-15: a tethered window only confirms ONE
        # specific real-world point (the tester) was below cutoff -- every
        # OTHER thing detected in the same frames (background parked cars,
        # passing traffic) is not, and including them unfiltered pulled in
        # ~9x too many samples with wildly wrong/noisy phi values. Filtered
        # to args.tethered_label (default "person") to fix.
        windows.append((start, end, "tethered", None, args.tethered_label))

    truncation_path = session_dir / "ground_truth_hood_truncation.json"
    if truncation_path.exists():
        segments = json.loads(truncation_path.read_text())
        for seg in segments:
            end = seg["end_t"] if seg["end_t"] is not None else float("inf")
            windows.append((seg["start_t"], end, "labeled_onset", seg["trackID"], None))
        print(f"Found {len(segments)} hood_truncation segment(s) in {truncation_path.name}")

    if not windows:
        sys.exit(
            "No sample windows: pass --tethered start:end [...] and/or make sure "
            f"{truncation_path.name} exists in {session_dir}."
        )

    existing = set()
    if args.output.exists():
        with open(args.output) as f:
            for line in f:
                s = json.loads(line)
                existing.add((s["session"], s["trackID"], s["t"]))

    new_samples = []
    for e in entries:
        vt = e["t"] - start_epoch
        aspect_ratio_parts = e["resolution"].split("x")
        aspect_ratio = int(aspect_ratio_parts[0]) / int(aspect_ratio_parts[1])
        for d in e.get("detections", []):
            track_id = d.get("trackID")
            if track_id is None:
                continue
            for start, end, source, track_filter, label_filter in windows:
                if not (start <= vt <= end):
                    continue
                if track_filter is not None and track_id != track_filter:
                    continue
                if label_filter is not None and d["label"] != label_filter:
                    continue
                key = (session_name, track_id, vt)
                if key in existing:
                    continue
                row = d["y"] + d["h"]
                col = d["x"] + d["w"] / 2
                reference_pitch_deg = e.get("referencePitchDegrees") or 0.0
                alpha_deg, phi_deg, bearing_deg = compute_alpha_phi_and_bearing(
                    row, col, aspect_ratio,
                    reference_pitch_deg,
                    e.get("referenceRollDegrees") or 0.0,
                )
                new_samples.append({
                    "session": session_name,
                    "source": source,
                    "trackID": track_id,
                    "label": d["label"],
                    "t": vt,
                    "confidence": d["confidence"],
                    "col": col,
                    "bottomY": row,
                    "bearingDegrees": bearing_deg,
                    "alphaDegrees": alpha_deg,
                    "phiDegrees": phi_deg,
                    "referencePitchDegrees": reference_pitch_deg,
                })
                existing.add(key)
                break  # don't double-count a detection matched by >1 window

    if not new_samples:
        print("No new samples (already extracted, or nothing in range).")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as f:
        for s in new_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Appended {len(new_samples)} new samples -> {args.output}")


def cmd_analyze(args) -> None:
    if not args.samples_file.exists():
        sys.exit(f"{args.samples_file} doesn't exist -- run 'extract' on at least one session first.")

    all_samples = []
    with open(args.samples_file) as f:
        for line in f:
            all_samples.append(json.loads(line))

    if not all_samples:
        sys.exit(f"{args.samples_file} is empty.")

    # Samples extracted before 2026-08-26 have no alphaDegrees/
    # referencePitchDegrees fields at all (the original 107-sample
    # 2026-08-15 walkaround study, from the OLD mount) -- can't be
    # retroactively converted without the raw session data, and shouldn't
    # be pooled with new-mount samples even if they could (the mount has
    # since moved, so alpha itself -- a fixed property of THIS camera-to-
    # hood relationship -- is a physically different number now). Excluded
    # from analysis, not silently dropped.
    old_schema = [s for s in all_samples if "alphaDegrees" not in s]
    samples = [s for s in all_samples if "alphaDegrees" in s]
    if old_schema:
        print(
            f"NOTE: {len(old_schema)} sample(s) predate alpha-based tracking "
            f"(pre-2026-08-26, OLD mount) -- excluded from this analysis, not deleted.\n"
        )
    if not samples:
        sys.exit(
            f"No alpha-tracked samples in {args.samples_file} -- only the old-schema, "
            "OLD-mount ones above. Run 'extract' on a session recorded on the current mount first."
        )

    bearings = [s["bearingDegrees"] for s in samples]
    lo, hi = min(bearings), max(bearings)
    bin_width = (hi - lo) / args.bins if hi > lo else 1.0

    def bin_index(b):
        if bin_width == 0:
            return 0
        idx = int((b - lo) / bin_width)
        return min(idx, args.bins - 1)

    buckets = {}
    for s in samples:
        buckets.setdefault(bin_index(s["bearingDegrees"]), []).append(s)

    def mean(xs): return sum(xs) / len(xs)
    def stdev(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0

    pitches = [s["referencePitchDegrees"] for s in samples]
    print(f"{len(samples)} total samples from {len(set(s['session'] for s in samples))} session(s) "
          f"({sum(1 for s in samples if s['source'] == 'tethered')} tethered, "
          f"{sum(1 for s in samples if s['source'] == 'labeled_onset')} labeled_onset)")
    print(f"referencePitchDegrees across pooled sessions: {min(pitches):.3f} to {max(pitches):.3f} "
          f"(spread of {max(pitches) - min(pitches):.3f} deg -- this is exactly the pitch-dependence "
          f"alpha, not phi, is supposed to make irrelevant; a wide spread here is fine, not a warning "
          f"sign, as long as it's alpha being binned below)\n")
    print(f"{'bearing bin':>16}  {'n':>5}  {'alpha mean':>10}  {'alpha std':>9}  {'phi mean':>9}")
    for i in range(args.bins):
        bucket = buckets.get(i, [])
        bin_lo = lo + i * bin_width
        bin_hi = bin_lo + bin_width
        if len(bucket) < 3:
            print(f"{bin_lo:6.1f} to {bin_hi:5.1f}     n={len(bucket)} (too few)")
            continue
        alphas = [s["alphaDegrees"] for s in bucket]
        phis = [s["phiDegrees"] for s in bucket]
        print(f"{bin_lo:6.1f} to {bin_hi:5.1f}    {len(bucket):>5}  {mean(alphas):>10.3f}  {stdev(alphas):>9.3f}  {mean(phis):>9.3f}")

    all_alphas = [s["alphaDegrees"] for s in samples]
    all_phis = [s["phiDegrees"] for s in samples]
    print(f"\nOVERALL: n={len(samples)}  alpha mean={mean(all_alphas):.3f}  alpha std={stdev(all_alphas):.3f}"
          f"  (phi mean={mean(all_phis):.3f}, for reference against the old 9.899deg scalar only)")
    print("(alpha -- angle below the camera's own optical axis, NOT phi -- is the pitch-independent "
          "quantity that's actually poolable across sessions recorded at different pitches; convert "
          "back to phi for a specific session via phi = alpha + that session's referencePitchDegrees "
          "when DistanceEstimator.hoodCutoffAngleDegrees actually needs a phi-style threshold)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Pull hood-clipped samples from one session into the shared dataset")
    p_extract.add_argument("session_dir", type=Path)
    p_extract.add_argument(
        "--tethered", nargs="+", metavar="start:end",
        help="Video-relative time window(s), confirmed below any plausible cutoff distance",
    )
    p_extract.add_argument(
        "--tethered-label", default="person",
        help="Class label the tethered tester's own detections carry -- restricts --tethered "
             "windows to just that class, so unrelated background objects in the same frames "
             "(parked cars, passing traffic) aren't wrongly included as hood-clipped (default: person)",
    )
    p_extract.add_argument("--output", type=Path, default=DEFAULT_SAMPLES_PATH)
    p_extract.add_argument(
        "--detections", type=Path, default=None,
        help="Defaults to <session_dir>/detections.jsonl -- override for a multi-recording "
             "directory, where package_session.py names it <video>-detections.jsonl instead.",
    )
    p_extract.add_argument(
        "--debug-log", type=Path, default=None,
        help="Defaults to <session_dir>/overlay-debug.log -- same multi-recording override as --detections.",
    )
    p_extract.set_defaults(func=cmd_extract)

    p_analyze = sub.add_parser("analyze", help="Report cutoff angle binned by bearing angle")
    p_analyze.add_argument("samples_file", type=Path, nargs="?", default=DEFAULT_SAMPLES_PATH)
    p_analyze.add_argument("--bins", type=int, default=7)
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
