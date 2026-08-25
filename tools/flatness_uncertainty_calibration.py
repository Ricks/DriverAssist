#!/usr/bin/env python3
"""
Estimates sigma_pitch_flatness / sigma_roll_flatness -- the prior
uncertainty distance_fusion.py's planned flatness-uncertainty term needs --
from ORDINARY DRIVING FOOTAGE already on disk, no cone setup required.

Real request 2026-08-25: the 26_08_25_Day_Cones test showed the session's
stored referencePitchDegrees/referenceRollDegrees can be off by ~1.8 degrees
of roll from the true local ground plane (car parked on a crowned lot), and
we want to know how BIG and how COMMON that kind of deviation is across many
real locations -- but repeating a dedicated cone test at 5-10 more locations
is a lot of setup for Rick to redo. This script gets the same signal for
free from footage already recorded, using a trick: widthDistanceMeters
(already logged per detection, from box width + KITTI class-width priors)
doesn't depend on pitch/roll AT ALL, while row_based_distance_meters does --
so for any ordinary, reasonably head-on car detection, the discrepancy
between the two is exactly the same signature the cone test's column-
dependent bias was, just noisier per-sample (real car width variance,
~10cm std, vs a cone's known-exact position) and available in bulk from
whatever's already been recorded.

Per session: fits (pitch, roll) that minimize the relative residual between
row_based_distance_meters (evaluated at trial pitch/roll) and each
qualifying detection's own widthDistanceMeters, across every usable car
detection in that session -- the exact same 2-parameter joint least-squares
approach used on the cone data, just with widthDistanceMeters standing in
for a tape measurement at each point instead of one shared true distance.
Compares the fit against that session's own stored reference pitch/roll to
get one (delta_pitch, delta_roll) sample per session -- aggregating those
samples across many real sessions is what actually estimates
sigma_pitch_flatness/sigma_roll_flatness, the same way the cone test gave
exactly one such sample from one parking lot.

Gating (mirrors corrected_distance_meters'/is_oblique_view's existing
logic, but stricter: excludes anything that could corrupt row_dist OR
width_dist, not just one with a fallback to the other):
  - label == "car" only (best-populated, most accurate KITTI width prior)
  - not edge-truncated (EDGE_TRUNCATION_MARGIN_NORMALIZED on either side)
  - not hood-truncated, with a safety margin below HOOD_CUTOFF_ANGLE_DEGREES
    (not just excluding the geometrically-impossible zone, but staying
    clear of its edge where the hood already partially encroaches)
  - not oblique (is_oblique_view) -- widthDistanceMeters assumes a
    head-on/tail-on view
  - confidence >= MIN_CONFIDENCE
  - widthDistanceMeters in a sane range (too close: hood-cutoff-adjacent
    effects some other gate might miss; too far: box width in pixels gets
    small enough that widthDistanceMeters itself gets noisy)

A session only gets a reported (delta_pitch, delta_roll) if it clears
MIN_DETECTIONS and MIN_COLUMN_SPREAD -- mirroring why the cone test spread
points from column 0.06 to 0.98 rather than clustering them: pitch and roll
aren't separable from narrow-column data (see the 2026-08-25 chat: fitting
the ORIGINAL 8-point near/far cone set, which sat close to the centerline,
left roll basically unconstrained even though pitch fit tightly).

Usage:
    python3 flatness_uncertainty_calibration.py [--data-dir ../data] [--min-detections 30] [--min-column-spread 0.3]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from reconstruct_annotated import (
    EDGE_TRUNCATION_MARGIN_NORMALIZED,
    HOOD_CUTOFF_ANGLE_DEGREES,
    _row_based_phi_degrees,
    is_oblique_view,
    row_based_distance_meters,
)

ASSUMED_ASPECT = 1920 / 1080  # every session recorded so far is 16:9, regardless of exact pixel count
MIN_CONFIDENCE = 0.5
HOOD_CUTOFF_SAFETY_MARGIN_DEGREES = 2.0  # stay clear of the hood-cutoff edge, not just past it
WIDTH_DISTANCE_MIN_METERS = 4.0
WIDTH_DISTANCE_MAX_METERS = 45.0


def qualifying_points(entries: list) -> list:
    """[(bottom_y, center_x, width_dist), ...] for every detection in this
    session trustworthy enough for BOTH row_dist and width_dist to be
    meaningful at once."""
    points = []
    for entry in entries:
        aspect = ASSUMED_ASPECT
        for det in entry.get("detections", []):
            if det.get("label") != "car":
                continue
            if det.get("confidence", 0.0) < MIN_CONFIDENCE:
                continue
            width_dist = det.get("widthDistanceMeters")
            if width_dist is None or not (WIDTH_DISTANCE_MIN_METERS <= width_dist <= WIDTH_DISTANCE_MAX_METERS):
                continue
            left, right = det["x"], det["x"] + det["w"]
            if left <= EDGE_TRUNCATION_MARGIN_NORMALIZED or right >= 1.0 - EDGE_TRUNCATION_MARGIN_NORMALIZED:
                continue
            bottom_y = det["y"] + det["h"]
            center_x = det["x"] + det["w"] / 2
            phi_deg = _row_based_phi_degrees(
                bottom_y, center_x, aspect,
                entry.get("referencePitchDegrees", 0.0), entry.get("referenceRollDegrees", 0.0),
            )
            if phi_deg > HOOD_CUTOFF_ANGLE_DEGREES - HOOD_CUTOFF_SAFETY_MARGIN_DEGREES:
                continue
            if is_oblique_view(det, entry, aspect):
                continue
            points.append((bottom_y, center_x, width_dist))
    return points


def fit_pitch_roll(points: list) -> tuple:
    """Joint (pitch, roll) that minimizes the relative residual between
    row_based_distance_meters and each point's own widthDistanceMeters --
    same Nelder-Mead approach used on the cone data, width_dist standing in
    for a tape measurement at each point."""
    def loss(params):
        pitch, roll = params
        total = 0.0
        for bottom_y, center_x, width_dist in points:
            dist = row_based_distance_meters(
                bottom_y=bottom_y, center_x=center_x, aspect=ASSUMED_ASPECT,
                reference_pitch_deg=pitch, reference_roll_deg=roll,
            )
            if dist is None:
                total += 4.0  # a relative-residual scale penalty, not an arbitrary large constant
                continue
            rel = (dist - width_dist) / width_dist
            total += rel * rel
        return total

    res = minimize(loss, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 5000})
    return tuple(res.x)


def relative_rms(points: list, pitch: float, roll: float) -> float:
    residuals = []
    for bottom_y, center_x, width_dist in points:
        dist = row_based_distance_meters(
            bottom_y=bottom_y, center_x=center_x, aspect=ASSUMED_ASPECT,
            reference_pitch_deg=pitch, reference_roll_deg=roll,
        )
        if dist is not None:
            residuals.append((dist - width_dist) / width_dist)
    if not residuals:
        return float("nan")
    return (sum(r * r for r in residuals) / len(residuals)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--min-detections", type=int, default=30)
    parser.add_argument("--min-column-spread", type=float, default=0.3)
    args = parser.parse_args()

    detection_files = sorted(args.data_dir.glob("*/*detections.jsonl"))
    # Skip dedicated calibration sessions -- those are handled by the real
    # cone/tape-measure fit directly (exact ground truth beats a width-prior
    # proxy), not this passive-mining approach.
    skip_name_fragments = ("Calibration", "Cone")
    detection_files = [
        f for f in detection_files if not any(frag in f.parent.name for frag in skip_name_fragments)
    ]

    print(f"Found {len(detection_files)} session detections.jsonl file(s) to scan.\n")

    deltas_pitch, deltas_roll = [], []
    for path in detection_files:
        session = path.parent.name if path.name == "detections.jsonl" else f"{path.parent.name}/{path.stem}"
        try:
            entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except Exception as e:
            print(f"{session:<55} SKIP (couldn't read: {e})")
            continue

        points = qualifying_points(entries)
        if len(points) < args.min_detections:
            print(f"{session:<55} SKIP ({len(points)} qualifying detections, need {args.min_detections})")
            continue

        columns = [p[1] for p in points]
        spread = max(columns) - min(columns)
        if spread < args.min_column_spread:
            print(f"{session:<55} SKIP (column spread {spread:.2f}, need {args.min_column_spread:.2f}; "
                  f"{len(points)} detections)")
            continue

        reference_pitch = next((e.get("referencePitchDegrees") for e in entries if e.get("referencePitchDegrees") is not None), 0.0)
        reference_roll = next((e.get("referenceRollDegrees") for e in entries if e.get("referenceRollDegrees") is not None), 0.0)

        rms_before = relative_rms(points, reference_pitch, reference_roll)
        pitch_fit, roll_fit = fit_pitch_roll(points)
        rms_after = relative_rms(points, pitch_fit, roll_fit)

        delta_pitch = pitch_fit - reference_pitch
        delta_roll = roll_fit - reference_roll
        deltas_pitch.append(delta_pitch)
        deltas_roll.append(delta_roll)

        print(
            f"{session:<55} n={len(points):>5} spread={spread:.2f}  "
            f"rms {rms_before*100:5.1f}%->{rms_after*100:4.1f}%  "
            f"d_pitch={delta_pitch:+6.2f} deg  d_roll={delta_roll:+6.2f} deg"
        )

    print()
    if len(deltas_pitch) >= 2:
        print(f"--- Aggregate over {len(deltas_pitch)} usable sessions ---")
        print(f"delta_pitch: mean={statistics.mean(deltas_pitch):+.3f} deg  "
              f"std={statistics.stdev(deltas_pitch):.3f} deg")
        print(f"delta_roll:  mean={statistics.mean(deltas_roll):+.3f} deg  "
              f"std={statistics.stdev(deltas_roll):.3f} deg")
    else:
        print(f"Only {len(deltas_pitch)} usable session(s) -- not enough to estimate a standard deviation yet.")


if __name__ == "__main__":
    main()
