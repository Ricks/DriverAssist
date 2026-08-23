#!/usr/bin/env python3
"""
Estimates the camera's yaw misalignment relative to the vehicle's true
forward direction, from real driving footage -- the piece the following-
distance/optic-flow design discussion flagged as still missing: the
wiper-marker calibration confirms the mount is "roughly right," but there's
no measured ANGLE for how many degrees the camera's boresight is actually
off from the vehicle's own forward direction.

Method: during a stretch of confirmed-straight driving (yaw rate near zero,
sustained, at real speed -- see find_straight_windows), the background's
optic flow radiates outward from a single point, the Focus of Expansion
(FOE), which for pure translation is exactly the camera's direction of
travel projected into the image. If the camera is perfectly aligned with
the vehicle, the FOE sits on the calibrated principal point
(PRINCIPAL_COLUMN_NORMALIZED/PRINCIPAL_ROW_NORMALIZED, mirroring
DistanceEstimator.swift's `calibrated` instance). Any horizontal offset
between the two, converted through the calibrated focal length, IS the
yaw misalignment angle -- no new hardware or calibration screen needed,
just real straight-line driving footage already being recorded anyway.

Sparse optical flow (goodFeaturesToTrack + calcOpticalFlowPyrLK) mirrors
tools/gmc.py's own approach -- deliberately re-implemented here rather than
importing GMC directly, since this needs the raw matched point pairs GMC's
own apply() computes internally but doesn't expose (it only returns the
final fitted transform).

The FOE itself is fit via RANSAC line-intersection: each flow vector, drawn
as a line through its own start point, should pass through the FOE under
pure translation -- moving objects (other traffic) violate this and are
exactly the outliers RANSAC is for. Near-zero-magnitude flow vectors
(parked-relative-to-camera points -- the ego hood, or genuine noise) are
dropped first since a zero-length vector has no defined direction and can't
usefully constrain the fit.

Usage:
    python3 tools/camera_yaw_alignment.py <session_dir_or_video> [--detections ...] [--debug-log ...]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch

# Calibrated intrinsics -- mirrors DistanceEstimator.swift's `calibrated`
# instance exactly (focalLengthNormalized/principalRowNormalized are FIT
# values, assumedPrincipalColumnNormalized is a stated assumption -- see
# that struct's own doc comments). Row-normalized: fraction of frame HEIGHT
# for principalRow, fraction of frame WIDTH for principalColumn.
FOCAL_LENGTH_NORMALIZED = 1.322673
PRINCIPAL_ROW_NORMALIZED = 0.481681
PRINCIPAL_COLUMN_NORMALIZED = 0.5

# "Confirmed straight": smoothed yaw rate under this, sustained, at real
# speed -- see find_straight_windows. Deliberately tight (the whole point
# is isolating segments where rotation contributes ~nothing to the flow
# field, not just "not obviously turning").
DEFAULT_MAX_YAW_RATE_DEG_S = 1.5
DEFAULT_MIN_SPEED_MPS = 5.0
DEFAULT_MIN_WINDOW_S = 2.0

# A full drive can easily contain 100+ qualifying windows -- each one costs a
# frame-by-frame cv2 seek+read+LK pass, so processing all of them turns a
# validation run into an hours-long one for no real accuracy benefit past a
# double-digit sample. Longest-first because a longer window pools more flow
# correspondences per RANSAC fit (steadier single-window estimate), not
# because long windows are otherwise more representative.
DEFAULT_MAX_WINDOWS = 15

DEFAULT_MAX_CORNERS = 300
# Flow vectors shorter than this are dropped before fitting -- both a
# noise floor and what excludes ego-hood points (which move with the
# camera, near-zero flow) without needing a separate hood-cutoff mask.
DEFAULT_MIN_FLOW_MAG_PX = 1.5
DEFAULT_RANSAC_THRESHOLD_PX = 3.0
DEFAULT_RANSAC_ITERS = 500


def find_straight_windows(
    entries: list, max_yaw_rate: float, min_speed: float, min_window_s: float,
) -> list:
    """Contiguous index ranges [start, end] (inclusive) into `entries` where
    smoothedYawRateDegreesPerSecond stays within +-max_yaw_rate and
    egoSpeedMps stays at or above min_speed, for at least min_window_s.
    Both fields nil (no reading yet) counts as NOT qualifying, same as out
    of range -- silently treating "no data" as "straight and fast" would be
    exactly backwards for a check whose whole point is confidence."""
    windows = []
    start = None
    for i, e in enumerate(entries):
        yaw = e.get("smoothedYawRateDegreesPerSecond")
        speed = e.get("egoSpeedMps")
        ok = yaw is not None and abs(yaw) <= max_yaw_rate and speed is not None and speed >= min_speed
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if entries[i - 1]["t"] - entries[start]["t"] >= min_window_s:
                windows.append((start, i - 1))
            start = None
    if start is not None and entries[-1]["t"] - entries[start]["t"] >= min_window_s:
        windows.append((start, len(entries) - 1))
    return windows


def sparse_flow_correspondences(
    video: Path, entries: list, window: tuple, video_start_epoch: float, max_corners: int,
) -> tuple:
    """cv2 goodFeaturesToTrack + calcOpticalFlowPyrLK between consecutive
    LOGGED-frame timestamps within `window` (not every raw video frame --
    matching the app's own ~15fps logging cadence is plenty dense for this
    and keeps this fast), pooling all matched point pairs across the whole
    window into one (prev_points, curr_points) pair. Seeked to each entry's
    estimated *capture* time (logged completion time minus elapsedMs), same
    convention as every other offline tool in this project that needs to
    match a video frame to a detections.jsonl entry."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {video}")

    prev_gray = None
    prev_pts = None
    all_prev, all_curr = [], []
    for i in range(window[0], window[1] + 1):
        e = entries[i]
        capture_time = e["t"] - e["elapsedMs"] / 1000.0
        offset_ms = (capture_time - video_start_epoch) * 1000.0
        if offset_ms < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and prev_pts is not None and len(prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            status = status.reshape(-1).astype(bool)
            if status.any():
                all_prev.append(prev_pts[status].reshape(-1, 2))
                all_curr.append(curr_pts[status].reshape(-1, 2))

        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)

    cap.release()
    if not all_prev:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.concatenate(all_prev), np.concatenate(all_curr)


def fit_foe(
    prev_pts: np.ndarray, curr_pts: np.ndarray, min_flow_mag: float,
    ransac_threshold: float, ransac_iters: int, rng: np.random.Generator,
):
    """RANSAC line-intersection fit for the Focus of Expansion. Each flow
    vector (curr - prev), as a line through `prev`, should pass through the
    true FOE under pure translation -- solved via the standard normal-form
    least-squares "point closest to a set of lines" formulation: for a line
    through point p with unit direction d, n = perpendicular(d) gives
    n . (F - p) = 0, i.e. n . F = n . p, one linear equation in F = (fx,
    fy) per line. Stacking all of them and solving via least-squares (2
    points is the minimal RANSAC sample) gives the point minimizing total
    squared perpendicular distance to every line.

    Returns (foe, inlier_count, total_count), or None if there aren't
    enough usable (non-degenerate) flow vectors to fit at all.
    """
    if len(prev_pts) < 2:
        return None
    flow = curr_pts - prev_pts
    mag = np.linalg.norm(flow, axis=1)
    keep = mag >= min_flow_mag
    prev_pts, flow = prev_pts[keep], flow[keep]
    n = len(prev_pts)
    if n < 2:
        return None

    dirs = flow / np.linalg.norm(flow, axis=1, keepdims=True)
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    b = np.sum(normals * prev_pts, axis=1)

    def solve(idx):
        sol, *_ = np.linalg.lstsq(normals[idx], b[idx], rcond=None)
        return sol

    best_inliers = None
    best_count = -1
    for _ in range(ransac_iters):
        idx = rng.choice(n, size=2, replace=False)
        try:
            F = solve(idx)
        except np.linalg.LinAlgError:
            continue
        residual = np.abs(normals @ F - b)
        inliers = residual <= ransac_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        return None
    F_final = solve(np.where(best_inliers)[0])
    return F_final, int(best_inliers.sum()), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to analyze")
    parser.add_argument("--detections", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--debug-log", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=DEFAULT_MAX_YAW_RATE_DEG_S)
    parser.add_argument("--min-speed-mps", type=float, default=DEFAULT_MIN_SPEED_MPS)
    parser.add_argument("--min-window-s", type=float, default=DEFAULT_MIN_WINDOW_S)
    parser.add_argument(
        "--max-windows", type=int, default=DEFAULT_MAX_WINDOWS,
        help=f"Cap on how many straight-driving windows to process, longest first, 0 for no cap (default: {DEFAULT_MAX_WINDOWS})",
    )
    parser.add_argument("--max-corners", type=int, default=DEFAULT_MAX_CORNERS)
    parser.add_argument("--min-flow-mag-px", type=float, default=DEFAULT_MIN_FLOW_MAG_PX)
    parser.add_argument("--ransac-threshold-px", type=float, default=DEFAULT_RANSAC_THRESHOLD_PX)
    parser.add_argument("--ransac-iters", type=int, default=DEFAULT_RANSAC_ITERS)
    parser.add_argument("--seed", type=int, default=0, help="RANSAC's own RNG seed, for reproducible runs")
    args = parser.parse_args()

    start_epoch, _ = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)
    detections = [e for e in detections if e["t"] >= start_epoch]
    if not detections:
        sys.exit(f"No detections at or after {args.video.name}'s start time were found in {args.detections}.")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {args.video}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    aspect_ratio = frame_width / frame_height
    focal_length_column_normalized = FOCAL_LENGTH_NORMALIZED / aspect_ratio
    principal_x_px = PRINCIPAL_COLUMN_NORMALIZED * frame_width
    principal_y_px = PRINCIPAL_ROW_NORMALIZED * frame_height

    windows = find_straight_windows(detections, args.max_yaw_rate_deg_s, args.min_speed_mps, args.min_window_s)
    if not windows:
        # CONFIRMED gap 2026-08-21: an older session (predating
        # smoothedYawRateDegreesPerSecond's addition) silently produces zero
        # windows -- every entry's yaw field is just missing, indistinguishable
        # from "never drives straight" without actually checking. Diagnose it
        # here rather than making that a repeat of the same manual dig.
        n_missing_yaw = sum(1 for e in detections if e.get("smoothedYawRateDegreesPerSecond") is None)
        if n_missing_yaw == len(detections):
            sys.exit(
                f"No entry in this session has smoothedYawRateDegreesPerSecond -- this recording "
                "predates that field being logged (see PitchSensor.swift). This tool can't run "
                "against it; pick a session recorded after smoothedYawRateDegreesPerSecond was added."
            )
        sys.exit(
            f"No confirmed-straight windows found (yaw rate <= {args.max_yaw_rate_deg_s} deg/s, "
            f"speed >= {args.min_speed_mps} m/s, sustained >= {args.min_window_s}s). "
            "Nothing to estimate the FOE from -- try loosening the thresholds, or use a session "
            "with more sustained straight-line highway/road driving."
        )
    total_found = len(windows)
    if args.max_windows > 0 and total_found > args.max_windows:
        windows = sorted(windows, key=lambda w: detections[w[1]]["t"] - detections[w[0]]["t"], reverse=True)[: args.max_windows]
        windows.sort()
        print(f"Found {total_found} confirmed-straight window(s); processing the {len(windows)} longest:")
    else:
        print(f"Found {total_found} confirmed-straight window(s):")

    rng = np.random.default_rng(args.seed)
    window_results = []
    for w in windows:
        t0, t1 = detections[w[0]]["t"], detections[w[1]]["t"]
        prev_pts, curr_pts = sparse_flow_correspondences(
            args.video, detections, w, start_epoch, args.max_corners
        )
        fit = fit_foe(prev_pts, curr_pts, args.min_flow_mag_px, args.ransac_threshold_px, args.ransac_iters, rng)
        if fit is None:
            print(f"  [{t0 - start_epoch:7.1f}s - {t1 - start_epoch:7.1f}s] ({t1 - t0:5.1f}s) -- FAILED, not enough usable flow")
            continue
        (fx, fy), inliers, total = fit
        yaw_offset_deg = np.degrees(np.arctan((fx - principal_x_px) / (focal_length_column_normalized * frame_width)))
        window_results.append({"t0": t0, "t1": t1, "fx": fx, "fy": fy, "inliers": inliers, "total": total, "yaw_offset_deg": yaw_offset_deg})
        print(
            f"  [{t0 - start_epoch:7.1f}s - {t1 - start_epoch:7.1f}s] ({t1 - t0:5.1f}s) "
            f"FOE=({fx:7.1f}, {fy:7.1f})px  inliers={inliers:4d}/{total:4d}  yaw_offset={yaw_offset_deg:+.3f} deg"
        )

    if not window_results:
        sys.exit("Every confirmed-straight window failed to produce a usable FOE fit -- nothing to report.")

    weights = np.array([r["inliers"] for r in window_results], dtype=float)
    offsets = np.array([r["yaw_offset_deg"] for r in window_results])
    weighted_mean = float(np.average(offsets, weights=weights))
    median = float(np.median(offsets))
    std = float(np.std(offsets))

    print(f"\n{len(window_results)}/{len(windows)} windows produced a usable FOE fit.")
    print(f"Principal point: ({principal_x_px:.1f}, {principal_y_px:.1f})px of {frame_width}x{frame_height} "
          f"(calibrated PRINCIPAL_COLUMN_NORMALIZED={PRINCIPAL_COLUMN_NORMALIZED}, PRINCIPAL_ROW_NORMALIZED={PRINCIPAL_ROW_NORMALIZED})")
    print(f"\nCamera yaw misalignment estimate:")
    print(f"  inlier-weighted mean : {weighted_mean:+.3f} deg")
    print(f"  median across windows: {median:+.3f} deg")
    print(f"  spread (std dev)     : {std:.3f} deg")
    print(
        f"\nSign convention: positive means the FOE sits to the RIGHT of the "
        f"calibrated principal point -- the camera's boresight is pointed "
        f"slightly LEFT of the vehicle's true forward direction (background "
        f"appears to expand from a point right of center, same 'pan left "
        f"makes the world slide right' relationship used elsewhere in this "
        f"project, e.g. ByteTracker.yawFallbackTransform)."
    )
    if len(window_results) < 3 or std > 1.0:
        print(
            "\nCONFIDENCE WARNING: fewer than 3 windows and/or spread above 1 deg -- "
            "treat this as a rough first estimate, not a number to calibrate anything "
            "against yet. Run against more/longer straight-driving sessions before trusting it."
        )


if __name__ == "__main__":
    main()
