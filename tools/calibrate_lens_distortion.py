#!/usr/bin/env python3
"""Computes real lens intrinsics (fx, fy, cx, cy) and distortion coefficients
(k1, k2, p1, p2, k3) from a video of a checkerboard pattern, via OpenCV's
standard cv2.calibrateCamera.

WHY THIS EXISTS: the entire distance system (DistanceEstimator.swift /
reconstruct_annotated.py's row_based_distance_meters) assumes an IDEAL
pinhole camera -- straight lines project to straight lines, no lens bending.
Real phone lenses don't quite do that, especially toward the frame edges.
CONFIRMED 2026-08-24 via a real tethered walkaround test (person held at a
constant true 3D distance from the camera while walking side to side): even
after fixing row_based_distance_meters' lateral-angle bug (returning true
radial distance instead of forward depth only), a real ~20m-radius sweep
still showed a reproducible, REVERSIBLE ~25-27% distance understatement at
the frame's far edge (same screen position -> same wrong distance, both
outbound and on the way back -- ruling out a sloppy human path). Already
ruled out before landing here: a math bug in the ray-ground-intersection
formula (proven exact against a synthetic 3D projection model), pitch/roll
drift during the walk (both were bit-for-bit constant), and a wrong assumed
principal column (swept 0.50-0.80, zero effect on the error). What's left is
exactly the failure mode an ideal-pinhole model can't represent: radial lens
distortion, worst at the edges -- consistent with the observed signature.

CALIBRATION PROTOCOL (see this repo's own chat/notes for the full writeup):
record a checkerboard with the SAME app/resolution/stabilization settings as
a real drive, phone mounted and PHYSICALLY STILL (only the checkerboard
moves) so stabilization has nothing to compensate for and doesn't introduce
frame-to-frame intrinsic drift. Cover the checkerboard across the whole
frame -- center, all four edges, all four corners (especially the edges,
where this bug actually lives) -- at a few different distances and tilt
angles. A phone held/moved by hand while the mount stays fixed is fine.

USAGE:
    python3 tools/calibrate_lens_distortion.py checkerboard_video.mp4 \\
        --square-size-mm 20 --pattern-cols 9 --pattern-rows 6

Output: calibration/lens_distortion.json (camera matrix + distortion
coefficients + the resolution they're valid for -- reusing them at a
DIFFERENT recording resolution/FOV than they were fit at is not valid).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def find_corners(frame: np.ndarray, pattern_size: tuple):
    """Detects checkerboard internal corners in one frame, sub-pixel refined.
    Tries the newer, more robust findChessboardCornersSB first (handles
    varied lighting/blur better); falls back to the classic detector +
    manual cornerSubPix refinement, which SB does internally already, for
    OpenCV builds where SB isn't available or fails on a marginal frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if found:
        return True, corners
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording of the checkerboard pattern")
    parser.add_argument(
        "--pattern-cols", type=int, default=9,
        help="Internal corners across the board's LONG side (default 9 -- matches the "
             "10x7-square pattern this tool's own checkerboard SVG generates)",
    )
    parser.add_argument(
        "--pattern-rows", type=int, default=6,
        help="Internal corners across the board's SHORT side (default 6)",
    )
    parser.add_argument(
        "--square-size-mm", type=float, required=True,
        help="Real, ruler-measured size of one printed square's edge, in mm -- NOT the "
             "nominal size you told the printer to use. Printer scaling error is a common "
             "calibration pitfall; measure the actual printout.",
    )
    parser.add_argument(
        "--sample-every", type=int, default=5,
        help="Only attempt corner detection on every Nth frame (default 5) -- consecutive "
             "video frames are nearly identical, so this skips redundant detection work "
             "without losing coverage.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=40,
        help="Stop once this many GOOD (corners found) frames are collected (default 40) "
             "-- more helps precision but with diminishing returns past ~30-40 well-spread "
             "detections.",
    )
    parser.add_argument("--output", type=Path, default=Path("calibration/lens_distortion.json"))
    parser.add_argument(
        "--debug-dir", type=Path, default=None,
        help="Save each accepted frame with its detected corners drawn on top, here -- a "
             "visual sanity check that detections are landing on the real board, not a "
             "false-positive pattern elsewhere in the scene.",
    )
    args = parser.parse_args()

    pattern_size = (args.pattern_cols, args.pattern_rows)
    # Real-world (mm) coordinates of each internal corner, board-local, Z=0
    # (the board is planar by construction) -- OpenCV solves for the board's
    # pose per frame AND the camera's fixed intrinsics/distortion jointly
    # across every frame's correspondences.
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * args.square_size_mm

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source video: {width}x{height}")

    objpoints, imgpoints = [], []
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    good = 0
    while good < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_every == 0:
            found, corners = find_corners(frame, pattern_size)
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                good += 1
                print(f"  frame {frame_idx}: found ({good}/{args.max_frames})")
                if args.debug_dir:
                    vis = frame.copy()
                    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                    cv2.imwrite(str(args.debug_dir / f"frame_{frame_idx:05d}.jpg"), vis)
        frame_idx += 1
    cap.release()

    if good < 10:
        sys.exit(
            f"Only found the checkerboard in {good} frame(s) -- need at least 10-15 spread "
            f"across the frame for a reliable fit. Check --debug-dir output if unsure "
            f"whether detection is working at all; otherwise capture more coverage, "
            f"especially near the frame edges/corners (that's the region this calibration "
            f"is actually for)."
        )

    print(f"Calibrating from {good} accepted frames...")
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"RMS reprojection error: {rms:.4f} px (well under 1px is good; over ~1px, "
          f"suspect a bad square-size measurement or blurry/motion-smeared frames)")
    print(f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"distortion coefficients (k1,k2,p1,p2,k3): {dist_coeffs.ravel()}")

    result = {
        "sourceVideo": str(args.video),
        "sourceWidth": width,
        "sourceHeight": height,
        "patternCols": args.pattern_cols,
        "patternRows": args.pattern_rows,
        "squareSizeMm": args.square_size_mm,
        "framesUsed": good,
        "rmsReprojectionErrorPx": rms,
        "cameraMatrix": camera_matrix.tolist(),
        "distortionCoefficients": dist_coeffs.ravel().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
