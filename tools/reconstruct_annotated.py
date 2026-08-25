#!/usr/bin/env python3
"""
Reconstructs an annotated copy of a clean DriverAssist recording by drawing
detection boxes + HUD text (from detections.jsonl, and optionally thermal
state/percent from the app's debug log) onto each frame. The source .mov is
never modified — this always writes a new file.

Usage:
    python3 reconstruct_annotated.py <clean.mov>

--detections and --debug-log both default to ~/DriverAssist/logs/ (the
directory tools/pull_logs.sh keeps filled with everything pulled off the
device), and both accept either a single file or a directory of them — so
the common case is just the video path, and the right data is found
automatically regardless of what the video's been renamed to. Override
either with an explicit file/directory, and --output, if needed:
    python3 reconstruct_annotated.py <clean.mov> \
        [--detections detections.jsonl] [--debug-log overlay-debug.log] \
        [--output annotated.mp4]

See driverassist_sync.py for how frame-to-timestamp sync and log matching
actually work.
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from driverassist_sync import (
    DEFAULT_LOGS_DIR,
    MODEL_DISPLAY_NAMES,
    build_key_index,
    load_detections,
    load_thermal,
    nearest_at_or_before,
    resolve_start_epoch,
)
from leading_vehicle import (
    DEFAULT_BAND_HALF_WIDTH,
    DEFAULT_CENTER_X,
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_GRACE_FRAMES,
    DEFAULT_MAX_ABS_VELOCITY,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MAX_BOTTOM_Y,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_SYMMETRY,
    DEFAULT_MIN_WIDTH,
    DEFAULT_THRESHOLD,
    LeadingVehicleLock,
    VelocityEstimator,
    classify_leading,
)
from tracker import DEFAULT_REID_MODEL, ByteTracker, build_reid_encoder
from tune_leading_vehicle import ground_truth_at

# BGR (OpenCV convention) approximations of the on-device box colors — these
# don't need to be pixel-identical to iOS's system colors, just visually
# distinct per class, matching OverlayStyle.color(for:) in the app.
BOX_COLORS_BGR = {
    "person": (0, 255, 255),      # yellow
    "bicycle": (255, 255, 0),     # cyan
    "motorcycle": (255, 255, 0),  # cyan
    "car": (0, 255, 0),           # green
    "bus": (0, 0, 255),           # red
    "truck": (0, 0, 255),         # red
}
DEFAULT_BOX_COLOR_BGR = (255, 255, 255)
HUD_DEFAULT_COLOR_BGR = (191, 191, 191)  # ~white @ 75% opacity, matches on-device HUD
HUD_YELLOW_BGR = (0, 215, 255)
HUD_RED_BGR = (0, 0, 255)

# Magenta -- matches label_leading_vehicle_frontend.html's own highlight
# color for a human-labeled followed vehicle, so this reads the same way in
# both tools. Reserved for ground truth specifically (see --ground-truth)
# now that there's a second, distinct tint for the algorithm's own pick --
# don't reuse it for that, the whole point is telling the two apart at a
# glance.
GROUND_TRUTH_TINT_BGR = (255, 0, 255)
# Orange -- distinct from every BOX_COLORS_BGR value (yellow/cyan/green/red)
# and from GROUND_TRUTH_TINT_BGR, so "human said yes" vs "algorithm said
# yes" never get confused even when they land on the same box.
ALGORITHM_TINT_BGR = (0, 165, 255)
TINT_ALPHA = 0.35

# Top-row text baseline — nudged down from a bare margin so QuickTime
# Player's title bar/menu chrome doesn't sit directly on top of it when
# reviewing a reconstructed clip.
TOP_ROW_Y = 70

# Black-on-white caption text, for every per-box/tint label going forward --
# colored text drawn directly over a busy/bright frame (sky, foliage,
# pavement) was frequently unreadable regardless of which class/tint color
# was used. The box outline/tint fill still carries the color-coding; only
# the caption text itself moved to a fixed, always-legible style.
LABEL_TEXT_BG_BGR = (255, 255, 255)
LABEL_TEXT_BG_ALPHA = 0.5
LABEL_TEXT_FG_BGR = (0, 0, 0)
LABEL_TEXT_PAD = 3


def _blend_translucent_rect(frame, top_left: tuple, bottom_right: tuple, color_bgr, alpha: float) -> None:
    """Alpha-blends a solid color_bgr fill into frame[top_left:bottom_right]
    IN PLACE, cropped to just that rectangle (clamped to the frame's own
    bounds) -- NOT a whole-frame copy+addWeighted.

    CONFIRMED 2026-08-23 via profiling (cProfile over a real 60s/900-frame
    render): the original approach both draw_label_text and draw_tint used
    (`overlay = frame.copy()`; `cv2.addWeighted(overlay, alpha, frame, ...,
    dst=frame)`) copies and blends the ENTIRE 1920x1080 frame to change a
    caption/tint rectangle that's typically a few thousand pixels -- 31.6%
    of this file's total render time (23.8% in addWeighted itself, 7.8% in
    the .copy() feeding it), more than the ReID model's own inference cost.
    Slicing a VIEW into `frame` first (not a copy -- `region` shares
    `frame`'s underlying memory) and blending only that crop is the
    identical visual result at a small fraction of the pixel count, not a
    behavior change. Silently does nothing for a rectangle that clamps to
    zero area (fully off-frame), matching cv2.rectangle's own tolerance for
    off-frame coordinates rather than raising."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, min(w, top_left[0])), max(0, min(h, top_left[1]))
    x1, y1 = max(0, min(w, bottom_right[0])), max(0, min(h, bottom_right[1]))
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    overlay = np.empty_like(region)
    overlay[:] = color_bgr
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def draw_label_text(frame, text: str, x: int, y: int, font_scale: float = 0.6, thickness: int = 1) -> None:
    """Draws `text` with its baseline at (x, y) (same origin convention as
    cv2.putText) on a translucent white backing rectangle sized to the text
    -- legible against any background without fully hiding whatever's
    underneath it."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top_left = (x - LABEL_TEXT_PAD, y - th - LABEL_TEXT_PAD)
    bottom_right = (x + tw + LABEL_TEXT_PAD, y + baseline + LABEL_TEXT_PAD)
    _blend_translucent_rect(frame, top_left, bottom_right, LABEL_TEXT_BG_BGR, LABEL_TEXT_BG_ALPHA)
    cv2.putText(frame, text, (x, y), font, font_scale, LABEL_TEXT_FG_BGR, thickness, cv2.LINE_AA)


def draw_box(frame, det: dict, track_id=None, distance_meters: Optional[float] = None) -> None:
    h, w = frame.shape[:2]
    x, y = int(det["x"] * w), int(det["y"] * h)
    bw, bh = int(det["w"] * w), int(det["h"] * h)
    color = BOX_COLORS_BGR.get(det["label"], DEFAULT_BOX_COLOR_BGR)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    id_prefix = f"#{track_id} " if track_id is not None else ""
    dist_text = f"{distance_meters:.1f} m" if distance_meters is not None else "? m"
    label = f"{id_prefix}{det['label']} {dist_text}"
    draw_label_text(frame, label, x + 4, y + 18)


def draw_tint(frame, det: dict, color_bgr, label: str, label_below: bool = False) -> None:
    """Fills a box with a translucent tint, on top of its already-drawn
    class-colored box/label. Alpha-blended (not a solid fill) so the vehicle
    itself stays visible underneath, and drawn as a thick outline too so the
    highlight still reads even on a very thin/distant box where the fill
    area is tiny.

    `label_below` puts the caption under the box instead of above it -- used
    to keep the ground-truth and algorithm labels from overlapping when both
    tints land on the same box (see the two call sites in the main loop)."""
    h, w = frame.shape[:2]
    x, y = int(det["x"] * w), int(det["y"] * h)
    bw, bh = int(det["w"] * w), int(det["h"] * h)

    _blend_translucent_rect(frame, (x, y), (x + bw, y + bh), color_bgr, TINT_ALPHA)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color_bgr, 3)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
    if label_below:
        label_y = y + bh + th + 8
    else:
        label_y = y - 8 if y - 8 - th > 0 else y + bh + th + 8
    draw_label_text(frame, label, x + 4, label_y, thickness=2)


# Calibrated intrinsics + the ego-motion flow model itself now live in
# flow_model.py (extracted 2026-08-22 so tracker.py can share it too, as a
# matching-time position-prediction gate) -- aliased back to their original
# FLOW_*-prefixed names here so nothing else in this file needs to change.
from flow_model import (  # noqa: E402
    FOCAL_LENGTH_ROW_NORMALIZED as FLOW_FOCAL_LENGTH_ROW_NORMALIZED,
    PRINCIPAL_ROW_NORMALIZED as FLOW_PRINCIPAL_ROW_NORMALIZED,
    PRINCIPAL_COLUMN_NORMALIZED as FLOW_PRINCIPAL_COLUMN_NORMALIZED,
    LEVER_ARM_FORWARD_M as FLOW_LEVER_ARM_FORWARD_M,
    LEVER_ARM_LEFT_M as FLOW_LEVER_ARM_LEFT_M,
    camera_velocity_from_yaw,
    angular_coords,
    predicted_flow_angular_raw,
)

FLOW_ARROW_COLOR_BGR = (255, 255, 0)  # cyan -- predicted ego-motion flow, this frame
PREVIOUS_FLOW_ARROW_COLOR_BGR = (0, 255, 0)  # green -- predicted ego-motion flow, previous frame
OBSERVED_RATE_ARROW_COLOR_BGR = (255, 0, 0)  # blue -- raw observed base-center displacement rate, unadjusted
MOTION_ARROW_COLOR_BGR = (255, 0, 255)  # magenta -- the object's own independent motion (flow subtracted out)
# Solid red -- NEW 2026-08-23, the tracker's own per-class-EMA + physical-
# outlier-gated flow_velocity (see tracker.py's Track.update_flow_state),
# deliberately a SEPARATE arrow from the magenta one above rather than a
# replacement: the magenta arrow is a single-step, exactly-derivable
# quantity (observed == avg(previous flow, current flow) + motion, an
# EXACT algebraic identity confirmed 2026-08-22 -- see
# _draw_angular_rate_arrow's doc comment) kept intentionally raw so that
# identity stays checkable frame by frame; smoothing it in place would
# break that debugging property. This is the one actually worth trusting
# as "is this object really moving" -- multi-frame, class-aware, gated
# against physically-impossible single-frame jumps -- so it's drawn
# fully OPAQUE (see draw_arrow's alpha, default 1.0) while every other
# rate arrow in this file draws at RAW_FLOW_ARROW_ALPHA instead, letting
# this one read clearly as the arrow that matters at a glance.
SMOOTHED_MOTION_ARROW_COLOR_BGR = (0, 0, 255)
FLOW_ARROW_THICKNESS = 2
# Fixed pixel size, not proportional to the arrow's own length (cv2
# .arrowedLine's own tipLength is a fraction of the shaft, so a short arrow
# got a tiny head and a long one an oversized one) -- keeps every
# arrowhead the same visual weight regardless of flow magnitude.
FLOW_ARROWHEAD_LENGTH_PX = 16
FLOW_ARROWHEAD_ANGLE_DEG = 32


PIXEL_SHIFT_BITS = 4
"""cv2.line/cv2.fillConvexPoly's own sub-pixel drawing mechanism: pass
coordinates pre-multiplied by 2**PIXEL_SHIFT_BITS (rounded to the nearest
integer) alongside shift=PIXEL_SHIFT_BITS, and OpenCV anti-aliases each
shape at its true fractional-pixel position instead of snapping every
point to the pixel grid first. Added 2026-08-22, real request -- coordinate
rounding was previously done in float space (int(...) or int(round(...)))
BEFORE handing off to cv2.line/fillConvexPoly, discarding sub-pixel
precision that mattered for exactly the kind of small, precise vectors
this file draws (see draw_arrow's/the *_arrow draw functions' own
CONFIRMED-exact vector-algebra identities elsewhere in this file -- that
algebra is exact in the underlying floats; drawing should not reintroduce
error the math doesn't have). 4 bits (1/16 pixel) is far finer than a
display can resolve or antialiasing can meaningfully use, so this is not a
tuning parameter in practice."""
_PIXEL_SHIFT_SCALE = 1 << PIXEL_SHIFT_BITS


def _shifted_point(pt: tuple) -> tuple:
    """A float (x, y) point converted to PIXEL_SHIFT_BITS fixed-point
    integer coordinates, for cv2 drawing calls' own shift= parameter."""
    return int(round(pt[0] * _PIXEL_SHIFT_SCALE)), int(round(pt[1] * _PIXEL_SHIFT_SCALE))


def draw_arrow(frame, start: tuple, end: tuple, color, thickness: int, alpha: float = 1.0) -> None:
    """Plain shaft (cv2.line) plus a small SOLID (filled) triangular head --
    cv2.arrowedLine's own built-in head is an open two-line chevron, not a
    filled triangle, which is what was actually wanted here (smaller,
    solid arrowheads -- CONFIRMED 2026-08-22 real request, replacing the
    original cv2.arrowedLine/tipLength=0.3 version). `start`/`end` are
    FLOAT pixel coordinates -- kept as floats all the way through this
    function (including the arrowhead geometry) and only converted to
    OpenCV's fixed-point sub-pixel representation at the final draw calls
    (see PIXEL_SHIFT_BITS) -- no coordinate here is ever snapped to a
    whole pixel before that single, antialiased conversion.

    `alpha` (NEW 2026-08-23): draws onto a scratch copy and alpha-blends it
    back, same overlay/addWeighted pattern draw_tint already uses -- lets
    the four raw/instantaneous debug arrows (predicted flow, previous
    flow, observed rate, raw motion) recede visually behind the one arrow
    actually worth trusting at a glance (the smoothed, solid motion
    vector), without changing any of their own math. Left at the default
    1.0 (fully opaque, and skips the copy/blend cost entirely) for
    anything that doesn't ask for translucency.

    That copy/blend is now cropped to a small padded box around the whole
    shape (shaft + arrowhead) rather than the whole frame -- same
    profiling-confirmed fix as _blend_translucent_rect, and correct for
    the same reason despite the shape being a thin diagonal line/triangle
    rather than a filled rectangle: addWeighted's blend is a per-pixel
    lerp between the drawn copy and the untouched original, so wherever
    the shape didn't touch a pixel the copy still equals the original and
    the blend is a no-op there regardless of alpha -- only the pixels the
    line/arrowhead actually covers change. Cropping just shrinks how many
    of those no-op pixels get redundantly copied/blended."""
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-3:
        return
    angle = math.atan2(y1 - y0, x1 - x0)
    head_angle = math.radians(FLOW_ARROWHEAD_ANGLE_DEG)
    p2 = (
        x1 - FLOW_ARROWHEAD_LENGTH_PX * math.cos(angle - head_angle),
        y1 - FLOW_ARROWHEAD_LENGTH_PX * math.sin(angle - head_angle),
    )
    p3 = (
        x1 - FLOW_ARROWHEAD_LENGTH_PX * math.cos(angle + head_angle),
        y1 - FLOW_ARROWHEAD_LENGTH_PX * math.sin(angle + head_angle),
    )

    def _paint(target, s, e, a, b) -> None:
        cv2.line(target, _shifted_point(s), _shifted_point(e), color, thickness, cv2.LINE_AA, PIXEL_SHIFT_BITS)
        head_points = np.array([_shifted_point(e), _shifted_point(a), _shifted_point(b)], dtype=np.int32)
        cv2.fillConvexPoly(target, head_points, color, cv2.LINE_AA, PIXEL_SHIFT_BITS)

    if alpha >= 1.0:
        _paint(frame, start, end, p2, p3)
        return

    h, w = frame.shape[:2]
    pad = thickness + FLOW_ARROWHEAD_LENGTH_PX + 4
    xs, ys = (x0, x1, p2[0], p3[0]), (y0, y1, p2[1], p3[1])
    cx0, cy0 = max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad))
    cx1, cy1 = min(w, int(max(xs) + pad) + 1), min(h, int(max(ys) + pad) + 1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    region = frame[cy0:cy1, cx0:cx1]
    overlay = region.copy()
    ox, oy = cx0, cy0
    _paint(
        overlay, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy),
        (p2[0] - ox, p2[1] - oy), (p3[0] - ox, p3[1] - oy),
    )
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def camera_velocity_from_yaw(smoothed_yaw_rate_deg_s: float) -> tuple:
    """Ports DistanceEstimator.LeverArm.cameraVelocityFromYaw (Swift) --
    the camera's own velocity due to the vehicle's yaw rotation about the
    rear axle (v = omega x leverArm), in the vehicle's body frame (forward,
    left), meters/second. Takes the SMOOTHED yaw rate, not raw -- see that
    Swift function's own doc comment for why (raw is too noisy for a
    per-frame consumer, confirmed against a real drive)."""
    omega = math.radians(smoothed_yaw_rate_deg_s)
    return (-omega * FLOW_LEVER_ARM_LEFT_M, omega * FLOW_LEVER_ARM_FORWARD_M)  # (forward, left)


def angular_coords(col_norm: float, row_norm: float, aspect: float) -> tuple:
    """Converts a normalized [0,1] screen position to this project's
    calibrated-camera angular coordinate space -- the SAME parameterization
    DistanceEstimator's own de-roll step uses: x=(col-principalCol)*aspect/f,
    y=(row-principalRow)/f. A flow/displacement of 1.0 in this space is one
    focal length of angular motion; multiply by f_col*frame_width (x) or
    f_row*frame_height (y) to get pixels."""
    f_row = FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    x = (col_norm - FLOW_PRINCIPAL_COLUMN_NORMALIZED) * aspect / f_row
    y = (row_norm - FLOW_PRINCIPAL_ROW_NORMALIZED) / f_row
    return x, y


EDGE_TRUNCATION_MARGIN_NORMALIZED = 0.01
"""Same margin/rationale as WidthDistanceOverride's Swift-side constant --
kept in sync deliberately, not derived from it, since this is a separate
Python tool. Used by corrected_distance_meters's width-vs-row
trustworthiness decision, and also by predicted_flow_angular's own
truncation check (see that function's doc comment) -- a box this close to
either side edge has no reliable lateral-position anchor at all (tried and
abandoned a corner-anchor scheme for this same reason, see
base_center_normalized's doc comment), so no flow/motion arrow is computed
for it rather than trust either its own crop-drifting center or a
scale-drifting corner."""


def base_center_normalized(det: dict) -> tuple:
    """The center of a detection's BASE (bottom-center of its box), in
    normalized [0,1] screen fractions -- the ground-contact point
    distanceMeters is itself computed from (bottomY, centerX), not the
    box's own 2D centroid, which sits partway up the object's height and
    isn't at the depth distanceMeters actually describes. Used as the
    single consistent anchor for both the flow arrows and the motion
    arrows below -- CONFIRMED 2026-08-22, real request: the flow arrows
    originally anchored at box center, an inconsistency with distanceMeters'
    own bottom-anchored assumption, corrected here.

    Plain center, deliberately -- NOT corner-anchored for a truncated box,
    despite that having seemed like the right fix earlier THIS SAME
    session (see git history for the full, now-abandoned attempt: anchor
    at whichever corner isn't touching the frame edge, on the theory that
    a detector localizes an un-cropped edge correctly even when the
    opposite edge is clipped). REVERTED 2026-08-22 after a second real
    case proved that theory wrong in general: car #12's box stayed
    genuinely left-truncated across several frames while its WIDTH kept
    growing (0.1127 -> 0.1161 -> 0.1193, more of the car becoming visible,
    not the car moving) -- and a growing box's far corner drifts outward
    from the true center purely from that scale change, at TWICE the rate
    plain center drift does, even for a car that hasn't moved sideways at
    all. Corner-anchoring isn't a refinement of the center-drift problem
    it was built to fix, it's a DIFFERENT confound (scale-driven, not
    crop-driven) that's just as real. Neither anchor is trustworthy for a
    genuinely truncated box -- see predicted_flow_angular's own truncation
    check, which now declines to compute a flow/motion arrow at all for
    such a box, rather than pick between two comparably-unreliable anchor
    strategies (same "don't fabricate a number with no real basis"
    discipline as corrected_distance_meters' own doubly-truncated case)."""
    return det["x"] + det["w"] / 2, det["y"] + det["h"]


DISTANCE_CAMERA_HEIGHT_METERS = 1.02
"""Same value as DistanceEstimator.calibrated.cameraHeightMeters (Swift) --
measured 2026-08-15 for the current SHAPE clamp mount. Kept in sync
deliberately, not derived from it."""


# Same real AVCameraCalibrationData capture as LensCalibrationData
# .factoryMeasured (Swift, LensCalibration.swift) -- 2026-08-24, this exact
# physical device, via IsolatedLensCalibrationCapture (GDC disabled, 2+
# constituent devices requested, virtual-constituent-delivery enabled before
# startRunning -- see that Swift type's own doc comment for the full
# recipe). Kept in sync deliberately, not derived from it. Cross-checked
# there against the independent cone-calibration fit to within 4.4%. Raw
# capture backed up at calibration/lens_calibration_20260824.json.
_LENS_CALIB_FX = 3005.462158203125
_LENS_CALIB_FY = 3005.462158203125
_LENS_CALIB_CX = 2103.8779296875
_LENS_CALIB_CY = 1188.294921875
_LENS_CALIB_REFERENCE_WIDTH = 4224.0
_LENS_CALIB_REFERENCE_HEIGHT = 2376.0
_LENS_CALIB_DISTORTION_CENTER_X = 2103.962646484375
_LENS_CALIB_DISTORTION_CENTER_Y = 1188.3253173828125
_LENS_CALIB_DISTORTION_LOOKUP_TABLE = [
    0, 9.703804e-05, 0.0003860671, 0.000860824, 0.0015108482, 0.002321466, 0.0032738035,
    0.004344853, 0.005507639, 0.0067315097, 0.007982596, 0.00922447, 0.010419019,
    0.011527535, 0.012512018, 0.013336645, 0.01396934, 0.014383375, 0.014558875,
    0.014484134, 0.014156586, 0.013583343, 0.012781186, 0.011775943, 0.010601236,
    0.009296611, 0.00790514, 0.006470612, 0.0050345063, 0.0036329643, 0.0022940412,
    0.0010355223, -0.00013637631, -0.0012271275, -0.0022525825, -0.0032360968,
    -0.004204166, -0.005180965, -0.006182665, -0.0072130556, -0.008262857, -0.009316202,
]


def _corrected_normalized_point(bottom_y: float, center_x: float) -> tuple:
    """Ports LensCalibrationData.correctedNormalizedPoint (Swift) exactly --
    corrects a normalized [0,1], top-left-origin point from the real,
    distorted camera image to where it would appear in an ideal pinhole
    image, using Apple's own documented lookup-table algorithm (radial-
    distance-indexed magnification, linearly interpolated). Applied to EVERY
    row_based_distance_meters/_row_based_phi_degrees call so the same
    correction reaches both the distance formula and the hood-cutoff angle
    check, mirroring InferenceEngine.attachDistances applying it once,
    upstream of both Swift consumers, before today's fix existed on the
    device this recording predates. See that struct's own doc comment for
    the coordinate-space assumption this relies on (same field of view/crop
    as the calibration capture, just a different pixel count -- true here,
    since both are the same 16:9 aspect ratio)."""
    px = center_x * _LENS_CALIB_REFERENCE_WIDTH
    py = bottom_y * _LENS_CALIB_REFERENCE_HEIGHT
    ocx, ocy = _LENS_CALIB_DISTORTION_CENTER_X, _LENS_CALIB_DISTORTION_CENTER_Y
    w, h = _LENS_CALIB_REFERENCE_WIDTH, _LENS_CALIB_REFERENCE_HEIGHT
    r_max = math.hypot(max(ocx, w - ocx), max(ocy, h - ocy))
    vx, vy = px - ocx, py - ocy
    r_point = math.hypot(vx, vy)
    table = _LENS_CALIB_DISTORTION_LOOKUP_TABLE
    last_index = len(table) - 1
    if r_point == 0 or r_max == 0 or last_index < 0:
        return bottom_y, center_x
    scaled_r = min(last_index, (r_point / r_max) * last_index)
    lower_index = int(scaled_r)
    t = scaled_r - lower_index
    lower_value = table[lower_index]
    upper_value = table[min(lower_index + 1, last_index)]
    magnification = lower_value + t * (upper_value - lower_value)
    new_vx = vx + magnification * vx
    new_vy = vy + magnification * vy
    corrected_px = ocx + new_vx
    corrected_py = ocy + new_vy
    return corrected_py / h, corrected_px / w


def row_based_distance_meters(
    bottom_y: float, center_x: float, aspect: float,
    reference_pitch_deg: float, reference_roll_deg: float,
) -> Optional[float]:
    """Ports DistanceEstimator.distanceMeters's ground-plane (row-based)
    formula exactly. Used to RECOVER the pre-override distance for a
    detection where distanceMetersIsWidthOverridden is true and the box is
    edge-truncated (see corrected_distance_meters): the on-device pipeline
    mutates Detection.distanceMeters in place when the width override
    fires, so the original row-based value isn't itself logged -- but every
    input this formula needs (box geometry, reference pitch/roll) IS logged
    per-entry, so it's cheaply recomputable offline instead of needing a
    new recording.

    GENERALIZED TO FULL 3D -- FIXED 2026-08-23, matching the same-day Swift
    fix (see DistanceEstimator.distanceMeters's own doc comment for the full
    derivation/verification). Until this date this returned
    `cameraHeightMeters / tan(alpha + theta)` -- exactly correct only for a
    point on the frame's vertical centerline, since it discarded the
    lateral ("derolled_x") component of the ray entirely; for anything
    off-center it silently returned the ray's forward DEPTH, not the true
    ground distance, understating distance more the further off-axis the
    detection sat. CONFIRMED via real 26_08_15_Walkaround data: a person
    walking a constant-radius tether arc read as shrinking purely from
    walking off to the side. Fix: invert the full pitch+roll rotation to
    recover the ray in world coordinates, solve for where it crosses the
    ground plane, and return the full lateral+forward hypotenuse there.
    `_row_based_phi_degrees` below is UNCHANGED and still uses the old
    vertical-only angle -- HOOD_CUTOFF_ANGLE_DEGREES was calibrated against
    that specific angle, not the radial distance this function now
    returns.

    LENS DISTORTION CORRECTED -- ADDED 2026-08-24, applied to (bottom_y,
    center_x) before anything else below: see _corrected_normalized_point's
    own doc comment. This recording predates the real AVCameraCalibration
    Data capture, but lens distortion is a fixed hardware property of the
    physical lens, not a per-recording one, so applying it retroactively
    here is exactly as valid as it would have been live."""
    bottom_y, center_x = _corrected_normalized_point(bottom_y, center_x)
    theta = math.radians(reference_pitch_deg)
    psi = math.radians(reference_roll_deg)
    focal_col = FLOW_FOCAL_LENGTH_ROW_NORMALIZED / aspect
    x = (center_x - FLOW_PRINCIPAL_COLUMN_NORMALIZED) / focal_col
    y = (bottom_y - FLOW_PRINCIPAL_ROW_NORMALIZED) / FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    derolled_y = -x * math.sin(psi) + y * math.cos(psi)
    derolled_x = x * math.cos(psi) + y * math.sin(psi)

    ray_down = math.cos(theta) * derolled_y + math.sin(theta)
    if ray_down <= 0:
        return None
    ray_forward = math.cos(theta) - math.sin(theta) * derolled_y

    range_at_ground = DISTANCE_CAMERA_HEIGHT_METERS / ray_down
    lateral_m = range_at_ground * derolled_x
    forward_m = range_at_ground * ray_forward
    if forward_m <= 0:
        return None
    return math.hypot(lateral_m, forward_m)


HOOD_CUTOFF_ANGLE_DEGREES = 9.899
"""Same value as DistanceEstimator.hoodCutoffAngleDegrees (Swift) -- the
measured angle below which the ego vehicle's own hood physically blocks the
camera's view (mean of 107 real samples, std 0.771, see that constant's own
doc comment). A box whose row-based ground-contact angle phi EXCEEDS this
cannot be a genuine ground-contact reading -- an object that close would
have its true bottom hidden behind the hood, so phi cannot geometrically
exceed this angle for a real detection; a box that computes higher is
reporting some other row (frame edge, dashboard/HUD artifact) as its
"bottom," not the object's real extent. Kept in sync deliberately, not
derived from it. This project has a matching (real, still-unbuilt-until-now)
gap on the on-device side too -- see the project_ego_hood_rejection_gap
memory: the angle was measured 2026-08-15 but no rejection gate was ever
implemented there."""


def _row_based_phi_degrees(
    bottom_y: float, center_x: float, aspect: float,
    reference_pitch_deg: float, reference_roll_deg: float,
) -> float:
    """The ground-contact angle phi alone -- split out so
    HOOD_CUTOFF_ANGLE_DEGREES can be checked independent of whatever
    distance value ultimately gets used. Deliberately UNCHANGED by
    row_based_distance_meters' 2026-08-23 full-3D fix (see that function's
    doc comment): HOOD_CUTOFF_ANGLE_DEGREES was calibrated by converting
    real boxes' bottom rows to this exact vertical-only angle, so this
    still needs to compute the same thing, not the angle implied by the
    now-lateral-inclusive distance.

    LENS DISTORTION CORRECTED -- ADDED 2026-08-24, same as row_based_
    distance_meters, so the hood-cutoff check sees the same corrected point
    the distance formula does."""
    bottom_y, center_x = _corrected_normalized_point(bottom_y, center_x)
    theta = math.radians(reference_pitch_deg)
    psi = math.radians(reference_roll_deg)
    focal_col = FLOW_FOCAL_LENGTH_ROW_NORMALIZED / aspect
    x = (center_x - FLOW_PRINCIPAL_COLUMN_NORMALIZED) / focal_col
    y = (bottom_y - FLOW_PRINCIPAL_ROW_NORMALIZED) / FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    derolled_y = -x * math.sin(psi) + y * math.cos(psi)
    alpha = math.atan(derolled_y)
    return math.degrees(alpha + theta)


OBJECT_ASPECT_PRIORS = {
    # Same values as ObjectWidthPriors.byLabel (WidthDistanceOverride.swift)
    # -- kept in sync deliberately, not derived from it, since this is a
    # separate Python tool. width_mean/height_mean are real per-class KITTI
    # means (see that Swift file's own doc comment for how they were
    # computed -- not estimated or guessed); aspect_tolerance is that same
    # file's per-class `aspectTolerance` (car validated only against the
    # confirmed-bad track #37 case below; person's 1.8 is empirically
    # tuned against the real data/26_08_15_Walkaround 3m-window validation
    # session, preserving 98% of its known-good frames).
    "car": {"width_mean": 1.629, "height_mean": 1.526, "aspect_tolerance": 1.4},
    "person": {"width_mean": 0.660, "height_mean": 1.761, "aspect_tolerance": 1.8},
}

MIN_YAW_RATE_FOR_OBLIQUENESS_CHECK_DEG_S = 1.0
"""Mirrors WidthDistanceOverrideManager.minYawRateForObliquenessCheckDegS
(WidthDistanceOverride.swift) -- see is_oblique_view's own doc comment for
the two real cases (a confirmed bug and a confirmed false positive) this
was reasoned from. Kept in sync deliberately, not derived from it."""


def is_oblique_view(det: dict, entry: dict, aspect: float) -> bool:
    """True when a box's pixel-space aspect ratio (width/height) is wider
    than a genuine head-on/tail-on view of its class could plausibly
    produce, AND the ego is actually turning meaningfully -- mirrors
    WidthDistanceOverrideManager's obliqueness gate (WidthDistanceOverride
    .swift) exactly, so already-logged sessions recorded before that
    on-device fix existed can still be corrected retroactively here, same
    idea as this function's other fallbacks.

    CONFIRMED 2026-08-23 (real drive, data/26_08_21_Day_Small, track #37 in
    the raw log -- rendered as #24 in this tool's own offline-retracked
    flow-arrow video, which assigns its own IDs): a parked car swept across
    frame by the ego vehicle's own yaw during a slow turn was viewed at a
    persistent oblique angle the entire ~1.5s window, its box picking up
    foreshortened LENGTH on top of width. Its widthDistanceMeters read
    ~2x closer than an independently-recomputed row-based distance the
    whole time (observed pixel aspect ~2.85 vs car's expected head-on
    ~1.07) -- not caught by either of the two truncation cases below,
    since this box was neither edge- nor hood-truncated. Feeding that
    understated depth into predicted_flow_angular's 1/z term produced a
    predicted flow ~1.9x too large and a large spurious "independent
    motion" arrow on an object confirmed, from the source video, to never
    have moved at all -- see the project_width_based_distance_override
    memory for the full writeup.

    CONFIRMED FALSE POSITIVE, same day: a genuinely PARKED, dead-ahead car
    (data/26_08_21_Day_Small, track #13, t~0s -- confirmed via the source
    frame, squarely facing the camera) was rejected by the aspect check
    anyway (observed pixel aspect 2.36 vs a ~1.49 threshold), substituting
    a WORSE row-based reading (6.45m) for a correct width-based one
    (3.03m) -- confirmed against a real user report that the car was right
    in front of the vehicle, not ~6m away. Its box height came out short
    for a reason unrelated to viewing angle: it sat right at the edge of
    the near-field regime (row-based phi 8.97 degrees, just 0.93 degrees
    under HOOD_CUTOFF_ANGLE_DEGREES), where the ego's own hood/dash edge
    was already encroaching on its visible bottom without yet crossing the
    HARD cutoff the hood-truncation case catches. Its
    smoothedYawRateDegreesPerSecond was ~0.003 deg/s and egoSpeedMps was 0
    -- parked, not turning -- whereas the confirmed-bad track #37 case only
    ever showed this failure while yaw rate ramped through 2.6-6.2 deg/s.
    The oblique-view mechanism above physically REQUIRES the camera to be
    rotating relative to the object, so gating on yaw rate separates the
    two real cases with wide margin.

    KNOWN REMAINING GAP: doesn't catch a car at a genuinely constant
    oblique angle viewed from a STATIONARY or straight-driving ego (no yaw
    at all) -- untested territory, left for whenever real data surfaces
    it, same "don't fix what isn't confirmed yet" discipline as the rest
    of this file."""
    yaw_rate = entry.get("smoothedYawRateDegreesPerSecond")
    if yaw_rate is None or abs(yaw_rate) < MIN_YAW_RATE_FOR_OBLIQUENESS_CHECK_DEG_S:
        return False
    prior = OBJECT_ASPECT_PRIORS.get(det["label"])
    if prior is None or det["h"] <= 0:
        return False
    observed_pixel_aspect = (det["w"] / det["h"]) * aspect
    expected_head_on_aspect = prior["width_mean"] / prior["height_mean"]
    return observed_pixel_aspect > expected_head_on_aspect * prior["aspect_tolerance"]


def corrected_distance_meters(det: dict, entry: dict, aspect: float) -> Optional[float]:
    """distanceMeters, corrected for FIVE CONFIRMED corruption cases
    (cases 1-3 found 2026-08-22, case 4 found 2026-08-23, case 5 found
    2026-08-24, all via this session's flow-arrow visualization or a real
    tethered walkaround test surfacing distance errors as visibly wrong
    motion or wrong numbers):

    1. LEFT/RIGHT edge truncation with the width-based override active
       (distanceMetersIsWidthOverridden) -- the logged value is the
       ballooning width-based misread this session first surfaced, not a
       real distance (apparent width shrinks from cropping, not real
       distance change). Falls back to a freshly recomputed row-based
       distance -- the same fallback WidthDistanceOverride's own on-device
       edge-truncation guard now takes for FUTURE recordings, applied here
       retroactively to already-logged data.

    2. BOTTOM (hood) truncation -- CONFIRMED via a real box whose row-based
       phi came out to 19.3 degrees, nearly double
       HOOD_CUTOFF_ANGLE_DEGREES (9.899), a geometrically impossible
       ground-contact reading (see that constant's own doc comment). This
       is INDEPENDENT of distanceMetersIsWidthOverridden -- case 1 above
       only ever looks at bottom-truncated boxes when the override
       happened to be active, but this failure corrupts the RAW row-based
       distanceMeters directly, override or not. Falls back to
       widthDistanceMeters (doesn't depend on bottomY at all).

    3. BOTH at once (a box truncated on the bottom AND a side, e.g. a
       corner) -- neither fallback is trustworthy (case 2's own
       width-based fallback is itself corrupted by the SAME left/right
       truncation case 1 exists to catch), so this returns None rather
       than pick between two comparably-unreliable numbers -- "don't
       fabricate a number with no real basis," same discipline as the rest
       of this file.

    4. Width-based override active on a box that's too OBLIQUE (see
       is_oblique_view) for widthDistanceMeters's head-on/tail-on
       assumption to hold, independent of any truncation -- falls back to
       a freshly recomputed row-based distance, same remedy as case 1
       (both are "the width-based reading can't be trusted here," just
       from different causes: cropping vs viewing angle).

    5. Not width-overridden at all -- the common case. The logged value here
       is whatever row_based_distance_meters computed ON THE DEVICE at
       record time, which for any recording predating that formula's
       2026-08-24 fix is the OLD forward-depth-only reading for any
       off-center detection. Always recomputed fresh instead of trusted, so
       this function reflects CURRENT understanding of the formula on
       historical logs, not whatever was true when they were recorded.

    A width-overridden reading with neither truncation nor obliqueness (a
    legitimate "genuinely closer" width-based reading, unaffected by any of
    the row-based formula's history -- widthBasedDistanceMeters is a
    different computation entirely) passes distanceMeters through
    unchanged."""
    left = det["x"]
    right = det["x"] + det["w"]
    bottom = det["y"] + det["h"]
    side_truncated = (
        left <= EDGE_TRUNCATION_MARGIN_NORMALIZED or right >= 1.0 - EDGE_TRUNCATION_MARGIN_NORMALIZED
    )

    phi_deg = _row_based_phi_degrees(
        bottom, det["x"] + det["w"] / 2, aspect,
        entry.get("referencePitchDegrees", 0.0), entry.get("referenceRollDegrees", 0.0),
    )
    if phi_deg > HOOD_CUTOFF_ANGLE_DEGREES:
        if side_truncated:
            return None
        return det.get("widthDistanceMeters")

    if not det.get("distanceMetersIsWidthOverridden"):
        # CASE 5, found 2026-08-24: the logged value here is whatever
        # row_based_distance_meters produced ON THE DEVICE, at record time --
        # for any recording made before that formula's 2026-08-24 lateral-
        # angle fix (every recording this project has so far), that's the
        # OLD, buggy forward-depth-only reading for any off-center detection,
        # not the corrected radial distance. CONFIRMED via a real tethered
        # walkaround test: recomputing fresh here (as this branch now does)
        # recovers the fixed formula's ~20m reading where the stale logged
        # value read as low as ~15.4m at the frame edge -- a gap that first
        # looked like leftover lens distortion until direct comparison
        # showed `corrected_distance_meters` was returning the untouched
        # pre-fix logged value on every single frame, not a freshly
        # recomputed one. Recomputing fresh is exactly what row_based_
        # distance_meters exists for (see its own doc comment: "cheaply
        # recomputable offline instead of needing a new recording") --
        # falls back to the logged value only if recomputation itself
        # can't produce one (invalid geometry), same as cases 1/4 below.
        recomputed = row_based_distance_meters(
            bottom_y=bottom,
            center_x=det["x"] + det["w"] / 2,
            aspect=aspect,
            reference_pitch_deg=entry.get("referencePitchDegrees", 0.0),
            reference_roll_deg=entry.get("referenceRollDegrees", 0.0),
        )
        return recomputed if recomputed is not None else det.get("distanceMeters")
    if not side_truncated and not is_oblique_view(det, entry, aspect):
        return det.get("distanceMeters")
    recomputed = row_based_distance_meters(
        bottom_y=bottom,
        center_x=det["x"] + det["w"] / 2,
        aspect=aspect,
        reference_pitch_deg=entry.get("referencePitchDegrees", 0.0),
        reference_roll_deg=entry.get("referenceRollDegrees", 0.0),
    )
    return recomputed if recomputed is not None else det.get("distanceMeters")


def predicted_flow_angular(det: dict, entry: dict, aspect: float, z_override: Optional[float] = None) -> Optional[tuple]:
    """Predicted per-second optic flow (u, v) in normalized ANGULAR
    coordinates (see angular_coords) at this detection's own BASE-center
    screen position and depth, for a point STATIC in the world, due to the
    camera's OWN ego-motion this frame (translation + rotation) alone --
    the standard Longuet-Higgins/Prazdny calibrated-camera flow
    decomposition, evaluated with this project's own signals. This is the
    flow the object's IMMEDIATE SURROUNDINGS (or the object itself, if it
    happens to be genuinely stationary) would show; a real moving object's
    OBSERVED screen motion differs from this by exactly its own independent
    motion -- see compute_motion_arrow_angular for that subtraction, done
    in this same angular space (not pixels -- a given angular flow doesn't
    map to a constant pixel displacement across the frame, see the (1+x^2)
    terms below, so pixel-space subtraction would be a worse approximation
    for anything not near the principal point).

    Derivation (camera axes X=right, Y=down, Z=forward -- confirmed
    right-handed): for a static point at camera-relative (X,Y,Z), with
    camera translational velocity V=(Vx,Vy,Vz) and angular velocity
    Omega=(wx,wy,wz) in its own frame, d(X,Y,Z)/dt = -V - Omega x (X,Y,Z).
    Differentiating the normalized-image-coordinate projection (x,y) =
    (X/Z, Y/Z) through that and simplifying gives the standard form:
        u = dx/dt = (-Vx + x*Vz)/Z - wy*(1+x^2) + wz*y + wx*x*y
        v = dy/dt = (-Vy + y*Vz)/Z + wx*(1+y^2) - wy*x*y - wz*x
    (x, y) use the exact same normalized angular-coordinate parameterization
    as DistanceEstimator's own de-roll step: x=(col-principalCol)*aspect/f,
    y=(row-principalRow)/f -- so a flow of 1.0 in this space corresponds to
    one focal length of angular motion, converted to pixels at the end via
    the same f_row/f_col relationship used throughout this project.

    Signal mapping, each cross-checked against an ALREADY-VALIDATED
    convention in this codebase rather than derived fresh and trusted
    blindly (this project has a real history of pitch/roll sign bugs, see
    PitchSensor.swift's own file-level note):
      - Vz (forward) = egoSpeedMps + the yaw-lever-arm's own forward
        component (camera_velocity_from_yaw); Vx (right) = -that function's
        left component; Vy (down) = 0 -- no vertical-camera-velocity signal
        exists in this project yet (suspension bounce, pitch-rate lever-arm
        lift), a real but currently-unclosed gap, not an oversight.
        CONFIRMED 2026-08-22 on real footage: this shows up as a small but
        consistent systematic bias in compute_motion_arrow_angular's output
        (~0.01, -0.06 angular units/sec, uniform across every tracked
        object in a real 10s window -- consistency across objects, not
        varying per-object like real noise would, is what points to Vy
        rather than a bug in the subtraction itself). Deliberately left
        uncorrected for now, by request -- see the
        project_motion_arrow_vertical_bias memory for the two candidate
        fixes and why neither was applied yet.

        SEPARATELY, Vz's sign is always treated as forward -- egoSpeedMps
        (EgoSpeedManager.swift) comes straight from CLLocation.speed,
        Apple's raw GPS ground-speed MAGNITUDE (guarded >= 0 to reject
        CoreLocation's own "invalid fix" sentinel, not to enforce
        direction) -- so there is no signal anywhere in this pipeline that
        can tell this function forward from reverse. CONFIRMED 2026-08-22
        on a real parking-space backout: predicted flow pointed away from
        the FOE (the forward-motion expansion pattern) throughout a slow,
        smooth REVERSE maneuver, when the true pattern should have been
        contraction toward it. Not a bug in this formula -- it's doing
        exactly what a positive Vz implies; the implied direction is
        simply wrong for that maneuver. Left unfixed, by request -- see
        the project_egospeed_unsigned_reverse_gap memory for why, and the
        one real fix path considered (inferring direction from whether the
        scene's own aggregate flow is expanding or contracting, not a
        quick patch).
      - wy (yaw): NEGATED smoothedYawRateDegreesPerSecond. Cross-checked
        against ByteTracker.yawFallbackTransform's own bench-confirmed
        ("turning left -> background shifts right in frame") convention:
        with this mapping, u's yaw term (~-wy near x=0) comes out positive
        for a positive (turning-left) yaw rate, matching exactly.
      - wx (pitch): NEGATED smoothedPitchRateDegreesPerSecond. Cross-checked
        against DistanceEstimator.distanceMeters's own validated
        phi=alpha+theta relationship (a fixed ground point's alpha/y
        DECREASES as nose-down pitch increases) -- this mapping reproduces
        that same direction.
      - wz (roll): NEGATED smoothedRollRateDegreesPerSecond. Cross-checked
        only against DistanceEstimator.fit()'s STATIC de-roll rotation
        formula (not a live bench test) -- weaker footing than pitch/yaw
        above, same "not yet bench-confirmed" caveat rollDegrees itself
        already carries elsewhere in this project.

    Returns None if any required signal (depth, ego speed, all three
    smoothed rotation rates) is missing for this entry/detection -- no
    existing recording has smoothedPitchRateDegreesPerSecond/
    smoothedRollRateDegreesPerSecond yet (added 2026-08-22, no drive since),
    so this returns None for every detection until a session recorded after
    that exists.

    z_override, when given, is used INSTEAD of a fresh
    corrected_distance_meters(det, entry, aspect) call -- pass a track's own
    EMA-smoothed depth (ByteTracker.get_track(track_id).flow_z) here rather
    than a single-frame value. CONFIRMED 2026-08-22: a single frame's
    corrected_distance_meters can be noisy even for a correctly, smoothly
    tracked object (truncation-margin boundary flicker, or the row-based
    formula's own sensitivity to tiny bottom-edge noise), and that noise
    otherwise feeds straight into this function's 1/Z term, producing a
    jumpy flow prediction -- and hence a jumpy motion-arrow residual -- for
    an object that isn't actually moving jumpily at all. Falls back to the
    normal per-frame computation when no override is given (untracked
    detections, or callers that don't have a tracker to ask).

    Also returns None outright for a box truncated by the LEFT or RIGHT
    frame edge (within EDGE_TRUNCATION_MARGIN_NORMALIZED) -- CONFIRMED
    2026-08-22, real case: no anchor point on such a box is trustworthy
    for lateral position (see base_center_normalized's doc comment for the
    full story -- both plain center and a tried-and-abandoned corner
    anchor drift as the box's visible width changes, which it always does
    for a truncated box, purely from scale, independent of any real
    motion). Declining to compute a flow/motion arrow here is what makes
    base_center_normalized's own anchor choice moot for this case, rather
    than needing a second, separate check wherever base_center_normalized
    gets called.
    """
    left = det["x"]
    right = det["x"] + det["w"]
    if left <= EDGE_TRUNCATION_MARGIN_NORMALIZED or right >= 1.0 - EDGE_TRUNCATION_MARGIN_NORMALIZED:
        return None
    z = z_override if z_override is not None else corrected_distance_meters(det, entry, aspect)
    if z is None or z <= 0:
        return None
    ego_speed = entry.get("egoSpeedMps")
    yaw_rate = entry.get("smoothedYawRateDegreesPerSecond")
    pitch_rate = entry.get("smoothedPitchRateDegreesPerSecond")
    roll_rate = entry.get("smoothedRollRateDegreesPerSecond")
    if ego_speed is None or yaw_rate is None or pitch_rate is None or roll_rate is None:
        return None

    base_col_norm, base_row_norm = base_center_normalized(det)
    x, y = angular_coords(base_col_norm, base_row_norm, aspect)

    lever_forward, lever_left = camera_velocity_from_yaw(yaw_rate)
    tz = ego_speed + lever_forward
    vx = -lever_left
    vy = 0.0

    omega_x = -math.radians(pitch_rate)
    omega_y = -math.radians(yaw_rate)
    omega_z = -math.radians(roll_rate)

    u = (-vx + x * tz) / z - omega_y * (1 + x * x) + omega_z * y + omega_x * x * y
    v = (-vy + y * tz) / z + omega_x * (1 + y * y) - omega_y * x * y - omega_z * x

    return u, v


def angular_to_pixels(u: float, v: float, frame_width: int, frame_height: int) -> tuple:
    """Converts an angular (u, v) -- a flow rate, or any other angular
    displacement -- to pixels, for display only. Inverse of angular_coords'
    own scaling."""
    aspect = frame_width / frame_height
    f_row = FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    f_col = f_row / aspect
    return u * f_col * frame_width, v * f_row * frame_height


RAW_FLOW_ARROW_ALPHA = 0.45
"""NEW 2026-08-23: applied to the four RAW/instantaneous debug arrows
(predicted flow, previous flow, observed rate, raw motion) so the one
arrow actually worth trusting at a glance -- the smoothed, solid-red
motion vector, see SMOOTHED_MOTION_ARROW_COLOR_BGR -- reads clearly
without the four math-verification arrows competing for attention. Purely
visual; doesn't touch any of their underlying values or the EXACT
algebraic identity _draw_angular_rate_arrow's own doc comment describes."""


def _draw_angular_rate_arrow(
    frame, det: dict, angular_rate: Optional[tuple], color, alpha: float = 1.0,
) -> None:
    """Shared draw path for every one-second-equivalent angular-rate arrow
    this file draws (predicted flow, previous flow, raw observed rate,
    subtracted motion) -- all anchored at the SAME point, this detection's
    CURRENT base center (base_center_normalized(det)), even for a
    quantity like the previous flow that was technically evaluated at the
    PREVIOUS frame's own position. Deliberate: these are all being
    compared as rates at the same instant for the viewer's benefit (see
    predicted_flow_angular/compute_motion_arrow_angular's own doc
    comments), not plotted as literal trajectories, so anchoring them all
    together is what makes the vector algebra (observed == avg(previous
    flow, current flow) + motion -- EXACT, not approximate, a direct
    algebraic rearrangement of compute_motion_arrow_angular's own
    subtraction, confirmed 2026-08-22) visually legible in one place --
    added 2026-08-22, real request, specifically to make this project's
    own flow-arrow debugging (done by hand, via ad-hoc scripts, all this
    session) visible directly in the render instead. A no-op when
    angular_rate is None, same as every other optional overlay here.
    Coordinates stay FLOAT all the way to draw_arrow (see
    PIXEL_SHIFT_BITS) -- no int(...)/round(...) here, so an exact identity
    in the underlying numbers isn't reintroduced as pixel-grid error by
    the drawing step."""
    if angular_rate is None:
        return
    h, w = frame.shape[:2]
    dx, dy = angular_to_pixels(angular_rate[0], angular_rate[1], w, h)
    base_col_norm, base_row_norm = base_center_normalized(det)
    cx = base_col_norm * w
    cy = base_row_norm * h
    draw_arrow(frame, (cx, cy), (cx + dx, cy + dy), color, FLOW_ARROW_THICKNESS, alpha)


def draw_flow_arrow(frame, det: dict, flow_angular: Optional[tuple]) -> None:
    """Draws a one-second-equivalent predicted-flow arrow from this
    detection's base center -- see predicted_flow_angular's doc comment.
    `flow_angular` is precomputed by main()'s own entry-processing loop
    (needed again there for the motion-arrow subtraction, so it's computed
    once and passed in rather than redone here). A no-op (nothing drawn)
    when None, same as every other optional overlay in this file."""
    _draw_angular_rate_arrow(frame, det, flow_angular, FLOW_ARROW_COLOR_BGR, RAW_FLOW_ARROW_ALPHA)


def draw_previous_flow_arrow(frame, det: dict, prev_flow_angular: Optional[tuple]) -> None:
    """Draws the PREVIOUS observation's predicted-flow arrow -- one of the
    two inputs averaged together and subtracted out inside
    compute_motion_arrow_angular (the other being this same frame's own
    flow_angular, drawn in cyan by draw_flow_arrow) -- so a viewer can see
    both halves of that average, not just its result. A no-op when None
    (no previous observation of this track yet)."""
    _draw_angular_rate_arrow(frame, det, prev_flow_angular, PREVIOUS_FLOW_ARROW_COLOR_BGR, RAW_FLOW_ARROW_ALPHA)


def draw_observed_rate_arrow(frame, det: dict, observed_rate_angular: Optional[tuple]) -> None:
    """Draws the RAW observed base-center displacement rate -- the
    un-adjusted quantity compute_motion_arrow_angular starts from, before
    subtracting the averaged predicted flow. Comparing this directly
    against the averaged cyan+green arrows is what makes the motion
    arrow's own subtraction legible: a small magenta residual should mean
    this blue arrow lands close to that average, not that it's small on
    its own (a stationary object under heavy ego-motion parallax can show
    a LARGE observed rate here and still net a near-zero motion arrow,
    correctly). A no-op when None (same gap/validity conditions as the
    motion arrow itself -- see compute_motion_arrow_angular)."""
    _draw_angular_rate_arrow(frame, det, observed_rate_angular, OBSERVED_RATE_ARROW_COLOR_BGR, RAW_FLOW_ARROW_ALPHA)


ANCHOR_DOT_COLOR_BGR = (0, 0, 0)  # black
ANCHOR_DOT_RADIUS_PX = 3


def draw_anchor_dot(frame, det: dict) -> None:
    """Marks base_center_normalized(det) -- the single point every
    constituent arrow (green/cyan/blue/magenta, see
    _draw_angular_rate_arrow) is measured from and drawn out of -- with a
    small filled black dot. Added 2026-08-22, real request, so it's never
    ambiguous which point on the box the vectors correspond to (this
    matters specifically because predicted_flow_angular now declines to
    compute anything at all for an edge-truncated box -- see that
    function's own doc comment -- so a box WITH a dot is one this file is
    actually confident measuring, not just guessing at)."""
    h, w = frame.shape[:2]
    base_col_norm, base_row_norm = base_center_normalized(det)
    cv2.circle(
        frame, _shifted_point((base_col_norm * w, base_row_norm * h)),
        ANCHOR_DOT_RADIUS_PX << PIXEL_SHIFT_BITS, ANCHOR_DOT_COLOR_BGR, -1, cv2.LINE_AA, PIXEL_SHIFT_BITS,
    )


MOTION_ARROW_MAX_DT_S = 1.0  # beyond this gap between observations, the averaged-flow approximation is too stale to trust


def compute_motion_arrow_angular(
    prev_base: tuple, prev_flow: tuple, prev_time: float,
    cur_base: tuple, cur_flow: tuple, cur_time: float,
) -> Optional[tuple]:
    """The tracked object's own independent motion (predicted ego-flow
    subtracted out), as a one-second-equivalent DISPLAY rate in normalized
    angular coordinates -- CONFIRMED 2026-08-22, real proposed algorithm:
    observed base-center displacement between two consecutive real
    observations of the SAME track, minus the average of the predicted
    ego-flow at each observation's own position/depth, integrated over the
    ACTUAL elapsed time between them (not assumed to be 1 second, or
    1/fps -- real capture-time gap, same "use real timestamps, not assumed
    frame intervals" discipline this project already applies everywhere
    else). The 1-second normalization applied at the very end is ONLY for
    visual display (matching the flow arrows' own convention) -- it is NOT
    used, and must never be used, for any actual motion/collision-relevant
    computation; the real quantity here is the angular displacement over
    the real dt, before that final division.

    Done in ANGULAR coordinates, not pixels -- a given angular flow doesn't
    correspond to a constant pixel displacement across the frame (see the
    (1+x^2)/(1+y^2) terms in predicted_flow_angular), so subtracting in
    pixel space would be a worse approximation off-center.

    Returns None if the gap between observations is non-positive or
    exceeds MOTION_ARROW_MAX_DT_S -- beyond that, the object may as well be
    a different resumed identity, and the averaged-flow approximation
    (implicitly assuming near-constant ego-motion across the gap) is too
    stale to trust.
    """
    dt = cur_time - prev_time
    if dt <= 0 or dt > MOTION_ARROW_MAX_DT_S:
        return None
    observed_u = cur_base[0] - prev_base[0]
    observed_v = cur_base[1] - prev_base[1]
    avg_flow_u = (prev_flow[0] + cur_flow[0]) / 2
    avg_flow_v = (prev_flow[1] + cur_flow[1]) / 2
    true_u = observed_u - avg_flow_u * dt
    true_v = observed_v - avg_flow_v * dt
    return true_u / dt, true_v / dt  # 1-second-equivalent DISPLAY rate only


def draw_motion_arrow(frame, det: dict, motion_angular: Optional[tuple]) -> None:
    """Draws the object's own independent-motion arrow from its base
    center -- see compute_motion_arrow_angular's doc comment. A no-op when
    None (no previous observation of this track yet, or the gap was too
    large/invalid)."""
    _draw_angular_rate_arrow(frame, det, motion_angular, MOTION_ARROW_COLOR_BGR, RAW_FLOW_ARROW_ALPHA)


def draw_smoothed_motion_arrow(frame, det: dict, smoothed_motion_angular: Optional[tuple]) -> None:
    """Draws the tracker's own smoothed flow_velocity -- see
    SMOOTHED_MOTION_ARROW_COLOR_BGR's doc comment for why this is a
    separate arrow from draw_motion_arrow's raw one, not a replacement.
    A no-op when None (no track, e.g. an unmatched low-confidence
    detection)."""
    _draw_angular_rate_arrow(frame, det, smoothed_motion_angular, SMOOTHED_MOTION_ARROW_COLOR_BGR)


FOE_DOT_COLOR_BGR = (255, 0, 0)  # blue
FOE_DOT_RADIUS = 8

# Combined FOE-based camera-to-vehicle yaw misalignment estimate -- see
# tools/camera_yaw_alignment.py. Average of the two current-calibration-
# regime sessions' cleanest central estimates: 26_08_20 median -0.608deg,
# 26_08_21 (2-outlier-cleaned) weighted mean -0.605deg -- these two
# independent sessions agreeing this closely is why this figure is trusted
# at all (see CameraManager.defaultYawMarkerNormalizedX's own 2026-08-22
# doc comment for the same figure used there). Positive = FOE right of
# principal point = camera pointed left of true vehicle-forward.
CALIBRATED_YAW_OFFSET_DEG = -0.6065


def calibrated_foe_pixels(frame_width: int, frame_height: int) -> tuple:
    """The FIXED, calibrated Focus of Expansion -- the camera's own static
    mounting misalignment relative to the vehicle (CALIBRATED_YAW_OFFSET_DEG),
    not any one frame's live ego-motion. Deliberately does NOT move frame to
    frame -- CONFIRMED 2026-08-22, real device report ("why does the blue
    dot move around? I'd expect it to be fixed"): an earlier per-frame
    version (using that frame's own live smoothedYawRateDegreesPerSecond)
    showed a real ~23px wobble even during confirmed-straight driving, from
    genuine frame-to-frame yaw-rate noise (steering micro-corrections, road
    camber) -- correct for what IT was computing (the instantaneous
    direction of travel), but the wrong thing to compare the flow arrows
    against as a stable reference. This is the OTHER "FOE" this project
    computes: a single calibration constant, not a live per-frame reading,
    same relationship as CameraManager.defaultYawMarkerNormalizedX's own
    nominal-mount derivation. No vertical calibration offset exists (the
    FOE tool's own analysis only ever measured/trusted the horizontal
    offset -- see its own module docstring), so this sits at the
    calibrated principal ROW, shifted only horizontally."""
    aspect = frame_width / frame_height
    f_row = FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    f_col = f_row / aspect
    yaw_offset_rad = math.radians(CALIBRATED_YAW_OFFSET_DEG)
    col_norm = FLOW_PRINCIPAL_COLUMN_NORMALIZED + math.tan(yaw_offset_rad) * f_col
    row_norm = FLOW_PRINCIPAL_ROW_NORMALIZED
    return col_norm * frame_width, row_norm * frame_height


def draw_foe_dot(frame, entry: dict) -> None:
    """Draws a filled blue dot at the FIXED, calibrated FOE -- see
    calibrated_foe_pixels's doc comment. `entry` is unused (kept so this
    call site doesn't need to change) -- the calibrated FOE is the same
    every frame, unlike the live per-frame version this replaced. Skipped
    (not clamped to the edge) if it ever fell off-frame, though in practice
    it won't -- a sub-1-degree offset lands well inside any real frame."""
    h, w = frame.shape[:2]
    x, y = calibrated_foe_pixels(w, h)
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (int(round(x)), int(round(y))), FOE_DOT_RADIUS, FOE_DOT_COLOR_BGR, -1)


def draw_hud(frame, entry: dict, thermal: Optional[tuple], frame_index: Optional[int] = None) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.0, 2

    def put(text: str, x: int, y: int, color) -> None:
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def right_aligned(text: str) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        return w - tw - 12

    if thermal is not None:
        _, state, percent = thermal
        state_text = "OK" if state == "nominal" else state.upper()
        color = HUD_DEFAULT_COLOR_BGR
        if state == "fair":
            color = HUD_YELLOW_BGR
        elif state in ("serious", "critical"):
            color = HUD_RED_BGR  # blinking on-device isn't reproduced here
        put(f"thermal: {state_text} {percent}%", 12, TOP_ROW_Y, color)

    # Replaced the old "two-pass" line (2026-08-15, by request -- twoPass is
    # stale/no longer a meaningful setting to surface here) with the same
    # "res: ..." reading the live app HUD shows (ContentView.resolutionLabel
    # / modelManager.currentResolutionLabel, logged per-frame here as
    # entry["resolution"]) -- this is the detector's input-buffer size
    # (e.g. "1152x640"/"1920x1088"), NOT the video's raw capture resolution;
    # matching the live screen's own label was the explicit ask, not adding
    # a new reading that has no on-screen counterpart.
    put(f"res: {entry['resolution']}", 12, h - 50, HUD_DEFAULT_COLOR_BGR)
    model_text = MODEL_DISPLAY_NAMES.get(entry["model"], entry["model"])
    put(model_text, 12, h - 12, HUD_DEFAULT_COLOR_BGR)

    low_state = "on" if entry["lowLightEnabled"] else "off"
    low_text = (
        f"low-light: auto ({low_state})" if entry["autoLowLightEnabled"] else f"low-light: {low_state}"
    )
    put(low_text, right_aligned(low_text), h - 12, HUD_DEFAULT_COLOR_BGR)

    stab_text = f"stabilization: {'on' if entry['stabilizationEnabled'] else 'off'}"
    put(stab_text, right_aligned(stab_text), TOP_ROW_Y, HUD_DEFAULT_COLOR_BGR)

    if frame_index is not None:
        # Black-on-white (not the HUD's usual plain color) -- this exists
        # specifically so an exact frame can be referenced back precisely
        # (see the #733/#746 mixup this was added after), so it needs to
        # stay legible against any background, not just blend in as
        # ambient HUD text.
        frame_text = f"frame {frame_index}"
        (tw, _), _ = cv2.getTextSize(frame_text, font, scale, thickness)
        draw_label_text(frame, frame_text, w - tw - 12, TOP_ROW_Y + 40, font_scale=scale, thickness=thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to annotate, named however you like")
    parser.add_argument(
        "--detections", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"detections.jsonl, or a directory of them (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--debug-log", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"overlay-debug.log, or a directory of them, for sync + the thermal HUD line (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <video>-annotated.mp4")
    parser.add_argument(
        "--reid", dest="reid", action="store_true", default=True,
        help="Use appearance/ReID matching in the tracker, in addition to motion/IoU (default: on)",
    )
    parser.add_argument("--no-reid", dest="reid", action="store_false", help="Geometry-only tracking")
    parser.add_argument("--reid-model", default=DEFAULT_REID_MODEL)
    parser.add_argument("--reid-device", default="mps")
    parser.add_argument(
        "--gmc", dest="gmc", action="store_true", default=False,
        help="Compensate track predictions for estimated camera motion (see gmc.py). Off by default -- experimental.",
    )
    parser.add_argument("--no-gmc", dest="gmc", action="store_false")
    parser.add_argument(
        "--flow-arrows", dest="flow_arrows", action="store_true", default=False,
        help="Draw, per tracked object, every constituent of the motion-arrow computation (see "
             "compute_motion_arrow_angular's doc comment), all from its base center: a semi-transparent GREEN "
             "arrow for the "
             "PREVIOUS observation's predicted ego-motion flow, a semi-transparent CYAN arrow for THIS frame's "
             "predicted "
             "ego-motion flow (see predicted_flow_angular's doc comment) -- the flow a STATIC point at that "
             "object's position/depth would show from the camera's own ego-motion alone -- a semi-transparent "
             "BLUE arrow for "
             "the RAW observed base-center displacement rate (unadjusted for ego-motion), and a "
             "semi-transparent MAGENTA arrow "
             "for the object's own independent motion (the blue arrow minus the averaged green+cyan flow, an "
             "EXACT single-step identity, kept deliberately raw/unsmoothed for that reason), plus a solid, "
             "fully opaque RED "
             "arrow for the tracker's own per-class-smoothed, physically-outlier-gated version of that same "
             "quantity (Track.flow_velocity -- the one actually worth trusting as 'is this object really "
             "moving', drawn opaque and in a distinct color, see SMOOTHED_MOTION_ARROW_COLOR_BGR's doc "
             "comment, so it reads clearly at a glance against the four translucent debug arrows); a "
             "small BLACK dot marks the exact point (base_center_normalized) all four raw arrows are measured "
             "from and drawn out of. Also a fixed blue dot at the CALIBRATED Focus of Expansion (see "
             "calibrated_foe_pixels). Every "
             "arrow is a one-second-equivalent DISPLAY rate only (the real motion-arrow math uses the actual "
             "elapsed time between observations, not 1 second). Off by default -- experimental; the arrows "
             "need smoothedPitchRateDegreesPerSecond/smoothedRollRateDegreesPerSecond (added 2026-08-22, not "
             "present in any recording predating that), though the FOE dot alone works on older footage too.",
    )
    parser.add_argument(
        "--smoothed-only", dest="smoothed_only", action="store_true", default=False,
        help="With --flow-arrows, draw ONLY the solid-red smoothed motion arrow (tracker.py's "
             "Track.flow_velocity) plus box/label -- suppresses the four translucent raw debug arrows "
             "(predicted flow x2, observed rate, raw motion) and the FOE/anchor dots, matching "
             "tools/flow_debug_viewer.py's own default (non-selected) display. No effect without "
             "--flow-arrows.",
    )
    parser.add_argument(
        "--flow-debug-json", type=Path, default=None,
        help="Write all five per-track arrow vectors (see --flow-arrows) plus corrected distance, to this "
             "path as JSON -- one entry per detections.jsonl entry, video-relative timestamps -- for "
             "tools/flow_debug_viewer.py's interactive frame-by-frame viewer, instead of/alongside the "
             "baked-in --flow-arrows render. Implies the same per-entry computation --flow-arrows does "
             "(so that need not also be passed), but is NOT any cheaper than a full --flow-arrows render -- "
             "still requires the same full tracker/ReID pass over the whole video regardless of whether "
             "--output is also written.",
    )
    parser.add_argument(
        "--highlight-leading", dest="highlight_leading", action="store_true", default=True,
        help="Tint the classified forward-leading vehicle (see leading_vehicle.py) (default: on)",
    )
    parser.add_argument("--no-highlight-leading", dest="highlight_leading", action="store_false")
    parser.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    parser.add_argument("--band-half-width", type=float, default=DEFAULT_BAND_HALF_WIDTH)
    parser.add_argument("--leading-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--leading-confirm-frames", type=int, default=DEFAULT_CONFIRM_FRAMES)
    parser.add_argument("--leading-grace-frames", type=int, default=DEFAULT_GRACE_FRAMES)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-width", type=float, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--max-bottom-y", type=float, default=DEFAULT_MAX_BOTTOM_Y)
    parser.add_argument("--max-abs-velocity", type=float, default=DEFAULT_MAX_ABS_VELOCITY)
    parser.add_argument("--max-aspect-ratio", type=float, default=DEFAULT_MAX_ASPECT_RATIO)
    parser.add_argument("--min-symmetry", type=float, default=DEFAULT_MIN_SYMMETRY)
    parser.add_argument(
        "--symmetry-cache", type=Path, default=None,
        help="compute_symmetry.py output -- needed for --min-symmetry to actually gate anything.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="Stop after this many seconds of source video, for a quick spot-check render instead of the full clip.",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="label_leading_vehicle.py ground truth (e.g. ground_truth_close_range.json) -- when given, "
             "tints the human-labeled followed vehicle in magenta alongside the algorithm's own pick "
             "(orange), for side-by-side visual review.",
    )
    args = parser.parse_args()

    output = args.output or args.video.with_name(args.video.stem + "-annotated.mp4")
    if output.resolve() == args.video.resolve():
        sys.exit("Refusing to overwrite the source recording — pass a different --output.")

    start_epoch, thermal_log_path = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)

    # CONFIRMED 2026-08-22 (real drive, this same session): --detections
    # defaulting to DEFAULT_LOGS_DIR pulls in EVERY session ever pulled to
    # that shared directory, not just this video's own. The render loop
    # below starts next_idx at 0 and unconditionally consumes every entry
    # whose t <= the current frame's epoch -- so on the very first frame it
    # burns through every entry from any EARLIER, unrelated session before
    # ever reaching this video's own data, calling tracker.update() (a real
    # reid forward pass, if --reid is on) on each one. Cheap enough to miss
    # with reid off; catastrophic with it on -- one real case took 800K+
    # wasted entries from a shared 1.3GB logs directory before reaching a
    # 67-minute video's own ~27K. This doesn't change behavior (the wasted
    # entries get pruned by the tracker's own max_age and never corrupt this
    # video's real tracking, they're just slow) -- just surfaces it instead
    # of silently eating the wall-clock cost. --detections/--debug-log
    # pointed at this video's own session directory avoids the waste
    # entirely.
    n_before = bisect.bisect_left([e["t"] for e in detections], start_epoch)
    if n_before > 2000:
        print(
            f"WARNING: {n_before} of {len(detections)} loaded detections entries are from BEFORE "
            f"this video even starts -- {args.detections} likely wasn't scoped to this session. "
            "Pass --detections/--debug-log pointed at this video's own session directory to avoid "
            "burning through them (slow, especially with --reid on).",
            file=sys.stderr,
        )

    symmetry_scores = json.loads(args.symmetry_cache.read_text()) if args.symmetry_cache else None
    ground_truth_segments = json.loads(args.ground_truth.read_text()) if args.ground_truth else None

    reid_encoder = build_reid_encoder(args.reid_model, device=args.reid_device) if args.reid else None
    tracker = ByteTracker(reid_encoder=reid_encoder, use_gmc=args.gmc)
    leading_lock = LeadingVehicleLock(args.leading_confirm_frames, args.leading_grace_frames)
    leading_velocity = VelocityEstimator()

    thermal = load_thermal(thermal_log_path)
    thermal_keys = build_key_index(thermal, lambda e: e[0])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"Couldn't open {output} for writing")

    # Tracking runs inline with this same sequential decode, not as a separate
    # seek-based pre-pass -- a pre-pass that re-seeks the video once per
    # detection entry (rather than once per raw frame here) turned out to
    # cost 30-40x more per lookup than a plain sequential read (measured: a
    # 5-6x slowdown overall on a real session), because OpenCV/ffmpeg must
    # decode forward from the nearest keyframe on every arbitrary seek. Each
    # entry is instead fed to the tracker using whichever already-decoded
    # frame is current when that entry's own timestamp is first reached --
    # entries land close enough to this video's own frame rate that this is
    # a fine approximation of the frame the entry actually corresponds to.
    next_idx = 0
    current_entry, current_track_ids = None, []
    # Persists across video frames the same way current_entry does -- the
    # lock is only updated when a *new* detection entry is processed below,
    # at the detections.jsonl cadence, not once per rendered video frame.
    current_locked_id = None
    # trackID -> (base_angular, flow_angular, capture_time) from that
    # track's MOST RECENT prior observation -- see
    # compute_motion_arrow_angular's doc comment. Only ever updated when a
    # full (base + flow) observation exists; a track with missing flow
    # signals this entry simply doesn't advance its stored state (see the
    # entry-processing loop below), same "don't fabricate a number with no
    # real basis" discipline as the rest of this feature.
    track_flow_state: dict = {}
    flow_debug_frames: list = []
    aspect = width / height

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_seconds is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > args.max_seconds:
            break
        # Actual embedded PTS for the frame just read (ms, relative to the
        # start of the recording) — not `i / fps`, which would assume a
        # perfectly constant frame rate and can drift over a long drive if
        # any frames were ever dropped or timing wasn't perfectly steady.
        frame_epoch = start_epoch + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        while next_idx < len(detections) and detections[next_idx]["t"] <= frame_epoch:
            current_entry = detections[next_idx]
            # Real elapsed-time-corrected capture time (see below for why
            # not current_entry["t"] alone) -- needed by the tracker's own
            # flow-based matching gate now too, not just the flow-arrows
            # visualization, so it's computed unconditionally here instead
            # of only under `if args.flow_arrows`.
            capture_time = current_entry["t"] - current_entry["elapsedMs"] / 1000.0
            detection_depths = [
                corrected_distance_meters(det, current_entry, aspect) for det in current_entry["detections"]
            ]
            current_track_ids = tracker.update(
                current_entry["detections"], frame=frame,
                capture_time=capture_time,
                ego_speed_mps=current_entry.get("egoSpeedMps"),
                yaw_rate_deg_s=current_entry.get("smoothedYawRateDegreesPerSecond"),
                pitch_rate_deg_s=current_entry.get("smoothedPitchRateDegreesPerSecond"),
                roll_rate_deg_s=current_entry.get("smoothedRollRateDegreesPerSecond"),
                aspect=aspect,
                detection_depths=detection_depths,
            )
            # classify_leading reads trackID off each det -- draw_box below
            # only takes it as a separate zip'd arg, so stamp it on here too.
            for det, track_id in zip(current_entry["detections"], current_track_ids):
                det["trackID"] = track_id
            if args.flow_arrows or args.flow_debug_json:
                # Computed once per detections.jsonl entry (not once per
                # rendered video frame, matching everything else in this
                # loop) and stashed on each det -- draw_flow_arrow/
                # draw_motion_arrow below just read it back, no
                # recomputation at render time.
                for det, track_id in zip(current_entry["detections"], current_track_ids):
                    # Prefer the track's own EMA-smoothed depth over a fresh
                    # single-frame corrected_distance_meters call -- see
                    # predicted_flow_angular's own doc comment on
                    # z_override for why (single-frame depth noise feeding
                    # a jumpy flow prediction for an object that isn't
                    # actually moving jumpily).
                    smoothed_z = None
                    track = None
                    if track_id is not None:
                        track = tracker.get_track(track_id)
                        smoothed_z = track.flow_z if track is not None else None
                    flow_angular = predicted_flow_angular(det, current_entry, aspect, z_override=smoothed_z)
                    det["flowAngular"] = flow_angular
                    det["motionAngular"] = None
                    det["prevFlowAngular"] = None
                    det["observedRateAngular"] = None
                    # The tracker's own per-class-EMA + physical-outlier-
                    # gated estimate (see Track.update_flow_state) --
                    # already updated for this entry by tracker.update()
                    # above, so this just reads it back, no recomputation.
                    # A no-op (0.0, 0.0) is Track's own "assumed
                    # stationary" cold-start default, not missing data.
                    det["smoothedMotionAngular"] = track.flow_velocity if track is not None else None
                    if track_id is None:
                        continue
                    base_angular = angular_coords(*base_center_normalized(det), aspect)
                    prev = track_flow_state.get(track_id)
                    if prev is not None and flow_angular is not None:
                        prev_base, prev_flow, prev_time = prev
                        det["motionAngular"] = compute_motion_arrow_angular(
                            prev_base, prev_flow, prev_time, base_angular, flow_angular, capture_time,
                        )
                        if det["motionAngular"] is not None:
                            # Same dt validity gate compute_motion_arrow_angular
                            # itself just passed -- see draw_previous_flow_arrow/
                            # draw_observed_rate_arrow's own doc comments for why
                            # these are worth showing alongside the final result.
                            dt = capture_time - prev_time
                            det["prevFlowAngular"] = prev_flow
                            det["observedRateAngular"] = (
                                (base_angular[0] - prev_base[0]) / dt,
                                (base_angular[1] - prev_base[1]) / dt,
                            )
                    if flow_angular is not None:
                        track_flow_state[track_id] = (base_angular, flow_angular, capture_time)
            if symmetry_scores is not None:
                # Aligned by list index, same order as load_detections -- see
                # compute_symmetry.py's docstring. NOT trackID-keyed.
                for det, sym in zip(current_entry["detections"], symmetry_scores[next_idx]):
                    det["sym"] = sym
            if args.highlight_leading:
                leading_velocity.update(current_entry["detections"], current_entry["t"])
                raw_leading = classify_leading(
                    current_entry["detections"], args.center_x, args.band_half_width, args.leading_threshold,
                    min_confidence=args.min_confidence, min_width=args.min_width,
                    max_bottom_y=args.max_bottom_y, max_abs_velocity=args.max_abs_velocity,
                    max_aspect_ratio=args.max_aspect_ratio, min_symmetry=args.min_symmetry,
                )
                current_locked_id = leading_lock.update(
                    raw_leading["trackID"] if raw_leading is not None else None
                )
            if args.flow_debug_json:
                # Video-relative "t" (not the raw epoch current_entry["t"])
                # so tools/flow_debug_viewer.py's frontend can binary-search
                # this the exact same way label_leading_vehicle_frontend
                # .html already does against <video>.currentTime -- no
                # separate epoch/start_epoch bookkeeping needed client-side.
                # Placed AFTER the highlight_leading block above (not
                # before, where it originally sat) so lockedLeadingTrackID
                # reflects THIS entry's own classification, not the
                # previous entry's -- an off-by-one that would otherwise
                # highlight the wrong frame's leading vehicle in the viewer.
                thermal_entry_for_json = nearest_at_or_before(thermal, thermal_keys, frame_epoch) if thermal else None
                flow_debug_frames.append({
                    "t": current_entry["t"] - start_epoch,
                    # Same fields/leading text as draw_hud's on-screen HUD
                    # (which itself mirrors the live on-device app's own
                    # HUD labels -- see draw_hud's own doc comments) --
                    # the frontend reproduces these verbatim rather than
                    # inventing its own wording, real request 2026-08-23.
                    "resolution": current_entry["resolution"],
                    "model": MODEL_DISPLAY_NAMES.get(current_entry["model"], current_entry["model"]),
                    "lowLightEnabled": current_entry["lowLightEnabled"],
                    "autoLowLightEnabled": current_entry["autoLowLightEnabled"],
                    "stabilizationEnabled": current_entry["stabilizationEnabled"],
                    "thermal": (
                        {"state": thermal_entry_for_json[1], "percent": thermal_entry_for_json[2]}
                        if thermal_entry_for_json is not None else None
                    ),
                    "lockedLeadingTrackID": current_locked_id if args.highlight_leading else None,
                    "detections": [
                        {
                            "trackID": track_id, "label": det["label"], "confidence": det["confidence"],
                            "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"],
                            "correctedDistanceMeters": corrected_distance_meters(det, current_entry, aspect),
                            "widthOverridden": det.get("distanceMetersIsWidthOverridden", False),
                            "flowAngular": det.get("flowAngular"),
                            "prevFlowAngular": det.get("prevFlowAngular"),
                            "observedRateAngular": det.get("observedRateAngular"),
                            "motionAngular": det.get("motionAngular"),
                            "smoothedMotionAngular": det.get("smoothedMotionAngular"),
                        }
                        for det, track_id in zip(current_entry["detections"], current_track_ids)
                    ],
                })
            next_idx += 1

        if current_entry is not None:
            for det, track_id in zip(current_entry["detections"], current_track_ids):
                draw_box(frame, det, track_id, corrected_distance_meters(det, current_entry, aspect))
                if args.flow_arrows:
                    if not args.smoothed_only:
                        draw_previous_flow_arrow(frame, det, det.get("prevFlowAngular"))
                        draw_flow_arrow(frame, det, det.get("flowAngular"))
                        draw_observed_rate_arrow(frame, det, det.get("observedRateAngular"))
                        draw_motion_arrow(frame, det, det.get("motionAngular"))
                        if det.get("observedRateAngular") is not None:
                            draw_anchor_dot(frame, det)
                    draw_smoothed_motion_arrow(frame, det, det.get("smoothedMotionAngular"))
            if args.flow_arrows and not args.smoothed_only:
                draw_foe_dot(frame, current_entry)

            if args.highlight_leading and current_locked_id is not None:
                # The locked vehicle may not be this exact entry's raw
                # classify_leading winner (grace-period stickiness) -- look
                # it up by trackID among this entry's own detections instead.
                # If it's not there at all (mid-gap, nothing detected for it
                # this entry), skip drawing rather than fabricate a box.
                locked_det = next(
                    (d for d in current_entry["detections"] if d.get("trackID") == current_locked_id), None
                )
                if locked_det is not None:
                    draw_tint(frame, locked_det, ALGORITHM_TINT_BGR, "ALGORITHM", label_below=True)

            if ground_truth_segments is not None:
                # Drawn AFTER the algorithm tint (not before) -- a labeled
                # vehicle should always render on top when both tints land on
                # the same box, since the human label is the ground truth
                # being checked against, not a peer of equal visual priority.
                t_rel = current_entry["t"] - start_epoch
                gt_track_id = ground_truth_at(ground_truth_segments, t_rel)
                if gt_track_id is not None:
                    gt_det = next(
                        (d for d in current_entry["detections"] if d.get("trackID") == gt_track_id), None
                    )
                    if gt_det is not None:
                        draw_tint(frame, gt_det, GROUND_TRUTH_TINT_BGR, "LABELED", label_below=False)

            thermal_entry = nearest_at_or_before(thermal, thermal_keys, frame_epoch)
            draw_hud(frame, current_entry, thermal_entry, frame_index=i)

        writer.write(frame)
        i += 1
        if i % max(1, int(fps) * 30) == 0:
            # Not shown as a fraction of frame_count: OpenCV's frame-count
            # estimate for a live-recorded/fragmented .mov is unreliable (seen
            # under-reporting by 3x on a real file), so this is just an
            # elapsed-progress heartbeat, not a percentage.
            print(f"  {i} frames processed ({i / fps:.0f}s of video)", file=sys.stderr)

    cap.release()
    writer.release()
    print(f"Wrote {i} frames to {output}")
    print(f"Original recording untouched: {args.video}")
    if args.flow_debug_json:
        args.flow_debug_json.write_text(json.dumps({
            # The video's own raw CAPTURE resolution (e.g. "3840x2160" for
            # 4K, "1920x1080" for 1080p) -- distinct from each frame's own
            # "resolution" field below, which is the DETECTOR's input
            # buffer size (e.g. "1152x640"), not the recording's real
            # resolution. Constant for the whole session, so top-level
            # rather than repeated per frame.
            "videoResolution": f"{width}x{height}",
            "videoFps": fps,
            "frames": flow_debug_frames,
        }))
        print(f"Wrote {len(flow_debug_frames)} entries to {args.flow_debug_json}")


if __name__ == "__main__":
    main()
