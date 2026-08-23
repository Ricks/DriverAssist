#!/usr/bin/env python3
"""
Shared calibrated-camera optic-flow model: ego-motion-only predicted flow at
a given angular position/depth, plus the angular<->normalized-screen
coordinate conversions it's expressed in.

Extracted from reconstruct_annotated.py (2026-08-22) specifically so
tracker.py can also use it, as a matching-time position-prediction gate --
tracker.py is imported BY reconstruct_annotated.py, so the flow model
couldn't stay defined there without a circular import. Single source of
truth for both consumers; see reconstruct_annotated.py's git history for
predicted_flow_angular's original, extensively-commented derivation and
per-signal sign-convention cross-checks (Vz/Vx from the yaw lever-arm,
Vy=0 -- a known, deliberately-uncorrected gap, see the
project_motion_arrow_vertical_bias memory -- wy/wx/wz negated smoothed
rates against already-validated conventions elsewhere in this project) --
not repeated here to keep this file terse; the math below is unchanged from
that derivation, just relocated.
"""
import math

# Calibrated intrinsics -- DistanceEstimator.calibrated / camera_yaw_alignment.py's
# own copy of the same fitted values. Row-normalized: fraction of frame
# HEIGHT for the row-axis focal length/principal row, fraction of frame
# WIDTH for principal column (assumed, not fitted -- see DistanceEstimator's
# own doc comment on assumedPrincipalColumnNormalized).
FOCAL_LENGTH_ROW_NORMALIZED = 1.322673
PRINCIPAL_ROW_NORMALIZED = 0.481681
PRINCIPAL_COLUMN_NORMALIZED = 0.5

# DistanceEstimator.LeverArm -- camera's fixed offset from the rear axle
# (forward, left), meters.
LEVER_ARM_FORWARD_M = 2.17
LEVER_ARM_LEFT_M = 0.26


def camera_velocity_from_yaw(smoothed_yaw_rate_deg_s: float) -> tuple:
    """Ports DistanceEstimator.LeverArm.cameraVelocityFromYaw (Swift) --
    the camera's own velocity due to the vehicle's yaw rotation about the
    rear axle (v = omega x leverArm), in the vehicle's body frame (forward,
    left), meters/second. Takes the SMOOTHED yaw rate, not raw -- see that
    Swift function's own doc comment for why (raw is too noisy for a
    per-frame consumer, confirmed against a real drive)."""
    omega = math.radians(smoothed_yaw_rate_deg_s)
    return (-omega * LEVER_ARM_LEFT_M, omega * LEVER_ARM_FORWARD_M)  # (forward, left)


def angular_coords(col_norm: float, row_norm: float, aspect: float) -> tuple:
    """Converts a normalized [0,1] screen position to this project's
    calibrated-camera angular coordinate space -- the SAME parameterization
    DistanceEstimator's own de-roll step uses: x=(col-principalCol)*aspect/f,
    y=(row-principalRow)/f. A flow/displacement of 1.0 in this space is one
    focal length of angular motion; multiply by f_col*frame_width (x) or
    f_row*frame_height (y) to get pixels."""
    f_row = FOCAL_LENGTH_ROW_NORMALIZED
    x = (col_norm - PRINCIPAL_COLUMN_NORMALIZED) * aspect / f_row
    y = (row_norm - PRINCIPAL_ROW_NORMALIZED) / f_row
    return x, y


def inverse_angular_coords(x: float, y: float, aspect: float) -> tuple:
    """Inverse of angular_coords -- converts a calibrated angular (x, y)
    back to a normalized [0,1] screen (col, row). Added for tracker.py's
    flow-based matching gate: it predicts a track's next position in
    angular space (where the flow physics is valid), then needs that
    position back in screen space to compare against a real detection's
    box."""
    f_row = FOCAL_LENGTH_ROW_NORMALIZED
    col_norm = PRINCIPAL_COLUMN_NORMALIZED + x * f_row / aspect
    row_norm = PRINCIPAL_ROW_NORMALIZED + y * f_row
    return col_norm, row_norm


def predicted_flow_angular_raw(
    x: float, y: float, z: float,
    ego_speed_mps: float, yaw_rate_deg_s: float, pitch_rate_deg_s: float, roll_rate_deg_s: float,
) -> tuple:
    """Predicted per-second optic flow (u, v) in normalized ANGULAR
    coordinates at angular position (x, y), depth z meters, for a point
    STATIC in the world, due to the camera's OWN ego-motion alone (the
    standard Longuet-Higgins/Prazdny calibrated-camera flow decomposition).
    Pure-math core, taking raw scalars instead of det/entry dicts, so both
    reconstruct_annotated.py (evaluated at a detection's own position/depth,
    for the flow/motion-arrow visualization) and tracker.py (evaluated at a
    TRACK's last known position/depth, stepped forward to predict where it
    should be next frame, for matching) can share one implementation.
    Caller's responsibility: z > 0, and all four motion signals non-None."""
    lever_forward, lever_left = camera_velocity_from_yaw(yaw_rate_deg_s)
    tz = ego_speed_mps + lever_forward
    vx = -lever_left
    vy = 0.0

    omega_x = -math.radians(pitch_rate_deg_s)
    omega_y = -math.radians(yaw_rate_deg_s)
    omega_z = -math.radians(roll_rate_deg_s)

    u = (-vx + x * tz) / z - omega_y * (1 + x * x) + omega_z * y + omega_x * x * y
    v = (-vy + y * tz) / z + omega_x * (1 + y * y) - omega_y * x * y - omega_z * x
    return u, v
