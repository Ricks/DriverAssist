#!/usr/bin/env python3
"""
General-purpose "is this object in my vehicle's path" scoring -- deliberately
NOT specific to the leading-vehicle classifier. Any consumer that needs to
reason about path membership (leading_vehicle.py's classify_leading today,
a future pedestrian/cyclist collision-warning module, or any other hazard
type) should call into this module rather than duplicate the logic, since
the underlying question -- "does this image position correspond to
somewhere my vehicle is headed" -- doesn't depend on what kind of object is
there. Object-type-specific reasoning (vehicle shape gates, pedestrian size
assumptions, etc.) belongs in each consumer, layered on top of this.

Three things live here:
  - lateral_mount_offset_shift: shifts where "straight ahead" is in image
    space to account for the camera's own KNOWN, FIXED lateral offset from
    the vehicle's true centerline (DistanceEstimator.swift's LeverArm,
    r_y=0.26m) -- unlike the yaw-rate shift below, this doesn't depend on
    any live sensor reading, just the candidate's own estimated distance
    (already logged per-detection as distanceMeters), so it's exact (to the
    extent the lever-arm measurement and pinhole model are), not a
    placeholder pending validation data.
  - curve_adjusted_center_x: shifts where "straight ahead" is in image space
    based on current yaw rate, so a curve doesn't get treated as "everything
    drifted off to the side."
  - path_probability: a continuous [0, 1] score for how well a specific
    image x-position matches that (possibly curve-shifted) path center, for
    consumers that want a soft signal rather than a hard in/out gate (this is
    what was asked for as "an input to the followed-vehicle algo" rather than
    a replacement for its own gates).

curve_adjusted_center_x is the simple, image-space heuristic version --
shifting directly in normalized pixel-x space based on yaw rate alone -- not
a full 3D geometric model (real-world distance + camera field-of-view +
turning radius converted back to image space). That fuller model would be
more physically correct, but it needs calibration this app didn't have when
this reasoning was first written (camera horizontal FOV was never measured,
on top of the ground-plane distance calibration that was pending the mount
at the time). Stacking multiple unvalidated calibrations at once means a bad
result can't be attributed to any one of them -- same reasoning
classify_leading's own central-band test was chosen over IPM for (see that
module's docstring): an uncalibrated "principled" model is worse than a
simple one. Revisit the full geometric version for yaw only if real data
from this heuristic shows a gap it can't close -- same posture as the
standing CoreMotion-vs-vision-GMC question.

lateral_mount_offset_shift is DIFFERENT: the ground-plane distance
calibration that reasoning above deferred to now exists (DistanceEstimator's
`calibrated` instance, refit 2026-08-15 for the SHAPE clamp mount) and is
already logged per-detection as distanceMeters -- no NEW calibration is
being stacked on, just an already-trusted one (also already reused for an
analogous horizontal-shift computation, see ByteTracker.yawFallbackTransform)
applied to a value (r_y) that's a physical tape-measure fact, not a live
signal. So this one isn't a placeholder-pending-validation the way
curve_adjusted_center_x's constants are -- it's exact to the extent the
lever-arm measurement, focal-length calibration, and pinhole model are,
which the following-distance work already leans on for its own accuracy
claims.

curve_adjusted_center_x is NOT YET VALIDATED against real data -- no
recorded session has rotation rate logged yet (added to DetectionLogger.swift
after every session recorded so far, including the one all the
leading-vehicle validation in this file's sibling tools was done against).
Its constants below are placeholders pending real yaw-rate-vs-path-error
data from an actual drive with turns -- don't trust them blindly; tune once
that data exists (see tune_leading_vehicle.py's CLASSIFY_KEYS/SEARCH_RANGES,
which already covers these).
"""
import math

# DistanceEstimator.swift's LeverArm.leftOfCenterlineMeters -- MEASURED
# 2026-08-15 with a tape measure for the SHAPE clamp mount, camera sits this
# far to the DRIVER'S LEFT of the car's true centerline. Kept as a separate
# copy rather than importing across the Swift/Python boundary (there isn't
# one) -- if the mount moves again, update both together, same as every
# other duplicated-by-necessity constant in this codebase (e.g.
# DistanceEstimator.calibrated's own history of superseded mount figures).
DEFAULT_LEFT_OF_CENTERLINE_METERS = 0.26

# DistanceEstimator.calibrated.focalLengthNormalized -- refit 2026-08-15
# alongside the lever-arm measurement above, for the same SHAPE clamp mount.
DEFAULT_FOCAL_LENGTH_NORMALIZED = 1.322673

# Placeholder pending real yaw-rate-vs-band-error data from an actual drive
# with turns.
DEFAULT_YAW_SHIFT_PER_DEG_S = 0.01

# Beyond this, "in front of me" stops being a meaningful concept anyway
# (sharp turns, parking maneuvers) -- clamps the shift so those don't push
# the path center somewhere nonsensical.
DEFAULT_MAX_YAW_SHIFT = 0.15

# Standard deviation of the Gaussian path_probability score, in units of
# band_half_width -- 1.0 means "one band-width off center" scores ~0.61,
# matching a similar falloff shape to classify_leading's existing hard
# band test without inheriting its vehicle-specific fully-contained-vs-
# straddling distinction (that's about a wide box's two edges, which
# doesn't generalize to a point-like pedestrian).
DEFAULT_SCORE_SIGMA_IN_BAND_WIDTHS = 1.0


def lateral_mount_offset_shift(
    distance_meters, aspect_ratio: float,
    left_of_centerline_meters: float = DEFAULT_LEFT_OF_CENTERLINE_METERS,
    focal_length_normalized: float = DEFAULT_FOCAL_LENGTH_NORMALIZED,
) -> float:
    """How far "straight ahead of the CAR" (not the camera) sits from image
    x=0.5, for a candidate at `distance_meters`. Positive (camera left of
    centerline) means the car's true centerline appears to the RIGHT of
    frame-center from the camera's own vantage point, so this shift is
    ADDED to whatever base center_x is in use, not subtracted.

    Derivation: a ground point on the car's centerline is `distance_meters`
    ahead and `left_of_centerline_meters` to the RIGHT of the camera (the
    camera sits that far to the car's left), so its bearing off the
    camera's own optical axis is atan(left_of_centerline_meters /
    distance_meters). DistanceEstimator.swift's pinhole model relates a
    bearing angle theta to a normalized image offset via
    focal_length_column_normalized * tan(theta) -- and since
    tan(atan(x)) == x exactly, the two inverse trig functions cancel and
    this reduces to a plain ratio, no trig calls needed:
        shift = (focal_length_normalized / aspect_ratio)
                * left_of_centerline_meters / distance_meters
    `aspect_ratio` is width/height of the frame the candidate's box is
    normalized against (e.g. 1152/640) -- NOT a fixed constant, varies with
    ModelManager's resolution mode, same caveat DistanceEstimator.swift's
    own focalLengthColumnNormalized carries; pass whichever was actually
    active for this detection (see DetectionLogEntry.resolution).

    Returns 0.0 if `distance_meters` is None, non-positive, or otherwise
    not a usable ground-plane distance (e.g. the detection's own
    distanceMeters was nil in the log, same "at/above the horizon, straight
    down" cases DistanceEstimator.distanceMeters itself returns nil for) --
    falls back to the un-shifted base center rather than raising or
    guessing a distance, since a missing distance is a real, expected case
    (a box that failed the ground-plane geometry check), not a bug.
    """
    if distance_meters is None or distance_meters <= 0:
        return 0.0
    focal_length_column_normalized = focal_length_normalized / aspect_ratio
    return focal_length_column_normalized * left_of_centerline_meters / distance_meters


def curve_adjusted_center_x(
    base_center_x: float, signed_yaw_rate_deg_s: float,
    shift_per_deg_s: float = DEFAULT_YAW_SHIFT_PER_DEG_S, max_shift: float = DEFAULT_MAX_YAW_SHIFT,
) -> float:
    """Shifts where "straight ahead" is in image space based on current yaw
    rate, so the path center tracks where the road actually goes during a
    curve, rather than assuming straight-ahead always means x=0.5.

    `signed_yaw_rate_deg_s` must already be resolved to the correct
    CoreMotion axis, with a sign convention of positive = turning right (path
    center shifts right) -- that mapping isn't confirmed yet for any real
    mount (see PitchSensor.swift's own axis/sign caveat), so this function is
    deliberately axis-agnostic and takes the resolved scalar as input rather
    than guessing which raw axis to read.
    """
    shift = signed_yaw_rate_deg_s * shift_per_deg_s
    shift = max(-max_shift, min(max_shift, shift))
    return base_center_x + shift


def path_probability(
    object_center_x: float, path_center_x: float, band_half_width: float,
    sigma_in_band_widths: float = DEFAULT_SCORE_SIGMA_IN_BAND_WIDTHS,
) -> float:
    """Continuous [0, 1] score for how well `object_center_x` (the object's
    own horizontal center, not its full box -- deliberately point-like so
    this works the same for a wide vehicle or a narrow pedestrian) matches
    `path_center_x` (typically curve_adjusted_center_x's output). 1.0 at
    exact match, smoothly falling off with horizontal distance -- a soft
    signal for consumers that want to weigh/rank candidates, not just gate
    them in/out.

    This is a shape, not a calibrated probability -- there's no real data
    yet to calibrate it against (see module docstring). Treat the actual
    numeric value as "higher is better," not as a real likelihood, until
    validated.
    """
    sigma = band_half_width * sigma_in_band_widths
    if sigma <= 0:
        return 1.0 if object_center_x == path_center_x else 0.0
    return math.exp(-0.5 * ((object_center_x - path_center_x) / sigma) ** 2)
