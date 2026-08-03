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

Two things live here:
  - curve_adjusted_center_x: shifts where "straight ahead" is in image space
    based on current yaw rate, so a curve doesn't get treated as "everything
    drifted off to the side."
  - path_probability: a continuous [0, 1] score for how well a specific
    image x-position matches that (possibly curve-shifted) path center, for
    consumers that want a soft signal rather than a hard in/out gate (this is
    what was asked for as "an input to the followed-vehicle algo" rather than
    a replacement for its own gates).

Both are the simple, image-space heuristic version -- shifting/scoring
directly in normalized pixel-x space based on yaw rate alone -- not a full
3D geometric model (real-world distance + camera field-of-view + turning
radius converted back to image space). That fuller model would be more
physically correct, but it needs calibration this app doesn't have yet
(camera horizontal FOV was never measured, on top of the ground-plane
distance calibration that's already pending the mount) -- stacking three
unvalidated calibrations at once means a bad result can't be attributed to
any one of them. Same reasoning classify_leading's own central-band test
was chosen over IPM for (see that module's docstring): an uncalibrated
"principled" model is worse than a simple one. Revisit the full geometric
version only if real data from this heuristic shows a gap it can't close --
same posture as the standing CoreMotion-vs-vision-GMC question.

NOT YET VALIDATED against real data -- no recorded session has rotation
rate logged yet (added to DetectionLogger.swift after every session
recorded so far, including the one all the leading-vehicle validation in
this file's sibling tools was done against). The constants below are
placeholders pending real yaw-rate-vs-path-error data from an actual drive
with turns -- don't trust them blindly; tune once that data exists (see
tune_leading_vehicle.py's CLASSIFY_KEYS/SEARCH_RANGES, which already
covers these).
"""
import math

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
