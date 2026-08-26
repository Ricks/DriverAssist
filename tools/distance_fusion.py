"""Probabilistic distance fusion -- replaces the binary row-based/width-based
switch (with its confirm/grace-frame hysteresis) with an inverse-variance
weighted combination of both estimates, each carrying its own uncertainty.

WHY THIS EXISTS: real 2026-08-24 data surfaced two failure modes the binary
switch can't represent well:
  1. A "bus" detection read as 1962m (phi=0.03 deg, a box bottom sitting
     ~30px below the calibrated horizon) with zero fallback, since bus had
     no width prior at the time. Row-based error isn't constant -- it's
     driven by d(distance)/d(phi) = -H/sin^2(phi), which blows up smoothly
     near the horizon. A continuous uncertainty model says "don't trust
     this" automatically, without needing a hand-picked cutoff angle.
  2. Two adjacent real cars (tracks #52/#57, same session) showed
     wildly different distances at one frame (61.1 vs 15.0 yds) purely
     because one track's confirm-frame streak had completed a few frames
     before the other's -- the SAME underlying noise, filtered through a
     binary hysteresis gate, produced a visible discontinuity between two
     objects that were actually the same true distance apart. Continuous
     variance-weighting has no discrete state to flicker.

WHAT THIS DOES NOT REPLACE: bias sources (oblique viewing angle, hood
truncation, occlusion by another object) are NOT noise -- inflating sigma
around a systematically wrong center doesn't fix a wrong center. Those stay
as separate gates that this module consumes as inputs (excluding a term
from the fusion, or inflating its sigma toward infinity), not something the
variance math absorbs on its own. See this file's own functions' doc
comments for which gates are real (oblique -- reuses is_oblique_view) vs.
still a gap (occlusion-by-other-object -- NOT built, see the
2026-08-24 chat discussion; not attempted here).

CALIBRATION STATUS, honestly, per term:
  - SIGMA_EDGE_ROW_PX: REAL, back-solved from the 8-point cone-calibration
    data (calibration/cone_calibration_points.json, known true distances)
    -- see calibrate_sigma_edge_row_px() below, which reproduces the
    computation this constant came from. 2.28px, mean bias +0.09px
    (~unbiased, consistent with a noise model rather than a hidden bias).
  - Real per-class width variance (KITTI): REAL, already in
    reconstruct_annotated.py's OBJECT_ASPECT_PRIORS.
  - SIGMA_EDGE_WIDTH_PX: REAL, back-solved 2026-08-25 from 10 genuinely
    stationary-car tracks mined across 4 already-recorded sessions -- see
    find_stationary_car_width_jitter() below. 1.075px, reassuringly close
    to (a bit smaller than) SIGMA_EDGE_ROW_PX's 2.28px, confirming rather
    than overturning the "same underlying phenomenon, different edge"
    assumption the old stand-in was built on.
  - Hood-truncation angular bins: REAL, 2026-08-26, from the
    26_08_26_Day_Walk garage session (717 samples, 8 walk passes, CURRENT
    mount) -- 7 bins (not 5, see HOOD_TRUNCATION_COLUMN_BINS), each with
    78-184 samples, storing alpha (pitch-independent) not phi. Confirmed
    real, non-flat curvature (not just noise): alpha rises toward both
    edges from a low near center, and per-bin std is much tighter than the
    old pooled global figure once bearing is accounted for. Still a single
    session/mount generation's worth of data -- see this constant's own
    doc comment for the reasoning and what would justify revisiting it.
  - SIGMA_ROLL_FLATNESS_DEGREES / SIGMA_PITCH_FLATNESS_DEGREES: a
    LITERATURE-ANCHORED PRIOR, not a fitted value -- exactly one real
    measurement exists (26_08_25_Day_Cones: +1.78deg roll / +0.24deg
    pitch delta from that session's stored reference), cross-checked
    against published road/parking-lot cross-slope design figures (see
    SIGMA_ROLL_FLATNESS_DEGREES's own doc comment for sources). An
    attempt to mine many existing driving sessions for more samples
    (flatness_uncertainty_calibration.py) turned out to be dominated by
    traffic-occlusion noise, not usable as-is -- abandoned, not deleted,
    in case occlusion-aware filtering makes it viable later. Revisit
    these two constants first once real multi-location data exists.

OVERALL STATUS, 2026-08-26: every numeric parameter above is now either
REAL (measured against real data) or a clearly-labeled, literature-
anchored PRIOR -- no more raw stand-ins masquerading as calibrated values.
Two real gaps remain, and neither is closable from a desk-side data-mining
pass the way SIGMA_EDGE_WIDTH_PX was:
  1. Occlusion by another object -- no detection mechanism exists at all
     (see WHAT THIS DOES NOT REPLACE above). Not hypothetical: the
     abandoned flatness_uncertainty_calibration.py attempt ran straight
     into it as the dominant confound in ordinary traffic, so this is a
     confirmed, real gap, not just a documented one.
  2. The flatness priors (SIGMA_ROLL_FLATNESS_DEGREES/
     SIGMA_PITCH_FLATNESS_DEGREES) -- still exactly one real location's
     worth of data, a defensible literature-anchored starting point, not
     a calibrated distribution.
Also worth being clear-eyed about: this module is still NOT wired into
anything that produces a distance a driver, or even reconstruct_
annotated.py's own displayed/exported distances, would actually see --
everything above has been validated through ad-hoc test scripts against
real cone-test and drive data, not through use. "Calibrated" and
"deployed" are still two different claims.
"""

import json
import math
import statistics
from pathlib import Path
from typing import Optional

import reconstruct_annotated as recon

# --- σ_row(phi): row-based distance uncertainty ---------------------------

SIGMA_EDGE_ROW_PX = 2.2777637089935387
"""Real, back-solved from calibration/cone_calibration_points.json -- see
calibrate_sigma_edge_row_px() below, which reproduces this exact number.
Effective pixel-level uncertainty in where a box's bottom edge really sits,
in a 1080px-tall frame."""

FRAME_HEIGHT_PX_REFERENCE = 1080.0
"""The frame height SIGMA_EDGE_ROW_PX was calibrated against -- pixel
counts don't matter for the NORMALIZED math below (sigma_edge_row_normalized
is resolution-independent), but this is kept explicit since a genuinely
different sensor/format could have different real pixel-level detector
precision, not just a different pixel count for the same precision."""


def calibrate_sigma_edge_row_px(
    points_path: Path = Path(__file__).parent.parent / "calibration" / "cone_calibration_points.json",
    reference_pitch_deg: float = -1.5137,
    reference_roll_deg: float = -0.4506,
    aspect: float = 1920 / 1080,
    frame_height_px: float = 1080.0,
) -> tuple:
    """Reproduces the calibration of SIGMA_EDGE_ROW_PX from the real 8-point
    cone data -- run this again (and update the constant above) if the cone
    dataset ever grows or the mount changes enough to need a refit. For each
    point: back out the angle phi the COMPUTED distance implies, compare
    against the angle the TRUE (tape-measured) distance implies, and convert
    that angular delta back to an equivalent row-pixel error via the local
    inverse of the same propagation used in sigma_row_meters. Returns
    (mean_px_error, std_px_error, per_point_errors) -- mean near zero
    confirms this behaves like noise, not a hidden bias.
    """
    H = recon.DISTANCE_CAMERA_HEIGHT_METERS
    f_row = recon.FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    pcol = recon.FLOW_PRINCIPAL_COLUMN_NORMALIZED
    prow = recon.FLOW_PRINCIPAL_ROW_NORMALIZED
    theta = math.radians(reference_pitch_deg)
    psi = math.radians(reference_roll_deg)

    with open(points_path) as f:
        cone_points = json.load(f)["points"]

    errors_px = []
    for p in cone_points:
        row, col = p["row"], p["col"]
        true_d, computed_d = p["trueDistanceMeters"], p["computedDistanceMeters"]
        phi_true = math.atan(H / true_d)
        phi_computed = math.atan(H / computed_d)
        delta_phi = phi_computed - phi_true

        focal_col = f_row / aspect
        x = (col - pcol) / focal_col
        y = (row - prow) / f_row
        derolled_y = -x * math.sin(psi) + y * math.cos(psi)

        delta_derolled_y = delta_phi * (1 + derolled_y ** 2)
        delta_y = delta_derolled_y / math.cos(psi)
        delta_row_normalized = delta_y * f_row
        errors_px.append(delta_row_normalized * frame_height_px)

    return statistics.mean(errors_px), statistics.pstdev(errors_px), errors_px


def sigma_row_meters(
    bottom_y: float, center_x: float, aspect: float,
    reference_pitch_deg: float, reference_roll_deg: float,
    frame_height_px: float = FRAME_HEIGHT_PX_REFERENCE,
) -> Optional[float]:
    """Combines THREE independent uncertainty sources in quadrature:

    1. SIGMA_EDGE_ROW_PX (pixel-level box-bottom-edge noise), propagated
       through this file's own simplified vertical-angle-only chain: pixel
       row -> tangent coordinate y -> angle alpha=atan(y) -> phi=alpha+theta
       -> distance=H/tan(phi). d(distance)/d(phi) = -H/sin^2(phi) is the
       whole story here -- this is what blows up near the horizon,
       deliberately unbounded (a very large but finite number, not
       inf/nan, so callers can still form a valid inverse-variance weight
       of ~0 without special-casing). This term historically ignored the
       (roll * center_x) cross-term's own contribution to phi's
       sensitivity, reasoning a real roll was typically under 1-2 degrees
       -- CONFIRMED 2026-08-25 that assumption breaks down at real mount
       roll error (see SIGMA_ROLL_FLATNESS_DEGREES), which is exactly why
       terms 2/3 below exist as their OWN numerical terms against the real
       full-3D formula, rather than patching this simplified one.
    2. SIGMA_PITCH_FLATNESS_DEGREES and 3. SIGMA_ROLL_FLATNESS_DEGREES:
       uncertainty from not knowing whether referencePitchDegrees/
       referenceRollDegrees (captured once, wherever the Level screen was
       last run) still match the ground the car is ACTUALLY on right now
       -- a session/location-level bias, not per-detection noise, that
       grows with lateral offset from center (see
       _flatness_sigma_component's own doc comment). Returns None if
       either of these can't be computed (phi out of range for the
       nudged trial angle), same "don't fabricate a number" discipline as
       row_based_distance_meters' own None returns.
    """
    theta = math.radians(reference_pitch_deg)
    psi = math.radians(reference_roll_deg)
    f_row = recon.FLOW_FOCAL_LENGTH_ROW_NORMALIZED
    pcol = recon.FLOW_PRINCIPAL_COLUMN_NORMALIZED
    prow = recon.FLOW_PRINCIPAL_ROW_NORMALIZED
    focal_col = f_row / aspect

    x = (center_x - pcol) / focal_col
    y = (bottom_y - prow) / f_row
    derolled_y = -x * math.sin(psi) + y * math.cos(psi)
    alpha = math.atan(derolled_y)
    phi = alpha + theta
    if not (0 < phi < math.pi / 2):
        return None

    sigma_edge_row_normalized = SIGMA_EDGE_ROW_PX / frame_height_px
    sigma_y = sigma_edge_row_normalized / f_row
    sigma_derolled_y = sigma_y * math.cos(psi)
    sigma_phi = sigma_derolled_y / (1 + derolled_y ** 2)

    H = recon.DISTANCE_CAMERA_HEIGHT_METERS
    sin_phi = math.sin(phi)
    sigma_dist_pixel = abs(H / (sin_phi * sin_phi)) * sigma_phi

    sigma_dist_pitch = _flatness_sigma_component(
        bottom_y, center_x, aspect, reference_pitch_deg, reference_roll_deg, "pitch", SIGMA_PITCH_FLATNESS_DEGREES,
    )
    sigma_dist_roll = _flatness_sigma_component(
        bottom_y, center_x, aspect, reference_pitch_deg, reference_roll_deg, "roll", SIGMA_ROLL_FLATNESS_DEGREES,
    )
    if sigma_dist_pitch is None or sigma_dist_roll is None:
        return None

    return math.sqrt(sigma_dist_pixel ** 2 + sigma_dist_pitch ** 2 + sigma_dist_roll ** 2)


# --- σ_flatness: uncertainty from NOT knowing whether the ground under the
# vehicle right now matches the reference pitch/roll calibration -----------

SIGMA_ROLL_FLATNESS_DEGREES = 1.8
"""Prior std dev on how much the car's ACTUAL roll (relative to the true
local ground plane) can differ from referenceRollDegrees (captured once,
wherever/whenever the Level screen was last run) -- a DIFFERENT failure
mode from SIGMA_EDGE_ROW_PX's per-detection pixel noise: this is a
SESSION/LOCATION-level bias that grows with lateral offset from center
(near-zero for a centered detection, largest at the frame edge), not
per-detection random noise. See sigma_row_meters' reference-not-live design
note and the 2026-08-25 chat for why "reference, not live" already handles
smooth sustained road grade correctly and this term is specifically about
what that design does NOT cover: local unevenness the reference calibration
wasn't captured on.

REAL DATA POINT: the 26_08_25_Day_Cones test (7 cones, known 20m, spread
column 0.06-0.98) measured an actual +1.78 degree roll delta from that
session's stored reference, on an ordinary (not unusually rough) parking
lot Rick visually confirmed was tilted. One measurement, not a calibrated
distribution -- but it lines up well with published pavement design
figures: AASHTO's standard road/parking-lot cross-slope is 2% (~1.15 deg),
with 2-5% (~1.15-2.86 deg) commonly used in practice for drainage
(sources below). Treating the STORED reference as itself captured on some
independently-sloped surface, the spread of the DIFFERENCE between two
such independent locations is call it sqrt(2) times a single location's
typical cross-slope -- landing in the same ~1.5-3 deg range our one real
sample sits in. 1.8 deg is chosen to match that real sample directly,
sitting centrally in the literature-informed range -- a conservative,
literature-anchored PRIOR, not a fitted value. Revisit once more real
locations' worth of data exist (see the parked, abandoned
flatness_uncertainty_calibration.py attempt -- confounded by traffic
occlusion, not usable as-is).

Sources (fetched 2026-08-25):
  https://en.wikipedia.org/wiki/Cross_slope
  https://www.cedengineering.com/userfiles/C06-017%20-%20Roadway%20Geometric%20Design%20II%20-%20Cross-sections%20and%20Road%20Types%20-%20US.pdf
  https://wrightconstructioninc.com/post/parking-lot-grading-specs-drainage-ada-guide/"""

SIGMA_PITCH_FLATNESS_DEGREES = 1.0
"""Same idea as SIGMA_ROLL_FLATNESS_DEGREES, for pitch -- deliberately
smaller, for two reasons. (1) Empirical: the SAME 26_08_25_Day_Cones fit
needed only +0.24 deg of pitch correction alongside the +1.78 deg of roll,
a ~7x smaller effect, from the identical real dataset. (2) Physical:
"reference, not live" already handles SUSTAINED longitudinal grade well (a
hill the road continues along, camera-relative-to-road stays near its
flat-ground value); what's left for this term is LOCAL longitudinal
unevenness within roughly one parking space's own length, which -- unlike
cross-slope, which is deliberately engineered into essentially every lot
for drainage -- has no equivalent "always present by design" reason to be
large over that short a span. ADA/AASHTO figures (see
SIGMA_ROLL_FLATNESS_DEGREES's sources) treat longitudinal and lateral
slope limits as comparably sized in regulated cases (~2%, ~1.15 deg each),
which would argue for a similar magnitude to roll -- 1.0 deg splits the
difference between that regulatory symmetry and the much smaller real
pitch delta actually measured, erring conservative (larger) rather than
trusting the single low measurement outright. Weaker grounding than the
roll value -- flag as the first one to revisit with more real data."""

_FLATNESS_FINITE_DIFF_STEP_DEGREES = 0.01


def _flatness_sigma_component(
    bottom_y: float, center_x: float, aspect: float,
    reference_pitch_deg: float, reference_roll_deg: float,
    param: str, sigma_degrees: float,
) -> Optional[float]:
    """Numerically propagates an assumed prior uncertainty in reference
    pitch or roll through recon.row_based_distance_meters directly (central
    finite difference) -- NOT through this file's own simplified
    vertical-angle-only phi used for the pixel term above, since this
    effect (see SIGMA_ROLL_FLATNESS_DEGREES) is dominated by exactly the
    lateral (derolled_x) term that simplified formula's own doc comment
    already flags as ignored. Naturally scales with lateral offset from
    center the same way the real cone data did: near-zero for a centered
    detection, largest at the frame edge, since that's the real shape of
    row_based_distance_meters' own sensitivity to roll -- not a hand-tuned
    falloff, a direct consequence of calling the real formula."""
    step = _FLATNESS_FINITE_DIFF_STEP_DEGREES
    if param == "pitch":
        d_plus = recon.row_based_distance_meters(bottom_y, center_x, aspect, reference_pitch_deg + step, reference_roll_deg)
        d_minus = recon.row_based_distance_meters(bottom_y, center_x, aspect, reference_pitch_deg - step, reference_roll_deg)
    else:
        d_plus = recon.row_based_distance_meters(bottom_y, center_x, aspect, reference_pitch_deg, reference_roll_deg + step)
        d_minus = recon.row_based_distance_meters(bottom_y, center_x, aspect, reference_pitch_deg, reference_roll_deg - step)
    if d_plus is None or d_minus is None:
        return None
    derivative_per_degree = (d_plus - d_minus) / (2 * step)
    return abs(derivative_per_degree) * sigma_degrees


# --- σ_width(w, class): width-based distance uncertainty -------------------

SIGMA_EDGE_WIDTH_PX = 1.075
"""REAL, back-solved 2026-08-25 -- see find_stationary_car_width_jitter()
below, which reproduces this measurement. Mines already-recorded sessions
for a genuinely parked car (ego stationary AND the target itself not
moving -- see that function's own doc comment for why both conditions are
needed) and measures frame-to-frame box-width jitter directly: with zero
real motion on either side, any width change left is pure detector edge
noise, the same phenomenon SIGMA_EDGE_ROW_PX measures for the bottom edge,
just the left/right edges instead.

10 clean tracks survived across 4 different sessions/days (26_08_20_Day_
Small, 26_08_21_Day_Small, 26_08_24_Day_Distance, 26_08_25_Day_Cones),
ranging 0.43-1.75px, mean 1.075px -- notably TIGHTER and more consistent
than expected, and reassuringly close to (a bit smaller than)
SIGMA_EDGE_ROW_PX's 2.28px, confirming the "same underlying phenomenon,
different edge" assumption the old stand-in was built on rather than
overturning it. Two additional candidate tracks were found and explicitly
EXCLUDED, not just filtered by threshold: both showed genuine multi-second
plateaus at two DIFFERENT stable widths (e.g. ~228px then a real step down
to ~201px) rather than frame-level jitter -- a real small physical
event (most likely a near-ID mixup between two close real objects), not
detector noise, confirmed by inspecting their raw sequences directly
before excluding them."""


def find_stationary_car_width_jitter(
    detections_path: Path, aspect: float = 1920 / 1080,
    min_frames: int = 30, min_duration_s: float = 10.0, max_range_ratio: float = 0.85,
) -> list:
    """Reproduces the SIGMA_EDGE_WIDTH_PX measurement against one session's
    detections.jsonl -- returns [(trackID, n_frames, duration_s, mean_width_px,
    std_width_px), ...] for every car track that survives BOTH of two
    conditions, each catching a different way a "stationary" track can
    still be moving:

    1. egoSpeedMps <= 0.15 at every entry used -- the ego vehicle itself
       isn't driving. Necessary but NOT sufficient: real traffic keeps
       passing a parked ego vehicle, and a passing car's box grows/shrinks
       fast regardless of what the ego is doing.
    2. The TARGET's own box width stays within max_range_ratio of its
       min-to-max range across its ENTIRE tracked lifetime (not just a
       locally-smooth-looking window) -- CONFIRMED NECESSARY 2026-08-25:
       an earlier attempt using only a "no single frame-to-frame jump
       bigger than 6%" heuristic let through cars in the middle of a fast,
       continuous approach/departure (width growing ~118px -> 658px in
       under 2 seconds, a real passing car, not a parked one) whenever
       that approach happened to have a brief low-velocity plateau, since
       consecutive SMALL steps can still add up to real bulk motion over
       a longer window. Requiring the ratio check across the WHOLE track
       life, not a local sub-window, is what actually filters these out.

    Same class/edge/hood/oblique gating as flatness_uncertainty_
    calibration.py's qualifying_points -- see that file's own doc comment
    for why each gate exists."""
    entries = [json.loads(line) for line in detections_path.read_text().splitlines() if line.strip()]
    by_track: dict = {}
    for entry in entries:
        ego_speed = entry.get("egoSpeedMps")
        if ego_speed is None or ego_speed > 0.15:
            continue
        for det in entry.get("detections", []):
            if det.get("label") != "car" or det.get("confidence", 0.0) < 0.5:
                continue
            left, right = det["x"], det["x"] + det["w"]
            if left <= recon.EDGE_TRUNCATION_MARGIN_NORMALIZED or right >= 1.0 - recon.EDGE_TRUNCATION_MARGIN_NORMALIZED:
                continue
            bottom_y, center_x = det["y"] + det["h"], det["x"] + det["w"] / 2
            phi_deg = recon._row_based_phi_degrees(
                bottom_y, center_x, aspect, entry.get("referencePitchDegrees", 0.0), entry.get("referenceRollDegrees", 0.0),
            )
            if phi_deg > recon.HOOD_CUTOFF_ANGLE_DEGREES - 2.0 or recon.is_oblique_view(det, entry, aspect):
                continue
            track_id = det.get("trackID")
            if track_id is None:
                continue
            frame_width_px = 1920.0  # every session recorded so far is 16:9, regardless of exact pixel count
            by_track.setdefault(track_id, []).append((entry["t"], det["w"] * frame_width_px))

    results = []
    for track_id, samples in by_track.items():
        samples.sort()
        if len(samples) < min_frames:
            continue
        duration = samples[-1][0] - samples[0][0]
        if duration < min_duration_s:
            continue
        widths = [w for _, w in samples]
        if min(widths) / max(widths) < max_range_ratio:
            continue
        results.append((track_id, len(samples), duration, statistics.mean(widths), statistics.pstdev(widths)))
    return results


def sigma_width_meters(
    distance_estimate: float, box_width_normalized: float, aspect: float, label: str,
    frame_width_px: Optional[float] = None,
) -> Optional[float]:
    """Combines two independent multiplicative error sources in quadrature:
    (1) the class's real population width variance (KITTI, exact -- e.g.
    car 6.3% relative, truck/bus 8.4%, person 21.7%) and (2) box-width pixel
    measurement noise, which GROWS as the box shrinks with distance (same
    absolute pixel error is a larger fraction of a smaller box) -- the
    slower-degrading counterpart to row-based's sin^2(phi) blowup.
    `frame_width_px` defaults to a 1920-wide frame (this project's standard
    recording width) if not given, consistent with SIGMA_EDGE_ROW_PX's own
    1080-tall reference at the same 16:9 aspect."""
    prior = recon.OBJECT_ASPECT_PRIORS.get(label)
    if prior is None or box_width_normalized <= 0:
        return None
    width_mean = prior["width_mean"]
    # Real per-class relative width std -- OBJECT_ASPECT_PRIORS only stores
    # the mean; pull the matching std from the same real KITTI pull
    # documented in that dict's own doc comment (project_kitti_width_stats).
    width_std_by_label = {"car": 0.102, "person": 0.143, "truck": 0.217, "bus": 0.217}
    width_std = width_std_by_label.get(label)
    if width_std is None:
        return None
    relative_population_variance = (width_std / width_mean) ** 2

    frame_width_px = frame_width_px or (FRAME_HEIGHT_PX_REFERENCE * aspect)
    box_width_px = box_width_normalized * frame_width_px
    if box_width_px <= 0:
        return None
    relative_pixel_variance = (SIGMA_EDGE_WIDTH_PX / box_width_px) ** 2

    relative_variance = relative_population_variance + relative_pixel_variance
    return distance_estimate * math.sqrt(relative_variance)


# --- P(truncated | phi, column): hood-truncation probability ---------------

HOOD_TRUNCATION_COLUMN_BINS = 7
"""Standardized on 7 (not the original 107-sample study's 5) 2026-08-26,
real request after the 26_08_26_Day_Walk garage session's 7-bin analysis
came back with 78-184 samples per bin -- comfortably enough data to
support the finer resolution, which matters here: the real curve isn't
quite symmetric (far-left alpha 11.14deg vs far-right 10.74deg), and 5
bins would have blurred that away. A future recalibration with thinner
data could leave some of these 7 under-populated where 5 wouldn't be --
hood_cutoff_calibration.py's analyze command already reports "(too few)"
for any bin under 3 samples rather than a bogus number, so this degrades
visibly rather than silently."""

HOOD_TRUNCATION_BEARING_BIN_EDGES_DEGREES = [-32.9, -23.7, -14.4, -5.1, 4.1, 13.4, 22.6, 31.9]
"""8 edges bounding HOOD_TRUNCATION_COLUMN_BINS's 7 bins, in BEARING degrees
(horizontal viewing angle from the camera's principal axis -- see
hood_cutoff_calibration.py's compute_alpha_phi_and_bearing), NOT normalized
column -- bearing is what the 26_08_26_Day_Walk calibration actually binned
by, and bearing/column aren't linearly related (bearing = atan(x)), so
reusing equal-width column bins here would silently misalign against the
real per-bin measurements below. These are exactly that session's own
observed min/max bearing split into 7 equal-width bins, not a fixed,
round-number grid -- a session covering a meaningfully different bearing
range would need these re-derived, not just the stats below replaced."""

HOOD_TRUNCATION_BIN_STATS = [
    (11.137, 0.468), (10.608, 0.351), (10.221, 0.437), (9.756, 0.302),
    (9.797, 0.312), (9.981, 0.210), (10.740, 0.214),
]
"""REAL, 2026-08-26 -- (alpha_mean_degrees, alpha_std_degrees) per bearing
bin, from 717 samples (8 walk passes, one real session on the CURRENT
mount) via hood_cutoff_calibration.py extract+analyze. ALPHA, NOT PHI --
see that file's own 2026-08-26 doc comment for why: alpha (angle below the
camera's own optical axis) is fixed by the camera-to-hood relationship
regardless of the car's momentary pitch, so these 7 numbers are meant to
stay valid across future sessions recorded at DIFFERENT pitches, unlike a
phi-style number which would silently drift with whatever pitch a given
session happens to have. hood_truncation_probability converts back to a
phi-style threshold at call time using that DETECTION's own
reference_pitch_deg, not whatever pitch was active during calibration.

Real curvature confirmed, not just noise: alpha rises toward both edges
(11.14deg far left, 10.74deg far right) from a low near center
(9.76-9.98deg) -- physically sensible (the wide dashboard/hood blocks the
view sooner straight ahead; off to the sides you can see steeper-down
before hitting an obstruction). Per-bin std (0.21-0.47deg) is also much
tighter than the old pooled global figure (0.771deg, n=107, OLD mount) --
most of that old spread was apparently real per-bearing curvature being
averaged away, not measurement noise. Still a single session/mount
generation's worth of data -- revisit if a future walk's samples disagree
meaningfully with these."""


def _bearing_bin_index(center_x: float, aspect: float) -> int:
    """Bearing (not raw column) is what HOOD_TRUNCATION_BIN_STATS is binned
    by -- see that constant's own doc comment. Same formula as
    hood_cutoff_calibration.py's compute_alpha_phi_and_bearing, minus the
    alpha/phi parts this call site doesn't need. Bearings outside the
    calibration session's own observed range clamp to the nearest edge bin
    rather than extrapolating -- the curve's shape beyond what was actually
    walked is unmeasured, not assumed flat."""
    focal_col = recon.FLOW_FOCAL_LENGTH_ROW_NORMALIZED / aspect
    x = (center_x - recon.FLOW_PRINCIPAL_COLUMN_NORMALIZED) / focal_col
    bearing_deg = math.degrees(math.atan(x))
    edges = HOOD_TRUNCATION_BEARING_BIN_EDGES_DEGREES
    for i in range(HOOD_TRUNCATION_COLUMN_BINS):
        if bearing_deg < edges[i + 1] or i == HOOD_TRUNCATION_COLUMN_BINS - 1:
            return max(0, i)
    return HOOD_TRUNCATION_COLUMN_BINS - 1


def hood_truncation_probability(phi_deg: float, center_x: float, aspect: float, reference_pitch_deg: float) -> float:
    """Density-shaped APPROXIMATION of P(truncated | phi), not a true
    Bayesian posterior -- see the 2026-08-24 chat discussion for why: a
    real posterior would also need P(phi | genuinely NOT truncated), the
    distribution of real ground-contact angles across ordinary driving,
    which isn't characterized anywhere in this project.

    CONFIRMED BUG, caught by this file's own bus/truck validation run: an
    earlier version of this function used a one-sided tail probability
    (1 - CDF(z)), which said P(truncated)~=1 for ANY phi below the mean --
    including phi=0.03deg (a real near-horizon-far-object case), which is
    physically backwards. A hood-truncation ARTIFACT always reads as
    approximately the mean angle (~9.899deg) regardless of the object's
    true, much-closer distance, since it's measuring the fixed hood edge,
    not the real ground contact -- so the AMBIGUOUS zone is a narrow band
    AROUND the mean, not "everything below it." This is a density-shaped
    (Gaussian bump, peak=1 at the mean) function instead: high right at the
    mean (genuinely ambiguous -- could be a real object at the matching
    distance, or a truncation artifact), and low in EITHER direction away
    from it, including toward zero (far objects, no truncation mechanism
    could produce that) and toward angles well above the mean (a much more
    severe, different failure -- see HOOD_CUTOFF_ANGLE_DEGREES' own "hard
    exclude" check in reconstruct_annotated.py, which stays a separate,
    non-probabilistic backstop for genuinely impossible phi, not something
    this bump is meant to replace).
    Uses the bearing bin's own (alpha_mean, alpha_std) -- see
    HOOD_TRUNCATION_BIN_STATS' own doc comment. alpha_mean is converted to
    a phi-style threshold using THIS call's own reference_pitch_deg (the
    detection's actual session pitch), not whatever pitch was active
    during calibration -- that conversion is the entire reason
    HOOD_TRUNCATION_BIN_STATS stores alpha instead of phi in the first
    place: a session at a different pitch than 26_08_26_Day_Walk's 0.926deg
    still gets a correctly-shifted threshold instead of a stale one."""
    bin_index = _bearing_bin_index(center_x, aspect)
    alpha_mean_deg, std_deg = HOOD_TRUNCATION_BIN_STATS[bin_index]
    mean_deg = alpha_mean_deg + reference_pitch_deg
    if std_deg <= 0:
        return 1.0 if abs(phi_deg - mean_deg) < 1e-9 else 0.0
    z = (phi_deg - mean_deg) / std_deg
    return math.exp(-0.5 * z * z)


# --- Fusion ------------------------------------------------------------

def fuse_distance_meters(det: dict, entry: dict, aspect: float) -> dict:
    """Inverse-variance-weighted fusion of row-based and width-based
    distance, replacing corrected_distance_meters' binary switch. Returns a
    dict (not just a float) since the whole point is to also surface the
    uncertainty and which terms contributed -- a None/absent
    'fused_distance_meters' means neither estimate was usable at all (same
    "don't fabricate a number with no real basis" discipline as the rest of
    this file).

    Bias sources stay OUTSIDE the variance math, exactly as decided in the
    2026-08-24 design discussion:
      - Oblique angle: reuses is_oblique_view -- when it fires, sigma_width
        is treated as infinite (the width term drops out entirely), not
        inflated by some finite amount.
      - Hood truncation: inflates sigma_row by 1/(1 - P(truncated)) rather
        than a hard reject -- continuous, and reduces to "row term
        effectively excluded" as P(truncated) -> 1.
      - Occlusion by another object: NOT implemented -- no detection exists
        for this yet (see this file's own top doc comment). A fused result
        for an occluded box will be wrong in whatever direction that bias
        pushes it; this is a known, open gap, not a silent guarantee.
    """
    left = det["x"]
    right = det["x"] + det["w"]
    bottom = det["y"] + det["h"]
    center_x = det["x"] + det["w"] / 2
    side_truncated = (
        left <= recon.EDGE_TRUNCATION_MARGIN_NORMALIZED
        or right >= 1.0 - recon.EDGE_TRUNCATION_MARGIN_NORMALIZED
    )
    pitch = entry.get("referencePitchDegrees", 0.0)
    roll = entry.get("referenceRollDegrees", 0.0)

    row_dist = recon.row_based_distance_meters(bottom, center_x, aspect, pitch, roll)
    phi_deg = recon._row_based_phi_degrees(bottom, center_x, aspect, pitch, roll)
    p_truncated = hood_truncation_probability(phi_deg, center_x, aspect, pitch)
    # Separate, non-probabilistic backstop for phi values the truncation
    # BUMP alone would wrongly clear (it's centered on the mean, so a phi
    # well ABOVE the mean scores low on "looks like an ambiguous truncation
    # artifact" even though it's a DIFFERENT, more severe failure --
    # reporting some other row (frame edge, dashboard/HUD artifact) as the
    # box bottom entirely, not just an ordinary hood-clipped reading. Same
    # hard-exclude discipline as reconstruct_annotated.py's own case 2.
    # Uses THIS detection's own bearing bin (not the old global scalar) so
    # the backstop margin matches the same per-column curve the probability
    # bump above does, converted to phi via this call's own reference pitch.
    _bin_idx = _bearing_bin_index(center_x, aspect)
    _alpha_mean_deg, _alpha_std_deg = HOOD_TRUNCATION_BIN_STATS[_bin_idx]
    genuinely_impossible = phi_deg > (_alpha_mean_deg + pitch) + 2 * _alpha_std_deg

    row_sigma = None
    if row_dist is not None and not side_truncated and not genuinely_impossible:
        row_sigma = sigma_row_meters(bottom, center_x, aspect, pitch, roll)
        if row_sigma is not None:
            row_sigma = row_sigma / max(1e-6, (1.0 - p_truncated))

    width_dist = None
    width_sigma = None
    prior = recon.OBJECT_ASPECT_PRIORS.get(det["label"])
    if prior is not None and det["h"] > 0 and not side_truncated:
        focal_col = recon.FLOW_FOCAL_LENGTH_ROW_NORMALIZED / aspect
        width_dist = focal_col * prior["width_mean"] / det["w"]
        is_oblique = recon.is_oblique_view(det, entry, aspect)
        if not is_oblique:
            width_sigma = sigma_width_meters(width_dist, det["w"], aspect, det["label"])
        # else: width term stays None -- excluded, per the oblique bias gate.

    terms = []
    if row_dist is not None and row_sigma is not None and row_sigma > 0:
        terms.append(("row", row_dist, row_sigma))
    if width_dist is not None and width_sigma is not None and width_sigma > 0:
        terms.append(("width", width_dist, width_sigma))

    if not terms:
        return {
            "fused_distance_meters": None, "fused_sigma_meters": None,
            "row_distance_meters": row_dist, "row_sigma_meters": row_sigma,
            "width_distance_meters": width_dist, "width_sigma_meters": width_sigma,
            "phi_degrees": phi_deg, "p_truncated": p_truncated, "side_truncated": side_truncated,
        }

    inverse_variance_sum = sum(1.0 / (sigma * sigma) for _, _, sigma in terms)
    weighted_mean = sum(dist / (sigma * sigma) for _, dist, sigma in terms) / inverse_variance_sum
    fused_sigma = math.sqrt(1.0 / inverse_variance_sum)

    # RELIABILITY THRESHOLD -- caught by this file's own bus/truck
    # validation run: a single-term fusion (row-only, because the oblique
    # gate excluded width) can have a relative sigma so large it swamps the
    # point estimate itself (e.g. the real bus case: 1962m +/- 6497m) --
    # technically "fused," but reporting that point estimate on its own,
    # with no visible error bar, would just be the SAME unreliable number
    # dressed up as a fusion result. Same "don't fabricate a number with no
    # real basis" discipline as the rest of this project: past this
    # relative-uncertainty bound, report unreliable rather than a number.
    RELATIVE_SIGMA_UNRELIABLE_THRESHOLD = 1.0
    if weighted_mean > 0 and (fused_sigma / weighted_mean) > RELATIVE_SIGMA_UNRELIABLE_THRESHOLD:
        return {
            "fused_distance_meters": None, "fused_sigma_meters": fused_sigma,
            "row_distance_meters": row_dist, "row_sigma_meters": row_sigma,
            "width_distance_meters": width_dist, "width_sigma_meters": width_sigma,
            "phi_degrees": phi_deg, "p_truncated": p_truncated, "side_truncated": side_truncated,
            "terms_used": [name for name, _, _ in terms],
            "rejected_unreliable": True,
        }

    return {
        "fused_distance_meters": weighted_mean, "fused_sigma_meters": fused_sigma,
        "row_distance_meters": row_dist, "row_sigma_meters": row_sigma,
        "width_distance_meters": width_dist, "width_sigma_meters": width_sigma,
        "phi_degrees": phi_deg, "p_truncated": p_truncated, "side_truncated": side_truncated,
        "terms_used": [name for name, _, _ in terms],
    }


if __name__ == "__main__":
    mean_err, std_err, per_point = calibrate_sigma_edge_row_px()
    print(f"SIGMA_EDGE_ROW_PX calibration: mean={mean_err:.4f}px std={std_err:.4f}px")
    print(f"per-point implied errors: {[round(e, 3) for e in per_point]}")
