//
//  DistanceEstimator.swift
//  DriverAssist
//
//  Ground-plane / bottom-of-box distance estimation -- a vehicle's real-world
//  distance follows from the image row where its box's bottom (ground-contact
//  point) appears, given the camera's height and pitch.
//
//  Physical (pinhole) model: for a ground point at horizontal distance D,
//  camera height H above the ground, and camera pitched theta below true
//  horizontal, the point's angle below true horizontal is phi = atan(H/D)
//  (the right triangle formed by height and distance). The point's angle
//  below the camera's own optical axis is then (phi - theta), and that angle
//  maps to an image row through the focal length:
//
//      v = v0 + f * tan(phi - theta)           [row from angle]
//      D = H / tan(theta + atan((v - v0) / f))  [angle from row, inverted]
//
//  where v0 is the row the optical axis itself crosses (principal point,
//  roughly image center but not guaranteed exactly), and f is focal length
//  in the same normalized-row units as v0.
//
//  This replaces an earlier small-angle-approximation form (v = horizonRow +
//  k/D) that baked a single fixed pitch into its fit constants -- accurate
//  only near that one pitch, and, by its own small-angle assumption, least
//  accurate at exactly the close range this feature cares about most.
//
//  *** theta is a REFERENCE pitch, captured deliberately on flat ground --
//  *** NOT a continuous live CoreMotion reading. This is deliberate, not an
//  *** oversight: on a hill, the road tilts with the car (assuming the body
//  *** roughly follows the grade), so the camera's pitch RELATIVE TO THE ROAD
//  *** AHEAD stays close to its flat-ground value even though gravity-
//  *** referenced pitch changes with every grade change. Feeding raw live
//  *** pitch in here every frame would misattribute hill grade as mount
//  *** misalignment -- exactly backwards. What this format DOES let you fix
//  *** cheaply is genuine mount drift (reattachment slop, droop over time):
//  *** re-run a flat-ground pitch capture (`PitchSensor.captureReferencePitch()`
//  *** / `referencePitchDegrees`) and pass the new value in -- no need to
//  *** redo the tape-mark fit below, since v0/f don't change with pitch.
//  *** Known accepted gap: this doesn't correct for transient pitch changes
//  *** from vehicle dynamics (braking dive, acceleration squat) mid-drive --
//  *** deliberately, since there's no cheap way to distinguish "genuine
//  *** dynamic pitch change" from "gradual hill" without more machinery than
//  *** is justified without real drive data showing this actually matters.
//
//  v0 and f are still fit from real tape-mark reference points, not trusted
//  from the phone's nominal field of view -- video stabilization/crop makes
//  that unreliable, same reasoning as before. v0 and f are true intrinsic-
//  camera constants that don't change with pitch, so that fit only needs to
//  happen once, at one known reference pitch (captured on the same flat
//  ground as the tape marks) -- not once per pitch, and not once per drive.
//
//  *** PITCH SIGN CONVENTION -- CONFIRMED 2026-08-11 (was previously flagged
//  *** here as unconfirmed). Every angle here assumes theta is signed
//  *** positive when the camera's nose is tilted DOWN toward the road.
//  *** PitchSensor.pitchDegrees/referencePitchDegrees were found NOT to
//  *** match that convention -- caught via this file's own recommended
//  *** validation (fit on one pitch, cross-check the prediction at other
//  *** pitches without refitting: as-shipped gave up to +3806% error and a
//  *** physically-impossible negative fitted focal length), then confirmed
//  *** directly with the hand-tilt bench test this comment used to ask for.
//  *** Fixed at the source in PitchSensor.swift (the sign was already
//  *** "chosen" there for this exact convention, just implemented
//  *** backwards) rather than here, so no adjustment is needed in this
//  *** file -- pitchDegrees/referencePitchDegrees now already read
//  *** correctly for direct use below.
//
//  *** Stale-data note: any referencePitchDegrees captured (and persisted
//  *** to UserDefaults) BEFORE the 2026-08-11 PitchSensor fix holds the OLD
//  *** wrong-sign value. `skipCalibration()`'s "No" path reuses whatever's
//  *** persisted without recapturing -- press "Calibrate" (not skip) at
//  *** least once after updating to get a correctly-signed value stored.
//
//  There are no default calibration constants -- this type can't be
//  constructed without supplying real ones (via `fit`, from real tape-mark
//  measurements), so a not-yet-calibrated instance can't silently produce
//  numbers that look real (same reasoning as EgoSpeedManager/PitchSensor
//  logging nil instead of a placeholder 0).
//
//  ROLL: reads the reference roll the same way as pitch (captured together
//  by PitchSensor.captureReferenceAttitude(), NOT a live reading -- road
//  camber/banked curves are roll's version of a hill, same misattribution
//  risk described above for pitch). Unlike pitch, roll is meant to be
//  physically zeroed with a bubble level as part of the same pre-calibration
//  routine, not just referenced -- there's no design reason for roll to be
//  anything but zero, unlike pitch's intentional downward aim. This
//  correction exists mainly so a small residual roll (whatever the bubble
//  level couldn't fully zero) doesn't quietly bias distance for off-center
//  detections, not to correct for large roll.
//
//  De-rolling requires the detection's COLUMN position too, not just its
//  row, plus a column-axis principal point/focal length. Two things worth
//  knowing about how those are handled here:
//  - `focalLengthColumnNormalized` is derived from `focalLengthNormalized`
//    via the caller-supplied `aspectRatio`, assuming square pixels (safe --
//    standard for digital video) rather than fit separately, since there's
//    no independent lateral calibration data to fit it from. `aspectRatio`
//    is NOT a fixed constant: `Detection.boundingBox` is normalized
//    independently against the model's own width/height
//    (InferenceEngine.decodeDetections), which differ between the standard
//    (1152x640) and high-res (1920x1088) export -- pass whichever was
//    actually active for this detection, not a hardcoded value.
//  - The column-axis principal point is assumed to be exact frame center
//    (0.5), not fit -- see `assumedPrincipalColumnNormalized` below for why
//    that's a safe assumption here specifically, unlike the row axis.
//
//  The de-roll itself is a tangent-plane (small-angle) rotation of the row/
//  col offset around the principal point, not an exact spherical rotation --
//  accurate exactly when roll is small and the detection isn't far off-
//  center, which is the expected regime given roll is physically minimized
//  before calibration and the leading-vehicle classifier already selects
//  roughly-centered targets. Revisit with an exact rotation if either
//  assumption stops holding in practice.
//
//  *** ROLL SIGN CONVENTION -- CONFIRMED 2026-08-11 via bench test on the
//  *** level/calibrate screen: rolling the phone counter-clockwise reads
//  *** positive, clockwise reads negative -- matches the assumption below
//  *** exactly (positive referenceRollDegrees = image rotating counter-
//  *** clockwise as the camera sees it). No code change needed.
//

import Foundation

struct DistanceEstimator {
    /// Camera height above the ground, in meters -- measured directly (laser
    /// level), not fit. Unlike v0/f below, this is a real physical constant
    /// the formula needs independently.
    var cameraHeightMeters: Double

    /// Row (top-left-origin normalized, 0...1 -- same convention as
    /// `Detection.boundingBox`) where the camera's optical axis itself
    /// crosses the image -- the "principal point" row. Fit from real
    /// reference points alongside `focalLengthNormalized`, not assumed to be
    /// exact image center (stabilization/crop can shift it).
    var principalRowNormalized: Double

    /// Focal length in the same normalized-row units as
    /// `principalRowNormalized`, fit alongside it. Not derived from the
    /// phone's nominal field of view -- same distrust as the row above.
    var focalLengthNormalized: Double

    /// Column-axis counterpart to `principalRowNormalized`, ASSUMED rather
    /// than fit -- there's currently no lateral/yaw calibration data to fit
    /// it from (yaw is handled by physically zeroing the mount, not a
    /// software correction -- see the following-distance-measurement plan).
    /// Exact frame center (0.5) is a safe assumption here in a way nominal
    /// FOV isn't elsewhere in this file: a lens's true principal point is
    /// normally within a small fraction of a percent of geometric sensor
    /// center from manufacturing tolerance alone, unlike focal length/FOV,
    /// which video stabilization and cropping can shift substantially. Only
    /// feeds the roll correction below -- replace with a real fitted value
    /// if the lateral-calibration work referenced in that plan ever gets
    /// built.
    private static let assumedPrincipalColumnNormalized: Double = 0.5

    /// The live app's actual calibrated instance.
    ///
    /// *** UPDATED 2026-08-15 -- v0/f refit from the SHAPE clamp mount's
    /// *** cone-calibration session (data/26_08_14_ConeCalibration/), NOT
    /// *** just the height. This corrects an assumption stated here until
    /// *** today: that v0/f are purely lens-intrinsic and never need
    /// *** re-fitting for a mount change alone. That assumption is still
    /// *** correct IN PRINCIPLE -- mount position/orientation only affects
    /// *** height/pitch/roll/lever-arm, all handled separately -- but this
    /// *** specific refit wasn't actually about the mount. Investigated
    /// *** because the new cone data, checked against the OLD v0/f with the
    /// *** real (device-logged) reference pitch/roll/height for this
    /// *** session, predicted distances 21-55% too far, growing with
    /// *** distance -- too large and too systematic to be noise. Root
    /// *** cause, confirmed by pulling both sessions' actual device logs
    /// *** (not assumed): the 2026-08-11 tape-mark session was recorded at
    /// *** 4K with video stabilization OFF; this cone session was recorded
    /// *** at 1080p with stabilization ON (`stabilizationEnabled: true`
    /// *** throughout, confirmed in detections.jsonl) -- forgot to switch
    /// *** to the 4K calibration-recording mode that day. `.standard`
    /// *** stabilization crops the frame (see `setStabilizationEnabled`'s
    /// *** own comment), which shifts the effective FOV and therefore v0/f
    /// *** in normalized terms even though it's the same physical lens --
    /// *** resolution alone (4K vs 1080p) would NOT do this, normalizing
    /// *** cancels out pixel count; the crop from stabilization is what
    /// *** actually moved. `isStabilizationEnabled` is a persisted setting,
    /// *** not session-scoped, so `true` here is very likely just this
    /// *** app's normal current driving configuration, not something
    /// *** specially set for calibration -- meaning this cone-based fit,
    /// *** not the old tape-mark one, is probably the one that actually
    /// *** matches real drives. Lesson: when two calibration sessions
    /// *** disagree, check what was actually DIFFERENT about how each was
    /// *** captured (resolution, stabilization, crop) before assuming a
    /// *** constant that's supposed to be fixed has drifted for no reason.
    /// ***
    /// *** Fit method: least-squares over all 8 real cone points (4 near:
    /// *** 8/10/12/14m, 4 far: 14/16/18/20m, marked via
    /// *** tools/cone_pinpoint_tool.html against extracted frames), using
    /// *** the session's actual logged reference pitch -1.5137deg / roll
    /// *** -0.4506deg (both confirmed constant across the whole recording
    /// *** except a stale value in the first ~3s before Calibrate was
    /// *** pressed) and cameraHeightMeters=1.02. Same linear reduction
    /// *** `fit()` already uses for 2 points, extended to 8 via ordinary
    /// *** least squares instead of an exact solve. Residuals: -3.65% to
    /// *** +2.74% per point, no systematic pattern by distance -- mean 1.59%
    /// *** / max 3.65%, comparable to (slightly better than) the original
    /// *** tape-mark fit's 2.9%/6.4%. Column-axis principal point is still
    /// *** the assumed 0.5, not independently fit -- this was a v0/f-only
    /// *** regression, same two unknowns as the original 2-point fit.
    ///
    /// (2026-08-11 tape-mark fit, superseded above -- kept for the
    /// pitch-sign-bug history, not current values: v0=0.500504, f=1.412226,
    /// from data/26_08_11_DistanceCalibration/ round 0, reference pitch
    /// -2.561deg/roll -2.322deg. That session's own real lesson, still
    /// valid: the raw debug-logged pitch for round 0 was +2.561deg, logged
    /// BEFORE the PitchSensor sign fix earlier in this file -- the
    /// corrected value is that raw value negated. Re-deriving this from a
    /// written summary after a context-compaction boundary once lost track
    /// of which pitch values already had the negation applied, silently
    /// reintroducing the same sign bug one level removed. Fit on that
    /// round alone predicted 6 held-out points from other rounds/pitches
    /// to 2.9% mean / 6.4% max error once the sign was actually correct.)
    ///
    /// cameraHeightMeters=1.02 -- measured 2026-08-15 for the SHAPE clamp
    /// mount (supersedes the 1.015m RAM-mount figure).
    static let calibrated = DistanceEstimator(
        cameraHeightMeters: 1.02,
        principalRowNormalized: 0.481681,
        focalLengthNormalized: 1.322673
    )

    /// Distance in meters to the ground-contact point of a detection.
    ///
    /// - Parameters:
    ///   - bottomY: normalized row (0 = top of frame, 1 = bottom) -- pass a
    ///     detection's `boundingBox.maxY`.
    ///   - centerX: normalized column (0 = left, 1 = right) of the same
    ///     ground-contact point -- pass a detection's `boundingBox.midX`
    ///     (bottom-center of the box, not the left edge).
    ///   - referencePitchDegreesBelowHorizontal: camera pitch in degrees,
    ///     positive when the nose is tilted down toward the road -- NOT a
    ///     live per-frame reading. Pass `PitchSensor.referencePitchDegrees`
    ///     (captured deliberately on flat ground), not `pitchDegrees` (live).
    ///     See the file-level comment above for why: live pitch conflates
    ///     road grade with genuine mount misalignment. Also see the pitch
    ///     sign-convention warning at the top of this file.
    ///   - referenceRollDegrees: same reference-not-live reasoning as pitch,
    ///     from `PitchSensor.referenceRollDegrees`. See the roll
    ///     sign-convention warning at the top of this file -- independently
    ///     unconfirmed from pitch's.
    ///   - aspectRatio: width/height of the frame `bottomY`/`centerX` are
    ///     normalized against -- NOT a fixed constant, varies with
    ///     `ModelManager`'s resolution mode. Pass whichever was actually
    ///     active when this detection was produced.
    /// - Returns: nil if the resulting ray doesn't cross the ground plane in
    ///   front of and below the camera (points at/above the horizon) --
    ///   geometrically not a finite forward distance, so treat as "can't
    ///   estimate" rather than clamping to some default.
    ///
    /// *** GENERALIZED TO FULL 3D -- FIXED 2026-08-23. Until this date, this
    /// *** method computed `alpha = atan(derolledY)` and `phi = alpha +
    /// *** theta`, then returned `cameraHeightMeters / tan(phi)` -- a formula
    /// *** that silently discards `derolledX` (the point's lateral angle
    /// *** once roll is undone) entirely. That's exactly correct for a point
    /// *** on the frame's vertical centerline (derolledX == 0), where it
    /// *** reduces to the right-triangle relationship in this file's header
    /// *** comment -- but for anything off-center, it was returning the
    /// *** ray's forward DEPTH component, not the true straight-line ground
    /// *** distance to the object, understating distance more the further
    /// *** off-axis the detection sat.
    /// ***
    /// *** CONFIRMED via real 2026-08-15 walkaround-test data: a person
    /// *** walking a constant-radius tether arc read as shrinking from
    /// *** ~16m to ~10m purely from walking off to the side, not from any
    /// *** real change in distance. CONFIRMED independently via an exact
    /// *** synthetic 3D projection check: a known ground point 20m away but
    /// *** 19m off-axis returned 6.24m under the old formula (its true
    /// *** forward depth), recovering exactly 20.000000m under the fix
    /// *** below across every lateral offset and roll angle tried.
    /// ***
    /// *** Fix: instead of reading off an angle that implicitly assumes
    /// *** zero lateral offset, invert the full pitch+roll rotation to
    /// *** recover the camera ray in WORLD coordinates, solve for where that
    /// *** ray crosses the ground plane (using cameraHeightMeters as the
    /// *** known vertical drop), and return the full lateral+forward
    /// *** hypotenuse at that ground point -- not just its forward
    /// *** component. See `groundContactAngleDegrees` below for the old
    /// *** vertical-only angle, kept as its own method: WidthDistanceOverride
    /// *** needs that specific angle (calibrated into
    /// *** `hoodCutoffAngleDegrees`), not the one this method now returns.
    func distanceMeters(
        bottomY: Double,
        centerX: Double,
        referencePitchDegreesBelowHorizontal: Double,
        referenceRollDegrees: Double,
        aspectRatio: Double
    ) -> Double? {
        let theta = referencePitchDegreesBelowHorizontal * .pi / 180
        let psi = referenceRollDegrees * .pi / 180

        let focalLengthColumnNormalized = focalLengthNormalized / aspectRatio
        let x = (centerX - Self.assumedPrincipalColumnNormalized) / focalLengthColumnNormalized
        let y = (bottomY - principalRowNormalized) / focalLengthNormalized

        // De-roll: rotate the (col, row) offset by -psi to undo the camera's
        // own roll -- see the roll sign-convention warning at the top of
        // this file. derolledY is the vertical angle once roll is undone
        // (same quantity `groundContactAngleDegrees` uses); derolledX is the
        // lateral counterpart this method now keeps instead of discarding.
        let derolledY = -x * sin(psi) + y * cos(psi)
        let derolledX = x * cos(psi) + y * sin(psi)

        // Un-pitch into world coordinates: rayDown/rayForward are the
        // downward/forward components (world frame) of the ray through this
        // pixel, after rotating by pitch theta about the camera's lateral
        // axis. The ray must point downward (rayDown > 0) to ever cross the
        // ground plane below the camera.
        let rayDown = cos(theta) * derolledY + sin(theta)
        guard rayDown > 0 else { return nil }
        let rayForward = cos(theta) - sin(theta) * derolledY

        // Scale the ray so its downward component covers cameraHeightMeters
        // -- that's where it crosses the ground -- then read off both
        // horizontal components at that same scale.
        let rangeAtGround = cameraHeightMeters / rayDown
        let lateralMeters = rangeAtGround * derolledX
        let forwardMeters = rangeAtGround * rayForward
        guard forwardMeters > 0 else { return nil }
        return (lateralMeters * lateralMeters + forwardMeters * forwardMeters).squareRoot()
    }

    /// The ground-contact angle phi alone (degrees) -- what `distanceMeters`
    /// itself computed and inverted before the 2026-08-23 fix above, kept as
    /// its own method because it's still the correct quantity for a
    /// different purpose: `hoodCutoffAngleDegrees` was measured (see that
    /// constant's own doc comment) by converting real boxes' bottom rows to
    /// this exact vertical-only angle, so comparing against it needs this
    /// same formula, not the angle implied by the new (lateral-inclusive)
    /// `distanceMeters`. Using `atan(cameraHeightMeters / distanceMeters)`
    /// after the fix above would silently understate phi for any off-center
    /// detection, since distanceMeters now grows with lateral offset while
    /// the true hood-encroachment angle doesn't.
    ///
    /// - Returns: nil under the same at/above-horizon condition
    ///   `distanceMeters` returns nil for.
    func groundContactAngleDegrees(
        bottomY: Double,
        centerX: Double,
        referencePitchDegreesBelowHorizontal: Double,
        referenceRollDegrees: Double,
        aspectRatio: Double
    ) -> Double? {
        let theta = referencePitchDegreesBelowHorizontal * .pi / 180
        let psi = referenceRollDegrees * .pi / 180

        let focalLengthColumnNormalized = focalLengthNormalized / aspectRatio
        let x = (centerX - Self.assumedPrincipalColumnNormalized) / focalLengthColumnNormalized
        let y = (bottomY - principalRowNormalized) / focalLengthNormalized

        let derolledY = -x * sin(psi) + y * cos(psi)
        let alpha = atan(derolledY)
        let phi = alpha + theta
        guard phi > 0, phi < .pi / 2 else { return nil }
        return phi * 180 / .pi
    }

    func distanceFeet(
        bottomY: Double,
        centerX: Double,
        referencePitchDegreesBelowHorizontal: Double,
        referenceRollDegrees: Double,
        aspectRatio: Double
    ) -> Double? {
        distanceMeters(
            bottomY: bottomY,
            centerX: centerX,
            referencePitchDegreesBelowHorizontal: referencePitchDegreesBelowHorizontal,
            referenceRollDegrees: referenceRollDegrees,
            aspectRatio: aspectRatio
        ).map { $0 * 3.28084 }
    }

    /// Distance implied by a detection's box WIDTH instead of its ground-
    /// contact row -- see WidthDistanceOverride.swift for why this exists
    /// (row-based distance is unreliable below the hood-truncation cutoff,
    /// see `hoodCutoffAngleDegrees`) and how this is actually used (one-
    /// directional override, sanity-capped against real per-class width
    /// stats, confidence-gated, hysteresis-confirmed -- none of that lives
    /// here, this is just the geometry).
    ///
    /// Pinhole similar-triangles formula: `boxWidthNormalized ≈
    /// focalLengthColumnNormalized * realWidthMeters / distanceMeters`,
    /// solved for distance. Small-angle approximation (treats the object's
    /// angular width as `width/distance` rather than `2*atan(width/(2*
    /// distance))`) -- acceptable here since this is only ever consulted at
    /// close range where a wrong-but-plausible row-based reading needs
    /// correcting, not as a general-purpose replacement for the row-based
    /// model at all distances. Reuses `focalLengthNormalized` from the SAME
    /// calibrated instance `distanceMeters` uses -- one calibration, two
    /// consumers, not a second independent fit.
    ///
    /// - Parameters:
    ///   - boxWidthNormalized: a detection's `boundingBox.width` (normalized
    ///     [0, 1], same convention as `distanceMeters`' `bottomY`/`centerX`).
    ///   - realWidthMeters: the object class's real-world width -- see
    ///     `ObjectWidthPriors` for real (KITTI-derived, not guessed) values.
    ///   - aspectRatio: same as `distanceMeters` -- whichever was actually
    ///     active for this detection.
    /// CONFIRMED 2026-08-16 (real 69-minute test drive,
    /// data/26_08_16_TestDrive_LowRes_Nano_Day): a plain `> 0` floor isn't
    /// enough -- a pathologically thin box (a decode/tracking edge case, not
    /// a real detection: a track about to be dropped, a heavily truncated
    /// box at the frame edge) divides down to an absurd distance (observed:
    /// single detections implying tens of thousands of meters, one track's
    /// distance std alone was over 1000m against a ~55m mean). This floor
    /// rejects any box wide enough to be a real division-by-near-zero risk
    /// but not wide enough to plausibly be a real car/pedestrian/cyclist at
    /// any distance the detector could actually resolve one at.
    private static let minPlausibleBoxWidthNormalized: Double = 0.003

    /// - Returns: nil if the box has no meaningful width -- either exactly
    ///   zero (shouldn't happen for a real detection, but would divide by
    ///   zero) or below `minPlausibleBoxWidthNormalized` (see its own doc
    ///   comment -- a real division-by-near-zero risk from a degenerate
    ///   box, not a genuine detection).
    func widthBasedDistanceMeters(
        boxWidthNormalized: Double,
        realWidthMeters: Double,
        aspectRatio: Double
    ) -> Double? {
        guard boxWidthNormalized >= Self.minPlausibleBoxWidthNormalized else { return nil }
        let focalLengthColumnNormalized = focalLengthNormalized / aspectRatio
        return focalLengthColumnNormalized * realWidthMeters / boxWidthNormalized
    }

    /// Angle below horizontal (degrees) at which the ego vehicle's own hood
    /// clips a detection's box -- below this, a detection's row-based
    /// distance no longer reflects real ground contact (the box bottom is
    /// pinned to the hood edge, not the object's actual feet/wheels), so
    /// `distanceMeters` reads a "phantom" distance around 5.9m regardless of
    /// how much closer the real object actually is.
    ///
    /// MEASURED 2026-08-15 from real walkaround-test footage
    /// (data/26_08_15_Walkaround), NOT the earlier tape/eye-measured "~7m
    /// minimum distance" figure it supersedes for this purpose -- that
    /// figure was measured differently and disagreed with this one by over
    /// a meter; this is the one that matches what the detector itself
    /// actually produces, which is what a code-level cutoff needs to agree
    /// with. Method: pooled 107 real YOLO person-detection boxes from the
    /// confirmed-tethered 3m and 5m windows (both well below any plausible
    /// true minimum, so every box in both windows is genuinely hood-
    /// clipped, not just close) -- each box's bottom row was converted to
    /// an angle via `atan(cameraHeightMeters / distanceMeters)`, the exact
    /// inverse of `distanceMeters`' own `height / tan(phi)`. Binned across
    /// the full horizontal width of the frame (5 column bins) to check for
    /// hood-slope dependence: found remarkably flat (row varies by only
    /// ~0.019 normalized units across the ENTIRE frame width), so a single
    /// scalar threshold is adequate -- no column-dependent function needed.
    /// Result: mean 9.899°, std 0.771° (n=107). This constant is the mean;
    /// callers needing a margin above it (e.g. the near-minimum-distance
    /// tier switch) should add their own buffer on top, not treat this as
    /// already-padded.
    static let hoodCutoffAngleDegrees: Double = 9.899

    /// Solves for `principalRowNormalized`/`focalLengthNormalized` from two
    /// (row, column, distance) tape-mark reference points, at whatever
    /// reference pitch AND roll the phone happens to sit at in the mount --
    /// record both with `PitchSensor.captureReferenceAttitude()` at the same
    /// time as the tape-mark measurements and pass them in here. Ground does
    /// NOT need to be flat/level for this -- see the file-level comment on
    /// why a reference (not live) pitch/roll already handles a sloped or
    /// cambered calibration/test area the same way it handles a hill or
    /// banked curve at prediction time. What DOES still matter: the tape
    /// marks and the reference capture should be on the same road surface
    /// (so the captured pitch/roll genuinely describes the camera's angle
    /// relative to THAT ground plane), same as every other reference-pitch
    /// reasoning in this file.
    ///
    /// De-rolling a calibration point requires knowing its column
    /// (`centerX`), not just its row -- CONFIRMED 2026-08-11 as a real gap,
    /// not a theoretical one: an earlier roll-blind version of this function
    /// (row-only, roll assumed zero) was fit against real tape-mark data
    /// captured on a cambered test street (real roll around -2deg, not
    /// zero), and the resulting v0/f varied by ~20% across three different
    /// reference pitches in a way that looked like a pitch bug but wasn't
    /// fully explained by one -- unmodeled roll, correlated with pitch
    /// across those three rounds, was the leading remaining suspect. This
    /// version corrects for that by de-rolling each point the same way
    /// `distanceMeters` does before solving for v0/f, rather than assuming
    /// away roll's effect during calibration and only correcting for it
    /// afterward at prediction time.
    ///
    /// With 3+ tape-mark points, fit on two and check the model's prediction
    /// against the held-out one(s) rather than averaging all of them into a
    /// single fit -- the whole point of the extra point is to catch this
    /// model being wrong, which an all-points fit would hide (same
    /// discipline as the leading-vehicle classifier's train/held-out split).
    ///
    /// Once this is fit, the strongest validation isn't another held-out
    /// point at the SAME reference pitch -- it's retilting the phone in the
    /// mount (still on the same ground), capturing a NEW reference
    /// pitch/roll, and feeding that through `distanceMeters` with these same
    /// v0/f constants to check the prediction against the same tape marks
    /// without refitting anything. That's what actually tests whether v0/f
    /// are genuinely pitch/roll-independent, not just whether the two-point
    /// fit is self-consistent at one attitude.
    static func fit(
        cameraHeightMeters: Double,
        referencePitchDegreesBelowHorizontal: Double,
        referenceRollDegrees: Double,
        aspectRatio: Double,
        row1: Double, centerX1: Double, distance1Meters: Double,
        row2: Double, centerX2: Double, distance2Meters: Double
    ) -> DistanceEstimator? {
        guard distance1Meters > 0, distance2Meters > 0 else { return nil }
        let theta = referencePitchDegreesBelowHorizontal * .pi / 180
        let psi = referenceRollDegrees * .pi / 180
        let cosPsi = cos(psi)
        guard cosPsi != 0 else { return nil }

        // Same derivation as `distanceMeters`, solved backwards: starting
        // from derolledY = -x*sin(psi) + y*cos(psi) = tan(alpha), substitute
        // x = (centerX - 0.5)*aspectRatio/f and y = (row - v0)/f, then solve
        // for v0 in terms of f. The result stays linear in (v0, f) -- each
        // point's roll/column contribution folds into an adjusted row and a
        // rescaled x, after which this is the same 2-point linear solve as
        // the roll-blind version, just with these adjusted inputs instead of
        // raw row/x.
        func knownX(_ distanceMeters: Double) -> Double {
            tan(atan(cameraHeightMeters / distanceMeters) - theta) / cosPsi
        }
        func adjustedRow(_ row: Double, _ centerX: Double) -> Double {
            row - (centerX - Self.assumedPrincipalColumnNormalized) * aspectRatio * tan(psi)
        }

        let x1 = knownX(distance1Meters)
        let x2 = knownX(distance2Meters)
        let denom = x1 - x2
        guard denom != 0 else { return nil }
        let v1 = adjustedRow(row1, centerX1)
        let v2 = adjustedRow(row2, centerX2)
        let f = (v1 - v2) / denom
        let v0 = v1 - f * x1
        return DistanceEstimator(
            cameraHeightMeters: cameraHeightMeters,
            principalRowNormalized: v0,
            focalLengthNormalized: f
        )
    }

    // MARK: - Lever arm (camera-to-rear-axle offset)
    //
    // The rotational half of the translational+rotational optic-flow
    // decomposition from the following-distance design discussion
    // (Longuet-Higgins/Prazdny): when the vehicle yaws, a camera mounted
    // off the yaw rotation center (the rear axle, standard bicycle-model
    // assumption) has its own velocity contribution from that rotation,
    // v = omega x r, on top of the vehicle's translational (forward) speed.
    // Groundwork for the still-unwritten full ego-motion optic-flow work --
    // not consumed anywhere yet.
    //
    // Vehicle body frame: x = forward, y = left, z = up (right-handed,
    // matches PitchSensor.yawRateDegreesPerSecond's confirmed sign
    // convention: positive omega = counter-clockwise viewed from above =
    // turning left). For pure yaw (omega purely vertical, omega = (0,0,
    // omega_z)) and r = (r_x, r_y, r_z), the cross product drops the
    // vertical component of r entirely:
    //     v = omega x r = (-omega_z * r_y, omega_z * r_x, 0)
    // -- camera height plays no part, only the horizontal offsets do.
    //
    // UPDATED 2026-08-15 for the SHAPE clamp mount -- supersedes the
    // 2026-08-09 figures below (old RAM mount), which are kept here for
    // history/context, not current use.
    //
    // r_x: MEASURED 2026-08-15 with a tape measure -- 2.17m forward of the
    // rear axle (also measured: 68cm to the front axle, i.e. 2.85m implied
    // wheelbase -- a sanity-check figure, not itself used here. Matches the
    // published 2021 Nissan Altima SR wheelbase, 282.4cm, to within 2.6cm --
    // same margin as the old mount's equivalent check, a good consistency
    // sign for both measurements).
    // r_y: MEASURED 2026-08-15 -- 26cm to the driver's-left of center.
    // Methodology (direct centerline measurement vs. an assumption-based
    // derivation, as the old 6cm figure was) not specified this time --
    // taken at face value as given, not re-derived. The jump from 6cm to
    // 26cm is real and expected, not a red flag -- this mount physically
    // positions the camera differently than the old dash-centered RAM
    // mount did, not a measurement error.
    //
    // (2026-08-09, old RAM mount, superseded: r_x=2.03m forward of rear
    // axle, 82cm to front axle, ~2.85m implied wheelbase. r_y=6cm left of
    // center, derived not measured -- assumed the iPhone itself, not the
    // lens, was centered on the dash, then added the confirmed 5.92cm
    // lens-to-phone-body lateral offset.)
    enum LeverArm {
        static let forwardOfRearAxleMeters: Double = 2.17
        static let leftOfCenterlineMeters: Double = 0.26

        /// Camera's own velocity due to yaw rotation about the rear axle,
        /// in the vehicle's body frame (x = forward, y = left, meters/
        /// second). Pass `PitchSensor.smoothedYawRateDegreesPerSecond`, NOT
        /// the raw `yawRateDegreesPerSecond` -- both are live readings
        /// (unlike pitch/roll's reference-not-live values -- yaw RATE is a
        /// dynamic quantity, there's no "hill grade" equivalent
        /// misattribution risk a static angle has), but CONFIRMED 2026-08-16
        /// against a real drive that the raw per-frame signal is too noisy
        /// for a per-frame consumer like this one (see that property's own
        /// doc comment) -- this is exactly the smoothed value's intended use.
        ///
        /// This is only the rotational term -- for the camera's TOTAL
        /// body-frame velocity, add the vehicle's own forward (translational)
        /// speed to the forward component separately, e.g. GPS
        /// `EgoSpeedManager` speed: `(egoSpeedMetersPerSecond +
        /// result.forwardMetersPerSecond, result.leftMetersPerSecond)`.
        static func cameraVelocityFromYaw(
            yawRateDegreesPerSecond: Double
        ) -> (forwardMetersPerSecond: Double, leftMetersPerSecond: Double) {
            let omega = yawRateDegreesPerSecond * .pi / 180
            return (
                forwardMetersPerSecond: -omega * leftOfCenterlineMeters,
                leftMetersPerSecond: omega * forwardOfRearAxleMeters
            )
        }
    }
}
