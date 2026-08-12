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

    /// The live app's actual calibrated instance -- fit 2026-08-11 from
    /// data/26_08_11_DistanceCalibration/ round 0 (near=8.1m/far=19.92m tape
    /// marks, reference pitch -2.561deg/roll -2.322deg) via the roll-aware
    /// `DistanceEstimator.fit()`. Tape-mark rows were re-read by hand
    /// (visual arrow-key line-up tool against the source video frames), and
    /// columns were read via targeted color-sampling on the same frames.
    ///
    /// *** Reference pitch is NEGATIVE here on purpose -- worth flagging
    /// *** since it looks backwards at a glance. The raw debug-logged pitch
    /// *** for this round was +2.561deg, captured BEFORE the PitchSensor
    /// *** sign fix earlier in this file; the corrected value (what a fixed
    /// *** sensor would have read) is that raw value negated, i.e. -2.561.
    /// *** A same-session bug (not a formula or hardware issue) came from
    /// *** re-deriving these constants from a written summary after a
    /// *** context-compaction boundary and losing track of which pitch
    /// *** values already had this negation applied vs which were still
    /// *** raw -- every fit run afterward used the raw (wrong-sign) value
    /// *** directly, which alone produced the ~20% cross-round v0 spread
    /// *** that briefly looked like unmodeled roll, or even like each
    /// *** separate calibration recording having a different effective FOV.
    /// *** Neither of those was real -- using the correctly-negated pitch
    /// *** resolves it completely: fitting round0 alone now predicts all 6
    /// *** held-out points from rounds 1/2 (different pitches, different
    /// *** recordings) to within 2.9% mean / 6.4% max error, down from the
    /// *** ~20-30%+ errors seen before this fix. Lesson for next time: when
    /// *** resuming work on a signed quantity after a compaction boundary,
    /// *** re-derive the sign from the actual source script/call site, not
    /// *** from a written description of it.
    /// cameraHeightMeters is the 2026-08-11 remeasurement with the new mount.
    static let calibrated = DistanceEstimator(
        cameraHeightMeters: 1.015,
        principalRowNormalized: 0.500504,
        focalLengthNormalized: 1.412226
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
    /// - Returns: nil if the resulting ground-plane angle isn't a valid point
    ///   in front of and below the camera (at/above the horizon, or beyond
    ///   the straight-down degenerate case) -- geometrically not a finite
    ///   forward distance, so treat as "can't estimate" rather than clamping
    ///   to some default.
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
        // own roll before reading off the vertical angle -- see the roll
        // sign-convention warning at the top of this file.
        let derolledY = -x * sin(psi) + y * cos(psi)

        let alpha = atan(derolledY)
        let phi = alpha + theta
        guard phi > 0, phi < .pi / 2 else { return nil }
        return cameraHeightMeters / tan(phi)
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
    // r_x: MEASURED 2026-08-09 with a tape measure -- 2.03m forward of the
    // rear axle (also measured: 82cm to the front axle, i.e. ~2.85m
    // wheelbase -- a sanity-check figure, not itself used here).
    // r_y: DERIVED 2026-08-09, not an independent measurement -- computed
    // by assuming the iPhone itself (not the camera lens) is centered on
    // the dash, then adding the confirmed 5.92cm lens-to-phone-body lateral
    // offset (see PitchSensor/ContentView's camera-orientation-warning
    // history). ~6cm to the driver's-left of the vehicle's true centerline,
    // +/-1cm combined uncertainty -- softer than r_x, since the dash-
    // centering half is still an assumption, not a caliper measurement.
    enum LeverArm {
        static let forwardOfRearAxleMeters: Double = 2.03
        static let leftOfCenterlineMeters: Double = 0.06

        /// Camera's own velocity due to yaw rotation about the rear axle,
        /// in the vehicle's body frame (x = forward, y = left, meters/
        /// second). Pass `PitchSensor.yawRateDegreesPerSecond` (a live
        /// reading, unlike pitch/roll's reference-not-live values -- yaw
        /// RATE is a dynamic quantity, there's no "hill grade" equivalent
        /// misattribution risk the way a static angle has).
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
