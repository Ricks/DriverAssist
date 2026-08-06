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
//  *** SIGN CONVENTION WARNING, NOT YET EMPIRICALLY CONFIRMED ***
//  Every angle here assumes theta is signed positive when the camera's nose
//  is tilted DOWN toward the road. PitchSensor.pitchDegrees/referencePitchDegrees
//  are CMAttitude's raw pitch, which depends on this mount's specific
//  physical orientation (landscape/portrait, which edge faces which way)
//  composed with CameraManager's own video-rotation handling -- neither of
//  those compositions has been empirically checked yet. If the real sign is
//  backwards, this formula doesn't error, it just silently inverts the pitch
//  correction (amplifying drift instead of cancelling it). Verify by tilting
//  the mounted phone's nose down by hand and confirming pitchDegrees
//  increases BEFORE wiring reference pitch in here.
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
//  *** ROLL SIGN CONVENTION, NOT YET EMPIRICALLY CONFIRMED (same caveat as
//  *** pitch above, independently) *** -- assumes positive
//  *** referenceRollDegrees corresponds to the image rotating counter-
//  *** clockwise as the camera sees it. Verify by rolling the mounted phone
//  *** a known way by hand and confirming a known off-center point's
//  *** de-rolled position moves the predicted direction, before trusting
//  *** this for anything safety-relevant.
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
    /// (row, distance) tape-mark reference points measured on flat ground, at
    /// whatever reference pitch the phone happens to sit at in the mount --
    /// record that pitch with `PitchSensor.captureReferenceAttitude()` at the
    /// same time as the tape-mark measurements and pass it in here. This fit
    /// itself doesn't take a roll parameter -- it assumes roll was near zero
    /// at calibration time (bubble-level step of the same routine), same as
    /// every other tape-mark measurement implicitly assumes a level phone.
    ///
    /// With 3+ tape-mark points, fit on two and check the model's prediction
    /// against the held-out one(s) rather than averaging all of them into a
    /// single fit -- the whole point of the extra point is to catch this
    /// model being wrong, which an all-points fit would hide (same
    /// discipline as the leading-vehicle classifier's train/held-out split).
    ///
    /// Once this is fit, the strongest validation isn't another held-out
    /// point at the SAME reference pitch -- it's retilting the phone in the
    /// mount (still on the same flat ground), capturing a NEW reference
    /// pitch, and feeding that through `distanceMeters` with these same
    /// v0/f constants to check the prediction against the same tape marks
    /// without refitting anything. That's what actually tests whether v0/f
    /// are genuinely pitch-independent, not just whether the two-point fit
    /// is self-consistent at one pitch. Do this on flat ground only --
    /// retilting on a slope would conflate the thing being tested (pitch
    /// independence) with the hill-grade effect described at the top of this
    /// file.
    static func fit(
        cameraHeightMeters: Double,
        referencePitchDegreesBelowHorizontal: Double,
        row1: Double, distance1Meters: Double,
        row2: Double, distance2Meters: Double
    ) -> DistanceEstimator? {
        guard distance1Meters > 0, distance2Meters > 0 else { return nil }
        let theta = referencePitchDegreesBelowHorizontal * .pi / 180
        // v_i = v0 + f * x_i, where x_i = tan(atan(H/D_i) - theta) is fully
        // known once H, D_i, and theta are -- reduces to the same 2-point
        // linear solve as the old model, just with this nonlinear x instead
        // of the old model's 1/D.
        let x1 = tan(atan(cameraHeightMeters / distance1Meters) - theta)
        let x2 = tan(atan(cameraHeightMeters / distance2Meters) - theta)
        let denom = x1 - x2
        guard denom != 0 else { return nil }
        let f = (row1 - row2) / denom
        let v0 = row1 - f * x1
        return DistanceEstimator(
            cameraHeightMeters: cameraHeightMeters,
            principalRowNormalized: v0,
            focalLengthNormalized: f
        )
    }
}
