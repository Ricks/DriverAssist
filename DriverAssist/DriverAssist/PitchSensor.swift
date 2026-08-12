//
//  PitchSensor.swift
//  DriverAssist
//
//  Publishes the phone's own pitch/roll, raw gyro rotation rate, raw linear
//  (gravity-subtracted) acceleration, and the raw gravity vector, from
//  CoreMotion. Deliberately logs more than any single planned use needs --
//  see the design discussion this came out of: a few extra Double fields
//  per log line costs nothing, while not having a signal when a future
//  hypothesis needs it means an entire wasted test drive.
//
//  Pitch is a static/DC reading derived from the gravity vector, not an
//  integration, so unlike ego-speed this doesn't drift over time. Two
//  intended uses, neither built yet: (1) the mount-reattachment drift check
//  -- compare a live reading against a reference captured once at
//  calibration time, to catch the phone coming back at a different angle
//  after being unclipped/reclipped (see the mount calibration design
//  discussion -- plain 1/4"-20 threads have no rotational index, so this is
//  a real, expected risk, not hypothetical); (2) potentially live-correcting
//  the future ground-plane distance formula for transient pitch changes
//  under braking/acceleration, if real data ever shows that matters enough
//  to be worth it.
//
//  Rotation rate is intended for guessing whether the car is currently
//  going through a curve/turn -- classify_leading's central-band assumption
//  (the followed vehicle stays near frame-center) breaks down on a curve,
//  and a real yaw-rate signal is a much cheaper way to detect that than
//  vision-based lane curvature. Not consumed yet; see the file-level note on
//  rotationRateXDegreesPerSecond etc. below for why this logs all three raw
//  axes AND a resolved yawRateDegreesPerSecond (rotating rotationRate into
//  world coordinates via the current attitude, rather than guessing which
//  raw axis is yaw) -- the raw axes stay logged for cross-checking the
//  resolved value against on a real drive, not because the resolved value
//  is provisional.
//
//  Deliberately uses .xArbitraryZVertical (gravity + gyro only) rather than
//  a magnetic-north reference frame -- pitch and roll derive from gravity
//  direction alone and don't need the magnetometer, which is unreliable
//  inside a vehicle (see the CoreMotion-for-GMC design discussion). No
//  permission needed: CMMotionManager's raw device motion doesn't require
//  the Motion & Fitness prompt CMPedometer/CMMotionActivity do.
//
//  CONFIRMED BROKEN 2026-08-09, then fixed: raw CMDeviceMotion `.pitch`
//  (a CMAttitude Euler angle) doesn't mean "nose tilted down toward the
//  road" in this app's landscape mount at all -- see pitchDegrees's own
//  derivation below for the gimbal-lock reason and the gravity-vector fix
//  (same underlying issue rollDegrees already hit and was fixed for).
//  pitchDegrees's new sign is reasoned, not yet bench-confirmed -- don't
//  trust it blindly either, same discipline as everything else here.
//
//  YAW RATE SIGN CONVENTION: CONFIRMED 2026-08-06 via bench test (hand-
//  rotating the mounted phone, watching yawRateDegreesPerSecond live) --
//  turning right gives negative, turning left gives positive, matching the
//  documented convention (positive = counter-clockwise viewed from above).
//  This validates the CMAttitude.rotationMatrix device-to-world assumption
//  the projection relies on. NOTE: this only confirms the sign, not the
//  magnitude/accuracy -- still needs the real-drive cross-checks (hand-
//  labeled turns, GPS-course differentiation, accelerometer a_lateral ~= v*w)
//  from the validation plan before trusting it for anything quantitative,
//  e.g. classify_leading's central-band loosening.
//

import CoreMotion
import Foundation

@MainActor
final class PitchSensor: ObservableObject {
    @Published private(set) var pitchDegrees: Double?
    @Published private(set) var referencePitchDegrees: Double?
    @Published private(set) var referenceRollDegrees: Double?

    /// Raw gyro rotation rate (deg/s), in the device's own x/y/z axes --
    /// deliberately NOT resolved down to a single "yaw rate" here, because
    /// which physical axis actually corresponds to the vehicle's turning
    /// (vertical/yaw) axis depends on how the phone sits in the mount, which
    /// isn't finalized yet (see the mount/calibration design discussion).
    /// Logging all three now and picking the right axis (or projecting
    /// through the attitude quaternion into world-frame yaw, which is the
    /// mathematically correct way to do this regardless of mount
    /// orientation) once real mounted-drive data exists to check it against
    /// -- same reasoning as pitchDegrees's own axis/sign caveat below.
    @Published private(set) var rotationRateXDegreesPerSecond: Double?
    @Published private(set) var rotationRateYDegreesPerSecond: Double?
    @Published private(set) var rotationRateZDegreesPerSecond: Double?

    /// World-frame yaw rate (deg/s, positive = counter-clockwise viewed from
    /// above -- standard right-hand rule about the vertical axis) -- the
    /// resolved value the three raw axes above exist to make possible.
    /// Computed by rotating device-frame `rotationRate` into the
    /// `.xArbitraryZVertical` reference frame via the current attitude's
    /// rotation matrix and reading off the vertical (Z) component:
    /// `omega_world = R(attitude) * omega_device`. This is exact regardless
    /// of mount tilt/orientation -- unlike picking one of the three raw axes
    /// by guessing, it doesn't assume anything about how the phone sits in
    /// the mount, since `attitude` already captures that. Sign confirmed via
    /// bench test 2026-08-06 (see the top-of-file note) -- magnitude/accuracy
    /// still needs real-drive validation before this is trustworthy for
    /// anything quantitative.
    @Published private(set) var yawRateDegreesPerSecond: Double?

    /// Rotation about the camera's own forward (boresight) axis -- how
    /// tilted the video frame's horizon is, i.e. "degrees from level" for
    /// the mount. Deliberately NOT `CMAttitude.roll`: that Euler angle
    /// rotates about the device's long/portrait axis, which in this app's
    /// landscape mount is horizontal, not the boresight axis this needs --
    /// and more importantly, the mount's SHORT axis sits vertical (see
    /// cameraOrientationWarning's confirmed gravityX reading), which is
    /// exactly CMAttitude's PITCH axis. With pitch parked near +/-90 in
    /// normal mounted use, the yaw-pitch-roll decomposition sits in (or
    /// next to) gimbal lock and `.roll` comes out as unstable garbage --
    /// CONFIRMED bad 2026-08-08 via bench test: flat on a desk (away from
    /// the singularity) read a correct 0, but stood up on its landscape
    /// edge (its actual mounted orientation) read -88 instead of ~0.
    /// Computed directly from the gravity vector below instead, which has
    /// no such singularity.
    @Published private(set) var rollDegrees: Double?

    /// Real linear acceleration with gravity subtracted out (in g, i.e. 1.0 =
    /// ~9.8 m/s^2), device-local x/y/z axes -- same axis-mapping caveat as
    /// rotationRate above (not resolved to a single "forward/lateral" axis
    /// here). Not consumed by anything yet; intended future uses are
    /// detecting braking (feeds the closing-rate/following-distance warning
    /// work) and cross-checking hard cornering against rotationRate (two
    /// independent sensors of the same maneuver).
    @Published private(set) var userAccelerationX: Double?
    @Published private(set) var userAccelerationY: Double?
    @Published private(set) var userAccelerationZ: Double?

    /// Raw gravity vector (device-local x/y/z, unit vector) -- what
    /// pitchDegrees/rollDegrees are themselves derived from. Logged
    /// separately, raw, so pitch/roll could be recomputed differently later
    /// (different smoothing, a different axis convention once the mount
    /// orientation is confirmed) without needing a new drive -- same
    /// "log raw, don't pre-resolve" reasoning as rotationRate's three axes.
    @Published private(set) var gravityX: Double?
    @Published private(set) var gravityY: Double?
    @Published private(set) var gravityZ: Double?

    /// nil until both a live reading and a saved reference exist.
    var pitchDriftDegrees: Double? {
        guard let pitch = pitchDegrees, let reference = referencePitchDegrees else { return nil }
        return pitch - reference
    }

    /// Same idea as `pitchDriftDegrees` -- nil until both a live reading and
    /// a saved reference exist.
    var rollDriftDegrees: Double? {
        guard let roll = rollDegrees, let reference = referenceRollDegrees else { return nil }
        return roll - reference
    }

    private let motionManager = CMMotionManager()
    private static let referencePitchDefaultsKey = "settings.referencePitchDegrees"
    private static let referenceRollDefaultsKey = "settings.referenceRollDegrees"

    init() {
        if UserDefaults.standard.object(forKey: Self.referencePitchDefaultsKey) != nil {
            referencePitchDegrees = UserDefaults.standard.double(forKey: Self.referencePitchDefaultsKey)
        }
        if UserDefaults.standard.object(forKey: Self.referenceRollDefaultsKey) != nil {
            referenceRollDegrees = UserDefaults.standard.double(forKey: Self.referenceRollDefaultsKey)
        }
    }

    func start() {
        guard motionManager.isDeviceMotionAvailable else {
            print("[PitchSensor] device motion not available")
            return
        }
        // Matches the camera's own ~15fps capture cadence -- this used to be a
        // deliberate slow 0.5s poll (fine for pitch's original static-
        // calibration use), but rotationRate is now meant to inform per-frame
        // curve detection (see path_awareness.py), and a stale-by-up-to-0.5s
        // reading would often belong to the wrong frame entirely. CoreMotion
        // sampling at 15Hz is negligible cost next to camera/ML inference
        // already running -- not the kind of resolution/model-size tradeoff
        // that needed real measurement earlier this project.
        motionManager.deviceMotionUpdateInterval = 1.0 / 15.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, error in
            guard let motion else {
                if let error { print("[PitchSensor] update failed: \(error)") }
                return
            }
            // Same gimbal-lock problem rollDegrees already hit, just for
            // pitch: CMAttitude.pitch rotates about the device's LOCAL X
            // axis, but that axis is real-world VERTICAL in this landscape
            // mount (confirmed via gravityX) -- so raw `.pitch` was really
            // reading something closer to yaw/heading, not nose-up/down
            // tilt. CONFIRMED bad 2026-08-09 via bench test: phone standing
            // upright in its mounted landscape orientation, tilted flat
            // onto a table (a genuine ~90 degree nose-down tilt), only
            // moved pitchDegrees a few degrees. Fixed the same way roll
            // was: derive directly from the gravity vector instead. sqrt(x^2
            // + y^2) is the gravity magnitude in the horizontal (X-Y) plane
            // -- rolling about the boresight (Z) axis only exchanges
            // gravity between X and Y, leaving this combined magnitude (and
            // Z itself) unchanged, so atan2(z, sqrt(x^2+y^2)) comes out
            // independent of roll, by the same kind of trig identity as
            // rollDegrees's own derivation.
            //
            // SIGN CONFIRMED WRONG 2026-08-11 via hand-tilt bench test (see
            // [[project-following-distance-measurement]]): the un-negated
            // formula below read ~0deg held level and DECREASED toward -90deg
            // tilted nose-down -- backwards from the nose-down-positive
            // convention this was originally "chosen" for for (see git
            // history/prior comment) and that DistanceEstimator.swift's
            // formula assumes. Confirmed via DistanceEstimator.fit() cross-
            // pitch validation first (un-negated pitch gave up to +3806%
            // error and a physically-impossible negative fitted focal
            // length; negating it dropped errors to low double digits),
            // then verified directly by hand-tilting the mounted phone.
            // Negated here (not in DistanceEstimator) since this is the
            // actual root cause -- every consumer of pitchDegrees/
            // referencePitchDegrees (display, logging, DistanceEstimator)
            // already expected nose-down-positive, this formula just wasn't
            // delivering that.
            let gx = motion.gravity.x, gy = motion.gravity.y, gz = motion.gravity.z
            self?.pitchDegrees = -atan2(gz, (gx * gx + gy * gy).squareRoot()) * 180 / .pi
            // Rotating about the boresight (device Z) axis exchanges gravity
            // between X and Y while leaving Z untouched, so atan2(y, -x)
            // isolates exactly that rotation -- and, per the small-angle
            // trig identity, comes out independent of pitch/nose-tilt too
            // (see pitchDegrees's derivation above, same underlying trig).
            // -x because the confirmed level+correct-orientation reading is
            // gravityX ~= -1, which this needs to map to 0 degrees. Sign
            // not yet bench-confirmed -- same discipline as gravityX's own
            // mount-orientation check.
            self?.rollDegrees = atan2(gy, -gx) * 180 / .pi
            self?.rotationRateXDegreesPerSecond = motion.rotationRate.x * 180 / .pi
            self?.rotationRateYDegreesPerSecond = motion.rotationRate.y * 180 / .pi
            self?.rotationRateZDegreesPerSecond = motion.rotationRate.z * 180 / .pi

            // omega_world = R(attitude) * omega_device -- see
            // yawRateDegreesPerSecond's doc comment. Only the Z row of R is
            // needed since we only want the world-vertical component.
            let r = motion.attitude.rotationMatrix
            let omega = motion.rotationRate
            let yawRateRadPerSecond = r.m31 * omega.x + r.m32 * omega.y + r.m33 * omega.z
            self?.yawRateDegreesPerSecond = yawRateRadPerSecond * 180 / .pi
            self?.userAccelerationX = motion.userAcceleration.x
            self?.userAccelerationY = motion.userAcceleration.y
            self?.userAccelerationZ = motion.userAcceleration.z
            self?.gravityX = motion.gravity.x
            self?.gravityY = motion.gravity.y
            self?.gravityZ = motion.gravity.z
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
    }

    /// Call once the mount is freshly set up (phone reattached, aligned to
    /// the witness mark) and the vehicle is on known-flat, level ground --
    /// saves the current pitch AND roll together as the reference both
    /// `DistanceEstimator` and future drift checks use. Pitch and roll are
    /// captured together, not separately, since both need the same flat-
    /// ground precondition to be meaningful (see DistanceEstimator.swift's
    /// file-level comment on why live/road-tilted readings can't substitute
    /// for this). Returns the captured values so a caller (the UI button)
    /// can show them for an immediate sanity check -- nil if no motion
    /// reading is available yet to capture.
    @discardableResult
    func captureReferenceAttitude() -> (pitch: Double, roll: Double)? {
        guard let pitch = pitchDegrees, let roll = rollDegrees else { return nil }
        referencePitchDegrees = pitch
        referenceRollDegrees = roll
        UserDefaults.standard.set(pitch, forKey: Self.referencePitchDefaultsKey)
        UserDefaults.standard.set(roll, forKey: Self.referenceRollDefaultsKey)
        return (pitch, roll)
    }
}
