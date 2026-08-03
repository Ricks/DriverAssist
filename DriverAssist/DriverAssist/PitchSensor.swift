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
//  vision-based lane curvature. Not consumed yet either; see the file-level
//  note on rotationRateXDegreesPerSecond etc. below for why this logs all
//  three raw axes rather than a single resolved "yaw rate".
//
//  Deliberately uses .xArbitraryZVertical (gravity + gyro only) rather than
//  a magnetic-north reference frame -- pitch and roll derive from gravity
//  direction alone and don't need the magnetometer, which is unreliable
//  inside a vehicle (see the CoreMotion-for-GMC design discussion). No
//  permission needed: CMMotionManager's raw device motion doesn't require
//  the Motion & Fitness prompt CMPedometer/CMMotionActivity do.
//
//  NOT YET VERIFIED against a real mounted phone: whether CMDeviceMotion's
//  pitch axis/sign convention matches "camera tilted down toward the road"
//  in this app's actual landscape mounted orientation is unconfirmed --
//  check this against reality once the real mount is in use, don't trust
//  the sign blindly.
//

import CoreMotion
import Foundation

@MainActor
final class PitchSensor: ObservableObject {
    @Published private(set) var pitchDegrees: Double?
    @Published private(set) var referencePitchDegrees: Double?

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

    /// Same gravity-derived, non-drifting property as pitch -- logged for the
    /// same reason: cheap to capture now, and a road-banking/vehicle-lean
    /// signal on a curve isn't available any other way. Not consumed by
    /// anything yet.
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

    private let motionManager = CMMotionManager()
    private static let referenceDefaultsKey = "settings.referencePitchDegrees"

    init() {
        if UserDefaults.standard.object(forKey: Self.referenceDefaultsKey) != nil {
            referencePitchDegrees = UserDefaults.standard.double(forKey: Self.referenceDefaultsKey)
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
            self?.pitchDegrees = motion.attitude.pitch * 180 / .pi
            self?.rollDegrees = motion.attitude.roll * 180 / .pi
            self?.rotationRateXDegreesPerSecond = motion.rotationRate.x * 180 / .pi
            self?.rotationRateYDegreesPerSecond = motion.rotationRate.y * 180 / .pi
            self?.rotationRateZDegreesPerSecond = motion.rotationRate.z * 180 / .pi
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
    /// the witness mark) to save the current reading as the reference future
    /// drives compare against. Exposed here as a plain method -- not yet
    /// wired to any UI/voice trigger beyond VoiceCommandManager's "calibrate
    /// pitch" command.
    func captureReferencePitch() {
        guard let pitch = pitchDegrees else { return }
        referencePitchDegrees = pitch
        UserDefaults.standard.set(pitch, forKey: Self.referenceDefaultsKey)
    }
}
