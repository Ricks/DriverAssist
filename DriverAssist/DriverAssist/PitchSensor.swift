//
//  PitchSensor.swift
//  DriverAssist
//
//  Publishes the phone's own pitch from CoreMotion -- a static/DC reading
//  derived from the gravity vector, not an integration, so unlike ego-speed
//  this doesn't drift over time. Two intended uses, neither built yet: (1)
//  the mount-reattachment drift check -- compare a live reading against a
//  reference captured once at calibration time, to catch the phone coming
//  back at a different angle after being unclipped/reclipped (see the mount
//  calibration design discussion -- plain 1/4"-20 threads have no
//  rotational index, so this is a real, expected risk, not hypothetical);
//  (2) potentially live-correcting the future ground-plane distance formula
//  for transient pitch changes under braking/acceleration, if real data
//  ever shows that matters enough to be worth it.
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
        // Slow poll deliberately -- this needs no frame-rate cadence, unlike
        // the frame-to-frame rotation compensation GMC.swift already does;
        // conflating the two was a real mistake worth not repeating (see
        // design discussion on why "wiring up CoreMotion anyway" doesn't
        // meaningfully reduce the cost of a GMC replacement).
        motionManager.deviceMotionUpdateInterval = 0.5
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, error in
            guard let motion else {
                if let error { print("[PitchSensor] update failed: \(error)") }
                return
            }
            self?.pitchDegrees = motion.attitude.pitch * 180 / .pi
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
