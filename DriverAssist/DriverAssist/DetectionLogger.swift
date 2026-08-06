//
//  DetectionLogger.swift
//  DriverAssist
//
//  Writes one JSON line per completed inference (model, timing, and every
//  detection's label/confidence/box) so a drive's real-time detections can be
//  compared offline against a stronger reference model run on the same
//  recording — visual inspection can't reliably judge small accuracy
//  differences between model tiers.
//

import Foundation

struct DetectionLogEntry: Codable {
    struct Box: Codable {
        let label: String
        let confidence: Float
        // Normalized [0, 1], top-left origin — matches Detection.boundingBox.
        let x: Double
        let y: Double
        let w: Double
        let h: Double
        // The on-device ByteTracker's actual live-assigned identity for this
        // box (nil for a low-confidence detection that matched nothing) —
        // logged so a drive's *real* tracking output can be compared offline
        // against tools/tracker.py's from-scratch recomputation on the same
        // raw detections, not just assumed to match it. Not comparable
        // across drives/app launches — IDs are only stable within one
        // continuous run of the tracker.
        let trackID: Int?
    }

    let t: Double
    let model: String
    // Actual input resolution used for this frame's inference ("1152x640" or
    // "1920x1088") -- see ModelManager.actualInputResolutionLabel. Logged
    // per-entry, same as `model`, since this can also change mid-drive.
    let resolution: String
    let twoPass: Bool
    let elapsedMs: Double
    let lowLightEnabled: Bool
    let autoLowLightEnabled: Bool
    let stabilizationEnabled: Bool
    // Which TrackingLevel was active for this entry -- mirrors `model`/
    // `twoPass` in being per-entry, since this can change mid-drive too.
    let trackingLevel: String
    // Wall-clock cost of trackingManager.track(...) (Kalman predict/update,
    // GMC, Hungarian assignment) -- NOT included in elapsedMs above, which is
    // captured before tracking runs. Logged separately since this is the
    // number that actually answers "what does tracking/GMC cost", which
    // elapsedMs alone can't.
    let trackingElapsedMs: Double
    // GMC's per-frame quality signal (see GMCStats) -- nil when GMC didn't
    // run this frame (disabled, no live tracks yet, or the very first frame).
    // Logged so degraded correspondence/inlier counts on low-texture footage
    // (e.g. at night) show up as data instead of a visual guess from playback.
    let gmcCorrespondenceCount: Int?
    let gmcInlierCount: Int?
    let gmcElapsedMs: Double?
    // Ego speed (GPS, m/s) and phone pitch (degrees) -- see EgoSpeedManager.swift
    // and PitchSensor.swift. Neither is consumed by anything yet; logged now so
    // real drive data exists once the leading-vehicle classifier and distance
    // work actually need it. nil whenever no reading is available yet (no GPS
    // fix, or device motion unavailable) -- not defaulted to 0, which would
    // misrepresent "no data" as "stationary"/"level".
    let egoSpeedMps: Double?
    let egoSpeedAccuracyMps: Double?
    let pitchDegrees: Double?
    let pitchDriftDegrees: Double?
    // Raw gyro rotation rate (deg/s, device-local x/y/z axes) -- see
    // PitchSensor.swift. Logged raw (not resolved to a single "yaw rate")
    // since which axis is actually yaw depends on final mount orientation,
    // not yet confirmed. Not consumed by anything yet; intended future use
    // is guessing whether the car is in a curve/turn, for classify_leading.
    let rotationRateXDegreesPerSecond: Double?
    let rotationRateYDegreesPerSecond: Double?
    let rotationRateZDegreesPerSecond: Double?
    // Same gravity-derived, non-drifting property as pitch -- see PitchSensor.swift.
    let rollDegrees: Double?
    // Real linear acceleration, gravity subtracted, in g (1.0 = ~9.8 m/s^2),
    // device-local x/y/z -- see PitchSensor.swift. Not consumed by anything
    // yet; intended future uses are braking detection (closing-rate warning
    // work) and cross-checking hard cornering against rotationRate.
    let userAccelerationX: Double?
    let userAccelerationY: Double?
    let userAccelerationZ: Double?
    // Raw gravity vector (device-local x/y/z, unit vector) -- what
    // pitchDegrees/rollDegrees are themselves derived from. Logged raw so
    // they could be recomputed differently later without a new drive.
    let gravityX: Double?
    let gravityY: Double?
    let gravityZ: Double?
    let detections: [Box]
}

enum DetectionLogger {
    private static let url: URL = {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return dir.appendingPathComponent("detections.jsonl")
    }()

    private static let queue = DispatchQueue(label: "DetectionLogger")
    private static let encoder = JSONEncoder()

    /// Archives (never deletes) the previous run's log — same reasoning as
    /// `DebugFileLogger.reset()`: a later unrelated launch shouldn't destroy a
    /// drive's data before it's been pulled off the device.
    static func reset() {
        queue.async {
            let fm = FileManager.default
            guard fm.fileExists(atPath: url.path) else { return }
            let archiveURL = url.deletingLastPathComponent()
                .appendingPathComponent("detections-\(Int(Date().timeIntervalSince1970)).jsonl")
            try? fm.moveItem(at: url, to: archiveURL)
        }
    }

    static func log(
        timestamp: Double,
        model: String,
        resolution: String,
        twoPass: Bool,
        elapsedMs: Double,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        stabilizationEnabled: Bool,
        trackingLevel: String,
        trackingElapsedMs: Double,
        gmcStats: GMCStats?,
        egoSpeedMps: Double?,
        egoSpeedAccuracyMps: Double?,
        pitchDegrees: Double?,
        pitchDriftDegrees: Double?,
        rotationRateXDegreesPerSecond: Double?,
        rotationRateYDegreesPerSecond: Double?,
        rotationRateZDegreesPerSecond: Double?,
        rollDegrees: Double?,
        userAccelerationX: Double?,
        userAccelerationY: Double?,
        userAccelerationZ: Double?,
        gravityX: Double?,
        gravityY: Double?,
        gravityZ: Double?,
        detections: [Detection]
    ) {
        let entry = DetectionLogEntry(
            t: timestamp,
            model: model,
            resolution: resolution,
            twoPass: twoPass,
            elapsedMs: elapsedMs,
            lowLightEnabled: lowLightEnabled,
            autoLowLightEnabled: autoLowLightEnabled,
            stabilizationEnabled: stabilizationEnabled,
            trackingLevel: trackingLevel,
            trackingElapsedMs: trackingElapsedMs,
            gmcCorrespondenceCount: gmcStats?.correspondenceCount,
            gmcInlierCount: gmcStats?.inlierCount,
            gmcElapsedMs: gmcStats?.elapsedMs,
            egoSpeedMps: egoSpeedMps,
            egoSpeedAccuracyMps: egoSpeedAccuracyMps,
            pitchDegrees: pitchDegrees,
            pitchDriftDegrees: pitchDriftDegrees,
            rotationRateXDegreesPerSecond: rotationRateXDegreesPerSecond,
            rotationRateYDegreesPerSecond: rotationRateYDegreesPerSecond,
            rotationRateZDegreesPerSecond: rotationRateZDegreesPerSecond,
            rollDegrees: rollDegrees,
            userAccelerationX: userAccelerationX,
            userAccelerationY: userAccelerationY,
            userAccelerationZ: userAccelerationZ,
            gravityX: gravityX,
            gravityY: gravityY,
            gravityZ: gravityZ,
            detections: detections.map {
                DetectionLogEntry.Box(
                    label: $0.label,
                    confidence: $0.confidence,
                    x: $0.boundingBox.minX,
                    y: $0.boundingBox.minY,
                    w: $0.boundingBox.width,
                    h: $0.boundingBox.height,
                    trackID: $0.trackID
                )
            }
        )
        queue.async {
            guard var data = try? encoder.encode(entry) else { return }
            data.append(UInt8(ascii: "\n"))
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            } else {
                try? data.write(to: url)
            }
        }
    }
}
