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
    }

    let t: Double
    let model: String
    let twoPass: Bool
    let elapsedMs: Double
    let lowLightEnabled: Bool
    let autoLowLightEnabled: Bool
    let stabilizationEnabled: Bool
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
        twoPass: Bool,
        elapsedMs: Double,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        stabilizationEnabled: Bool,
        detections: [Detection]
    ) {
        let entry = DetectionLogEntry(
            t: timestamp,
            model: model,
            twoPass: twoPass,
            elapsedMs: elapsedMs,
            lowLightEnabled: lowLightEnabled,
            autoLowLightEnabled: autoLowLightEnabled,
            stabilizationEnabled: stabilizationEnabled,
            detections: detections.map {
                DetectionLogEntry.Box(
                    label: $0.label,
                    confidence: $0.confidence,
                    x: $0.boundingBox.minX,
                    y: $0.boundingBox.minY,
                    w: $0.boundingBox.width,
                    h: $0.boundingBox.height
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
