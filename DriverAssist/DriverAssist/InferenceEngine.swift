//
//  InferenceEngine.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//
//  The InferenceEngine is the app’s real-time bridge between raw camera frames and usable detections:
//  it receives CVPixelBuffers from the camera, runs them through the active Core ML YOLO model off the
//  main thread, decodes the model’s outputs into a clean list of normalized bounding boxes with labels
//  and confidences, and publishes those detections back on the main thread for SwiftUI to render overlays —
//  all while preventing overlapping inferences to keep latency and UI responsiveness under control.
//
//  This engine:
//   - Keeps a reference to ModelManager.
//   - Tracks whether an inference is in-flight (isBusy) to avoid piling up frames.
//   - Publishes the latest detections.
//   - Runs model prediction on a background queue and publishes on main.
//   - Is pinned to @MainActor for its published state; the actual Core ML work runs in
//     a non-isolated `Sendable` decoder (`YOLODecoder`) so it can execute synchronously
//     off-main without crossing actor boundaries.

import Foundation
import CoreML
import Combine
import CoreVideo
import CoreGraphics

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let boundingBox: CGRect   // normalized: origin + size in [0, 1]
}

/// Wraps a non-Sendable value so it can cross a `@Sendable` closure boundary.
///
/// `CVPixelBuffer` is a Core Foundation type that is not marked `Sendable`,
/// but its retain/release and read access are safe across threads as long as
/// nobody mutates it concurrently. Here the buffer is handed off once to the
/// background queue and not touched afterwards on the calling side, so
/// asserting `@unchecked Sendable` is safe.
private struct UncheckedSendableBox<Value>: @unchecked Sendable {
    let value: Value
}

/// Performs the actual Core ML inference and output decoding.
///
/// This type is intentionally *not* `@MainActor` so it can run synchronously
/// on a background `DispatchQueue` without needing to hop actors. It holds
/// no mutable state, so it's safely `Sendable`.
struct YOLODecoder: Sendable {
    func makeInput(from pixelBuffer: CVPixelBuffer) throws -> MLFeatureProvider {
        // For a real Core ML model, you either:
        // - use the auto-generated input class, or
        // - wrap the pixel buffer in an `MLDictionaryFeatureProvider`.
        //
        // Replace "image" with the actual input name from your .mlpackage.
        let imageValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        let dict = ["image": imageValue]
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    func decodeDetections(from output: MLFeatureProvider) throws -> [Detection] {
        // This is intentionally generic; you will adapt it to your YOLO26 export.
        // Many exported detectors provide an `MLMultiArray` with rows [x, y, w, h, confidence, classIndex].
        guard let raw = output.featureValue(for: "output") else {
            throw InferenceError.missingOutput("output")
        }

        guard let array = raw.multiArrayValue else {
            throw InferenceError.invalidOutput("output is not a multiArray")
        }

        let labels = ["person", "bicycle", "car", "motorcycle", "bus", "truck"] // from COCO / your mapping
        var results: [Detection] = []

        // Example: assume shape (N, 6) where N rows of [x, y, w, h, conf, classIndex].
        let rows = array.shape[0].intValue
        let cols = array.shape[1].intValue
        guard cols >= 6 else {
            throw InferenceError.invalidOutput("expected 6 columns per detection, found \(cols)")
        }

        for row in 0..<rows {
            let base = row * cols
            let x = array[base + 0].floatValue
            let y = array[base + 1].floatValue
            let w = array[base + 2].floatValue
            let h = array[base + 3].floatValue
            let conf = array[base + 4].floatValue
            let cls = Int(array[base + 5].floatValue)

            // Basic confidence and class filter; tune later.
            guard conf >= 0.4 else { continue }
            guard cls >= 0 && cls < labels.count else { continue }

            let label = labels[cls]

            // center (x, y) + size (w, h) in normalized coordinates → CGRect
            let origin = CGPoint(x: CGFloat(x - w / 2), y: CGFloat(y - h / 2))
            let size = CGSize(width: CGFloat(w), height: CGFloat(h))
            let bbox = CGRect(origin: origin, size: size)

            results.append(Detection(label: label, confidence: conf, boundingBox: bbox))
        }

        return results
    }

    /// Runs the full pipeline: build input, predict, decode. Synchronous and
    /// safe to call from any thread/queue.
    func run(model: MLModel, pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        let input = try makeInput(from: pixelBuffer)
        let output = try model.prediction(from: input)
        return try decodeDetections(from: output)
    }
}

@MainActor
final class InferenceEngine: ObservableObject {
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var lastError: String?

    private let modelManager: ModelManager
    private let queue = DispatchQueue(label: "InferenceEngine.queue", qos: .userInitiated)
    private let decoder = YOLODecoder()
    private var isBusy = false

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    func process(pixelBuffer: CVPixelBuffer) {
        guard !isBusy else { return }
        guard let model = modelManager.model else {
            lastError = "Model not loaded"
            return
        }

        isBusy = true
        let box = UncheckedSendableBox(value: pixelBuffer)
        let decoder = self.decoder

        queue.async {
            do {
                let detections = try decoder.run(model: model, pixelBuffer: box.value)

                Task { @MainActor in
                    self.detections = detections
                    self.isBusy = false
                }
            } catch {
                Task { @MainActor in
                    self.lastError = error.localizedDescription
                    self.isBusy = false
                }
            }
        }
    }
}

enum InferenceError: LocalizedError {
    case missingOutput(String)
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .missingOutput(let name):
            return "Missing output feature '\(name)'"
        case .invalidOutput(let message):
            return "Invalid output: \(message)"
        }
    }
}
