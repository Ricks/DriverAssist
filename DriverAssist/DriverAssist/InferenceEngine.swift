//
//  InferenceEngine.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import Combine
import CoreGraphics
import CoreImage
import CoreML
import CoreVideo
import Foundation
import Vision

private let ciContext = CIContext()

// MARK: - Errors

enum InferenceError: LocalizedError {
    case missingOutput(String)
    case unexpectedShape([Int])

    var errorDescription: String? {
        switch self {
        case .missingOutput(let detail):
            return "Model output not found: \(detail)"
        case .unexpectedShape(let shape):
            return "Unexpected output shape: \(shape)"
        }
    }
}

// MARK: - Decoder

struct YOLODecoder: Sendable {
    private let cocoTargets: [Int: String] = [
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    ]

    private let modelInputSize: Float = 640
    private let confidenceThreshold: Float = 0.25

    func run(model: MLModel, pixelBuffer: CVPixelBuffer) throws -> [Detection] {
        guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw InferenceError.missingOutput("no image input constraint on model")
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw InferenceError.missingOutput("failed to create CGImage from pixel buffer")
        }

        // The camera delivers full-resolution (4K where supported) frames but the model
        // requires an exact 640x640 match (MLFeatureValue(pixelBuffer:) alone doesn't
        // resize) — scale to fit here. Capturing at the highest resolution available
        // means more real detail survives this downscale, which is what actually helps
        // detect small/distant objects — the model's input size is fixed regardless.
        let imageFeature = try MLFeatureValue(
            cgImage: cgImage,
            constraint: constraint,
            options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        )
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
        let output = try model.prediction(from: input)
        return try decodeDetections(from: output)
    }

    func decodeDetections(from output: MLFeatureProvider) throws -> [Detection] {
        guard let array = output.featureNames.lazy
            .compactMap({ output.featureValue(for: $0)?.multiArrayValue })
            .first
        else {
            throw InferenceError.missingOutput("no multiArray output in model response")
        }

        let shape = array.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, shape[2] >= 6 else {
            throw InferenceError.unexpectedShape(shape)
        }

        let numDetections = shape[1]
        var results: [Detection] = []

        for i in 0..<numDetections {
            let conf = value(in: array, at: [0, i, 4])
            guard conf >= confidenceThreshold else { continue }

            let cls = Int(value(in: array, at: [0, i, 5]).rounded())
            guard let label = cocoTargets[cls] else { continue }

            let x1 = value(in: array, at: [0, i, 0]) / modelInputSize
            let y1 = value(in: array, at: [0, i, 1]) / modelInputSize
            let x2 = value(in: array, at: [0, i, 2]) / modelInputSize
            let y2 = value(in: array, at: [0, i, 3]) / modelInputSize

            let box = CGRect(
                x: CGFloat(x1),
                y: CGFloat(y1),
                width: CGFloat(max(0, x2 - x1)),
                height: CGFloat(max(0, y2 - y1))
            )

            results.append(
                Detection(
                    label: label,
                    confidence: conf,
                    boundingBox: box
                )
            )
        }

        return DetectionFilter.nonMaxSuppression(results)
    }

    private func value(in array: MLMultiArray, at index: [Int]) -> Float {
        let key = index.map(NSNumber.init(value:))
        return array[key].floatValue
    }
}

// MARK: - Engine

@MainActor
final class InferenceEngine: ObservableObject {
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var lastError: String?
    /// True pixel dimensions of the most recent camera frame. `Detection.boundingBox`
    /// is normalized against this size, not against the screen — the overlay needs it
    /// to reproduce the preview's aspect-fill crop instead of naively stretching boxes
    /// across the screen bounds.
    @Published private(set) var sourceSize: CGSize = .zero
    @Published private(set) var isSmoothingEnabled = false

    private let modelManager: ModelManager
    private let queue = DispatchQueue(label: "InferenceEngine.queue", qos: .userInitiated)
    private let decoder = YOLODecoder()
    private let smoother = DetectionSmoother()
    private var isBusy = false
    private var frameCount = 0

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    /// Resets tracking state on change so a stale pre-toggle position can't get
    /// blended into the first frame after re-enabling.
    func setSmoothingEnabled(_ enabled: Bool) {
        guard enabled != isSmoothingEnabled else { return }
        isSmoothingEnabled = enabled
        smoother.reset()
    }

    func process(pixelBuffer: CVPixelBuffer) {
        sourceSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        DebugFileLogger.log("process() called, sourceSize set to \(sourceSize)")
        guard !isBusy else { return }
        guard let model = modelManager.model else {
            frameCount += 1
            if frameCount % 60 == 1 { print("[InferenceEngine] no model loaded yet, dropping frame") }
            return
        }

        isBusy = true

        let pixelBufferBox = UncheckedSendableBox(value: pixelBuffer)
        let modelBox = UncheckedSendableBox(value: model)
        let decoder = self.decoder

        queue.async { [weak self] in
            do {
                let detections = try decoder.run(
                    model: modelBox.value,
                    pixelBuffer: pixelBufferBox.value
                )

                Task { @MainActor [weak self] in
                    self?.finishSuccess(detections)
                }
            } catch {
                Task { @MainActor [weak self] in
                    self?.finishFailure(error)
                }
            }
        }
    }

    private func finishSuccess(_ detections: [Detection]) {
        self.detections = isSmoothingEnabled ? smoother.smooth(detections) : detections
        self.lastError = nil
        self.isBusy = false
        frameCount += 1
        if frameCount % 60 == 1 || !detections.isEmpty {
            print("[InferenceEngine] frame \(frameCount): \(detections.count) detection(s)")
        }
    }

    private func finishFailure(_ error: Error) {
        self.lastError = error.localizedDescription
        self.isBusy = false
        print("[InferenceEngine] inference failed: \(error)")
    }
}
