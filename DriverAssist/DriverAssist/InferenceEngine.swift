//
//  InferenceEngine.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import Combine
import CoreGraphics
import CoreML
import CoreVideo
import Foundation

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
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
        )
        let output = try model.prediction(from: input)
        return try decodeDetections(from: output)
    }

    private func decodeDetections(from output: MLFeatureProvider) throws -> [Detection] {
        guard let raw = output.featureNames.lazy
            .compactMap({ output.featureValue(for: $0) })
            .first(where: { $0.multiArrayValue != nil })
        else {
            throw InferenceError.missingOutput("no multiArray output in model response")
        }

        guard let array = raw.multiArrayValue else {
            throw InferenceError.missingOutput("multiArray output was nil")
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

    private let modelManager: ModelManager
    private let queue = DispatchQueue(label: "InferenceEngine.queue", qos: .userInitiated)
    private let decoder = YOLODecoder()
    private var isBusy = false

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    func process(pixelBuffer: CVPixelBuffer) {
        guard !isBusy else { return }
        guard let model = modelManager.model else { return }

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
        self.detections = detections
        self.lastError = nil
        self.isBusy = false
    }

    private func finishFailure(_ error: Error) {
        self.lastError = error.localizedDescription
        self.isBusy = false
    }
}
