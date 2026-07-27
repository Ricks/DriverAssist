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

    private let confidenceThreshold: Float = 0.25

    /// Centered region (top-left-origin normalized, matching `Detection.boundingBox`'s
    /// convention) used for the second "zoom" pass in two-pass mode — a 2x digital
    /// zoom on the road ahead. Cropped symmetrically so it keeps the source's 16:9
    /// aspect ratio and doesn't reintroduce distortion when scaled to the model input.
    private let zoomRegion = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)

    /// Runs the model once on the full frame and, when `twoPass` is on, a second time
    /// on a cropped/zoomed center region — a distant object occupies far more of the
    /// model's fixed input pixels in the zoomed pass, which is what actually extends
    /// detection range, at roughly double the compute cost per frame. Results from
    /// both passes are merged and de-duplicated by NMS.
    func run(model: MLModel, pixelBuffer: CVPixelBuffer, twoPass: Bool) throws -> [Detection] {
        guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw InferenceError.missingOutput("no image input constraint on model")
        }

        let fullImage = CIImage(cvPixelBuffer: pixelBuffer)
        var detections = try runSinglePass(fullImage, model: model, constraint: constraint)

        if twoPass {
            let croppedImage = fullImage.cropped(to: ciCropRect(for: zoomRegion, in: fullImage.extent))
            let zoomedDetections = try runSinglePass(croppedImage, model: model, constraint: constraint)
            detections.append(contentsOf: remap(zoomedDetections, into: zoomRegion))
        }

        return DetectionFilter.nonMaxSuppression(detections)
    }

    // The camera delivers full-resolution (4K where supported), 16:9 frames. The
    // model's input is exported at a matching 1152x640 (also 16:9, not square), so
    // .scaleFill no longer distorts proportions the way it would squashing 16:9 into
    // a square — it now just scales both axes by very nearly the same factor.
    // Capturing at the highest resolution available means more real detail survives
    // this downscale, which is what actually helps detect small/distant objects.
    private func runSinglePass(_ image: CIImage, model: MLModel, constraint: MLImageConstraint) throws -> [Detection] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            throw InferenceError.missingOutput("failed to create CGImage from pixel buffer")
        }
        let imageFeature = try MLFeatureValue(
            cgImage: cgImage,
            constraint: constraint,
            options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
        )
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
        let output = try model.prediction(from: input)
        return try decodeDetections(
            from: output,
            modelWidth: Float(constraint.pixelsWide),
            modelHeight: Float(constraint.pixelsHigh)
        )
    }

    /// Converts a top-left-origin normalized region into Core Image's own bottom-left-
    /// origin pixel space (`CIImage.extent` has y increasing upward, the opposite of
    /// `Detection.boundingBox`'s convention) so it can be passed to `CIImage.cropped(to:)`.
    func ciCropRect(for region: CGRect, in extent: CGRect) -> CGRect {
        let yFromBottomFraction = 1 - (region.minY + region.height)
        return CGRect(
            x: extent.minX + region.minX * extent.width,
            y: extent.minY + yFromBottomFraction * extent.height,
            width: region.width * extent.width,
            height: region.height * extent.height
        )
    }

    /// Detections from the cropped "zoom" pass are normalized to the crop's own
    /// frame; remap into full-frame-normalized coordinates via the same (top-left-
    /// origin normalized) region used to create the crop — no axis flip needed here,
    /// unlike `ciCropRect`, since both sides are already in that convention.
    func remap(_ detections: [Detection], into region: CGRect) -> [Detection] {
        detections.map { detection in
            let box = detection.boundingBox
            return Detection(
                label: detection.label,
                confidence: detection.confidence,
                boundingBox: CGRect(
                    x: region.minX + box.minX * region.width,
                    y: region.minY + box.minY * region.height,
                    width: box.width * region.width,
                    height: box.height * region.height
                )
            )
        }
    }

    func decodeDetections(from output: MLFeatureProvider, modelWidth: Float, modelHeight: Float) throws -> [Detection] {
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

            // x/y are denormalized against the model's own (possibly non-square) input
            // dimensions independently — using a single shared divisor here would
            // silently reintroduce the aspect-ratio distortion the rectangular export
            // was meant to eliminate.
            let x1 = value(in: array, at: [0, i, 0]) / modelWidth
            let y1 = value(in: array, at: [0, i, 1]) / modelHeight
            let x2 = value(in: array, at: [0, i, 2]) / modelWidth
            let y2 = value(in: array, at: [0, i, 3]) / modelHeight

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
    @Published private(set) var isTwoPassEnabled = true

    private let modelManager: ModelManager
    private let queue = DispatchQueue(label: "InferenceEngine.queue", qos: .userInitiated)
    private let decoder = YOLODecoder()
    private let smoother = DetectionSmoother()
    private var isBusy = false
    private var frameCount = 0

    private static let smoothingDefaultsKey = "settings.smoothingEnabled"
    private static let twoPassDefaultsKey = "settings.twoPassEnabled"

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
        let defaults = UserDefaults.standard
        if defaults.object(forKey: Self.smoothingDefaultsKey) != nil {
            isSmoothingEnabled = defaults.bool(forKey: Self.smoothingDefaultsKey)
        }
        if defaults.object(forKey: Self.twoPassDefaultsKey) != nil {
            isTwoPassEnabled = defaults.bool(forKey: Self.twoPassDefaultsKey)
        }
    }

    /// Resets tracking state on change so a stale pre-toggle position can't get
    /// blended into the first frame after re-enabling.
    func setSmoothingEnabled(_ enabled: Bool) {
        guard enabled != isSmoothingEnabled else { return }
        isSmoothingEnabled = enabled
        smoother.reset()
        UserDefaults.standard.set(enabled, forKey: Self.smoothingDefaultsKey)
    }

    func setTwoPassEnabled(_ enabled: Bool) {
        isTwoPassEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: Self.twoPassDefaultsKey)
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
        let twoPass = isTwoPassEnabled

        queue.async { [weak self] in
            do {
                let detections = try decoder.run(
                    model: modelBox.value,
                    pixelBuffer: pixelBufferBox.value,
                    twoPass: twoPass
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
