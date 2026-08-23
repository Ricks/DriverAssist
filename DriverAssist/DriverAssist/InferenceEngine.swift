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
        7: "truck",
        17: "horse",
        36: "skateboard"
    ]

    private let confidenceThreshold: Float = 0.25

    /// Bottom-aligned region (top-left-origin normalized, matching `Detection.
    /// boundingBox`'s convention) used for the second "zoom" pass in two-pass mode.
    /// Originally centered on the horizon to extend detection range for distant
    /// objects; retargeted to the bottom strip after reviewing real footage showed
    /// almost all actual detections landing there — near-field hazards (the ones a
    /// warning system would actually care about), not the distant traffic the
    /// centered crop was built to reach. That's a deliberate trade: this pass no
    /// longer helps distant-object range, only near-field accuracy in the zone
    /// where detections actually occur.
    ///
    /// Width/height fractions are kept equal (as before) so the crop stays 16:9 in
    /// native pixels — a 16:9 source cropped by equal width/height fractions stays
    /// 16:9, avoiding the distortion a mismatched crop would reintroduce when scaled
    /// to the model's (also 16:9-ish) input. 0.6 is unchanged from the original
    /// centered crop: at 1080p, 1920x0.6=1152 and 1080x0.6=648 just meets the
    /// model's 1152x640 input, so this pass still downscales (real detail) rather
    /// than upscaling (interpolated, no added detail).
    private let zoomRegion = CGRect(x: 0.2, y: 0.4, width: 0.6, height: 0.6)

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

    // Capture is 1080p/15fps (see CameraManager -- moved off 4K/30fps after a real
    // drive showed that pipeline alone thermally throttled the device 10-15x,
    // independent of model choice). The model's input (1152x640 standard, 1920x1088
    // high-res) is close to but not exactly the same aspect ratio as that 1920x1080
    // capture, so .scaleFit (preserve aspect ratio, letterbox the rest) is used
    // instead of .scaleFill (stretch to exact size, distorting proportions) --
    // matches how the offline resolution/accuracy validation actually ran
    // (tools/benchmark.py's run_reference_model calls ultralytics' own
    // model.predict(imgsz:), which letterboxes by default), and avoids even the
    // small residual stretch .scaleFill would otherwise apply. See
    // decodeDetections for the corresponding un-letterbox math this requires --
    // without it, every detection's reported position would silently shift by the
    // padding offset.
    private func runSinglePass(_ image: CIImage, model: MLModel, constraint: MLImageConstraint) throws -> [Detection] {
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            throw InferenceError.missingOutput("failed to create CGImage from pixel buffer")
        }
        let imageFeature = try MLFeatureValue(
            cgImage: cgImage,
            constraint: constraint,
            options: [.cropAndScale: VNImageCropAndScaleOption.scaleFit.rawValue]
        )
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
        let output = try model.prediction(from: input)
        return try decodeDetections(
            from: output,
            modelWidth: Float(constraint.pixelsWide),
            modelHeight: Float(constraint.pixelsHigh),
            sourceWidth: Float(image.extent.width),
            sourceHeight: Float(image.extent.height)
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

    func decodeDetections(
        from output: MLFeatureProvider,
        modelWidth: Float,
        modelHeight: Float,
        sourceWidth: Float,
        sourceHeight: Float
    ) throws -> [Detection] {
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

        // .scaleFit (see runSinglePass) scales the source uniformly -- by
        // whichever axis is the tighter fit -- to preserve its aspect ratio
        // within the model's input buffer, then centers it, leaving letterbox
        // bars on the loose axis. Raw model-space coordinates have to be
        // un-letterboxed (subtract the padding offset, divide by the
        // *scaled* content size, not the full buffer size) before they're
        // valid normalized source-image coordinates -- dividing by
        // modelWidth/modelHeight directly, like the old .scaleFill path did,
        // would silently include the padding bars in the denominator and
        // shift every box.
        let letterboxScale = min(modelWidth / sourceWidth, modelHeight / sourceHeight)
        let scaledContentWidth = sourceWidth * letterboxScale
        let scaledContentHeight = sourceHeight * letterboxScale
        let xOffset = (modelWidth - scaledContentWidth) / 2
        let yOffset = (modelHeight - scaledContentHeight) / 2

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
            let rawX1 = value(in: array, at: [0, i, 0])
            let rawY1 = value(in: array, at: [0, i, 1])
            let rawX2 = value(in: array, at: [0, i, 2])
            let rawY2 = value(in: array, at: [0, i, 3])

            // Clamp to [0, 1] -- a box that extends into the letterbox padding
            // (a real object partially at the frame edge) would otherwise
            // decode to a coordinate outside the source image entirely, which
            // .scaleFill's full-buffer-is-real-content assumption never had
            // to guard against.
            let x1 = min(1, max(0, (rawX1 - xOffset) / scaledContentWidth))
            let y1 = min(1, max(0, (rawY1 - yOffset) / scaledContentHeight))
            let x2 = min(1, max(0, (rawX2 - xOffset) / scaledContentWidth))
            let y2 = min(1, max(0, (rawY2 - yOffset) / scaledContentHeight))

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
    @Published private(set) var isTwoPassEnabled = false
    /// Wall-clock time the most recently completed frame took end to end,
    /// in ms -- feeds `CameraManager.recordInferenceLatency` for the
    /// thermal "% of full speed" metric. UPDATED 2026-08-16: measured
    /// directly as (now - startTime) in `finishSuccess`, not as
    /// elapsedMs + trackingElapsedMs -- inference and GMC now run
    /// CONCURRENTLY (see `process`'s `gmcTask`), so summing their
    /// individually-measured costs would double-count the overlapped
    /// portion and overstate real latency. A direct start-to-finish
    /// measurement stays correct regardless of how much internal overlap
    /// there is.
    @Published private(set) var lastFrameElapsedMs: Double = 0

    private let modelManager: ModelManager
    let trackingManager: TrackingManager
    let egoSpeedManager: EgoSpeedManager
    let pitchSensor: PitchSensor
    /// Combines person + bicycle/motorcycle detections into cyclist/
    /// motorcyclist detections for display -- see ComboManager.swift.
    /// Applied AFTER DetectionLogger.log (see finishSuccess) so the raw,
    /// per-object detections keep getting logged unchanged; only the
    /// published `detections` (overlay/UI) reflect the merge. Combo
    /// detections currently publish `distanceMeters == nil` -- recomputing
    /// a real distance for a synthesized union box needs pitch/roll/
    /// aspectRatio context ComboManager doesn't have yet, deliberately
    /// deferred rather than restructuring attachDistances' ordering for it
    /// in this pass.
    private let comboManager = ComboManager(strategies: [BicycleComboStrategy(), MotorcycleComboStrategy()])
    private let widthDistanceOverrideManager = WidthDistanceOverrideManager()
    private let queue = DispatchQueue(label: "InferenceEngine.queue", qos: .userInitiated)
    private let decoder = YOLODecoder()
    private var isBusy = false
    private var frameCount = 0

    private static let twoPassDefaultsKey = "settings.twoPassEnabled"

    init(modelManager: ModelManager, trackingManager: TrackingManager, egoSpeedManager: EgoSpeedManager, pitchSensor: PitchSensor) {
        self.modelManager = modelManager
        self.trackingManager = trackingManager
        self.egoSpeedManager = egoSpeedManager
        self.pitchSensor = pitchSensor
        // Two-pass is hardcoded off for the time being -- deliberately NOT restoring
        // a persisted on-state here (same reasoning as isAutoLowLightEnabled not
        // persisting), so a stale "two pass on" from earlier testing can't silently
        // carry forward into every later launch/rebuild the way it just did.
    }

    /// No-op for now -- disabled along with the voice command's effect, not just
    /// its HUD display, since the resolution label replaced the old "two-pass:
    /// on/off" text (see ContentView). Without that field a misheard "two pass
    /// on" would silently change behavior with no on-screen indication at all.
    /// Trivially reversible: restore the two lines below once two-pass is back
    /// on the HUD (or otherwise made visible) and worth testing again.
    func setTwoPassEnabled(_ enabled: Bool) {
        _ = enabled
        // isTwoPassEnabled = enabled
        // UserDefaults.standard.set(enabled, forKey: Self.twoPassDefaultsKey)
    }

    func process(
        pixelBuffer: CVPixelBuffer,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        normalAEEnabled: Bool,
        stabilizationEnabled: Bool,
        parametersLocked: Bool,
        yawReferenceNormalizedX: CGFloat
    ) {
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
        let startTime = CFAbsoluteTimeGetCurrent()
        let modelLabel = modelManager.selectedModel.rawValue
        let resolutionLabel = modelManager.actualInputResolutionLabel

        // Kicked off now, CONCURRENTLY with the CoreML inference dispatched
        // below -- GMC (see TrackingManager.computeGMC) only ever needs
        // this frame's pixel buffer, never the detections inference
        // produces, so there's no real dependency forcing these to run one
        // after the other. Runs on MainActor, same as tracking always has
        // (this only changes WHEN GMC runs, not what thread it runs on),
        // so it genuinely overlaps with inference's background `queue`
        // work below without requiring ByteTracker/TrackingManager to
        // become thread-safe for arbitrary background-queue access.
        //
        // CONFIRMED 2026-08-16 against real drive data: before this split,
        // GMC only started once inference's detections already existed --
        // pure unforced serialization, costing roughly 40% more combined
        // per-frame latency than necessary (mean 80.9ms measured vs.
        // ~46.5ms estimated once overlapped).
        let gmcTask = Task { @MainActor [weak self] in
            self?.trackingManager.computeGMC(pixelBuffer: pixelBufferBox.value)
        }

        queue.async { [weak self] in
            do {
                let detections = try decoder.run(
                    model: modelBox.value,
                    pixelBuffer: pixelBufferBox.value,
                    twoPass: twoPass
                )
                let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

                Task { @MainActor [weak self] in
                    let gmcResult = await gmcTask.value
                    self?.finishSuccess(
                        detections,
                        pixelBuffer: pixelBufferBox.value,
                        gmcResult: gmcResult,
                        elapsedMs: elapsedMs,
                        startTime: startTime,
                        modelLabel: modelLabel,
                        resolutionLabel: resolutionLabel,
                        twoPass: twoPass,
                        lowLightEnabled: lowLightEnabled,
                        autoLowLightEnabled: autoLowLightEnabled,
                        normalAEEnabled: normalAEEnabled,
                        stabilizationEnabled: stabilizationEnabled,
                        parametersLocked: parametersLocked,
                        yawReferenceNormalizedX: yawReferenceNormalizedX
                    )
                }
            } catch {
                // gmcTask still runs to completion on its own even though
                // its result is discarded here -- GMC's internal
                // "previous frame" reference advances regardless of
                // whether inference succeeded, which is a small,
                // deliberate behavior change from before this split (GMC
                // used to only run inside a successful finishSuccess, so
                // it silently skipped frames where inference failed).
                // Camera motion between two real captured frames doesn't
                // depend on whether the detector managed to run on them,
                // so this is arguably more correct, not less -- and
                // inference failures are rare (a genuine CoreML runtime
                // error, not the "no model loaded yet" case, which returns
                // before isBusy/gmcTask even start).
                Task { @MainActor [weak self] in
                    self?.finishFailure(error)
                }
            }
        }
    }

    private func finishSuccess(
        _ detections: [Detection],
        pixelBuffer: CVPixelBuffer,
        gmcResult: GMCResult?,
        elapsedMs: Double,
        startTime: CFAbsoluteTime,
        modelLabel: String,
        resolutionLabel: String,
        twoPass: Bool,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        normalAEEnabled: Bool,
        stabilizationEnabled: Bool,
        parametersLocked: Bool,
        yawReferenceNormalizedX: CGFloat
    ) {
        let trackingStart = CFAbsoluteTimeGetCurrent()
        let result = trackingManager.track(
            detections,
            pixelBuffer: pixelBuffer,
            gmcResult: gmcResult,
            yawRateDegreesPerSecond: pitchSensor.smoothedYawRateDegreesPerSecond
        )
        let trackingElapsedMs = (CFAbsoluteTimeGetCurrent() - trackingStart) * 1000
        let rangedDetections = Self.attachDistances(
            to: result.detections,
            sourceSize: sourceSize,
            pitchSensor: pitchSensor
        )
        // Runs BEFORE DetectionLogger.log below, so the override's effect
        // (widthDistanceMeters/distanceMetersIsWidthOverridden) is itself
        // logged -- see WidthDistanceOverrideManager.apply's doc comment.
        let overriddenDetections = widthDistanceOverrideManager.apply(
            to: rangedDetections,
            cameraHeightMeters: DistanceEstimator.calibrated.cameraHeightMeters,
            aspectRatio: Double(sourceSize.width / sourceSize.height)
        )
        self.lastError = nil
        self.isBusy = false
        // Real overlapped wall-clock latency (startTime -> now), NOT
        // elapsedMs + trackingElapsedMs -- see this property's own doc
        // comment for why that sum would overstate real latency now that
        // GMC runs concurrently with inference instead of inside tracking.
        self.lastFrameElapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        DetectionLogger.log(
            timestamp: Date().timeIntervalSince1970,
            drivingSide: DrivingSideSetting.current,
            model: modelLabel,
            resolution: resolutionLabel,
            twoPass: twoPass,
            elapsedMs: elapsedMs,
            lowLightEnabled: lowLightEnabled,
            autoLowLightEnabled: autoLowLightEnabled,
            normalAEEnabled: normalAEEnabled,
            stabilizationEnabled: stabilizationEnabled,
            parametersLocked: parametersLocked,
            trackingLevel: trackingManager.trackingLevel.rawValue,
            trackingElapsedMs: trackingElapsedMs,
            gmcStats: result.gmcStats,
            egoSpeedMps: egoSpeedManager.speedMps,
            egoSpeedAccuracyMps: egoSpeedManager.speedAccuracyMps,
            courseDegrees: egoSpeedManager.courseDegrees,
            courseAccuracyDegrees: egoSpeedManager.courseAccuracyDegrees,
            pitchDegrees: pitchSensor.pitchDegrees,
            pitchDriftDegrees: pitchSensor.pitchDriftDegrees,
            referencePitchDegrees: pitchSensor.referencePitchDegrees,
            rotationRateXDegreesPerSecond: pitchSensor.rotationRateXDegreesPerSecond,
            rotationRateYDegreesPerSecond: pitchSensor.rotationRateYDegreesPerSecond,
            rotationRateZDegreesPerSecond: pitchSensor.rotationRateZDegreesPerSecond,
            yawRateDegreesPerSecond: pitchSensor.yawRateDegreesPerSecond,
            smoothedYawRateDegreesPerSecond: pitchSensor.smoothedYawRateDegreesPerSecond,
            pitchRateDegreesPerSecond: pitchSensor.pitchRateDegreesPerSecond,
            rollRateDegreesPerSecond: pitchSensor.rollRateDegreesPerSecond,
            smoothedPitchRateDegreesPerSecond: pitchSensor.smoothedPitchRateDegreesPerSecond,
            smoothedRollRateDegreesPerSecond: pitchSensor.smoothedRollRateDegreesPerSecond,
            rollDegrees: pitchSensor.rollDegrees,
            rollDriftDegrees: pitchSensor.rollDriftDegrees,
            referenceRollDegrees: pitchSensor.referenceRollDegrees,
            referenceYawMarkerNormalizedX: Double(yawReferenceNormalizedX),
            userAccelerationX: pitchSensor.userAccelerationX,
            userAccelerationY: pitchSensor.userAccelerationY,
            userAccelerationZ: pitchSensor.userAccelerationZ,
            gravityX: pitchSensor.gravityX,
            gravityY: pitchSensor.gravityY,
            gravityZ: pitchSensor.gravityZ,
            detections: overriddenDetections
        )
        // Combo merge happens AFTER logging, so detections.jsonl always
        // has the raw per-object detections -- see comboManager's doc
        // comment above.
        self.detections = comboManager.combine(overriddenDetections)
        frameCount += 1
        if frameCount % 60 == 1 || !detections.isEmpty {
            print(String(format: "[InferenceEngine] frame %d: %.0fms model=%@ twoPass=%@ detections=%d",
                          frameCount, elapsedMs, modelLabel, String(twoPass), detections.count))
        }
    }

    private func finishFailure(_ error: Error) {
        self.lastError = error.localizedDescription
        self.isBusy = false
        print("[InferenceEngine] inference failed: \(error)")
    }

    /// Attaches DistanceEstimator.calibrated's estimate to each detection.
    /// Needs a reference pitch to produce anything -- stays nil (not the
    /// stale previous value) for every detection until the user has pressed
    /// "Calibrate" at least once, same nil-over-placeholder reasoning as the
    /// rest of PitchSensor/EgoSpeedManager's logging.
    private static func attachDistances(
        to detections: [Detection],
        sourceSize: CGSize,
        pitchSensor: PitchSensor
    ) -> [Detection] {
        guard let referencePitch = pitchSensor.referencePitchDegrees,
              sourceSize.width > 0, sourceSize.height > 0 else {
            return detections
        }
        let referenceRoll = pitchSensor.referenceRollDegrees ?? 0
        let aspectRatio = Double(sourceSize.width / sourceSize.height)
        return detections.map { detection in
            var detection = detection
            detection.distanceMeters = DistanceEstimator.calibrated.distanceMeters(
                bottomY: Double(detection.boundingBox.maxY),
                centerX: Double(detection.boundingBox.midX),
                referencePitchDegreesBelowHorizontal: referencePitch,
                referenceRollDegrees: referenceRoll,
                aspectRatio: aspectRatio
            )
            return detection
        }
    }
}
