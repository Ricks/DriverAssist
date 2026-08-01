//
//  GMC.swift
//  DriverAssist
//
//  Global Motion Compensation -- estimates the camera's own frame-to-frame
//  motion so KalmanBoxTracker can correct a track's predicted position for
//  how much the *scene* shifted due to the vehicle turning/tilting, not just
//  how the tracked object itself moved. Ports tools/gmc.py's approach and
//  rationale (see that file's docstring); the estimation method differs
//  because it's built on Vision's dense optical flow rather than OpenCV's
//  sparse corner-tracking, since that's the native-and-accelerated option on
//  iOS. Only rotation/pan/tilt is corrected this way -- forward-motion
//  parallax needs per-object depth to compensate properly and isn't
//  attempted here.
//
//  IMPORTANT: unlike gmc.py (validated against synthetic and real footage
//  offline), this Vision-based implementation has NOT been build-and-run
//  verified -- this environment can write and reason about the Swift/Vision
//  API but can't execute it against real camera frames to confirm the
//  request's exact input/output conventions hold as documented. Build and
//  run this against real driving footage before trusting its output.
//

import CoreVideo
import Foundation
import Vision

/// x' = a*x - b*y + tx
/// y' = b*x + a*y + ty
/// -- a similarity transform (rotation + uniform scale + translation),
/// consistent with tools/gmc.py's use of cv2.estimateAffinePartial2D.
struct SimilarityTransform {
    var a: Double = 1
    var b: Double = 0
    var tx: Double = 0
    var ty: Double = 0

    static let identity = SimilarityTransform()

    var scale: Double { (a * a + b * b).squareRoot() }

    func apply(to point: (Double, Double)) -> (Double, Double) {
        (a * point.0 - b * point.1 + tx, b * point.0 + a * point.1 + ty)
    }

    func applyLinearOnly(to vector: (Double, Double)) -> (Double, Double) {
        (a * vector.0 - b * vector.1, b * vector.0 + a * vector.1)
    }
}

/// Per-frame quality signal for a `GMC.apply(to:)` call -- logged (see
/// DetectionLogger.swift) so degraded correspondence/inlier counts on
/// low-texture footage (e.g. at night) show up as data instead of a visual
/// guess from playback.
struct GMCStats {
    let correspondenceCount: Int
    let inlierCount: Int
    /// Wall-clock cost of the optical-flow request + RANSAC fit, in ms --
    /// the actual per-frame compute/thermal contribution of vision-based
    /// GMC, which nothing else currently measures (see InferenceEngine's
    /// lastFrameElapsedMs, which is captured before tracking/GMC even runs).
    let elapsedMs: Double
}

struct GMCResult {
    let transform: SimilarityTransform
    /// nil only on the very first frame seen (nothing to compare against yet)
    /// -- distinct from a real attempt that came up with 0 correspondences/
    /// inliers, which is the failure mode actually worth watching for.
    let stats: GMCStats?
}

final class GMC {
    /// Grid resolution flow is sampled at, per axis -- 16x16 = 256 candidate
    /// correspondences per frame, comfortably enough for a robust RANSAC fit
    /// without needing to walk every pixel of the dense flow field.
    private let gridSize = 16
    private let ransacIterations = 50
    private let inlierThresholdPx: Double = 2.0

    private let sequenceHandler = VNSequenceRequestHandler()
    private var havePreviousFrame = false

    /// Returns the estimated previous-frame-to-current-frame transform, in
    /// *pixel space at the buffer's actual resolution* -- identity if this is
    /// the first frame seen, or estimation fails/has too few inliers to trust.
    func apply(to pixelBuffer: CVPixelBuffer) -> GMCResult {
        defer { havePreviousFrame = true }
        guard havePreviousFrame else { return GMCResult(transform: .identity, stats: nil) }

        let startTime = CFAbsoluteTimeGetCurrent()
        let (transform, correspondenceCount, inlierCount) = estimate(pixelBuffer)
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        return GMCResult(
            transform: transform,
            stats: GMCStats(correspondenceCount: correspondenceCount, inlierCount: inlierCount, elapsedMs: elapsedMs)
        )
    }

    private func estimate(_ pixelBuffer: CVPixelBuffer) -> (transform: SimilarityTransform, correspondenceCount: Int, inlierCount: Int) {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: pixelBuffer)
        request.computationAccuracy = .medium

        do {
            try sequenceHandler.perform([request], on: pixelBuffer)
        } catch {
            print("[GMC] optical flow request failed: \(error)")
            return (.identity, 0, 0)
        }

        guard let observation = request.results?.first as? VNPixelBufferObservation else {
            return (.identity, 0, 0)
        }

        let correspondences = sampleFlowGrid(from: observation.pixelBuffer, sourceWidth: CVPixelBufferGetWidth(pixelBuffer), sourceHeight: CVPixelBufferGetHeight(pixelBuffer))
        guard correspondences.count >= 6 else {
            return (.identity, correspondences.count, 0)
        }

        let (transform, inlierCount) = fitRANSAC(correspondences)
        return (transform ?? .identity, correspondences.count, inlierCount)
    }

    /// Samples the dense flow field (kCVPixelFormatType_TwoComponent32Float:
    /// per-pixel (dx, dy)) on a coarse grid, returning (source, destination)
    /// point pairs in the *source* pixel buffer's own resolution -- the flow
    /// buffer's dimensions can differ slightly from the source's, so each
    /// sample is rescaled to source-pixel coordinates.
    private func sampleFlowGrid(from flowBuffer: CVPixelBuffer, sourceWidth: Int, sourceHeight: Int) -> [((Double, Double), (Double, Double))] {
        CVPixelBufferLockBaseAddress(flowBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flowBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(flowBuffer) else { return [] }
        let flowWidth = CVPixelBufferGetWidth(flowBuffer)
        let flowHeight = CVPixelBufferGetHeight(flowBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(flowBuffer)
        let scaleX = Double(sourceWidth) / Double(flowWidth)
        let scaleY = Double(sourceHeight) / Double(flowHeight)

        var pairs: [((Double, Double), (Double, Double))] = []
        for gy in 0..<gridSize {
            for gx in 0..<gridSize {
                let fx = Int((Double(gx) + 0.5) / Double(gridSize) * Double(flowWidth))
                let fy = Int((Double(gy) + 0.5) / Double(gridSize) * Double(flowHeight))
                guard fx >= 0, fx < flowWidth, fy >= 0, fy < flowHeight else { continue }

                let rowPtr = base.advanced(by: fy * bytesPerRow)
                let pixelPtr = rowPtr.assumingMemoryBound(to: Float32.self).advanced(by: fx * 2)
                let dx = Double(pixelPtr[0])
                let dy = Double(pixelPtr[1])
                guard dx.isFinite, dy.isFinite else { continue }

                let srcX = Double(fx) * scaleX
                let srcY = Double(fy) * scaleY
                // Vision reports flow in the flow-buffer's own pixel scale;
                // rescale the displacement itself to source-pixel units too.
                let dstX = srcX + dx * scaleX
                let dstY = srcY + dy * scaleY
                pairs.append(((srcX, srcY), (dstX, dstY)))
            }
        }
        return pairs
    }

    /// Fits a similarity transform via least squares over a minimal random
    /// sample, scores inliers, repeats, then refits over the best model's
    /// full inlier set -- standard RANSAC, guarding against the flow field
    /// containing genuine object motion (not just camera-induced background
    /// motion) among the sampled points.
    private func fitRANSAC(_ correspondences: [((Double, Double), (Double, Double))]) -> (transform: SimilarityTransform?, inlierCount: Int) {
        guard correspondences.count >= 6 else { return (nil, 0) }
        var best: SimilarityTransform?
        var bestInlierCount = 0

        for _ in 0..<ransacIterations {
            let sample = (0..<correspondences.count).shuffled().prefix(6).map { correspondences[$0] }
            guard let candidate = leastSquaresFit(sample) else { continue }

            let inlierCount = correspondences.filter { residual($0, candidate) < inlierThresholdPx }.count
            if inlierCount > bestInlierCount {
                bestInlierCount = inlierCount
                best = candidate
            }
        }

        guard let bestModel = best, bestInlierCount >= 6 else { return (nil, bestInlierCount) }
        let inliers = correspondences.filter { residual($0, bestModel) < inlierThresholdPx }
        return (leastSquaresFit(inliers) ?? bestModel, bestInlierCount)
    }

    private func residual(_ pair: ((Double, Double), (Double, Double)), _ transform: SimilarityTransform) -> Double {
        let predicted = transform.apply(to: pair.0)
        let dx = predicted.0 - pair.1.0
        let dy = predicted.1 - pair.1.1
        return (dx * dx + dy * dy).squareRoot()
    }

    /// Solves for (a, b, tx, ty) minimizing sum of squared residuals over:
    ///   a*x - b*y + tx = x'
    ///   b*x + a*y + ty = y'
    /// via normal equations (A^T A) p = A^T z.
    private func leastSquaresFit(_ correspondences: [((Double, Double), (Double, Double))]) -> SimilarityTransform? {
        guard correspondences.count >= 2 else { return nil }

        var ata = Matrix(rows: 4, cols: 4)
        var atz = [0.0, 0.0, 0.0, 0.0]

        for ((x, y), (xPrime, yPrime)) in correspondences {
            let row1: [Double] = [x, -y, 1, 0]
            let row2: [Double] = [y, x, 0, 1]
            for i in 0..<4 {
                for j in 0..<4 {
                    ata[i, j] += row1[i] * row1[j] + row2[i] * row2[j]
                }
                atz[i] += row1[i] * xPrime + row2[i] * yPrime
            }
        }

        guard let ataInv = ata.inverse else { return nil }
        let p = ataInv * atz
        return SimilarityTransform(a: p[0], b: p[1], tx: p[2], ty: p[3])
    }
}
