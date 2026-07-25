//
//  DetectionSmoother.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/25/26.
//

import CoreGraphics

/// Blends each new detection toward its nearest match from the previous frame (same
/// label, highest IoU) so boxes glide rather than jump — masks both per-frame model
/// noise and the discrete jumps caused by dropped frames during inference latency.
/// Unmatched detections (new objects) pass through unsmoothed on their first frame.
final class DetectionSmoother {
    private let alpha: CGFloat
    private let matchIoUThreshold: Float
    private var previous: [Detection] = []

    init(alpha: CGFloat = 0.4, matchIoUThreshold: Float = 0.3) {
        self.alpha = alpha
        self.matchIoUThreshold = matchIoUThreshold
    }

    /// Clears tracking state. Call when smoothing is toggled off/on so a stale
    /// position from before the gap doesn't get blended into the next frame.
    func reset() {
        previous = []
    }

    func smooth(_ detections: [Detection]) -> [Detection] {
        var remainingPrevious = previous
        var result: [Detection] = []

        for detection in detections {
            if let index = bestMatch(for: detection, in: remainingPrevious) {
                let match = remainingPrevious.remove(at: index)
                result.append(blend(from: match, toward: detection))
            } else {
                result.append(detection)
            }
        }

        previous = result
        return result
    }

    private func bestMatch(for detection: Detection, in candidates: [Detection]) -> Int? {
        var bestIndex: Int?
        var bestIoU: Float = matchIoUThreshold
        for (index, candidate) in candidates.enumerated() where candidate.label == detection.label {
            let overlap = DetectionFilter.iou(candidate.boundingBox, detection.boundingBox)
            if overlap > bestIoU {
                bestIoU = overlap
                bestIndex = index
            }
        }
        return bestIndex
    }

    private func blend(from previous: Detection, toward current: Detection) -> Detection {
        let box = CGRect(
            x: lerp(previous.boundingBox.minX, current.boundingBox.minX),
            y: lerp(previous.boundingBox.minY, current.boundingBox.minY),
            width: lerp(previous.boundingBox.width, current.boundingBox.width),
            height: lerp(previous.boundingBox.height, current.boundingBox.height)
        )
        return Detection(label: current.label, confidence: current.confidence, boundingBox: box)
    }

    private func lerp(_ from: CGFloat, _ to: CGFloat) -> CGFloat {
        from + (to - from) * alpha
    }
}
