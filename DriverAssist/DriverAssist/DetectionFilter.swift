//
//  DetectionFilter.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import Foundation
import CoreGraphics

enum DetectionFilter {

    /// Class-specific non-maximum suppression. Suppresses lower-confidence boxes
    /// that overlap a higher-confidence box of the same class by more than `iouThreshold`.
    static func nonMaxSuppression(
        _ detections: [Detection],
        iouThreshold: Float = 0.45
    ) -> [Detection] {
        var kept: [Detection] = []
        for cls in Set(detections.map(\.label)) {
            kept += suppressOverlapping(
                detections.filter { $0.label == cls },
                iouThreshold: iouThreshold
            )
        }
        return kept.sorted { $0.confidence > $1.confidence }
    }

    private static func suppressOverlapping(
        _ detections: [Detection],
        iouThreshold: Float
    ) -> [Detection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var suppressed = Set<UUID>()
        var kept: [Detection] = []

        for detection in sorted {
            guard !suppressed.contains(detection.id) else { continue }
            kept.append(detection)
            for other in sorted where other.id != detection.id && !suppressed.contains(other.id) {
                if iou(detection.boundingBox, other.boundingBox) > iouThreshold {
                    suppressed.insert(other.id)
                }
            }
        }
        return kept
    }

    private static func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        guard !intersection.isNull, !intersection.isEmpty else { return 0 }
        let inter = Float(intersection.width * intersection.height)
        let union = Float(a.width * a.height + b.width * b.height) - inter
        return union > 0 ? inter / union : 0
    }
}
