//
//  DetectionFilterTests.swift
//  DriverAssistTests
//

import Testing
import CoreGraphics
@testable import DriverAssist

@Suite struct DetectionFilterTests {

    private func detection(
        label: String,
        confidence: Float,
        x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat
    ) -> Detection {
        Detection(
            label: label,
            confidence: confidence,
            boundingBox: CGRect(x: x, y: y, width: width, height: height)
        )
    }

    @Test func suppressesLowerConfidenceOverlapOfSameClass() {
        // Two "car" boxes with IoU ≈ 0.82 — the weaker one should be suppressed.
        let strong = detection(label: "car", confidence: 0.9, x: 0.1,  y: 0.1,  width: 0.2, height: 0.2)
        let weak   = detection(label: "car", confidence: 0.4, x: 0.11, y: 0.11, width: 0.2, height: 0.2)

        let result = DetectionFilter.nonMaxSuppression([strong, weak])

        #expect(result.count == 1)
        #expect(result.first?.confidence == 0.9)
    }

    @Test func doesNotSuppressAcrossDifferentClasses() {
        // Identical geometry, different classes — class-specific NMS must not cross-suppress.
        let car    = detection(label: "car",    confidence: 0.9,  x: 0.1, y: 0.1, width: 0.2, height: 0.2)
        let person = detection(label: "person", confidence: 0.85, x: 0.1, y: 0.1, width: 0.2, height: 0.2)

        let result = DetectionFilter.nonMaxSuppression([car, person])

        #expect(result.count == 2)
        #expect(result.contains { $0.label == "car" })
        #expect(result.contains { $0.label == "person" })
    }

    @Test func keepsNonOverlappingBoxesOfSameClass() {
        let left  = detection(label: "car", confidence: 0.6, x: 0.0, y: 0.0, width: 0.1, height: 0.1)
        let right = detection(label: "car", confidence: 0.7, x: 0.9, y: 0.9, width: 0.1, height: 0.1)

        #expect(DetectionFilter.nonMaxSuppression([left, right]).count == 2)
    }

    @Test func resultIsSortedByDescendingConfidence() {
        let low  = detection(label: "car",   confidence: 0.3,  x: 0.0, y: 0.0, width: 0.1, height: 0.1)
        let high = detection(label: "bus",   confidence: 0.95, x: 0.5, y: 0.5, width: 0.1, height: 0.1)
        let mid  = detection(label: "truck", confidence: 0.6,  x: 0.8, y: 0.1, width: 0.1, height: 0.1)

        let result = DetectionFilter.nonMaxSuppression([low, high, mid])

        #expect(result.map(\.confidence) == [0.95, 0.6, 0.3])
    }

    @Test func emptyInputProducesEmptyOutput() {
        #expect(DetectionFilter.nonMaxSuppression([]).isEmpty)
    }

    @Test func customIoUThresholdIsRespected() {
        // These two "car" boxes have a real IoU of ~0.14 (partial overlap).
        let a = detection(label: "car", confidence: 0.9, x: 0.0, y: 0.0, width: 0.2, height: 0.2)
        let b = detection(label: "car", confidence: 0.5, x: 0.1, y: 0.1, width: 0.2, height: 0.2)

        #expect(
            DetectionFilter.nonMaxSuppression([a, b]).count == 2,
            "IoU ≈ 0.14 should not trigger suppression at the default 0.45 threshold"
        )
        #expect(
            DetectionFilter.nonMaxSuppression([a, b], iouThreshold: 0.1).count == 1,
            "Lowering the threshold below the actual IoU should trigger suppression"
        )
    }
}
