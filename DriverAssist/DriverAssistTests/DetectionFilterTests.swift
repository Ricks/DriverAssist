//
//  DetectionFilterTests.swift
//  DriverAssistTests
//

import XCTest
import CoreGraphics
@testable import DriverAssist

final class DetectionFilterTests: XCTestCase {

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

    func testSuppressesLowerConfidenceOverlapOfSameClass() {
        // Two "car" boxes that overlap almost completely (IoU ≈ 0.82) -> the
        // lower-confidence one should be suppressed at the default 0.45 threshold.
        let strong = detection(label: "car", confidence: 0.9, x: 0.1, y: 0.1, width: 0.2, height: 0.2)
        let weak = detection(label: "car", confidence: 0.4, x: 0.11, y: 0.11, width: 0.2, height: 0.2)

        let result = DetectionFilter.nonMaxSuppression([strong, weak])

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result.first?.confidence, 0.9)
    }

    func testDoesNotSuppressAcrossDifferentClasses() {
        // Identical, fully overlapping geometry but different classes -> class-specific
        // NMS must not cross-suppress, so both should be kept.
        let car = detection(label: "car", confidence: 0.9, x: 0.1, y: 0.1, width: 0.2, height: 0.2)
        let person = detection(label: "person", confidence: 0.85, x: 0.1, y: 0.1, width: 0.2, height: 0.2)

        let result = DetectionFilter.nonMaxSuppression([car, person])

        XCTAssertEqual(result.count, 2)
        XCTAssertTrue(result.contains { $0.label == "car" })
        XCTAssertTrue(result.contains { $0.label == "person" })
    }

    func testKeepsNonOverlappingBoxesOfSameClass() {
        let left = detection(label: "car", confidence: 0.6, x: 0.0, y: 0.0, width: 0.1, height: 0.1)
        let right = detection(label: "car", confidence: 0.7, x: 0.9, y: 0.9, width: 0.1, height: 0.1)

        let result = DetectionFilter.nonMaxSuppression([left, right])

        XCTAssertEqual(result.count, 2)
    }

    func testResultIsSortedByDescendingConfidence() {
        let low = detection(label: "car", confidence: 0.3, x: 0.0, y: 0.0, width: 0.1, height: 0.1)
        let high = detection(label: "bus", confidence: 0.95, x: 0.5, y: 0.5, width: 0.1, height: 0.1)
        let mid = detection(label: "truck", confidence: 0.6, x: 0.8, y: 0.1, width: 0.1, height: 0.1)

        let result = DetectionFilter.nonMaxSuppression([low, high, mid])

        XCTAssertEqual(result.map(\.confidence), [0.95, 0.6, 0.3])
    }

    func testEmptyInputProducesEmptyOutput() {
        XCTAssertTrue(DetectionFilter.nonMaxSuppression([]).isEmpty)
    }

    func testCustomIoUThresholdIsRespected() {
        // These two "car" boxes have a real IoU of ~0.14 (partial overlap).
        let a = detection(label: "car", confidence: 0.9, x: 0.0, y: 0.0, width: 0.2, height: 0.2)
        let b = detection(label: "car", confidence: 0.5, x: 0.1, y: 0.1, width: 0.2, height: 0.2)

        let keptAtDefault = DetectionFilter.nonMaxSuppression([a, b])
        XCTAssertEqual(keptAtDefault.count, 2, "0.14 IoU should not trigger suppression at the default 0.45 threshold")

        let keptAtLowThreshold = DetectionFilter.nonMaxSuppression([a, b], iouThreshold: 0.1)
        XCTAssertEqual(keptAtLowThreshold.count, 1, "Lowering the threshold below the actual IoU should trigger suppression")
    }
}
