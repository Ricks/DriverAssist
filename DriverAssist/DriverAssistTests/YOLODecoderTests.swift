//
//  YOLODecoderTests.swift
//  DriverAssistTests
//

import XCTest
import CoreML
@testable import DriverAssist

final class YOLODecoderTests: XCTestCase {

    private let decoder = YOLODecoder()

    /// Builds a synthetic [1, N, 6] model output: rows of
    /// (x1, y1, x2, y2, confidence, classId) in pixel coordinates at the
    /// model's 640x640 input resolution — matching the real yolo26 export.
    private func makeOutput(rows: [[Float]]) throws -> MLFeatureProvider {
        let shape: [NSNumber] = [1, NSNumber(value: rows.count), 6]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        for (i, row) in rows.enumerated() {
            for (j, value) in row.enumerated() {
                let key: [NSNumber] = [0, NSNumber(value: i), NSNumber(value: j)]
                array[key] = NSNumber(value: value)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(multiArray: array)])
    }

    func testDecodesDetectionAboveConfidenceThresholdForTargetClass() throws {
        // car (class 2), pixel box (100,100)-(300,400) in 640-space, confidence 0.8
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.8, 2]])

        let detections = try decoder.decodeDetections(from: output)

        XCTAssertEqual(detections.count, 1)
        let detection = detections[0]
        XCTAssertEqual(detection.label, "car")
        XCTAssertEqual(detection.confidence, 0.8, accuracy: 0.0001)
        XCTAssertEqual(detection.boundingBox.origin.x, 100.0 / 640.0, accuracy: 0.0001)
        XCTAssertEqual(detection.boundingBox.origin.y, 100.0 / 640.0, accuracy: 0.0001)
        XCTAssertEqual(detection.boundingBox.width, 200.0 / 640.0, accuracy: 0.0001)
        XCTAssertEqual(detection.boundingBox.height, 300.0 / 640.0, accuracy: 0.0001)
    }

    func testDropsDetectionsBelowConfidenceThreshold() throws {
        // 0.1 confidence is below the model's 0.25 threshold.
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.1, 2]])

        let detections = try decoder.decodeDetections(from: output)

        XCTAssertTrue(detections.isEmpty)
    }

    func testDropsNonTargetClasses() throws {
        // Class 16 ("dog" in COCO) isn't one of the driving-relevant target classes.
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.9, 16]])

        let detections = try decoder.decodeDetections(from: output)

        XCTAssertTrue(detections.isEmpty)
    }

    func testRoundsFloatingPointClassIds() throws {
        // Real model output stores class id as float; small imprecision should still round correctly.
        let output = try makeOutput(rows: [[0, 0, 64, 64, 0.9, 4.998]])

        let detections = try decoder.decodeDetections(from: output)

        XCTAssertEqual(detections.first?.label, "bus") // class 5 = "bus"
    }

    func testAppliesNonMaximumSuppressionAcrossRawDetections() throws {
        // Two heavily overlapping "car" boxes -> only the higher-confidence one should survive.
        let output = try makeOutput(rows: [
            [100, 100, 300, 300, 0.9, 2],
            [105, 105, 305, 305, 0.5, 2],
        ])

        let detections = try decoder.decodeDetections(from: output)

        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections.first?.confidence, 0.9, accuracy: 0.0001)
    }

    func testThrowsOnUnexpectedShape() throws {
        // Only 5 columns instead of the required 6 (xyxy + confidence + class).
        let shape: [NSNumber] = [1, 1, 5]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let output = try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(multiArray: array)])

        XCTAssertThrowsError(try decoder.decodeDetections(from: output)) { error in
            guard let inferenceError = error as? InferenceError else {
                XCTFail("Expected InferenceError, got \(error)")
                return
            }
            switch inferenceError {
            case .unexpectedShape:
                break
            case .missingOutput:
                XCTFail("Expected .unexpectedShape, got .missingOutput")
            }
        }
    }

    func testThrowsOnMissingMultiArrayOutput() throws {
        // A feature provider whose only output isn't a multiArray.
        let output = try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(string: "not-an-array")])

        XCTAssertThrowsError(try decoder.decodeDetections(from: output)) { error in
            guard let inferenceError = error as? InferenceError else {
                XCTFail("Expected InferenceError, got \(error)")
                return
            }
            switch inferenceError {
            case .missingOutput:
                break
            case .unexpectedShape:
                XCTFail("Expected .missingOutput, got .unexpectedShape")
            }
        }
    }
}
