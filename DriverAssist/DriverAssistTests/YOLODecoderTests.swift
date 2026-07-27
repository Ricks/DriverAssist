//
//  YOLODecoderTests.swift
//  DriverAssistTests
//

import Testing
import CoreML
@testable import DriverAssist

@Suite struct YOLODecoderTests {

    private let decoder = YOLODecoder()

    // Builds a synthetic [1, N, 6] output: (x1, y1, x2, y2, confidence, classId)
    // in pixel coordinates at the model's 640x640 input resolution.
    private func makeOutput(rows: [[Float]]) throws -> MLFeatureProvider {
        let shape: [NSNumber] = [1, NSNumber(value: rows.count), 6]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        for (i, row) in rows.enumerated() {
            for (j, value) in row.enumerated() {
                array[[0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: value)
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(multiArray: array)])
    }

    @Test func decodesDetectionAboveConfidenceThresholdForTargetClass() throws {
        // car (class 2), pixel box (100,100)-(300,400) in 640-space, confidence 0.8
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.8, 2]])

        let detections = try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640)

        #expect(detections.count == 1)
        let detection = try #require(detections.first)
        #expect(detection.label == "car")
        #expect(abs(detection.confidence - 0.8) < 0.0001)
        #expect(abs(detection.boundingBox.origin.x - 100.0 / 640.0) < 0.0001)
        #expect(abs(detection.boundingBox.origin.y - 100.0 / 640.0) < 0.0001)
        #expect(abs(detection.boundingBox.width  - 200.0 / 640.0) < 0.0001)
        #expect(abs(detection.boundingBox.height - 300.0 / 640.0) < 0.0001)
    }

    @Test func dropsBelowConfidenceThreshold() throws {
        // 0.1 confidence is below the 0.25 threshold.
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.1, 2]])
        #expect(try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640).isEmpty)
    }

    @Test func dropsNonTargetClasses() throws {
        // Class 16 ("dog" in COCO) is not a driving-relevant target.
        let output = try makeOutput(rows: [[100, 100, 300, 400, 0.9, 16]])
        #expect(try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640).isEmpty)
    }

    @Test func roundsFloatingPointClassIds() throws {
        // Real model output stores class id as float; small imprecision must round correctly.
        let output = try makeOutput(rows: [[0, 0, 64, 64, 0.9, 4.998]])
        let detections = try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640)
        #expect(detections.first?.label == "bus") // class 5 = "bus"
    }

    @Test func appliesNonMaximumSuppression() throws {
        // Two heavily overlapping "car" boxes — only the higher-confidence one should survive.
        let output = try makeOutput(rows: [
            [100, 100, 300, 300, 0.9, 2],
            [105, 105, 305, 305, 0.5, 2],
        ])
        let detections = try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640)
        #expect(detections.count == 1)
        #expect(abs((detections.first?.confidence ?? 0) - 0.9) < 0.0001)
    }

    @Test func throwsOnUnexpectedShape() throws {
        // Only 5 columns instead of the required 6 (xyxy + confidence + class).
        let shape: [NSNumber] = [1, 1, 5]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let output = try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(multiArray: array)])

        #expect {
            try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640)
        } throws: { error in
            if case InferenceError.unexpectedShape = error { return true }
            return false
        }
    }

    @Test func throwsOnMissingMultiArrayOutput() throws {
        // A feature provider whose only output is not a multiArray.
        let output = try MLDictionaryFeatureProvider(dictionary: ["output": MLFeatureValue(string: "not-an-array")])

        #expect {
            try decoder.decodeDetections(from: output, modelWidth: 640, modelHeight: 640)
        } throws: { error in
            if case InferenceError.missingOutput = error { return true }
            return false
        }
    }
}
