//
//  TwoPassRemapTests.swift
//  DriverAssistTests
//
//  Created by Rick Clark on 7/27/26.
//

import CoreGraphics
import Testing
@testable import DriverAssist

struct TwoPassRemapTests {

    private let decoder = YOLODecoder()

    /// A detection filling the entire cropped "zoom" pass should map onto exactly
    /// the crop region within the full frame.
    @Test func remapsFullCropDetectionOntoCropRegion() {
        let region = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let detection = Detection(label: "person", confidence: 0.9, boundingBox: CGRect(x: 0, y: 0, width: 1, height: 1))

        let remapped = decoder.remap([detection], into: region)

        #expect(remapped.count == 1)
        let box = try! #require(remapped.first).boundingBox
        #expect(abs(box.minX - 0.25) < 0.0001)
        #expect(abs(box.minY - 0.25) < 0.0001)
        #expect(abs(box.width - 0.5) < 0.0001)
        #expect(abs(box.height - 0.5) < 0.0001)
    }

    /// A detection in the center of the crop should land in the center of the full
    /// frame too (the crop is centered), proving the y-axis wasn't accidentally
    /// flipped or offset in the wrong direction.
    @Test func remapsCenterOfCropOntoCenterOfFrame() {
        let region = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let detection = Detection(label: "car", confidence: 0.8, boundingBox: CGRect(x: 0.5, y: 0.5, width: 0, height: 0))

        let remapped = decoder.remap([detection], into: region)

        let box = try! #require(remapped.first).boundingBox
        #expect(abs(box.minX - 0.5) < 0.0001)
        #expect(abs(box.minY - 0.5) < 0.0001)
    }

    /// `ciCropRect` converts a top-left-origin normalized region into Core Image's
    /// bottom-left-origin pixel space. For a centered region this stays centered
    /// either way, so the fragile part — the y-axis flip direction — needs an
    /// off-center check to actually catch a sign error.
    @Test func ciCropRectFlipsYForOffCenterRegion() {
        // A region hugging the TOP of the frame (top-left-origin: y=0..0.25) should
        // land at the TOP of Core Image's bottom-left-origin extent too — i.e. its
        // maxY should equal the extent's maxY, not its minY.
        let topRegion = CGRect(x: 0, y: 0, width: 1, height: 0.25)
        let extent = CGRect(x: 0, y: 0, width: 1000, height: 500)

        let ciRect = decoder.ciCropRect(for: topRegion, in: extent)

        #expect(abs(ciRect.maxY - extent.maxY) < 0.01)
        #expect(abs(ciRect.height - 125) < 0.01)
    }
}
