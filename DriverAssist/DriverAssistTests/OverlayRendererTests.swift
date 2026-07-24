//
//  OverlayRendererTests.swift
//  DriverAssistTests
//
//  Created by Rick Clark on 7/24/26.
//

import CoreGraphics
import Testing
@testable import DriverAssist

struct OverlayRendererTests {

    /// Renders into a CGContext with the same top-left-origin flip used when baking
    /// the overlay into recorded pixel buffers (CameraManager.compositeOverlay), then
    /// checks the stroke landed at the expected pixel location rather than mirrored.
    @Test func drawPlacesBoxAtNormalizedTopLeftCoordinates() throws {
        let width = 100
        let height = 100
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 255, count: bytesPerRow * height) // opaque white

        let context = try #require(pixels.withUnsafeMutableBytes { rawBuffer -> CGContext? in
            CGContext(
                data: rawBuffer.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            )
        })

        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        // Box occupying the top-left quadrant in normalized (top-left origin) coordinates.
        let detection = Detection(label: "car", confidence: 0.9, boundingBox: CGRect(x: 0.1, y: 0.1, width: 0.3, height: 0.3))
        OverlayRenderer.draw([detection], in: context, size: CGSize(width: width, height: height))

        func pixel(x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
            let offset = y * bytesPerRow + x * 4
            // premultipliedFirst + byteOrder32Little => BGRA in memory.
            return (r: pixels[offset + 2], g: pixels[offset + 1], b: pixels[offset])
        }

        // The stroke should be near the top-left quadrant's edges (y=10 and y=40 for a
        // box spanning normalized y 0.1...0.4 at height 100), not mirrored to the bottom.
        let onTopEdge = pixel(x: 20, y: 10)
        let onBottomEdgeOfBox = pixel(x: 20, y: 40)
        let farFromAnyBox = pixel(x: 90, y: 90)

        #expect(onTopEdge != (255, 255, 255))
        #expect(onBottomEdgeOfBox != (255, 255, 255))
        #expect(farFromAnyBox == (255, 255, 255))
    }

    @Test func colorMappingIsStableForKnownLabels() {
        #expect(OverlayStyle.color(for: "person") != OverlayStyle.color(for: "car"))
        #expect(OverlayStyle.color(for: "unknown-label") == OverlayStyle.color(for: "also-unknown"))
    }
}
