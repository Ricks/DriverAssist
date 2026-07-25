//
//  AspectFillTransformTests.swift
//  DriverAssistTests
//
//  Created by Rick Clark on 7/24/26.
//

import CoreGraphics
import Testing
@testable import DriverAssist

struct AspectFillTransformTests {

    /// A landscape source into a taller/narrower destination should scale up until
    /// the source's height matches the destination's height, then crop the sides —
    /// the same rule `.resizeAspectFill` uses.
    @Test func cropsWidthWhenSourceIsWiderThanDestination() {
        let transform = AspectFillTransform(
            source: CGSize(width: 1280, height: 720),
            destination: CGSize(width: 390, height: 844)
        )

        // scale is chosen so the source fully covers the destination on both axes.
        let scaledWidth  = 1280 * transform.scale
        let scaledHeight = 720  * transform.scale
        #expect(scaledWidth >= 390)
        #expect(scaledHeight >= 844)
        // One axis should be an exact (non-cropped) fit.
        #expect(abs(scaledWidth - 390) < 0.01 || abs(scaledHeight - 844) < 0.01)

        // A box covering the full source frame should be centered and overflow evenly.
        let full = transform.rect(for: CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(abs(full.minX + full.maxX - 390) < 0.01) // symmetric about the center
        #expect(abs(full.minY + full.maxY - 844) < 0.01)
    }

    @Test func identityWhenAspectRatiosMatch() {
        let transform = AspectFillTransform(
            source: CGSize(width: 400, height: 300),
            destination: CGSize(width: 800, height: 600)
        )
        let box = transform.rect(for: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))
        #expect(abs(box.minX - 200) < 0.01)
        #expect(abs(box.minY - 150) < 0.01)
        #expect(abs(box.width - 400) < 0.01)
        #expect(abs(box.height - 300) < 0.01)
    }
}
