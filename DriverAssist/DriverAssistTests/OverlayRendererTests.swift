//
//  OverlayRendererTests.swift
//  DriverAssistTests
//
//  Created by Rick Clark on 7/24/26.
//

import Testing
@testable import DriverAssist

struct OverlayRendererTests {

    @Test func colorMappingIsStableForKnownLabels() {
        #expect(OverlayStyle.color(for: "person") != OverlayStyle.color(for: "car"))
        #expect(OverlayStyle.color(for: "unknown-label") == OverlayStyle.color(for: "also-unknown"))
    }
}
