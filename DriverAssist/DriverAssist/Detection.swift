//
//  Detection.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import Foundation
import CoreGraphics

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    // Normalized [0, 1] in top-left origin form: (x, y, width, height)
    let boundingBox: CGRect
}
