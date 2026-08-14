//
//  OverlayRenderer.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/24/26.
//

import CoreGraphics
import UIKit

/// Shared label→color mapping so the live SwiftUI overlay matches the colors
/// used when reconstructing an annotated recording offline.
enum OverlayStyle {
    static func color(for label: String) -> CGColor {
        switch label {
        case "person":                return UIColor.systemYellow.cgColor
        case "bicycle", "motorcycle":  return UIColor.systemCyan.cgColor
        case "cyclist", "motorcyclist": return UIColor.systemOrange.cgColor
        case "skateboard", "horse":    return UIColor.systemPurple.cgColor
        case "car":                    return UIColor.systemGreen.cgColor
        case "bus", "truck":           return UIColor.systemRed.cgColor
        default:                       return UIColor.white.cgColor
        }
    }
}
