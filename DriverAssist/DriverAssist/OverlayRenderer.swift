//
//  OverlayRenderer.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/24/26.
//

import CoreGraphics
import UIKit

/// Shared label→color mapping so the live SwiftUI overlay and the baked-in
/// recording overlay render identically.
enum OverlayStyle {
    static func color(for label: String) -> CGColor {
        switch label {
        case "person":                return UIColor.systemYellow.cgColor
        case "bicycle", "motorcycle":  return UIColor.systemCyan.cgColor
        case "car":                    return UIColor.systemGreen.cgColor
        case "bus", "truck":           return UIColor.systemRed.cgColor
        default:                       return UIColor.white.cgColor
        }
    }
}

/// Draws detection boxes/labels into a Core Graphics context, used to bake the
/// overlay into recorded video frames (mirrors `OverlayView`'s SwiftUI Canvas drawing).
enum OverlayRenderer {
    static func draw(_ detections: [Detection], in context: CGContext, size: CGSize) {
        for detection in detections {
            draw(detection, in: context, size: size)
        }
    }

    private static func draw(_ detection: Detection, in context: CGContext, size: CGSize) {
        let box = CGRect(
            x:      detection.boundingBox.minX * size.width,
            y:      detection.boundingBox.minY * size.height,
            width:  detection.boundingBox.width  * size.width,
            height: detection.boundingBox.height * size.height
        )
        let color = OverlayStyle.color(for: detection.label)

        context.setStrokeColor(color)
        context.setLineWidth(2)
        context.stroke(box)

        let pct = Int(detection.confidence * 100)
        let text = "\(detection.label) \(pct)%" as NSString
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.boldSystemFont(ofSize: 12),
            .foregroundColor: UIColor(cgColor: color)
        ]

        UIGraphicsPushContext(context)
        text.draw(at: CGPoint(x: box.minX + 4, y: box.minY + 2), withAttributes: attributes)
        UIGraphicsPopContext()
    }

    private enum Corner {
        case bottomLeading, bottomTrailing, topTrailing
    }

    /// Draws the model-name, low-light-state, and smoothing-state labels in the
    /// screen corners, mirroring the live HUD text in `ContentView` so recordings
    /// match what's on screen.
    static func drawHUD(
        modelLabel: String,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        smoothingEnabled: Bool,
        in context: CGContext,
        size: CGSize
    ) {
        let lowLightState = lowLightEnabled ? "on" : "off"
        let lowLightText = autoLowLightEnabled ? "low-light: auto (\(lowLightState))" : "low-light: \(lowLightState)"
        let smoothingText = "smoothing: \(smoothingEnabled ? "on" : "off")"
        drawCornerLabel(modelLabel, in: context, size: size, corner: .bottomLeading)
        drawCornerLabel(lowLightText, in: context, size: size, corner: .bottomTrailing)
        drawCornerLabel(smoothingText, in: context, size: size, corner: .topTrailing)
    }

    private static func drawCornerLabel(_ string: String, in context: CGContext, size: CGSize, corner: Corner) {
        let text = string as NSString
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 24, weight: .medium),
            .foregroundColor: UIColor.white.withAlphaComponent(0.75)
        ]
        let textSize = text.size(withAttributes: attributes)
        let margin: CGFloat = 12

        let x: CGFloat
        let y: CGFloat
        switch corner {
        case .bottomLeading:
            x = margin
            y = size.height - textSize.height - margin
        case .bottomTrailing:
            x = size.width - textSize.width - margin
            y = size.height - textSize.height - margin
        case .topTrailing:
            x = size.width - textSize.width - margin
            y = margin
        }

        UIGraphicsPushContext(context)
        context.saveGState()
        context.setShadow(offset: .zero, blur: 4, color: UIColor.black.withAlphaComponent(0.6).cgColor)
        text.draw(at: CGPoint(x: x, y: y), withAttributes: attributes)
        context.restoreGState()
        UIGraphicsPopContext()
    }
}
