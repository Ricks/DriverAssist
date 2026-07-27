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
        case topLeading, bottomLeading, bottomTrailing, topTrailing
    }

    /// Draws the model-name, low-light-state, smoothing-state, two-pass-state, and
    /// stabilization-state labels in the screen corners, mirroring the live HUD text
    /// in `ContentView` so recordings match what's on screen.
    static func drawHUD(
        modelLabel: String,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        smoothingEnabled: Bool,
        twoPassEnabled: Bool,
        stabilizationEnabled: Bool,
        in context: CGContext,
        size: CGSize
    ) {
        let lowLightState = lowLightEnabled ? "on" : "off"
        let lowLightText = autoLowLightEnabled ? "low-light: auto (\(lowLightState))" : "low-light: \(lowLightState)"
        let smoothingText = "smoothing: \(smoothingEnabled ? "on" : "off")"
        let twoPassText = "two-pass: \(twoPassEnabled ? "on" : "off")"
        let stabilizationText = "stabilization: \(stabilizationEnabled ? "on" : "off")"
        drawCornerLabels([modelLabel], in: context, size: size, corner: .bottomLeading)
        drawCornerLabels([lowLightText], in: context, size: size, corner: .bottomTrailing)
        drawCornerLabels([smoothingText, stabilizationText], in: context, size: size, corner: .topTrailing)
        drawCornerLabels([twoPassText], in: context, size: size, corner: .topLeading)
    }

    /// Draws one or more lines stacked at a screen corner (top corners stack
    /// downward from the margin; bottom corners stack upward so the last line sits
    /// at the margin), mirroring the `VStack` layout of the live HUD in `ContentView`.
    private static func drawCornerLabels(_ lines: [String], in context: CGContext, size: CGSize, corner: Corner) {
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 24, weight: .medium),
            .foregroundColor: UIColor.white.withAlphaComponent(0.75)
        ]
        let margin: CGFloat = 12
        let lineSpacing: CGFloat = 4
        let sizes = lines.map { ($0 as NSString).size(withAttributes: attributes) }

        var y: CGFloat
        switch corner {
        case .topLeading, .topTrailing:
            y = margin
        case .bottomLeading, .bottomTrailing:
            let totalHeight = sizes.reduce(0) { $0 + $1.height } + CGFloat(max(0, lines.count - 1)) * lineSpacing
            y = size.height - totalHeight - margin
        }

        UIGraphicsPushContext(context)
        context.saveGState()
        context.setShadow(offset: .zero, blur: 4, color: UIColor.black.withAlphaComponent(0.6).cgColor)

        for (line, textSize) in zip(lines, sizes) {
            let x: CGFloat
            switch corner {
            case .topLeading, .bottomLeading:
                x = margin
            case .topTrailing, .bottomTrailing:
                x = size.width - textSize.width - margin
            }
            (line as NSString).draw(at: CGPoint(x: x, y: y), withAttributes: attributes)
            y += textSize.height + lineSpacing
        }

        context.restoreGState()
        UIGraphicsPopContext()
    }
}
