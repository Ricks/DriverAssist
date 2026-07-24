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
}
