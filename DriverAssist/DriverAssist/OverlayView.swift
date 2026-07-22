//
//  OverlayView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI

/// Draws bounding boxes and labels for all current detections over a full-screen canvas.
struct OverlayView: View {
    let detections: [Detection]

    var body: some View {
        Canvas { context, size in
            for detection in detections {
                draw(detection, in: &context, size: size)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(_ detection: Detection, in context: inout GraphicsContext, size: CGSize) {
        let box = CGRect(
            x:      detection.boundingBox.minX * size.width,
            y:      detection.boundingBox.minY * size.height,
            width:  detection.boundingBox.width  * size.width,
            height: detection.boundingBox.height * size.height
        )
        let color = labelColor(detection.label)

        var path = Path()
        path.addRect(box)
        context.stroke(path, with: .color(color), lineWidth: 2)

        let pct   = Int(detection.confidence * 100)
        let label = Text("\(detection.label) \(pct)%")
            .font(.caption2).bold()
            .foregroundStyle(color)
        context.draw(label, at: CGPoint(x: box.minX + 4, y: box.minY + 2), anchor: .topLeading)
    }

    private func labelColor(_ label: String) -> Color {
        switch label {
        case "person":                  return .yellow
        case "bicycle", "motorcycle":   return .cyan
        case "car":                     return .green
        case "bus", "truck":            return .red
        default:                        return .white
        }
    }
}
