//
//  OverlayView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI

/// Draws bounding boxes and labels for all current detections over a full-screen canvas.
///
/// `Detection.boundingBox` is normalized against the camera buffer's own pixel
/// dimensions (`sourceSize`), not against the screen. The camera preview underneath
/// renders that buffer with `.resizeAspectFill` — a centered, uniform scale-and-crop —
/// so a naive `fraction * screenSize` stretch would drift from the true on-screen
/// position whenever the buffer and screen aspect ratios differ. `AspectFillTransform`
/// reproduces that same scale-and-crop for the boxes.
struct OverlayView: View {
    let detections: [Detection]
    let sourceSize: CGSize

    var body: some View {
        DebugFileLogger.log("body evaluated: sourceSize=\(sourceSize) detections=\(detections.count)")
        return Canvas { context, size in
            DebugFileLogger.log("CANVAS DRAW: canvas=\(size) sourceSize=\(sourceSize) detections=\(detections.count)")
            guard sourceSize.width > 0, sourceSize.height > 0 else { return }
            let transform = AspectFillTransform(source: sourceSize, destination: size)
            for detection in detections {
                draw(detection, in: &context, transform: transform)
            }
        }
        .allowsHitTesting(false)
    }

    private func draw(_ detection: Detection, in context: inout GraphicsContext, transform: AspectFillTransform) {
        let box = transform.rect(for: detection.boundingBox)
        let color = Color(cgColor: OverlayStyle.color(for: detection.label))

        var path = Path()
        path.addRect(box)
        context.stroke(path, with: .color(color), lineWidth: 2)

        // Falls back to confidence% when distance isn't available yet (no
        // reference pitch captured -- see Detection.distanceMeters) so the
        // overlay still shows something useful before "Calibrate" is pressed.
        // "w" suffix marks a width-based override (see WidthDistanceOverride
        // .swift) so it's visually distinguishable on a live drive from an
        // ordinary row-based reading, not just in the logs afterward.
        let metric = detection.distanceMeters.map {
            String(format: "%.2fm\(detection.distanceMetersIsWidthOverridden ? "w" : "")", $0)
        } ?? "\(Int(detection.confidence * 100))%"
        let idPrefix = detection.trackID.map { "#\($0) " } ?? ""
        let label = Text("\(idPrefix)\(detection.label) \(metric)")
            .font(.system(size: 18, weight: .bold))
            .foregroundStyle(color)
        context.draw(label, at: CGPoint(x: box.minX + 4, y: box.minY + 2), anchor: .topLeading)
    }
}

/// Maps normalized (0,1) coordinates in a `source`-sized buffer onto a `destination`
/// view using the same centered scale-and-crop as `AVLayerVideoGravity.resizeAspectFill`.
struct AspectFillTransform {
    let source: CGSize
    let scale: CGFloat
    let offsetX: CGFloat
    let offsetY: CGFloat

    init(source: CGSize, destination: CGSize) {
        self.source = source
        scale = max(destination.width / source.width, destination.height / source.height)
        offsetX = (destination.width  - source.width  * scale) / 2
        offsetY = (destination.height - source.height * scale) / 2
    }

    func rect(for normalized: CGRect) -> CGRect {
        CGRect(
            x:      offsetX + normalized.minX * source.width  * scale,
            y:      offsetY + normalized.minY * source.height * scale,
            width:  normalized.width  * source.width  * scale,
            height: normalized.height * source.height * scale
        )
    }
}
