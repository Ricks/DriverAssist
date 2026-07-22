//
//  DetectionView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

//  DetectionView is the SwiftUI overlay layer that renders model detections on top of the camera preview. Its job is
//  to take the normalized bounding boxes from your InferenceEngine, convert them into on-screen rectangles using the
//  actual size of the preview, and draw labeled boxes for things like cars and pedestrians.

//  What it does:

//  Each Detection contains a CGRect whose values are ratios in the range [0, 1], not pixel coordinates. DetectionView
//  uses GeometryReader to get the displayed size of the preview and multiplies the normalized x, y, width, and height
//  by that size so the box lands in the correct place on screen.

//  It also draws a label for each detection, such as car 91%, and color-codes categories so the overlay is easier to scan
//  in real time. In SwiftUI, an overlay or ZStack is a standard way to place this drawing layer on top of the underlying
//  camera view.

import SwiftUI
import CoreGraphics

struct DetectionView: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                ForEach(detections) { detection in
                    let rect = denormalize(
                        detection.boundingBox,
                        in: geometry.size
                    )

                    ZStack(alignment: .topLeading) {
                        Rectangle()
                            .stroke(borderColor(for: detection.label), lineWidth: 2)
                            .frame(width: rect.width, height: rect.height)
                            .position(
                                x: rect.midX,
                                y: rect.midY
                            )

                        labelView(for: detection)
                            .position(
                                x: rect.minX + 70,
                                y: max(12, rect.minY + 12)
                            )
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .allowsHitTesting(false)
        }
    }

    private func denormalize(_ rect: CGRect, in size: CGSize) -> CGRect {
        CGRect(
            x: rect.origin.x * size.width,
            y: rect.origin.y * size.height,
            width: rect.size.width * size.width,
            height: rect.size.height * size.height
        )
    }

    @ViewBuilder
    private func labelView(for detection: Detection) -> some View {
        Text("\(detection.label) \(Int(detection.confidence * 100))%")
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(borderColor(for: detection.label))
            )
    }

    private func borderColor(for label: String) -> Color {
        switch label {
        case "person":
            return .red
        case "bicycle":
            return .orange
        case "car":
            return .green
        case "motorcycle":
            return .yellow
        case "bus":
            return .blue
        case "truck":
            return .purple
        default:
            return .white
        }
    }
}
