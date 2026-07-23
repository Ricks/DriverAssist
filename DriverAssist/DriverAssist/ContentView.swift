//
//  ContentView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI

// MARK: — Root view

/// Creates the shared ModelManager and hands it to InferenceView, which owns
/// the InferenceEngine (requires ModelManager at init time) and CameraManager.
@MainActor
struct ContentView: View {
    @StateObject private var modelManager = ModelManager()

    var body: some View {
        InferenceView(modelManager: modelManager)
    }
}

// MARK: — Inference view

/// Wires ModelManager → InferenceEngine → CameraManager into a full-screen
/// live-detection UI: camera feed, detection overlay, and a small model-name
/// label in the lower-left corner. Swipe anywhere (any direction) to toggle
/// between the small and nano models.
@MainActor
struct InferenceView: View {
    @ObservedObject var modelManager: ModelManager
    @StateObject private var inferenceEngine: InferenceEngine
    @StateObject private var cameraManager = CameraManager()

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
        _inferenceEngine  = StateObject(wrappedValue: InferenceEngine(modelManager: modelManager))
    }

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            OverlayView(detections: inferenceEngine.detections)
                .ignoresSafeArea()

            VStack {
                Spacer()
                HStack {
                    Text(modelLabel)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                    Spacer()
                }
            }
            .padding(.leading, 12)
            .padding(.bottom, 12)
            .ignoresSafeArea(edges: .bottom)
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 24)
                .onEnded { _ in cycleModel() }
        )
        .onAppear {
            modelManager.loadInitialModel()
            cameraManager.onFrame = { [weak inferenceEngine] pixelBuffer in
                inferenceEngine?.process(pixelBuffer: pixelBuffer)
            }
            cameraManager.start()
        }
        .onDisappear {
            cameraManager.stop()
        }
    }

    // MARK: Helpers

    private var modelLabel: String {
        switch modelManager.selectedModel {
        case .small: return "small"
        case .nano: return "nano"
        }
    }

    private func cycleModel() {
        let models = DetectorModel.allCases
        guard let index = models.firstIndex(of: modelManager.selectedModel) else { return }
        let next = models[(index + 1) % models.count]
        modelManager.switchModel(to: next)
    }
}

#Preview {
    ContentView()
}
