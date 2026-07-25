//
//  ContentView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI

// MARK: — Root view

@MainActor
struct ContentView: View {
    @StateObject private var modelManager = ModelManager()

    var body: some View {
        InferenceView(modelManager: modelManager)
    }
}

// MARK: — Inference view

@MainActor
struct InferenceView: View {
    @ObservedObject var modelManager: ModelManager
    @StateObject private var inferenceEngine: InferenceEngine
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var voiceCommandManager = VoiceCommandManager()

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
        _inferenceEngine  = StateObject(wrappedValue: InferenceEngine(modelManager: modelManager))
    }

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            OverlayView(detections: inferenceEngine.detections, sourceSize: inferenceEngine.sourceSize)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Spacer()
                    Text(smoothingLabel)
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                }
                Spacer()
                HStack {
                    Text(modelLabel)
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                    Spacer()
                    Text(lowLightLabel)
                        .font(.system(size: 24, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
            .ignoresSafeArea(edges: [.top, .bottom])
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 24)
                .onEnded { value in
                    if abs(value.translation.width) > abs(value.translation.height) {
                        cycleModel()
                    } else {
                        cameraManager.toggleLowLightBoost()
                    }
                }
        )
        .onChange(of: inferenceEngine.detections) { _, newDetections in
            cameraManager.currentDetections = newDetections
        }
        .onChange(of: modelManager.selectedModel) { _, _ in
            cameraManager.currentModelLabel = modelLabel
        }
        .onChange(of: inferenceEngine.isSmoothingEnabled) { _, newValue in
            cameraManager.currentSmoothingEnabled = newValue
        }
        .onAppear {
            DebugFileLogger.reset()
            modelManager.loadInitialModel()
            cameraManager.currentModelLabel = modelLabel
            cameraManager.currentSmoothingEnabled = inferenceEngine.isSmoothingEnabled
            cameraManager.onFrame = { [weak inferenceEngine] pixelBuffer in
                inferenceEngine?.process(pixelBuffer: pixelBuffer)
            }
            cameraManager.start()
            voiceCommandManager.onCommand = { [weak modelManager, weak cameraManager, weak inferenceEngine] command in
                switch command {
                case .selectModel(let model):
                    modelManager?.switchModel(to: model)
                case .lowLight(let enabled):
                    cameraManager?.setLowLightBoost(enabled)
                case .smoothing(let enabled):
                    inferenceEngine?.setSmoothingEnabled(enabled)
                }
            }
            voiceCommandManager.start()
        }
        .onDisappear {
            cameraManager.stop()
            voiceCommandManager.stop()
        }
    }

    // MARK: Helpers

    private var modelLabel: String {
        switch modelManager.selectedModel {
        case .small: return "small"
        case .nano:  return "nano"
        }
    }

    private var lowLightLabel: String {
        "low-light: \(cameraManager.isLowLightBoostEnabled ? "on" : "off")"
    }

    private var smoothingLabel: String {
        "smoothing: \(inferenceEngine.isSmoothingEnabled ? "on" : "off")"
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
