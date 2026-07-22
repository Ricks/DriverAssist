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
/// live-detection UI.
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

            VStack(spacing: 6) {
                modelPicker
                statusBanner
                Spacer()
            }
            .padding(.top, 8)
        }
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

    // MARK: Sub-views

    private var modelPicker: some View {
        Picker("Model", selection: Binding(
            get: { modelManager.selectedModel },
            set: { modelManager.switchModel(to: $0) }
        )) {
            ForEach(DetectorModel.allCases, id: \.self) { model in
                Text(model.rawValue).tag(model)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .tint(.white)
    }

    @ViewBuilder
    private var statusBanner: some View {
        if !modelManager.isLoaded && modelManager.lastError == nil {
            Label("Loading model…", systemImage: "arrow.triangle.2.circlepath")
                .font(.caption)
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.black.opacity(0.55))
                .clipShape(Capsule())
        }
        if let error = modelManager.lastError ?? inferenceEngine.lastError {
            Text(error)
                .font(.caption)
                .foregroundStyle(.red)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(.black.opacity(0.55))
                .clipShape(Capsule())
        }
    }
}

#Preview {
    ContentView()
}
