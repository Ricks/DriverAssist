//
//  ContentView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI
import UIKit

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

    @State private var batteryLevel: Float = UIDevice.current.batteryLevel
    @State private var batteryState: UIDevice.BatteryState = UIDevice.current.batteryState
    @State private var hasPressedGo = false

    // Drives the critical-thermal blink — `cameraManager.thermalState` only changes
    // (and thus only re-evaluates the HUD) when the state itself changes, so without
    // a timer a sustained `.critical` would never actually blink on screen.
    @State private var thermalBlinkOn = true
    private let thermalBlinkTimer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    // Dashcam usage means the screen is mounted and glanced at, not stared at —
    // dimming it saves real power over a long drive without losing the HUD's
    // legibility. Restored on disappear rather than left dim system-wide.
    @State private var previousBrightness: CGFloat = UIScreen.main.brightness
    private static let dimmedBrightness: CGFloat = 0.6

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
        _inferenceEngine  = StateObject(
            wrappedValue: InferenceEngine(
                modelManager: modelManager,
                trackingManager: TrackingManager(),
                egoSpeedManager: EgoSpeedManager(),
                pitchSensor: PitchSensor()
            )
        )
    }

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            OverlayView(detections: inferenceEngine.detections, sourceSize: inferenceEngine.sourceSize)
                .ignoresSafeArea()

            VStack {
                HStack {
                    VStack(alignment: .leading) {
                        Text(recordingLabel)
                            .font(.system(size: 36, weight: .medium))
                            .foregroundStyle(isRecordingHealthy ? .white.opacity(0.75) : Color.red)
                            .shadow(color: .black.opacity(0.6), radius: 2)
                        Text(thermalLabel)
                            .font(.system(size: 36, weight: .medium))
                            .foregroundStyle(thermalLabelColor)
                            .shadow(color: .black.opacity(0.6), radius: 2)
                    }
                    Spacer()
                    Text(stabilizationLabel)
                        .font(.system(size: 36, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                }
                Spacer()
                HStack {
                    VStack(alignment: .leading) {
                        Text(resolutionLabel)
                            .font(.system(size: 36, weight: .medium))
                            .foregroundStyle(.white.opacity(0.75))
                            .shadow(color: .black.opacity(0.6), radius: 2)
                        Text(modelLabel)
                            .font(.system(size: 36, weight: .medium))
                            .foregroundStyle(.white.opacity(0.75))
                            .shadow(color: .black.opacity(0.6), radius: 2)
                    }
                    Spacer()
                    Text(lowLightLabel)
                        .font(.system(size: 36, weight: .medium))
                        .foregroundStyle(.white.opacity(0.75))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
            .ignoresSafeArea(edges: [.top, .bottom])

            if !hasPressedGo {
                Color.black.ignoresSafeArea()
                Button {
                    hasPressedGo = true
                    cameraManager.markReadyForFocusCalibration()
                } label: {
                    Text("GO")
                        .font(.system(size: 48, weight: .bold))
                        .foregroundStyle(.black)
                        .frame(width: 180, height: 180)
                        .background(Circle().fill(Color.yellow))
                }
            }
        }
        .contentShape(Rectangle())
        .gesture(
            // Two-axis grid: horizontal picks tier (nano/small), vertical picks
            // input resolution (low/high) -- a direct set based on direction each
            // time, not a cycle. Medium stays voice-only (see DetectorModel /
            // ModelManager.highResTiers -- it has no high-res export, and isn't
            // on this grid at all). Two-pass is also voice-only now.
            DragGesture(minimumDistance: 24)
                .onEnded { value in
                    if abs(value.translation.width) > abs(value.translation.height) {
                        // Left swipe (finger moves right-to-left) selects nano; right selects small.
                        modelManager.switchModel(to: value.translation.width < 0 ? .nano : .small)
                    } else {
                        // Swipe up sets high-res (1920x1088) input, swipe down sets standard (1152x640).
                        modelManager.setHighResEnabled(value.translation.height < 0)
                    }
                }
        )
        .onChange(of: inferenceEngine.lastFrameElapsedMs) { _, newValue in
            let configKey = "\(modelManager.selectedModel.rawValue)|\(inferenceEngine.isTwoPassEnabled)"
            cameraManager.recordInferenceLatency(newValue, configKey: configKey)
        }
        .onReceive(NotificationCenter.default.publisher(for: UIDevice.batteryLevelDidChangeNotification)) { _ in
            batteryLevel = UIDevice.current.batteryLevel
        }
        .onReceive(NotificationCenter.default.publisher(for: UIDevice.batteryStateDidChangeNotification)) { _ in
            batteryState = UIDevice.current.batteryState
        }
        .onReceive(thermalBlinkTimer) { _ in
            thermalBlinkOn.toggle()
        }
        .onAppear {
            DebugFileLogger.reset()
            DetectionLogger.reset()
            // Keeps the screen (and thus the camera/recording) awake for the whole
            // drive instead of auto-locking after the idle timeout.
            UIApplication.shared.isIdleTimerDisabled = true
            previousBrightness = UIScreen.main.brightness
            UIScreen.main.brightness = Self.dimmedBrightness
            UIDevice.current.isBatteryMonitoringEnabled = true
            batteryLevel = UIDevice.current.batteryLevel
            batteryState = UIDevice.current.batteryState
            modelManager.loadInitialModel()
            cameraManager.onFrame = { [weak inferenceEngine, weak cameraManager] pixelBuffer in
                inferenceEngine?.process(
                    pixelBuffer: pixelBuffer,
                    lowLightEnabled: cameraManager?.isLowLightBoostEnabled ?? false,
                    autoLowLightEnabled: cameraManager?.isAutoLowLightEnabled ?? true,
                    stabilizationEnabled: cameraManager?.isStabilizationEnabled ?? false
                )
            }
            cameraManager.start()
            inferenceEngine.egoSpeedManager.start()
            inferenceEngine.pitchSensor.start()
            voiceCommandManager.onCommand = { [weak modelManager, weak cameraManager, weak inferenceEngine] command in
                switch command {
                case .selectModel(let model):
                    modelManager?.switchModel(to: model)
                case .lowLight(let enabled):
                    cameraManager?.setLowLightBoost(enabled)
                case .lowLightAuto:
                    cameraManager?.enableAutoLowLight()
                case .twoPass(let enabled):
                    inferenceEngine?.setTwoPassEnabled(enabled)
                case .stabilization(let enabled):
                    cameraManager?.setStabilizationEnabled(enabled)
                case .highRes(let enabled):
                    modelManager?.setHighResEnabled(enabled)
                case .calibratePitch:
                    inferenceEngine?.pitchSensor.captureReferencePitch()
                }
            }
            voiceCommandManager.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            UIScreen.main.brightness = previousBrightness
            UIDevice.current.isBatteryMonitoringEnabled = false
            cameraManager.stop()
            voiceCommandManager.stop()
            inferenceEngine.egoSpeedManager.stop()
            inferenceEngine.pitchSensor.stop()
        }
    }

    // MARK: Helpers

    private var modelLabel: String {
        switch modelManager.selectedModel {
        case .small:  return "small"
        case .nano:   return "nano"
        case .medium: return "medium"
        }
    }

    private var lowLightLabel: String {
        let state = cameraManager.isLowLightBoostEnabled ? "on" : "off"
        return cameraManager.isAutoLowLightEnabled ? "low-light: auto (\(state))" : "low-light: \(state)"
    }

    private var resolutionLabel: String {
        "res: \(modelManager.currentResolutionLabel)"
    }

    private var stabilizationLabel: String {
        "stabilization: \(cameraManager.isStabilizationEnabled ? "on" : "off")"
    }

    private var thermalLabel: String {
        let state = cameraManager.thermalState == .nominal ? "OK" : CameraManager.thermalStateName(cameraManager.thermalState).uppercased()
        return "thermal: \(state) \(cameraManager.thermalSpeedPercent)%"
    }

    private var thermalLabelColor: Color {
        Color(uiColor: CameraManager.thermalColor(for: cameraManager.thermalState, blinkOn: thermalBlinkOn))
    }

    /// True only when unplugged and below 20% — plugged-in low battery isn't a risk
    /// to warn about, and -1 (monitoring not yet started) shouldn't read as low.
    private var isBatteryLow: Bool {
        batteryState == .unplugged && batteryLevel >= 0 && batteryLevel < 0.2
    }

    private var recordingLabel: String {
        if !cameraManager.isRecording {
            return "⚠ NOT RECORDING"
        } else if cameraManager.isStorageLow {
            return "⚠ LOW STORAGE"
        } else if isBatteryLow {
            return "⚠ LOW BATTERY"
        } else {
            return "● REC"
        }
    }

    private var isRecordingHealthy: Bool {
        cameraManager.isRecording && !cameraManager.isStorageLow && !isBatteryLow
    }

}

#Preview {
    ContentView()
}
