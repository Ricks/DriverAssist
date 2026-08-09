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

private extension View {
    /// Shared look for the settings HUD's text labels — legible over live
    /// video, dimmed slightly so it doesn't compete with detection overlays.
    func hudLabelStyle() -> some View {
        self
            .font(.system(size: 36, weight: .medium))
            .foregroundStyle(.white.opacity(0.75))
            .shadow(color: .black.opacity(0.6), radius: 2)
    }
}

// MARK: — Session lifecycle

/// A session goes level -> configuring -> driving, then either loops back to
/// level (exit, long-press + confirm) or the app is closed. Replaces the old
/// single "hasPressedGo" flag with an explicit state machine now that there
/// are three distinct screens instead of two.
private enum SessionPhase {
    /// Initial screen: "Calibrate?" Yes/No. Calibration only needs to happen
    /// when the mount was just attached (or feels like it may have moved) --
    /// not every session, and importantly NOT when the car itself is
    /// currently on unlevel ground (parked on a slope, roadside camber),
    /// since the whole point of a calibration is to capture a known-flat-
    /// ground reference. "No" reuses whatever reference pitch/roll was
    /// captured last time (persisted in PitchSensor via UserDefaults, not
    /// re-derived here) and skips straight to `.configuring`.
    case calibratePrompt
    /// Live tilt readout + "Calibrate" button -- only reached via "Yes"
    /// above. No recording yet -- this is the mount-leveling step, done
    /// standing still before anything else.
    case level
    /// Post-calibration screen: camera-orientation warning, plus the one-time
    /// "Lock Settings"/"Unlocked" choice that starts the drive. Settings
    /// themselves are still changed the old way -- swipe gesture and voice
    /// command, both active here too, not a dedicated settings UI.
    case configuring
    /// Normal driving HUD -- camera preview, detections, status labels.
    /// Settings are fixed for the remainder of the session (either frozen,
    /// if locked, or just not exposed anywhere to change, if unlocked --
    /// either way nothing here can change them anymore, only the settings
    /// screen could).
    case driving
}

// MARK: — Inference view

@MainActor
struct InferenceView: View {
    @ObservedObject var modelManager: ModelManager
    @StateObject private var inferenceEngine: InferenceEngine
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var voiceCommandManager = VoiceCommandManager()
    // Held directly (not just via inferenceEngine.pitchSensor/.egoSpeedManager)
    // so SwiftUI actually observes their @Published changes. InferenceEngine
    // holds the same instances as plain `let` properties, not `@Published` --
    // a nested ObservableObject's changes don't propagate through a parent's
    // objectWillChange, so reading them only via inferenceEngine.X never
    // triggered a re-render on its own (it only appeared to work during
    // active driving, piggybacking on the frequent detections-driven
    // re-renders from inferenceEngine itself). Confirmed as the cause of the
    // Level screen's "not updated continuously" readout, 2026-08-08.
    @StateObject private var pitchSensor: PitchSensor
    @StateObject private var egoSpeedManager: EgoSpeedManager

    @State private var batteryLevel: Float = UIDevice.current.batteryLevel
    @State private var batteryState: UIDevice.BatteryState = UIDevice.current.batteryState
    @State private var sessionPhase: SessionPhase = .calibratePrompt
    @State private var showExitConfirmation = false

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

    // Freezes model/resolution/low-light/stabilization for the whole drive once
    // chosen on the settings screen -- a one-time decision now (via "Lock
    // Settings"/"Unlocked" below), not a re-toggleable long-press anymore. In
    // practice this was already close to a one-time decision (this state
    // resets on every fresh launch), so making it explicit removes the one
    // real risk the old always-live toggle had: a sustained accidental touch
    // mid-drive silently flipping a locked session to unlocked.
    @State private var parametersLocked = false

    // Brief center-screen confirmation for one-shot actions (attitude
    // calibration) -- generalized to arbitrary text, since calibration needs
    // to show the actual captured pitch/roll values for an immediate sanity
    // check, not just "done".
    @State private var toastText: String?

    private func flashToast(_ text: String) {
        withAnimation { toastText = text }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation { toastText = nil }
        }
    }

    /// "Yes" on the calibrate-prompt screen -- go do the actual leveling
    /// step.
    private func confirmCalibrate() {
        withAnimation { sessionPhase = .level }
    }

    /// "No" on the calibrate-prompt screen -- skip straight to configuring,
    /// reusing whichever reference pitch/roll `PitchSensor` already loaded
    /// from a previous session (persisted via UserDefaults -- see its
    /// init()). Deliberately does NOT call `captureReferenceAttitude()`.
    private func skipCalibration() {
        let pitch = pitchSensor.referencePitchDegrees
        let roll = pitchSensor.referenceRollDegrees
        DebugFileLogger.log("calibrate: SKIPPED reusing persisted reference pitch=\(String(describing: pitch)) roll=\(String(describing: roll))")
        enterConfiguring()
    }

    /// Deliberate, UI-triggered pitch+roll calibration -- see
    /// PitchSensor.captureReferenceAttitude() and DistanceEstimator.swift's
    /// file-level comment for why this must be done standing still on known-
    /// flat, level ground, not while driving. Transitions to the settings
    /// screen on success; stays on the level screen (with a toast explaining
    /// why) if there's no motion reading yet.
    private func calibrateAttitude() {
        guard let result = pitchSensor.captureReferenceAttitude() else {
            DebugFileLogger.log("calibrate: FAILED no motion reading yet")
            flashToast("CALIBRATION FAILED\n(no sensor data yet)")
            return
        }
        DebugFileLogger.log("calibrate: MATCHED pitch=\(result.pitch) roll=\(result.roll)")
        enterConfiguring()
    }

    /// Shared tail of both calibration paths (captured fresh, or skipped and
    /// reusing the persisted reference) -- starts the preview-only camera
    /// session and transitions to the configuring screen. Preview-only lets
    /// the configuring screen show a live camera feed (and reflect swipe/
    /// tap/voice setting changes immediately) without writing anything to
    /// disk yet. See startDriving(locked:). Autofocus is restricted to the
    /// far field automatically as part of this (see
    /// CameraManager.restrictAutofocusToFarField) -- no separate one-shot
    /// calibration step anymore; that used to cause a multi-second freeze,
    /// see its doc comment for what changed.
    private func enterConfiguring() {
        cameraManager.start(recording: false)
        withAnimation { sessionPhase = .configuring }
    }

    /// Starts (or restarts, after a prior exit) the actual drive: applies the
    /// chosen lock state, begins recording, and transitions to the driving
    /// HUD. Recording deliberately doesn't start until this point -- not at
    /// app launch, not when the preview session starts on the configuring
    /// screen -- so the level/configuring screens never get recorded into
    /// the same file as the actual drive. The session itself is already
    /// running by now (calibrateAttitude started it preview-only), so this
    /// only needs to begin writing, not reconfigure/restart it.
    private func startDriving(locked: Bool) {
        parametersLocked = locked
        DebugFileLogger.log("lifecycle: MATCHED startDriving(locked: \(locked))")
        cameraManager.beginRecording()
        withAnimation { sessionPhase = .driving }
    }

    /// Long-press + confirmation, from configuring/driving -- finishes the
    /// current recording (blocking, via CameraManager.stop's completion, so
    /// the file/Photos save actually completes rather than relying on the
    /// mid-write-kill recovery path) and then terminates the app outright,
    /// per explicit request -- exiting a session means exiting, not
    /// returning to the level screen for another one.
    private func exitSession() {
        DebugFileLogger.log("lifecycle: MATCHED exitSession")
        cameraManager.stop {
            exit(0)
        }
    }

    // MARK: — Setting toggles (tap-to-change, see settingsHUD)

    private func toggleModel() {
        guard !parametersLocked else {
            DebugFileLogger.log("tap: IGNORED (locked) toggleModel")
            flashToast("LOCKED")
            return
        }
        // Cycles nano<->small only -- medium stays reachable by voice command
        // alone (see DetectorModel's doc comment), same boundary the old
        // swipe grid drew.
        let target: DetectorModel = modelManager.selectedModel == .nano ? .small : .nano
        DebugFileLogger.log("tap: MATCHED selectModel(\(target.rawValue))")
        modelManager.switchModel(to: target)
    }

    private func toggleResolution() {
        guard !parametersLocked else {
            DebugFileLogger.log("tap: IGNORED (locked) toggleResolution")
            flashToast("LOCKED")
            return
        }
        let enabled = !modelManager.isHighResEnabled
        DebugFileLogger.log("tap: MATCHED setHighRes(\(enabled))")
        modelManager.setHighResEnabled(enabled)
    }

    private func toggleLowLight() {
        guard !parametersLocked else {
            DebugFileLogger.log("tap: IGNORED (locked) toggleLowLight")
            flashToast("LOCKED")
            return
        }
        // Cycles auto -> on -> off -> auto -- all three states, since voice
        // commands (the only other way to reach "auto") are currently off.
        if cameraManager.isAutoLowLightEnabled {
            DebugFileLogger.log("tap: MATCHED setLowLightBoost(true)")
            cameraManager.setLowLightBoost(true)
        } else if cameraManager.isLowLightBoostEnabled {
            DebugFileLogger.log("tap: MATCHED setLowLightBoost(false)")
            cameraManager.setLowLightBoost(false)
        } else {
            DebugFileLogger.log("tap: MATCHED enableAutoLowLight")
            cameraManager.enableAutoLowLight()
        }
    }

    private func toggleStabilization() {
        guard !parametersLocked else {
            DebugFileLogger.log("tap: IGNORED (locked) toggleStabilization")
            flashToast("LOCKED")
            return
        }
        let enabled = !cameraManager.isStabilizationEnabled
        DebugFileLogger.log("tap: MATCHED setStabilization(\(enabled))")
        cameraManager.setStabilizationEnabled(enabled)
    }

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
        let pitchSensor = PitchSensor()
        let egoSpeedManager = EgoSpeedManager()
        _pitchSensor = StateObject(wrappedValue: pitchSensor)
        _egoSpeedManager = StateObject(wrappedValue: egoSpeedManager)
        _inferenceEngine  = StateObject(
            wrappedValue: InferenceEngine(
                modelManager: modelManager,
                trackingManager: TrackingManager(),
                egoSpeedManager: egoSpeedManager,
                pitchSensor: pitchSensor
            )
        )
    }

    var body: some View {
        ZStack {
            // Hoisted out of configuringScreen/drivingScreen so the SAME
            // CameraPreviewView (and its underlying AVCaptureVideoPreviewLayer)
            // stays mounted across the configuring -> driving transition,
            // instead of each screen creating its own. CONFIRMED 2026-08-09
            // via on-device timing logs as the actual cause of the ~9s
            // Lock-Settings freeze: captureOutput itself stopped firing (not
            // just our own recording code) for the whole gap, starting right
            // when the phase change would have torn down one preview layer
            // and attached a new one to the same running session -- a session-
            // level stall, not anything in CameraManager's recording path.
            if sessionPhase == .configuring || sessionPhase == .driving {
                CameraPreviewView(session: cameraManager.session)
                    .ignoresSafeArea()
            }
            switch sessionPhase {
            case .calibratePrompt:
                calibratePromptScreen
            case .level:
                levelScreen
            case .configuring:
                withSessionGestures(configuringScreen)
            case .driving:
                withSessionGestures(drivingScreen)
            }
        }
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
            // A Binding (not the plain Bool) so this long-lived closure reads
            // the lock's live value on every call, instead of freezing
            // whatever it was at onAppear time.
            let isLocked = $parametersLocked
            cameraManager.onFrame = { [weak inferenceEngine, weak cameraManager] pixelBuffer in
                inferenceEngine?.process(
                    pixelBuffer: pixelBuffer,
                    lowLightEnabled: cameraManager?.isLowLightBoostEnabled ?? false,
                    autoLowLightEnabled: cameraManager?.isAutoLowLightEnabled ?? true,
                    stabilizationEnabled: cameraManager?.isStabilizationEnabled ?? false,
                    parametersLocked: isLocked.wrappedValue
                )
            }
            // PitchSensor/EgoSpeedManager run for the app's whole lifetime, not
            // per-session -- the level screen needs live tilt data before any
            // session/recording has started, and starting GPS early gives it
            // more time to get a fix before it's actually needed while driving.
            // Camera/recording is NOT started here -- see startDriving(locked:).
            egoSpeedManager.start()
            pitchSensor.start()
            func ignoredIfLocked(_ label: String) -> Bool {
                guard isLocked.wrappedValue else { return false }
                DebugFileLogger.log("voice: IGNORED (locked) \(label)")
                return true
            }
            voiceCommandManager.onCommand = { [weak modelManager, weak cameraManager, weak inferenceEngine] command in
                switch command {
                case .selectModel(let model):
                    guard !ignoredIfLocked("selectModel(\(model.rawValue))") else { return }
                    modelManager?.switchModel(to: model)
                case .lowLight(let enabled):
                    guard !ignoredIfLocked("lowLight(\(enabled))") else { return }
                    cameraManager?.setLowLightBoost(enabled)
                case .lowLightAuto:
                    guard !ignoredIfLocked("lowLightAuto") else { return }
                    cameraManager?.enableAutoLowLight()
                case .twoPass(let enabled):
                    guard !ignoredIfLocked("twoPass(\(enabled))") else { return }
                    inferenceEngine?.setTwoPassEnabled(enabled)
                case .stabilization(let enabled):
                    guard !ignoredIfLocked("stabilization(\(enabled))") else { return }
                    cameraManager?.setStabilizationEnabled(enabled)
                case .highRes(let enabled):
                    guard !ignoredIfLocked("highRes(\(enabled))") else { return }
                    modelManager?.setHighResEnabled(enabled)
                case .calibrateAttitude:
                    // Not a config parameter -- a one-shot calibration action, stays
                    // available even while locked.
                    pitchSensor.captureReferenceAttitude()
                }
            }
            // Disabled for now (2026-08-05) -- see prior note; onCommand above
            // stays wired up so re-enabling this is just uncommenting the line.
            // voiceCommandManager.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            UIScreen.main.brightness = previousBrightness
            UIDevice.current.isBatteryMonitoringEnabled = false
            cameraManager.stop()
            voiceCommandManager.stop()
            egoSpeedManager.stop()
            pitchSensor.stop()
        }
    }

    // MARK: — Calibrate-prompt screen

    private var calibratePromptScreen: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 32) {
                Text("Calibrate?")
                    .font(.system(size: 40, weight: .bold))
                    .foregroundStyle(.white)
                Text("Only needed when the phone is freshly mounted --\nnot if the mount hasn't moved, and not if the car\nitself isn't on level ground right now.")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                HStack(spacing: 16) {
                    Button {
                        confirmCalibrate()
                    } label: {
                        Text("Yes")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.black)
                            .padding(.horizontal, 40)
                            .frame(height: 72)
                            .background(Color.yellow, in: Capsule())
                    }
                    Button {
                        skipCalibration()
                    } label: {
                        Text("No")
                            .font(.system(size: 28, weight: .bold))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 40)
                            .frame(height: 72)
                            .background(Color.black.opacity(0.5), in: Capsule())
                    }
                }
                .padding(.top, 8)
            }
        }
    }

    // MARK: — Level screen

    private var rollDegreesRounded: Int {
        Int((pitchSensor.rollDegrees ?? 0).rounded())
    }

    private var levelScreen: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 32) {
                Text("\(rollDegreesRounded)°")
                    .font(.system(size: 96, weight: .bold, design: .rounded))
                    .foregroundStyle(abs(rollDegreesRounded) <= 1 ? .green : .white)
                Text("from level")
                    .font(.system(size: 24, weight: .medium))
                    .foregroundStyle(.white.opacity(0.6))
                Button {
                    calibrateAttitude()
                } label: {
                    Text("Calibrate")
                        .font(.system(size: 32, weight: .bold))
                        .foregroundStyle(.black)
                        .frame(width: 220, height: 80)
                        .background(Color.yellow, in: Capsule())
                }
                .padding(.top, 24)
            }

            if let toastText {
                toastView(toastText)
            }
        }
    }

    // MARK: — Configuring screen

    /// Sign CONFIRMED 2026-08-08 via bench test: gravityX read -0.99 with the
    /// phone correctly oriented (camera at top, warning off) and +0.97 after
    /// rotating 180 deg by hand (camera at bottom, warning correctly on) --
    /// a clean flip near +/-1 magnitude in both positions, matching the
    /// predicted behavior exactly. In landscape mounting, the device's own
    /// portrait-frame X axis (the short/left-right axis) becomes the real-
    /// world vertical axis, which is what this reads.
    private var cameraOrientationWarning: Bool {
        guard let gx = pitchSensor.gravityX else { return false }
        return gx > 0.5
    }

    private var configuringScreen: some View {
        ZStack {
            // Camera preview itself is rendered once at the top-level body,
            // shared across configuring/driving -- see body's comment.
            settingsHUD { EmptyView() }

            if cameraOrientationWarning {
                VStack {
                    Text("⚠ Camera may be mounted upside down\n(rotated 90° from the top)")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .padding()
                        .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 12))
                        .padding(.top, 60)
                    Spacer()
                }
            }

            // No Spacers here -- ZStack's default center alignment is what
            // actually centers this vertically (and horizontally) over the
            // camera preview/HUD layers above.
            VStack(spacing: 16) {
                Button {
                    startDriving(locked: true)
                } label: {
                    Text("Lock Settings")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundStyle(.black)
                        .padding(.horizontal, 32)
                        .frame(height: 64)
                        .background(Color.yellow, in: RoundedRectangle(cornerRadius: 16))
                }
                Button {
                    startDriving(locked: false)
                } label: {
                    Text("Unlocked")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 32)
                        .frame(height: 64)
                        .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))
                }
            }
        }
    }

    // MARK: — Driving screen

    private var drivingScreen: some View {
        ZStack {
            // Camera preview itself is rendered once at the top-level body,
            // shared across configuring/driving -- see body's comment.
            OverlayView(detections: inferenceEngine.detections, sourceSize: inferenceEngine.sourceSize)
                .ignoresSafeArea()

            settingsHUD {
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
            }

            if let toastText {
                toastView(toastText)
            }
        }
    }

    /// Same corner-positioned settings display on both configuringScreen and
    /// drivingScreen -- `topLeading` supplies the extra top-left content
    /// drivingScreen shows (recording/thermal status), not meaningful before
    /// Lock/Unlocked starts the drive so configuringScreen passes EmptyView.
    /// Each label is individually tappable to toggle that setting (see
    /// toggleModel/toggleResolution/toggleLowLight/toggleStabilization
    /// below) -- replaces the old swipe gesture entirely: with settings
    /// actually visible now, tapping the one you want beats remembering
    /// swipe directions blind.
    @ViewBuilder
    private func settingsHUD<TopLeading: View>(@ViewBuilder topLeading: () -> TopLeading) -> some View {
        VStack {
            HStack {
                topLeading()
                Spacer()
                Text(stabilizationLabel)
                    .hudLabelStyle()
                    .onTapGesture { toggleStabilization() }
            }
            Spacer()
            HStack {
                VStack(alignment: .leading) {
                    Text(resolutionLabel)
                        .hudLabelStyle()
                        .onTapGesture { toggleResolution() }
                    Text(modelLabel)
                        .hudLabelStyle()
                        .onTapGesture { toggleModel() }
                }
                Spacer()
                Text(lowLightLabel)
                    .hudLabelStyle()
                    .onTapGesture { toggleLowLight() }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 12)
        .ignoresSafeArea(edges: [.top, .bottom])
    }

    /// Long-press-to-exit, shared by both configuringScreen and
    /// drivingScreen -- not levelScreen, where no session exists yet to
    /// exit. Factored out rather than duplicated so both screens stay in
    /// sync automatically.
    @ViewBuilder
    private func withSessionGestures<Content: View>(_ content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .simultaneousGesture(
                // Long-press anywhere -> confirm -> exit the session (stop
                // recording, back to the level screen). Replaces the old
                // long-press-toggles-lock behavior entirely -- locking is now a
                // one-time choice made on the configuring screen, not something
                // re-toggled mid-drive.
                LongPressGesture(minimumDuration: 0.8)
                    .onEnded { _ in
                        DebugFileLogger.log("gesture: MATCHED longPress(exit-prompt)")
                        showExitConfirmation = true
                    }
            )
            .confirmationDialog("Exit this session?", isPresented: $showExitConfirmation, titleVisibility: .visible) {
                Button("Yes", role: .destructive) { exitSession() }
                Button("No", role: .cancel) {}
            }
    }

    // MARK: Helpers

    private func toastView(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 48, weight: .bold))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 32)
            .padding(.vertical, 16)
            .background(Color.black.opacity(0.75), in: RoundedRectangle(cornerRadius: 20))
            .transition(.opacity)
    }

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
