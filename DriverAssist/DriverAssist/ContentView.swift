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

    /// Hidden distance-calibration flow state -- see
    /// `beginDistanceCalibration`'s doc comment and `levelScreen`'s
    /// long-press. Not part of `SessionPhase`: this is a side-flow off the
    /// level/configuring screens, not a step in the normal lifecycle.
    @State private var isCalibrationRecording = false
    /// nil = flow inactive. 0/1/2 = currently on that round of 3 (either
    /// showing "Adjust pitch" or actively recording that round's clip).
    @State private var distanceCalibrationRound: Int?
    @State private var isChoosingTapeMarkCount = false
    @State private var tapeMarkCount = 3
    /// Roll at the moment the flow began (`beginDistanceCalibration`) --
    /// shown alongside the live roll on the "Adjust pitch" screen as a
    /// fixed reference point, since only pitch is meant to change between
    /// rounds; roll drifting away from this baseline while handling the
    /// phone is exactly what it's there to catch.
    @State private var initialCalibrationRoll: Double?

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

    /// Hidden, deliberate long-press on the level screen (or tap on the
    /// configuring screen's small "Distance cal" label) -- NOT part of the
    /// normal session lifecycle (`sessionPhase` doesn't change). Kicks off
    /// the tape-mark distance-calibration flow: how many tape marks were
    /// placed, then three rounds of "adjust pitch, confirm, record a ~1s
    /// clip" -- gathering reference footage at three different pitches is
    /// what lets the eventual fit be validated for pitch-independence
    /// later (retilt + re-check against the same tape marks without
    /// refitting -- see the following-distance-measurement memory), not
    /// just fit once. Actual tape-mark distances are entered separately,
    /// later, offline -- this only captures the video + the reference
    /// pitch/roll for each round.
    private func beginDistanceCalibration() {
        guard !isCalibrationRecording, distanceCalibrationRound == nil else { return }
        initialCalibrationRoll = pitchSensor.rollDegrees
        isChoosingTapeMarkCount = true
    }

    private func cancelDistanceCalibration() {
        DebugFileLogger.log("distance-cal: MATCHED cancel")
        isChoosingTapeMarkCount = false
        distanceCalibrationRound = nil
        initialCalibrationRoll = nil
    }

    private func confirmTapeMarkCount() {
        DebugFileLogger.log("distance-cal: MATCHED confirmTapeMarkCount count=\(tapeMarkCount)")
        isChoosingTapeMarkCount = false
        distanceCalibrationRound = 0
    }

    /// "Ready" on the "Adjust pitch" screen -- captures the CURRENT live
    /// pitch/roll at full precision (the on-screen readout only shows
    /// these rounded to the nearest degree) and starts that round's ~1s
    /// recording.
    private func finishPitchAdjustment() {
        guard let round = distanceCalibrationRound else { return }
        let pitch = pitchSensor.pitchDegrees
        let roll = pitchSensor.rollDegrees
        DebugFileLogger.log("distance-cal: round=\(round) pitch=\(String(describing: pitch)) roll=\(String(describing: roll))")
        recordCalibrationRound(round: round)
    }

    /// Records one round's ~1s, 4K, locked-far-focus clip (see
    /// CameraManager.startCalibrationRecording's file-level comment),
    /// saved as calibration-<timestamp>.mov via Photos -- never confused
    /// with a real drive's recording-<timestamp>.mov. No stop control --
    /// one second of already-locked-focus 4K is plenty of frames, so it
    /// just auto-stops rather than needing a second deliberate action.
    ///
    /// After stopping, checks the detected tape-mark count (see
    /// CameraManager.countRedTapeMarks) against what was entered: on match,
    /// advances to the next round (or finishes after round 3); on
    /// mismatch, stays on the same round so the user can adjust and retry
    /// rather than restarting the whole three-round sequence. If a clean
    /// still frame from this ever isn't enough to make the tape out
    /// reliably, the planned fallback is a proper photo capture instead of
    /// a longer clip.
    private func recordCalibrationRound(round: Int) {
        guard !isCalibrationRecording else { return }
        // Set TRUE immediately, synchronously -- not after
        // startCalibrationRecording's async completion -- so a second tap
        // during the focus-settle window (which can take a few seconds,
        // with no visible feedback otherwise) can't start a second
        // overlapping attempt on the same session. CONFIRMED bug
        // 2026-08-09 via debug log: two overlapping "focus: settling"
        // calls 154ms apart, from the Ready button still being visible/
        // tappable during that window -- neither attempt's completion ever
        // reached the pass/fail comparison, reading as "stuck on round 1
        // with no error message" (there wasn't a missing-error bug -- the
        // comparison itself was never being reached). This also means the
        // "recording" banner now shows the instant Done is tapped instead
        // of after an unexplained multi-second pause, which was likely why
        // the user tapped again in the first place.
        isCalibrationRecording = true
        DebugFileLogger.log("distance-cal: round=\(round) MATCHED start recording")
        cameraManager.startCalibrationRecording { started in
            Task { @MainActor in
                guard started else {
                    DebugFileLogger.log("distance-cal: round=\(round) FAILED to start")
                    isCalibrationRecording = false
                    flashToast("CALIBRATION RECORDING\nFAILED TO START")
                    return
                }
                try? await Task.sleep(for: .seconds(1))
                cameraManager.stopCalibrationRecording { detectedCount in
                    Task { @MainActor in
                        isCalibrationRecording = false
                        // CameraManager.stopCalibrationRecording already
                        // restarts a normal preview-only session internally
                        // (atomically, before its completion fires) -- a
                        // separate `cameraManager.start(recording: false)`
                        // call here used to race with the next round's
                        // startCalibrationRecording if it started quickly,
                        // corrupting the session. CONFIRMED bug 2026-08-09:
                        // stuck on "Recording round 3/3" with a stray
                        // "recording-" (not "calibration-") prefixed file
                        // created mid-flow -- see stopCalibrationRecording's
                        // doc comment for the full explanation.
                        if detectedCount == tapeMarkCount {
                            DebugFileLogger.log("distance-cal: round=\(round) PASSED detectedCount=\(detectedCount)")
                            if round >= 2 {
                                distanceCalibrationRound = nil
                                initialCalibrationRoll = nil
                                flashToast("DISTANCE CALIBRATION\nCOMPLETE (3/3)")
                            } else {
                                distanceCalibrationRound = round + 1
                            }
                        } else {
                            DebugFileLogger.log("distance-cal: round=\(round) FAILED detectedCount=\(detectedCount) expected=\(tapeMarkCount)")
                            flashToast("COULDN'T CLEARLY SEE ALL\n\(tapeMarkCount) TAPE MARKS (found \(detectedCount))\nADJUST AND RETRY")
                        }
                    }
                }
            }
        }
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
            // isCalibrationRecording / distanceCalibrationRound: the hidden
            // distance-calibration side-flow also needs a live preview, so
            // the user can see the tape marks are actually in frame and in
            // focus while adjusting pitch and recording -- see
            // beginDistanceCalibration.
            if sessionPhase == .configuring || sessionPhase == .driving
                || isCalibrationRecording || distanceCalibrationRound != nil {
                CameraPreviewView(session: cameraManager.session)
                    .ignoresSafeArea()
            }
            // Mutually exclusive with sessionPhase's own screen, not
            // layered on top of it -- CONFIRMED bug 2026-08-09: layering
            // (even rendered after, on top in z-order) still let
            // configuringScreen's Lock Settings button and settingsHUD
            // show/receive touches through any part of the overlay that
            // wasn't itself opaque. The camera preview above is still
            // shown throughout (needed so the user can confirm all tape
            // marks are actually in frame), only sessionPhase's own screen
            // content is swapped out.
            if isChoosingTapeMarkCount {
                tapeMarkCountPickerOverlay
            } else if isCalibrationRecording {
                distanceCalibrationRecordingBanner
            } else if let round = distanceCalibrationRound {
                distanceCalibrationPitchScreen(round: round)
            } else {
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

            // Hoisted here (once) rather than duplicated per-screen --
            // CONFIRMED bug 2026-08-09: a per-screen toastView had been
            // missed on distanceCalibrationPitchScreen, so a failed tape-
            // mark detection retried silently with no visible error at all.
            if let toastText {
                toastView(toastText)
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
        }
        .contentShape(Rectangle())
        .simultaneousGesture(
            LongPressGesture(minimumDuration: 1.5)
                .onEnded { _ in
                    beginDistanceCalibration()
                }
        )
    }

    // MARK: — Distance-calibration side-flow (hidden, see beginDistanceCalibration)

    /// A singleton screen -- fully opaque, nothing else (camera preview,
    /// whichever screen triggered it) needs to show through, unlike
    /// `distanceCalibrationPitchScreen`, which genuinely needs the live
    /// preview visible behind it.
    private var tapeMarkCountPickerOverlay: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 32) {
                Text("How many tape marks?")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(.white)
                HStack(spacing: 32) {
                    Button {
                        tapeMarkCount = max(1, tapeMarkCount - 1)
                    } label: {
                        Text("−")
                            .font(.system(size: 40, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 64, height: 64)
                            .background(Color.black.opacity(0.5), in: Circle())
                    }
                    Text("\(tapeMarkCount)")
                        .font(.system(size: 64, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                        .frame(minWidth: 80)
                    Button {
                        tapeMarkCount += 1
                    } label: {
                        Text("+")
                            .font(.system(size: 40, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 64, height: 64)
                            .background(Color.black.opacity(0.5), in: Circle())
                    }
                }
                Button {
                    confirmTapeMarkCount()
                } label: {
                    Text("Confirm")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundStyle(.black)
                        .padding(.horizontal, 48)
                        .frame(height: 64)
                        .background(Color.yellow, in: Capsule())
                }
                Button {
                    cancelDistanceCalibration()
                } label: {
                    Text("Cancel")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
                .padding(.top, 8)
            }
        }
    }

    private var pitchDegreesRounded: Int {
        Int((pitchSensor.pitchDegrees ?? 0).rounded())
    }

    /// Camera feed still shows underneath (needed so the user can confirm
    /// all tape marks are actually in frame) -- only sessionPhase's own
    /// screen content (Lock Settings, settingsHUD, etc.) is swapped out
    /// while this is on screen, not layered under it -- see body's comment.
    /// User physically adjusts the phone's tilt in the mount, watching the
    /// live pitch/roll readout, then taps Ready once satisfied. Rounded to
    /// the nearest degree for display only -- `finishPitchAdjustment` logs
    /// the actual full-precision reading. Initial roll (captured once, at
    /// the very start of the whole flow) is shown alongside the live one
    /// as a fixed reference -- only pitch is meant to change between
    /// rounds, so this is what catches roll drifting away from where it
    /// started while handling the phone.
    ///
    /// Info on the left, actions on the right, center left empty --
    /// deliberately, so the red tape marks (roughly centered in frame,
    /// same as the settings HUD leaves the center clear) aren't obscured
    /// by any of this screen's own UI.
    private func distanceCalibrationPitchScreen(round: Int) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("Adjust Pitch")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.6), radius: 3)
                Text("Round \(round + 1) of 3")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.white.opacity(0.7))
                    .shadow(color: .black.opacity(0.6), radius: 3)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Pitch: \(pitchDegreesRounded)°")
                    Text("Roll: \(rollDegreesRounded)°")
                    if let initialCalibrationRoll {
                        Text("Initial roll: \(Int(initialCalibrationRoll.rounded()))°")
                            .font(.system(size: 18, weight: .medium))
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.6), radius: 3)
                .padding()
                .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))
            }
            Spacer()
            VStack(spacing: 16) {
                Button {
                    finishPitchAdjustment()
                } label: {
                    Text("Ready")
                        .font(.system(size: 28, weight: .bold))
                        .foregroundStyle(.black)
                        .padding(.horizontal, 40)
                        .frame(height: 64)
                        .background(Color.yellow, in: Capsule())
                }
                Button {
                    cancelDistanceCalibration()
                } label: {
                    Text("Cancel")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.white.opacity(0.8))
                        .shadow(color: .black.opacity(0.6), radius: 3)
                }
            }
        }
        .padding(24)
    }

    /// Shown in place of sessionPhase's own screen while a round's clip is
    /// actively recording -- see body's comment on why this replaces
    /// (rather than layers on top of) the level/configuring screen.
    private var distanceCalibrationRecordingBanner: some View {
        VStack {
            Text("● RECORDING ROUND \((distanceCalibrationRound ?? 0) + 1)/3 (4K)")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
                .padding()
                .background(Color.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
                .padding(.top, 50)
            Spacer()
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
            //
            // topLeading fills the same corner drivingScreen uses for
            // recording/thermal status, here with a small, deliberately
            // low-key trigger for the tape-mark distance-calibration flow
            // (see beginDistanceCalibration's doc comment) -- needed here,
            // not just the level screen's hidden long-press, because the
            // real workflow is: roll-calibrate in a known-level parking
            // spot, drive to a street with room for tape marks (staying on
            // this screen the whole time -- its preview-only session
            // doesn't need Lock Settings/Unlocked pressed yet), then start
            // the distance calibration once there.
            settingsHUD {
                Text(isCalibrationRecording ? "● Recording" : "Distance cal")
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(isCalibrationRecording ? .red : .white.opacity(0.4))
                    .shadow(color: .black.opacity(0.6), radius: 2)
                    .onTapGesture { beginDistanceCalibration() }
            }

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
