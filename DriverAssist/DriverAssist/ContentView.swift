//
//  ContentView.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI
import UIKit

/// The three states toggleLowLight() cycles through -- named here so a
/// pending confirmation (see ContentView.pendingLowLightTarget) can name its
/// target without re-deriving it from CameraManager's raw booleans.
private enum LowLightTarget {
    case auto, on, off

    var label: String {
        switch self {
        case .auto: return "auto"
        case .on: return "on"
        case .off: return "off"
        }
    }
}

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
    /// The translucent black backing (2026-08-16, by request) is what
    /// actually carries readability at a glance while driving -- text
    /// shadow alone wasn't enough contrast against a bright/busy background
    /// (sky, pavement, oncoming headlights).
    func hudLabelStyle() -> some View {
        self
            .font(.system(size: 36, weight: .medium))
            .foregroundStyle(.white.opacity(0.9))
            .shadow(color: .black.opacity(0.6), radius: 2)
            .hudBoxBackground()
    }

    /// Just the translucent box from `hudLabelStyle` above, without the
    /// fixed white foreground -- for HUD labels (recording/thermal status,
    /// 2026-08-16, by request) that need their own dynamic color (red when
    /// unhealthy, thermal-state color) but should still get the same
    /// at-a-glance-while-driving readability treatment.
    func hudBoxBackground() -> some View {
        self
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: — Session lifecycle

/// A session goes level -> configuring -> driving, then either loops back to
/// level (exit, long-press + confirm) or the app is closed. Replaces the old
/// single "hasPressedGo" flag with an explicit state machine now that there
/// are three distinct screens instead of two.
private enum SessionPhase {
    /// Initial screen: live tilt readout + yaw band/crosshair + "OK" button
    /// -- the mount-leveling step, done standing still before anything
    /// else, entered directly on launch (no more separate "Calibrate?"
    /// Yes/No prompt in front of it -- see the design discussion this came
    /// out of). Fine calibration (yaw auto-detect, and pitch/roll capture
    /// on "OK") now always runs fresh every session; there's no skip path
    /// left to reuse a stale persisted reference. "NUDGE MOUNT" state (see
    /// ContentView.isMountOK) gates "OK" behind a confirmation instead of
    /// silently proceeding with a mount that's out of tolerance.
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
    @State private var sessionPhase: SessionPhase = .level
    @State private var showExitConfirmation = false
    /// Confirms before actually enabling high-res -- added 2026-08-17 after
    /// real-drive data showed every high-res config either loses/ties a
    /// same-latency low-res option or costs 200+ms for marginal accuracy
    /// gain (see project_matrix_comparison_result), so an accidental tap on
    /// this HUD label mid-drive would otherwise silently tank frame rate
    /// with no warning (the label's own text updates, but nothing else
    /// flags the drop). Only gates turning it ON -- turning back OFF is the
    /// safe direction and doesn't need a confirmation in the way.
    @State private var showHighResConfirmation = false
    /// Confirmation gate for stabilization and low-light, same rationale as
    /// showHighResConfirmation above -- an accidental tap on either HUD label
    /// mid-drive would silently change the crop/exposure behavior with no
    /// warning beyond the label's own text. Both directions are gated here
    /// (unlike high-res, where only enabling was risky) since these are
    /// simple toggles/cycles with no clearly "safe" direction.
    @State private var showStabilizationConfirmation = false
    @State private var pendingStabilizationEnabled = false
    @State private var showLowLightConfirmation = false
    @State private var pendingLowLightTarget: LowLightTarget = .auto
    /// True while any `StepperButton` is mid-hold (repeat timer running) --
    /// lets `withSessionGestures`'s long-press-to-exit gesture ignore a
    /// stepper hold-to-repeat past its ~2s threshold instead of popping the
    /// exit dialog over it. See `withSessionGestures`'s doc comment.
    @State private var isStepperActive = false

    // Drives the critical-thermal blink — `cameraManager.thermalState` only changes
    // (and thus only re-evaluates the HUD) when the state itself changes, so without
    // a timer a sustained `.critical` would never actually blink on screen.
    @State private var thermalBlinkOn = true

    /// Brief highlight on YawMarker right after runYawAutoDetect() finds a
    /// marker -- lets a glance confirm the auto-detected position before
    /// trusting it, rather than the value silently changing with no visual
    /// acknowledgment.
    @State private var showYawDetectionFlash = false

    /// Gates the level screen's "OK" button when isMountOK is false -- see
    /// tapOK().
    @State private var showNudgeMountConfirmation = false
    private let thermalBlinkTimer = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    // Full brightness for the whole session, by explicit request (2026-08-14)
    // -- overrides the earlier dashcam-dimming tradeoff (dim while driving to
    // save power, full only for the level/calibration screens that need to
    // resolve fine detail). Restored on disappear rather than left changed
    // system-wide.
    //
    // Exception (2026-08-18): dimmed to dimmedBrightness while low-light
    // boost is actually active -- "On" or "Auto (on)", i.e.
    // isLowLightBoostEnabled, not isAutoLowLightEnabled -- since full
    // brightness fights night visibility/glare at exactly the moment the
    // boost is compensating for a dark scene. Kept reactive via onChange
    // below rather than only set once, since auto-detection can flip this
    // mid-drive.
    private static let dimmedBrightness: CGFloat = 0.7
    @State private var previousBrightness: CGFloat = UIScreen.main.brightness

    private func updateBrightnessForLowLight() {
        UIScreen.main.brightness = cameraManager.isLowLightBoostEnabled ? Self.dimmedBrightness : 1.0
    }

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

    /// Ground-truth distance entry, right after the tape-mark count is
    /// confirmed -- nil = inactive, 0..<tapeMarkCount = currently on that
    /// mark's input screen. Once the last mark is confirmed, flows straight
    /// into round 1 of pitch-adjustment/recording (`distanceCalibrationRound`)
    /// -- re-chained 2026-08-11 after briefly being decoupled on 2026-08-10;
    /// the decoupled version left no way to actually reach the recording
    /// rounds at all, which is what prompted re-chaining them. Entry screens
    /// don't need to be visited in physical near-to-far order (there's no
    /// way to enforce that from the UI alone); whatever order the user
    /// enters them in, `confirmTapeMarkDistance` sorts ascending (nearest
    /// first) before logging the final array.
    @State private var tapeMarkDistanceIndex: Int?
    /// Sized to `tapeMarkCount` as soon as the flow starts (see
    /// `confirmTapeMarkCount`), indexed by screen index -- lets Back/Next
    /// revisit and overwrite any mark's entry rather than only supporting
    /// forward-only append.
    @State private var tapeMarkDistancesMeters: [Double] = []
    @State private var currentDistanceMeters = 0
    @State private var currentDistanceCentimeters = 0

    /// Long-press-on-"Distance cal" menu (see `withDistanceCalGesture`) --
    /// offers "Tape marks" (the original flow, `beginDistanceCalibration`)
    /// vs "Walkaround" (`beginWalkaroundRecording`), added 2026-08-15 once
    /// both flows existed and needed a single shared entry point.
    @State private var isShowingDistanceCalModePicker = false
    /// True while the "Distance cal" label is physically being pressed --
    /// mirrors `isStepperActive`'s exact role: `withSessionGestures`'s
    /// long-press-to-exit is `.simultaneousGesture`, so it fires
    /// independently of whatever gesture is attached directly to this
    /// label, including a plain tap held slightly too long. Without this
    /// guard, opening the long-press menu on the configuring/driving
    /// screens (both wrapped in `withSessionGestures`) would also pop the
    /// exit-confirmation dialog at the same time. Set true on touch-down
    /// (not after any duration), so it's already up by the time the outer
    /// gesture's own ~2s could complete.
    @State private var isDistanceCalLabelPressActive = false
    /// Long-form recording for the "Walkaround" distance-cal flow -- see
    /// `beginWalkaroundRecording`. Deliberately separate from
    /// `isCalibrationRecording` (the tape-marks flow's own state): a
    /// walkaround recording is manually stopped, minutes long, and doesn't
    /// go through the tape-marks state machine's rounds/pitch-adjustment
    /// screens at all.
    @State private var isWalkaroundRecording = false

    private func flashToast(_ text: String) {
        withAnimation { toastText = text }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation { toastText = nil }
        }
    }

    /// "OK" on the level screen, when the mount is already within
    /// tolerance (isMountOK) -- proceeds straight to calibrateAttitude().
    /// When it's NOT (still showing "NUDGE MOUNT"), gates behind a
    /// confirmation instead of silently letting a known-out-of-tolerance
    /// mount through -- see showNudgeMountConfirmation's confirmationDialog
    /// on levelScreen.
    private func tapOK() {
        if isMountOK {
            calibrateAttitude()
        } else {
            DebugFileLogger.log("tap: MATCHED nudgeMountConfirmationPrompt")
            showNudgeMountConfirmation = true
        }
    }

    /// Deliberate, UI-triggered pitch+roll calibration -- see
    /// PitchSensor.captureReferenceAttitude() and DistanceEstimator.swift's
    /// file-level comment for why this must be done standing still on known-
    /// flat, level ground, not while driving. Always runs fresh now (no more
    /// skip-and-reuse-persisted path, see SessionPhase.level's doc comment)
    /// -- transitions to the settings screen on success; stays on the level
    /// screen (with a toast explaining why) if there's no motion reading
    /// yet. Yaw calibration is NOT part of this -- it runs unconditionally
    /// on level-screen appear (see runYawAutoDetect), independent of this.
    private func calibrateAttitude() {
        guard let result = pitchSensor.captureReferenceAttitude() else {
            DebugFileLogger.log("calibrate: FAILED no motion reading yet")
            flashToast("CALIBRATION FAILED\n(no sensor data yet)")
            return
        }
        DebugFileLogger.log("calibrate: MATCHED pitch=\(result.pitch) roll=\(result.roll)")
        enterConfiguring()
    }

    /// Tail of the level screen's "OK" flow (see tapOK/calibrateAttitude) --
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

    /// Reached only via "Tape marks" on the long-press Distance-cal mode
    /// picker (`withDistanceCalGesture` / `isShowingDistanceCalModePicker`)
    /// -- NOT part of the normal session lifecycle (`sessionPhase` doesn't
    /// change). 2026-08-11: the level screen's trigger used to be a 1.5s
    /// long-press instead of a tap, specifically so it wouldn't collide with
    /// the level screen also needing the standard 0.8s long-press-to-exit
    /// gesture (`withSessionGestures`) -- switched to a tap once that was
    /// the actual ask. 2026-08-15: replaced by the long-press mode picker,
    /// then (per explicit request) the plain-tap shortcut into this
    /// function was removed entirely -- a quick accidental tap on this
    /// small, permanently-visible label used to drop straight into the
    /// tape-marks flow with no confirmation, which was the actual problem.
    /// The label is now inert except for the deliberate long-press. Kicks off
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
        guard !isCalibrationRecording, distanceCalibrationRound == nil, !isWalkaroundRecording else { return }
        initialCalibrationRoll = pitchSensor.rollDegrees
        isChoosingTapeMarkCount = true
    }

    /// "Walkaround" choice from the Distance-cal long-press menu (see
    /// `withDistanceCalGesture`) -- added 2026-08-15 after a real
    /// walkaround session (data/26_08_15_Walkaround) came back unusable:
    /// accidentally recorded at 4K, and focus drifted onto the near-field
    /// tape-measure marker the tester holds to maintain each tethered
    /// distance (that marker is a deliberate, permanent part of this
    /// test's methodology -- unlike the tape-marks flow, it can't just be
    /// kept out of frame). Starts a real drive-style recording (1080p,
    /// "recording-" prefixed, manually stopped, no fixed duration) with
    /// focus explicitly settled and LOCKED far/infinity for the whole
    /// clip -- see `CameraManager.startWalkaroundRecording`'s doc comment
    /// for why normal driving's periodic focus recalibration isn't
    /// sufficient here. `isWalkaroundRecording` swaps the current screen
    /// for `walkaroundRecordingBanner` (mirroring `isCalibrationRecording`
    /// -> `distanceCalibrationRecordingBanner`) until `endWalkaroundRecording`
    /// is tapped.
    private func beginWalkaroundRecording() {
        guard !isCalibrationRecording, distanceCalibrationRound == nil, !isWalkaroundRecording else { return }
        DebugFileLogger.log("distance-cal: MATCHED beginWalkaroundRecording")
        cameraManager.startWalkaroundRecording { success in
            Task { @MainActor in
                if success {
                    isWalkaroundRecording = true
                } else {
                    DebugFileLogger.log("distance-cal: beginWalkaroundRecording FAILED to start")
                    flashToast("FAILED TO START RECORDING")
                }
            }
        }
    }

    /// "Stop" on `walkaroundRecordingBanner`. Deliberately not wrapped in
    /// `withSessionGestures`'s exit flow -- same reasoning as
    /// `isCalibrationRecording`'s own banner: exiting the session mid-
    /// recording would race `exitSession`'s `cameraManager.stop()` against
    /// this function's own in-flight stop/teardown.
    private func endWalkaroundRecording() {
        DebugFileLogger.log("distance-cal: MATCHED endWalkaroundRecording")
        cameraManager.stopWalkaroundRecording {
            Task { @MainActor in
                isWalkaroundRecording = false
            }
        }
    }

    /// Attaches the "Distance cal" label's gesture: ONLY a 0.5s long-press
    /// reaches it, opening `isShowingDistanceCalModePicker`'s "Tape
    /// marks"/"Walkaround" choice -- a plain tap is a no-op. 2026-08-15:
    /// used to also fire `beginDistanceCalibration` directly on a plain tap
    /// (via `.exclusively(before:)`, so a short tap and a long-press stayed
    /// mutually exclusive on the same view), but that meant a quick,
    /// accidental tap on this small, permanently-visible label silently
    /// dropped straight into the tape-marks flow -- removed by request, so
    /// the label is now inert except for the deliberate long-press.
    ///
    /// Blocked entirely on the level screen (2026-08-15, by request):
    /// `beginDistanceCalibration`/`beginWalkaroundRecording` only read
    /// `pitchSensor`'s LIVE roll for their own on-screen display -- the
    /// actual `referencePitchDegrees`/`referenceRollDegrees`
    /// `DistanceEstimator` uses for every distance calculation only gets
    /// (re)captured by `calibrateAttitude()` ("OK" on the level screen).
    /// Neither tape-marks nor walkaround recapture it, so starting either
    /// from the level screen -- before pressing OK -- would silently record
    /// against whatever reference was left over from the previous launch.
    /// Reaching the configuring screen means calibration has already run
    /// fresh (unconditional now, no skip path), so this only guards
    /// `.level`.
    ///
    /// The `.simultaneousGesture(DragGesture(minimumDistance: 0))` exists
    /// only to flip `isDistanceCalLabelPressActive` -- see that property's
    /// doc comment for why `withSessionGestures`'s exit gesture needs it
    /// (same pattern as `StepperButton`/`isStepperActive`).
    @ViewBuilder
    private func withDistanceCalGesture<Content: View>(_ label: Content) -> some View {
        label
            .gesture(
                LongPressGesture(minimumDuration: 0.5)
                    .onEnded { _ in
                        guard sessionPhase != .level else {
                            DebugFileLogger.log("gesture: IGNORED longPress(distanceCalModePicker) -- sessionPhase == .level, calibration not complete")
                            flashToast("CALIBRATE FIRST")
                            return
                        }
                        DebugFileLogger.log("gesture: MATCHED longPress(distanceCalModePicker)")
                        isShowingDistanceCalModePicker = true
                    }
            )
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in isDistanceCalLabelPressActive = true }
                    .onEnded { _ in isDistanceCalLabelPressActive = false }
            )
    }

    private func cancelDistanceCalibration() {
        DebugFileLogger.log("distance-cal: MATCHED cancel")
        isChoosingTapeMarkCount = false
        distanceCalibrationRound = nil
        initialCalibrationRoll = nil
        tapeMarkDistanceIndex = nil
        tapeMarkDistancesMeters = []
        currentDistanceMeters = 0
        currentDistanceCentimeters = 0
    }

    private func confirmTapeMarkCount() {
        DebugFileLogger.log("distance-cal: MATCHED confirmTapeMarkCount count=\(tapeMarkCount)")
        isChoosingTapeMarkCount = false
        // 10m is a reasonable starting guess for a following-distance tape
        // mark -- initializing here (rather than 0) means fewer stepper
        // taps for the common case.
        tapeMarkDistancesMeters = Array(repeating: 10, count: tapeMarkCount)
        currentDistanceMeters = 10
        currentDistanceCentimeters = 0
        tapeMarkDistanceIndex = 0
    }

    /// Splits a combined meters value back into its stepper parts, for
    /// loading a previously-entered (or default-zero) mark's distance into
    /// the current screen's fields -- used by both directions of
    /// navigation so Back/Next always show what was last entered for that
    /// mark rather than resetting to zero.
    private func loadDistanceIntoFields(_ distance: Double) {
        let meters = Int(distance)
        currentDistanceMeters = meters
        currentDistanceCentimeters = Int(((distance - Double(meters)) * 100).rounded())
    }

    /// "Next"/"Start Calibration" on the tape-mark distance entry screen --
    /// records the current mark's distance (meters + cm combined) at its
    /// own index (not appended -- Back can revisit and overwrite any
    /// mark), then either advances to the next mark or, after the last
    /// one, sorts the collected distances ascending (nearest first, since
    /// entry order needn't match physical order), logs them, and starts
    /// round 1 of pitch-adjustment/recording (`distanceCalibrationRound`).
    private func confirmTapeMarkDistance() {
        guard let index = tapeMarkDistanceIndex else { return }
        let distance = Double(currentDistanceMeters) + Double(currentDistanceCentimeters) / 100
        tapeMarkDistancesMeters[index] = distance
        DebugFileLogger.log("distance-cal: mark=\(index) distanceMeters=\(distance)")
        if index + 1 >= tapeMarkCount {
            let sorted = tapeMarkDistancesMeters.sorted()
            DebugFileLogger.log("distance-cal: MATCHED all distances entered, nearest-first \(sorted)")
            tapeMarkDistanceIndex = nil
            tapeMarkDistancesMeters = []
            currentDistanceMeters = 0
            currentDistanceCentimeters = 0
            // Re-chained 2026-08-11: distance entry now flows straight into
            // round 1 of pitch-adjustment/recording, same as before the
            // 2026-08-10 decoupling -- `initialCalibrationRoll` (captured
            // back in `beginDistanceCalibration`) stays set, since it's the
            // fixed roll reference `distanceCalibrationPitchScreen` shows
            // across all 3 rounds; only cleared once round 3 actually
            // completes (see `recordCalibrationRound`'s round >= 2 branch).
            distanceCalibrationRound = 0
        } else {
            tapeMarkDistanceIndex = index + 1
            loadDistanceIntoFields(tapeMarkDistancesMeters[index + 1])
        }
    }

    /// "Back" on the tape-mark distance entry screen -- saves whatever's
    /// currently entered (so a Back tap never silently discards it) and
    /// steps to the previous mark's screen, loading its own last-entered
    /// value. Logs the save the same way `confirmTapeMarkDistance` does --
    /// otherwise a value entered then left via Back (never revisited via
    /// Next) would only show up in the final sorted-array summary line,
    /// not in the per-mark log trail.
    private func backTapeMarkDistance() {
        guard let index = tapeMarkDistanceIndex, index > 0 else { return }
        let distance = Double(currentDistanceMeters) + Double(currentDistanceCentimeters) / 100
        tapeMarkDistancesMeters[index] = distance
        DebugFileLogger.log("distance-cal: mark=\(index) distanceMeters=\(distance) (via back)")
        tapeMarkDistanceIndex = index - 1
        loadDistanceIntoFields(tapeMarkDistancesMeters[index - 1])
    }

    /// "Ready" on the "Adjust pitch" screen -- captures the CURRENT live
    /// pitch/roll at full precision (the on-screen readout only shows
    /// these rounded to the nearest degree) and starts that round's ~1s
    /// recording.
    private func finishPitchAdjustment() {
        guard let round = distanceCalibrationRound else { return }
        let pitch = pitchSensor.pitchDegrees
        let roll = pitchSensor.rollDegrees
        // 2026-08-11: added after finding that implied per-round focal
        // length varied ~2x across three tape-mark calibration rounds in a
        // way pitch couldn't explain -- video stabilization crop (see
        // CameraManager.isStabilizationEnabled) changes pixel-space FOV, and
        // this wasn't being recorded per round, so a differing setting
        // between rounds couldn't be ruled in or out after the fact.
        DebugFileLogger.log("distance-cal: round=\(round) pitch=\(String(describing: pitch)) roll=\(String(describing: roll)) stabilizationEnabled=\(cameraManager.isStabilizationEnabled)")
        recordCalibrationRound(round: round)
    }

    /// Records one round's ~1s, 4K, locked-far-focus clip (see
    /// CameraManager.startCalibrationRecording's file-level comment),
    /// saved as calibration-<timestamp>.mov via Photos -- never confused
    /// with a real drive's recording-<timestamp>.mov. No stop control --
    /// one second of already-locked-focus 4K is plenty of frames, so it
    /// just auto-stops rather than needing a second deliberate action.
    ///
    /// 2026-08-11: no automatic tape-mark detection anymore -- previously
    /// ran `CameraManager.countRedTapeMarks` against the last frame and
    /// retried the round on a mismatch; removed by explicit request
    /// ("don't even bother checking... assume we can label tape mark edges
    /// post calibration"). Every round now unconditionally advances (or
    /// completes, after round 3) once its clip is saved -- the actual tape-
    /// mark rows get read off the recorded footage by hand afterward,
    /// which is also what `DistanceEstimator.fit()` needs regardless (it
    /// was never fed by the automatic count in the first place).
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
        // reached the round-advance logic below, reading as "stuck on
        // round 1 with no error message" (there wasn't a missing-error bug
        // -- that logic itself was never being reached). This also means
        // the "recording" banner now shows the instant Ready is tapped
        // instead of after an unexplained multi-second pause, which was
        // likely why the user tapped again in the first place.
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
                        DebugFileLogger.log("distance-cal: round=\(round) DONE (unchecked, detectedCount=\(detectedCount) for reference only)")
                        if round >= 2 {
                            distanceCalibrationRound = nil
                            initialCalibrationRoll = nil
                            flashToast("DISTANCE CALIBRATION\nCOMPLETE (3/3)")
                        } else {
                            distanceCalibrationRound = round + 1
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
        // Cycles nano -> small -> medium -> nano -- voice command (the
        // original way to reach medium, see DetectorModel's doc comment)
        // proved unreliable in practice, so this is the only path to medium now.
        let allModels = DetectorModel.allCases
        let currentIndex = allModels.firstIndex(of: modelManager.selectedModel) ?? 0
        let target = allModels[(currentIndex + 1) % allModels.count]
        DebugFileLogger.log("tap: MATCHED selectModel(\(target.rawValue))")
        modelManager.switchModel(to: target)
    }

    private func toggleResolution() {
        guard !parametersLocked else {
            DebugFileLogger.log("tap: IGNORED (locked) toggleResolution")
            flashToast("LOCKED")
            return
        }
        if modelManager.isHighResEnabled {
            // Turning off -- back to the safe default, no confirmation needed.
            DebugFileLogger.log("tap: MATCHED setHighRes(false)")
            modelManager.setHighResEnabled(false)
        } else {
            DebugFileLogger.log("tap: MATCHED highResConfirmationPrompt")
            showHighResConfirmation = true
        }
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
            pendingLowLightTarget = .on
        } else if cameraManager.isLowLightBoostEnabled {
            pendingLowLightTarget = .off
        } else {
            pendingLowLightTarget = .auto
        }
        DebugFileLogger.log("tap: MATCHED lowLightConfirmationPrompt(\(pendingLowLightTarget.label))")
        showLowLightConfirmation = true
    }

    private func applyPendingLowLightTarget() {
        switch pendingLowLightTarget {
        case .on:
            DebugFileLogger.log("tap: MATCHED setLowLightBoost(true)")
            cameraManager.setLowLightBoost(true)
        case .off:
            DebugFileLogger.log("tap: MATCHED setLowLightBoost(false)")
            cameraManager.setLowLightBoost(false)
        case .auto:
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
        pendingStabilizationEnabled = !cameraManager.isStabilizationEnabled
        DebugFileLogger.log("tap: MATCHED stabilizationConfirmationPrompt(\(pendingStabilizationEnabled))")
        showStabilizationConfirmation = true
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
            if sessionPhase == .level || sessionPhase == .configuring || sessionPhase == .driving
                || isCalibrationRecording || distanceCalibrationRound != nil || isWalkaroundRecording {
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
                withSessionGestures(tapeMarkCountPickerOverlay)
            } else if let index = tapeMarkDistanceIndex {
                withSessionGestures(tapeMarkDistanceScreen(index: index))
            } else if isCalibrationRecording {
                // Deliberately NOT withSessionGestures here -- exiting
                // mid-recording would race exitSession's cameraManager.stop()
                // against stopCalibrationRecording's own in-flight
                // teardown/restart chain, exactly the class of session race
                // this flow already went through several rounds of fixing
                // (see [[project-following-distance-measurement]]). Only a
                // few seconds long either way.
                distanceCalibrationRecordingBanner
            } else if isWalkaroundRecording {
                // Deliberately NOT withSessionGestures here either -- same
                // race-avoidance reasoning as isCalibrationRecording above,
                // just for endWalkaroundRecording's stop/teardown chain
                // instead. Unlike that banner, this one has no fixed
                // duration -- it's up to the user to tap Stop.
                walkaroundRecordingBanner
            } else if let round = distanceCalibrationRound {
                withSessionGestures(distanceCalibrationPitchScreen(round: round))
            } else {
                switch sessionPhase {
                case .level:
                    withSessionGestures(levelScreen)
                case .configuring:
                    withSessionGestures(configuringScreen)
                case .driving:
                    withSessionGestures(drivingScreen)
                }
            }

            // Structural SIBLING to the big if/else chain above, not nested
            // inside it -- see levelScreen's own doc comment for why:
            // YawMarker's drag gesture never fired at all while it was a
            // DESCENDANT of withSessionGestures's gesture wrapper, even
            // with .highPriorityGesture. Mirrors that chain's own gating
            // conditions so this only shows during the plain level screen,
            // not on top of the other calibration sub-flows.
            if sessionPhase == .level, !isChoosingTapeMarkCount, tapeMarkDistanceIndex == nil,
               !isCalibrationRecording, !isWalkaroundRecording, distanceCalibrationRound == nil {
                YawBand(isMountYawOK: isMountYawOK)
                YawMarker(cameraManager: cameraManager, isFlashing: showYawDetectionFlash)
            }

            // Hoisted here (once) rather than duplicated per-screen --
            // CONFIRMED bug 2026-08-09: a per-screen toastView had been
            // missed on distanceCalibrationPitchScreen, so a failed tape-
            // mark detection retried silently with no visible error at all.
            if let toastText {
                toastView(toastText)
            }

            // Hoisted once here (like toastView above) rather than duplicated
            // on both levelScreen and configuringScreen -- the two are the
            // only places `withDistanceCalGesture` attaches, but the picker
            // itself doesn't care which one triggered it, and a single shared
            // instance avoids duplicating this exact button list twice in
            // sync. CONFIRMED 2026-08-15 via a real device screenshot: a
            // system `.confirmationDialog` here rendered "Tape marks" and
            // "Walkaround" but silently dropped its own `role: .cancel`
            // button -- the exact same pattern works fine for the unrelated
            // exit-confirmation dialog elsewhere in this file (its "No" shows
            // up correctly), so this wasn't a general role:.cancel bug, just
            // something specific to this one dialog that wasn't worth
            // chasing further blind. Replaced with a plain custom overlay
            // instead, matching every other picker in this app
            // (`tapeMarkCountPickerOverlay`, etc.) -- fully self-drawn, so
            // there's no system component left to silently misbehave.
            if isShowingDistanceCalModePicker {
                distanceCalModePickerOverlay
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
        .onChange(of: cameraManager.isLowLightBoostEnabled) { _, _ in
            updateBrightnessForLowLight()
        }
        .onAppear {
            DebugFileLogger.reset()
            DetectionLogger.reset()
            // Starts the preview-only session immediately on launch -- used
            // to wait for "Yes" on the now-removed calibrate-prompt screen,
            // but the app now goes straight to the level screen, so the
            // live camera feed (and yaw auto-detect, see levelScreen's own
            // onAppear) needs to be up from the very first frame.
            cameraManager.start(recording: false)
            // Keeps the screen (and thus the camera/recording) awake for the whole
            // drive instead of auto-locking after the idle timeout.
            UIApplication.shared.isIdleTimerDisabled = true
            previousBrightness = UIScreen.main.brightness
            updateBrightnessForLowLight()
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
                    parametersLocked: isLocked.wrappedValue,
                    yawReferenceNormalizedX: cameraManager?.yawReferenceNormalizedX ?? CameraManager.defaultYawMarkerNormalizedX
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

    // MARK: — Level screen

    /// Shared video-to-screen x math for the yaw calibration overlays
    /// (YawBand, YawMarker) -- the camera preview scales to fill the view
    /// and center-crops on whichever axis overflows, so converting a
    /// normalized video-space x into a screen-space x needs the actual
    /// runtime view size, not a hardcoded device aspect ratio. Was inline
    /// in the old single-line YawReferenceLine; factored out since two
    /// separate overlays need the same math now.
    private enum YawOverlayGeometry {
        static let videoWidth: CGFloat = 1920
        static let videoHeight: CGFloat = 1080

        static func displayedWidth(in geo: GeometryProxy) -> CGFloat {
            let scale = max(geo.size.width / videoWidth, geo.size.height / videoHeight)
            return videoWidth * scale
        }

        static func screenX(forNormalizedX normalizedX: CGFloat, in geo: GeometryProxy) -> CGFloat {
            let displayed = displayedWidth(in: geo)
            let originX = (geo.size.width - displayed) / 2
            return originX + normalizedX * displayed
        }

        static func screenWidth(forNormalizedWidth normalizedWidth: CGFloat, in geo: GeometryProxy) -> CGFloat {
            normalizedWidth * displayedWidth(in: geo)
        }
    }

    /// Coarse, FIXED-position "is the mount roughly sane" sanity check --
    /// replaces the old single-pixel-precise YawReferenceLine, which is now
    /// two separate things: this band (a stable expectation about the
    /// mount hardware, never re-measured) and YawMarker below (the
    /// precise, per-session measurement). Deliberately not yellow, so the
    /// actual yellow marker doesn't visually blend into it when correctly
    /// aligned -- that's exactly the case where standing out matters most.
    /// Border color mirrors the roll readout's own green/white convention,
    /// extended to a third (orange) "needs attention" state -- see
    /// ContentView.isMountYawOK.
    private struct YawBand: View {
        // Placeholder pending real data on how much residual yaw offset the
        // fine per-session correction can actually absorb -- same "don't
        // pretend to precision we don't have yet" discipline as
        // path_awareness.py's own placeholder constants. Revisit once
        // there's a real basis to tune this against.
        static let halfWidthNormalized: CGFloat = 0.04

        var isMountYawOK: Bool

        var body: some View {
            GeometryReader { geo in
                let centerX = YawOverlayGeometry.screenX(forNormalizedX: CameraManager.defaultYawMarkerNormalizedX, in: geo)
                let width = YawOverlayGeometry.screenWidth(forNormalizedWidth: 2 * Self.halfWidthNormalized, in: geo)
                Rectangle()
                    .fill(Color.white.opacity(0.15))
                    .overlay(
                        Rectangle()
                            .stroke(isMountYawOK ? Color.green.opacity(0.7) : Color.orange.opacity(0.8), lineWidth: 2)
                    )
                    .frame(width: width, height: geo.size.height)
                    .position(x: centerX, y: geo.size.height / 2)
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)
        }
    }

    /// Precise, per-session yaw-calibration marker -- positioned at
    /// cameraManager.yawReferenceNormalizedX (auto-detected on level-screen
    /// appear, see runYawAutoDetect), draggable for manual fine adjustment
    /// when auto-detection needs correcting.
    ///
    /// Drag movement is scaled down from actual finger movement (dragScale
    /// below) rather than 1:1, so a full swipe across the screen only moves
    /// the marker a small, precise amount -- addresses the "fat finger"
    /// imprecision problem a direct drag would have against a target this
    /// narrow (see the design discussion this came out of). The tappable
    /// hit area is much wider than the visible line for the same reason,
    /// applied to STARTING the drag rather than moving it.
    private struct YawMarker: View {
        // How much of actual finger movement translates to marker
        // movement -- 0.25 means dragging 100pt across the screen moves
        // the marker by the normalized-x equivalent of 25pt of on-screen
        // travel. Placeholder, not yet validated against how much
        // precision this actually needs in practice.
        static let dragScale: CGFloat = 0.25
        private static let hitAreaWidth: CGFloat = 100

        @ObservedObject var cameraManager: CameraManager
        var isFlashing: Bool

        @State private var dragStartNormalizedX: CGFloat?
        @State private var liveNormalizedX: CGFloat?

        private var displayedNormalizedX: CGFloat {
            liveNormalizedX ?? cameraManager.yawReferenceNormalizedX
        }

        var body: some View {
            GeometryReader { geo in
                let x = YawOverlayGeometry.screenX(forNormalizedX: displayedNormalizedX, in: geo)
                ZStack {
                    // Generous drag catcher -- the right HALF of the screen
                    // (away from the OK button and "Distance cal" label,
                    // both on the left), not a narrow region around the
                    // line. THREE narrower attempts before this (a nested
                    // Color.clear+contentShape rectangle; layered .frame+
                    // .contentShape on the line itself; highPriorityGesture;
                    // a structural-sibling restructure out of
                    // withSessionGestures) all registered ZERO drag events
                    // on-device, confirmed via debug log each time -- this
                    // sidesteps hit-testing precision entirely rather than
                    // refining it further. Color.white.opacity(0.001), not
                    // Color.clear -- a known SwiftUI workaround for cases
                    // where a truly transparent color's hit-testing doesn't
                    // reliably register despite .contentShape.
                    //
                    // Includes a diagnostic .simultaneousGesture(TapGesture)
                    // logging separately from the drag: if a plain tap ALSO
                    // never registers, that points to something more
                    // fundamental about this screen region intercepting
                    // ALL touches, not a drag-specific issue.
                    Color.white.opacity(0.001)
                        .frame(width: geo.size.width / 2, height: geo.size.height)
                        .position(x: geo.size.width * 0.75, y: geo.size.height / 2)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    let base = dragStartNormalizedX ?? cameraManager.yawReferenceNormalizedX
                                    if dragStartNormalizedX == nil {
                                        dragStartNormalizedX = base
                                        DebugFileLogger.log("yaw-marker: drag STARTED base=\(base)")
                                    }
                                    let displayedWidth = YawOverlayGeometry.displayedWidth(in: geo)
                                    let delta = (value.translation.width * Self.dragScale) / displayedWidth
                                    liveNormalizedX = base + delta
                                }
                                .onEnded { _ in
                                    DebugFileLogger.log("yaw-marker: drag ENDED")
                                    if let final = liveNormalizedX {
                                        cameraManager.setYawReferenceNormalizedX(final)
                                    }
                                    dragStartNormalizedX = nil
                                    liveNormalizedX = nil
                                }
                        )
                        .simultaneousGesture(
                            TapGesture().onEnded {
                                DebugFileLogger.log("yaw-marker: TAP registered (diagnostic)")
                            }
                        )
                    Rectangle()
                        .fill(Color.yellow.opacity(isFlashing ? 1.0 : 0.85))
                        .frame(width: isFlashing ? 8 : 4, height: geo.size.height)
                        .position(x: x, y: geo.size.height / 2)
                        .animation(.easeOut(duration: 0.3), value: isFlashing)
                        .allowsHitTesting(false)
                }
            }
            // Same GeometryReader full-bounds fix as the old YawReferenceLine
            // needed -- see git history if this regresses ("doesn't go all
            // the way to the bottom").
            .ignoresSafeArea()
        }
    }

    private var rollDegreesRounded: Int {
        Int((pitchSensor.rollDegrees ?? 0).rounded())
    }

    /// Live pitch minus `PitchSensor.defaultMountPitchDegrees`, rounded --
    /// so the level screen can show "0" at the mount's known-good tilt
    /// instead of the raw absolute pitch (which is never near 0, see
    /// `defaultMountPitchDegrees`'s doc comment). Falls back to 0 (reads as
    /// "on target") when no reading exists yet, same convention as
    /// `rollDegreesRounded`.
    private var pitchOffsetDegreesRounded: Int {
        guard let pitch = pitchSensor.pitchDegrees else { return 0 }
        return Int((pitch - PitchSensor.defaultMountPitchDegrees).rounded())
    }

    /// The three mount-status checks, split out so YawBand's border color,
    /// the on-screen verdict text, and (eventually) any hard gate on
    /// starting a drive all read the same computed truth rather than
    /// three separately-eyeballed signals -- see the design discussion
    /// this came out of: "takes the guesswork out of whether to
    /// calibrate." Fine per-session calibration (pitch/roll reference
    /// capture, yaw auto-detect) always runs regardless of these --
    /// they're only about whether the PHYSICAL mount needs a manual nudge
    /// on top of that.
    private var isMountRollOK: Bool {
        abs(rollDegreesRounded) <= 1
    }

    private var isMountPitchOK: Bool {
        abs(pitchOffsetDegreesRounded) <= 1
    }

    private var isMountYawOK: Bool {
        abs(cameraManager.yawReferenceNormalizedX - CameraManager.defaultYawMarkerNormalizedX) <= YawBand.halfWidthNormalized
    }

    private var isMountOK: Bool {
        isMountRollOK && isMountPitchOK && isMountYawOK
    }

    /// Runs yaw auto-detection unconditionally on level-screen appear --
    /// unlike pitch/roll (which has a skip-and-reuse-persisted path, see
    /// `skipCalibration`), fine yaw calibration has no stationary/level
    /// precondition to worry about, so there's no reason to ever skip it.
    /// A short delay gives the camera session (just started by
    /// `confirmCalibrate`) time to actually produce a frame -- detection
    /// against no frame yet just leaves the previous value in place (see
    /// `CameraManager.detectYawMarker`), not a crash, but retrying blind
    /// immediately would likely just fail the first time every launch.
    private func runYawAutoDetect() {
        Task {
            try? await Task.sleep(for: .seconds(0.4))
            guard cameraManager.detectYawMarker() != nil else {
                DebugFileLogger.log("yaw-detect: FAILED no confident marker found")
                return
            }
            DebugFileLogger.log("yaw-detect: MATCHED x=\(cameraManager.yawReferenceNormalizedX)")
            withAnimation { showYawDetectionFlash = true }
            try? await Task.sleep(for: .seconds(0.4))
            withAnimation { showYawDetectionFlash = false }
        }
    }

    /// Circular arc + arrowhead (SF Symbol, not hand-drawn) pointing the
    /// direction to rotate the mount to null out roll -- shown only while
    /// isMountRollOK is false. `needsClockwise = rollDegrees > 0` is a
    /// CHOSEN, not bench-confirmed, convention -- rollDegrees's own sign
    /// (atan2(gy, -gx)) is itself flagged not-yet-bench-confirmed in
    /// PitchSensor.swift, so treat this arrow's direction the same way:
    /// confirm which way an actual mount nudge moves the reading toward
    /// zero before trusting it blindly.
    private struct RollNudgeIndicator: View {
        var rollDegrees: Double
        private var needsClockwise: Bool { rollDegrees > 0 }

        var body: some View {
            VStack(spacing: 4) {
                Image(systemName: needsClockwise ? "arrow.clockwise" : "arrow.counterclockwise")
                    .font(.system(size: 34, weight: .bold))
                    .foregroundStyle(.orange)
                Text("Roll")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.orange.opacity(0.8))
            }
            .shadow(color: .black.opacity(0.7), radius: 4)
        }
    }

    /// A thin plane rendered with a genuine 3D tilt (rotation3DEffect, not
    /// a flat skew) plus an up/down chevron -- shown only while
    /// isMountPitchOK is false, indicating which way to tilt the phone's
    /// nose. CONFIRMED 2026-08-18 on-device: the original `< reference`
    /// condition pointed the wrong direction -- flipped to `>`. Caller
    /// passes `needsNoseDown` computed from the live reading vs.
    /// `PitchSensor.defaultMountPitchDegrees`.
    private struct PitchNudgeIndicator: View {
        var needsNoseDown: Bool

        var body: some View {
            VStack(spacing: 4) {
                Image(systemName: needsNoseDown ? "chevron.down" : "chevron.up")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundStyle(.orange)
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.orange.opacity(0.85))
                    .frame(width: 56, height: 8)
                    .rotation3DEffect(
                        .degrees(needsNoseDown ? -25 : 25),
                        axis: (x: 1, y: 0, z: 0),
                        perspective: 0.6
                    )
                Text("Pitch")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.orange.opacity(0.8))
            }
            .shadow(color: .black.opacity(0.7), radius: 4)
        }
    }

    /// Live camera preview shows through from `body` (see the
    /// `sessionPhase == .level` case added to its visibility condition) --
    /// deliberately no opaque background here anymore, matching
    /// `distanceCalibrationPitchScreen`'s pattern, so the yaw overlays and
    /// the dash behind them are actually visible. Text shadows replace the
    /// old solid-black backdrop for legibility over live video.
    ///
    /// YawBand/YawMarker are NOT here anymore -- moved to a sibling in the
    /// outer ZStack (see body), OUTSIDE withSessionGestures's wrapping.
    /// CONFIRMED 2026-08-18 via on-device debug log: YawMarker's drag
    /// gesture never fired at all (not even once) while nested inside
    /// withSessionGestures's content, even with .highPriorityGesture --
    /// that modifier resolves parent/child priority within one view's own
    /// ancestor chain, not sibling z-order conflicts within a ZStack, and
    /// something about being a DESCENDANT of withSessionGestures's
    /// .contentShape(Rectangle()) + .simultaneousGesture(LongPressGesture)
    /// wrapper was swallowing the touch before YawMarker ever saw it.
    /// Making it a structural SIBLING instead of a descendant sidesteps
    /// the whole question of gesture priority rather than resolving it.
    private var levelScreen: some View {
        ZStack {
            // Top-left corner label, same low-key styling and tap trigger
            // as configuringScreen's -- not wrapped in the full settingsHUD
            // (its resolution/model/stabilization labels don't apply here,
            // there's no locked/unlocked settings state to show yet).
            VStack {
                HStack {
                    withDistanceCalGesture(
                        Text("Distance cal")
                            .font(.system(size: 28, weight: .medium))
                            .foregroundStyle(.white.opacity(0.4))
                            .shadow(color: .black.opacity(0.6), radius: 2)
                    )
                    Spacer()
                }
                Spacer()
            }
            .padding()
            VStack(spacing: 32) {
                // The explicit go/no-go verdict -- see isMountOK's doc
                // comment for why this exists as its own line instead of
                // leaving Rick to mentally combine the roll color and the
                // yaw band by eye.
                Text(isMountOK ? "MOUNT OK" : "NUDGE MOUNT")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(isMountOK ? .green : .orange)
                    .shadow(color: .black.opacity(0.7), radius: 4)
                if !isMountOK {
                    VStack(spacing: 4) {
                        if !isMountPitchOK {
                            Text("Pitch must be within 1° of reference")
                        }
                        if !isMountRollOK {
                            Text("Roll must be between -1° and 1°")
                        }
                        if !isMountYawOK {
                            Text("Yellow dash stick must be in band")
                        }
                    }
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.orange)
                    .shadow(color: .black.opacity(0.7), radius: 3)
                }
                Text("Pitch: \(pitchOffsetDegreesRounded)°   Roll: \(rollDegreesRounded)°")
                    .font(.system(size: 52, weight: .bold, design: .rounded))
                    .foregroundStyle(isMountPitchOK && isMountRollOK ? .green : .white)
                    .shadow(color: .black.opacity(0.7), radius: 4)
                if !isMountPitchOK || !isMountRollOK {
                    HStack(spacing: 32) {
                        if !isMountPitchOK {
                            PitchNudgeIndicator(
                                needsNoseDown: (pitchSensor.pitchDegrees ?? PitchSensor.defaultMountPitchDegrees)
                                    > PitchSensor.defaultMountPitchDegrees
                            )
                        }
                        if !isMountRollOK {
                            RollNudgeIndicator(rollDegrees: pitchSensor.rollDegrees ?? 0)
                        }
                    }
                }
                // Added 2026-08-11 to make the pitch-sign hand-tilt check
                // (DistanceEstimator.swift's file-level warning: "tilt the
                // mounted phone's nose down by hand and confirm pitchDegrees
                // increases") quick to do -- the only other live pitch
                // readout is buried behind the whole tape-mark-count +
                // distance-entry flow. One decimal place, not rounded to a
                // whole degree, so a small hand tilt visibly moves the
                // number instead of needing a large exaggerated motion to
                // see any change at all. Kept alongside the new reference-
                // relative "Pitch: X°" readout above, not replaced by it --
                // this is the raw sensor value that formula actually needs.
                Text(pitchSensor.pitchDegrees.map { String(format: "absolute pitch: %.1f°", $0) } ?? "absolute pitch: --")
                    .font(.system(size: 20, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.6))
                    .shadow(color: .black.opacity(0.7), radius: 4)
            }

            // 2026-08-12: moved out to the far left -- was previously
            // stacked centered with the roll/pitch readout above, which
            // covered the yaw marker right when the user needs an
            // unobstructed view of it (this screen's whole point is
            // checking yaw alignment against the band). CONFIRMED 2026-08-18
            // on-device: vertically centered (as it was) put it directly
            // behind the center readout's left edge once that readout grew
            // wider (pitch+roll combined, plus reasons/indicators) --
            // pinned to the bottom-left corner instead so it can't collide
            // with the center content regardless of that content's width.
            VStack {
                Spacer()
                HStack {
                    Button {
                        tapOK()
                    } label: {
                        Text("OK")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundStyle(.black)
                            .frame(width: 220, height: 80)
                            .background(Color.yellow, in: Capsule())
                    }
                    .padding(.leading, 12)
                    .padding(.bottom, 12)
                    .confirmationDialog(
                        "Mount out of tolerance -- continue anyway?",
                        isPresented: $showNudgeMountConfirmation,
                        titleVisibility: .visible
                    ) {
                        Button("Continue Anyway") {
                            DebugFileLogger.log("tap: MATCHED nudgeMountConfirmed")
                            calibrateAttitude()
                        }
                        Button("Cancel", role: .cancel) {}
                    }
                    Spacer()
                }
            }
        }
        .onAppear {
            runYawAutoDetect()
        }
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

    /// Shared circular +/- button, matching `tapeMarkCountPickerOverlay`'s
    /// original inline style -- factored out once the meters/cm entry
    /// screen below tripled the number of call sites. Thin wrapper around
    /// `StepperButton` so none of the 6 call sites needed to change when
    /// hold-to-repeat was added.
    private func stepperButton(_ label: String, action: @escaping () -> Void) -> some View {
        StepperButton(label: label, action: action, onHoldChange: { isStepperActive = $0 })
    }

    /// A single tap fires `action` once, same as before. Holding past 1s
    /// starts auto-repeating it every 0.15s until released -- added
    /// 2026-08-11 so running the meters/cm distance steppers up by a lot
    /// doesn't need dozens of individual taps. Deliberately not built on
    /// `Button` -- a plain view + `.onTapGesture` for the single-tap case
    /// plus a `simultaneousGesture(DragGesture(minimumDistance: 0))` to
    /// track press/release for the hold-repeat timer, which is the more
    /// reliable combination for a custom hold-repeat control than layering
    /// extra gestures on a real `Button` (whose own tap recognizer can
    /// interact unpredictably with a simultaneous one). No double-fire risk
    /// from combining `.onTapGesture` with the drag-tracked repeat: a quick
    /// tap's `onEnded` cancels the not-yet-fired repeat task well before
    /// its 1s delay elapses, and a genuine long hold never satisfies
    /// `.onTapGesture`'s own short-press recognition in the first place.
    /// `onHoldChange` reports press/release up to `isStepperActive` so
    /// `withSessionGestures`'s long-press-to-exit gesture (which lives
    /// several view levels up, on whichever screen embeds this button) can
    /// tell a stepper hold apart from an actual exit-intent long-press --
    /// see that function's doc comment.
    private struct StepperButton: View {
        let label: String
        let action: () -> Void
        var onHoldChange: (Bool) -> Void = { _ in }
        @State private var repeatTask: Task<Void, Never>?

        var body: some View {
            Text(label)
                .font(.system(size: 40, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 64, height: 64)
                .background(Color.black.opacity(0.5), in: Circle())
                .contentShape(Circle())
                .onTapGesture { action() }
                .simultaneousGesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { _ in
                            guard repeatTask == nil else { return }
                            onHoldChange(true)
                            repeatTask = Task {
                                try? await Task.sleep(for: .seconds(1))
                                while !Task.isCancelled {
                                    action()
                                    try? await Task.sleep(for: .seconds(0.15))
                                }
                            }
                        }
                        .onEnded { _ in
                            repeatTask?.cancel()
                            repeatTask = nil
                            onHoldChange(false)
                        }
                )
        }
    }

    /// One of N sequential screens (see `tapeMarkDistanceIndex`), shown
    /// before any of the 3 pitch/recording rounds, for entering the
    /// measured ground-truth distance from the camera to each tape mark --
    /// split into meters/cm steppers on one line rather than text entry,
    /// matching `tapeMarkCountPickerOverlay`'s tap-based convention, the
    /// only input style used anywhere else in the app. Fully opaque
    /// singleton, same reasoning as that screen: no camera preview needed
    /// for pure numeric entry. Back/Next let the user revisit and correct
    /// any mark in either direction -- entry order doesn't need to match
    /// physical near-to-far order since `confirmTapeMarkDistance` sorts
    /// before logging.
    private func tapeMarkDistanceScreen(index: Int) -> some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 32) {
                Text("Tape Mark \(index + 1)")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundStyle(.white)
                HStack(spacing: 48) {
                    VStack(spacing: 8) {
                        Text("meters")
                            .font(.system(size: 28, weight: .medium))
                            .foregroundStyle(.white.opacity(0.6))
                        HStack(spacing: 20) {
                            stepperButton("−") {
                                currentDistanceMeters = max(0, currentDistanceMeters - 1)
                            }
                            Text("\(currentDistanceMeters)")
                                .font(.system(size: 56, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                                .frame(minWidth: 72)
                            stepperButton("+") {
                                currentDistanceMeters += 1
                            }
                        }
                    }
                    VStack(spacing: 8) {
                        Text("cm")
                            .font(.system(size: 28, weight: .medium))
                            .foregroundStyle(.white.opacity(0.6))
                        HStack(spacing: 20) {
                            stepperButton("−") {
                                currentDistanceCentimeters = (currentDistanceCentimeters - 1 + 100) % 100
                            }
                            Text("\(currentDistanceCentimeters)")
                                .font(.system(size: 56, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                                .frame(minWidth: 72)
                            stepperButton("+") {
                                currentDistanceCentimeters = (currentDistanceCentimeters + 1) % 100
                            }
                        }
                    }
                }
                HStack(spacing: 24) {
                    if index > 0 {
                        Button {
                            backTapeMarkDistance()
                        } label: {
                            Text("Back")
                                .font(.system(size: 24, weight: .bold))
                                .foregroundStyle(.white)
                                .padding(.horizontal, 40)
                                .frame(height: 64)
                                .background(Color.white.opacity(0.15), in: Capsule())
                        }
                    }
                    Button {
                        confirmTapeMarkDistance()
                    } label: {
                        Text(index + 1 >= tapeMarkCount ? "Start Calibration" : "Next")
                            .font(.system(size: 24, weight: .bold))
                            .foregroundStyle(.black)
                            .padding(.horizontal, 40)
                            .frame(height: 64)
                            .background(Color.yellow, in: Capsule())
                    }
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
        ZStack {
            // Fixed reference only here (not the draggable YawMarker too --
            // this screen has its own tape-mark-based interaction model,
            // dragging a yaw crosshair on top of it would just be
            // confusing) -- same purpose as before: a stable visual anchor
            // to check the camera hasn't rotated between rounds.
            YawBand(isMountYawOK: isMountYawOK)
            distanceCalibrationPitchScreenContent(round: round)
        }
    }

    private func distanceCalibrationPitchScreenContent(round: Int) -> some View {
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

    /// Shown in place of sessionPhase's own screen for the whole
    /// "Walkaround" recording -- see `beginWalkaroundRecording`. Unlike
    /// `distanceCalibrationRecordingBanner`'s fixed ~1s auto-stop, this has
    /// no set duration, so it needs its own explicit Stop control.
    private var walkaroundRecordingBanner: some View {
        VStack {
            Text("● WALKAROUND RECORDING (1080p, focus locked far)")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
                .padding()
                .background(Color.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
                .padding(.top, 50)
            Spacer()
            Button {
                endWalkaroundRecording()
            } label: {
                Text("Stop")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 32)
                    .frame(height: 64)
                    .background(Color.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 16))
            }
            .padding(.bottom, 60)
        }
    }

    /// Custom replacement for the system `.confirmationDialog` that used to
    /// live here -- see body's comment (2026-08-15) for why. A dimmed scrim
    /// (tap anywhere on it to cancel, matching an action sheet's swipe-down/
    /// tap-outside dismiss) behind a centered card, styled like every other
    /// two-choice-plus-cancel screen in this file (`configuringScreen`'s
    /// "Lock Settings"/"Unlocked" for the two real choices,
    /// `tapeMarkCountPickerOverlay`'s plain-text Cancel for the third).
    /// Each button flips `isShowingDistanceCalModePicker` off itself before
    /// calling its action -- unlike a system dialog, a plain `Button` here
    /// doesn't auto-dismiss anything.
    private var distanceCalModePickerOverlay: some View {
        ZStack {
            Color.black.opacity(0.6)
                .ignoresSafeArea()
                .onTapGesture { isShowingDistanceCalModePicker = false }
            VStack(spacing: 20) {
                Text("Distance calibration")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(.white)
                Button {
                    isShowingDistanceCalModePicker = false
                    beginDistanceCalibration()
                } label: {
                    Text("Tape marks")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(.black)
                        .frame(maxWidth: .infinity)
                        .frame(height: 60)
                        .background(Color.yellow, in: RoundedRectangle(cornerRadius: 16))
                }
                Button {
                    isShowingDistanceCalModePicker = false
                    beginWalkaroundRecording()
                } label: {
                    Text("Walkaround")
                        .font(.system(size: 22, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 60)
                        .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))
                }
                Button {
                    isShowingDistanceCalModePicker = false
                } label: {
                    Text("Cancel")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(.white.opacity(0.6))
                }
                .padding(.top, 4)
            }
            .padding(28)
            .frame(maxWidth: 340)
            .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 20))
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
                withDistanceCalGesture(
                    Text(isCalibrationRecording ? "● Recording" : "Distance cal")
                        .font(.system(size: 28, weight: .medium))
                        .foregroundStyle(isCalibrationRecording ? .red : .white.opacity(0.4))
                        .shadow(color: .black.opacity(0.6), radius: 2)
                )
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
                        .hudBoxBackground()
                    Text(thermalLabel)
                        .font(.system(size: 36, weight: .medium))
                        .foregroundStyle(thermalLabelColor)
                        .shadow(color: .black.opacity(0.6), radius: 2)
                        .hudBoxBackground()
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
                    .confirmationDialog(
                        "Turn stabilization \(pendingStabilizationEnabled ? "on" : "off")?",
                        isPresented: $showStabilizationConfirmation,
                        titleVisibility: .visible
                    ) {
                        Button(pendingStabilizationEnabled ? "Turn On" : "Turn Off") {
                            DebugFileLogger.log("tap: MATCHED setStabilization(\(pendingStabilizationEnabled))")
                            cameraManager.setStabilizationEnabled(pendingStabilizationEnabled)
                        }
                        Button("Cancel", role: .cancel) {}
                    }
            }
            Spacer()
            HStack {
                VStack(alignment: .leading) {
                    Text(resolutionLabel)
                        .hudLabelStyle()
                        .onTapGesture { toggleResolution() }
                        .confirmationDialog(
                            "Enable high-res? This significantly increases per-frame latency.",
                            isPresented: $showHighResConfirmation,
                            titleVisibility: .visible
                        ) {
                            Button("Enable") {
                                DebugFileLogger.log("tap: MATCHED setHighRes(true)")
                                modelManager.setHighResEnabled(true)
                            }
                            Button("Cancel", role: .cancel) {}
                        }
                    Text(modelLabel)
                        .hudLabelStyle()
                        .onTapGesture { toggleModel() }
                }
                Spacer()
                Text(lowLightLabel)
                    .hudLabelStyle()
                    .onTapGesture { toggleLowLight() }
                    .confirmationDialog(
                        "Set low-light to \(pendingLowLightTarget.label)?",
                        isPresented: $showLowLightConfirmation,
                        titleVisibility: .visible
                    ) {
                        Button("Set to \(pendingLowLightTarget.label.capitalized)") {
                            applyPendingLowLightTarget()
                        }
                        Button("Cancel", role: .cancel) {}
                    }
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
                //
                // isStepperActive guard added 2026-08-11: this gesture is
                // `.simultaneousGesture`, which deliberately does NOT get
                // blocked by inner views' own gestures (so long-press-to-
                // exit still works no matter where on screen the touch
                // starts) -- but that means holding a StepperButton long
                // enough to trigger its own hold-to-repeat could ALSO
                // satisfy this long-press, popping the exit dialog over what
                // was meant as "keep incrementing the meters/cm value."
                // StepperButton flips `isStepperActive` true the instant its
                // own hold begins, so checking it here reliably suppresses
                // this gesture for that case without needing real gesture-
                // level exclusivity across several view-hierarchy levels.
                // isDistanceCalLabelPressActive (2026-08-15) does the
                // identical job for the "Distance cal" label's own
                // long-press menu -- see its doc comment.
                //
                // CONFIRMED bug 2026-08-15: isDistanceCalLabelPressActive
                // alone wasn't enough -- the label's own long-press fires at
                // 0.5s and presents `isShowingDistanceCalModePicker`'s
                // confirmationDialog, which (unlike StepperButton's plain
                // counter-increment) disrupts the DragGesture(minimumDistance:
                // 0) the flag depends on, so it could read back false by the
                // time this gesture's own onEnded fires -- "Exit this
                // session?" popped up right behind the picker. Checking
                // `isShowingDistanceCalModePicker` directly closes that gap:
                // it flips true at the SAME 0.5s mark, from real state (is the
                // picker actually showing), not from touch-tracking that a
                // dialog presentation can interrupt.
                //
                // Duration raised 0.8s -> 2s (2026-08-15, by request) -- a
                // deliberate exit action shouldn't be this easy to trigger by
                // accident during normal handling of the phone.
                LongPressGesture(minimumDuration: 2.0)
                    .onEnded { _ in
                        guard !isStepperActive, !isDistanceCalLabelPressActive, !isShowingDistanceCalModePicker else { return }
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
