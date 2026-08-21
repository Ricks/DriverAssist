//
//  CameraManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

@preconcurrency import AVFoundation
import CoreVideo
import os
import Photos
import SwiftUI
import UIKit

// MARK: — CameraManager

@MainActor
final class CameraManager: NSObject, ObservableObject {

    /// Called on the main actor for every captured frame.
    var onFrame: ((CVPixelBuffer) -> Void)?

    /// Whether the low-light exposure boost is active. Published for the HUD text;
    /// toggling also updates `currentLowLightEnabled` below for the recording bake.
    @Published private(set) var isLowLightBoostEnabled = false

    /// Whether the ISO-based auto-detector is currently in control of the boost. A
    /// manual voice override suspends this until `enableAutoLowLight()` is called
    /// again (voice: "low light auto"), so auto-detection doesn't immediately fight
    /// an explicit override.
    @Published private(set) var isAutoLowLightEnabled = true

    /// Whether frames are actively being written to a recording segment right now.
    /// Published so the HUD can surface a loud warning the moment this goes false —
    /// a silent recording failure is otherwise invisible, since live detection keeps
    /// working normally regardless of whether anything is being saved.
    @Published private(set) var isRecording = false

    /// True when free storage has dropped below `lowStorageThresholdBytes`. Published
    /// so the HUD can warn before recording actually fails from running out of space,
    /// rather than only reporting it after the fact.
    @Published private(set) var isStorageLow = false

    /// Whether `.standard` video stabilization is applied to the capture connection.
    /// Published for the HUD text; toggling also updates `currentStabilizationEnabled`
    /// below for applying the mode to the capture connection. Defaults on (not
    /// persisted, see init() -- every launch starts here regardless of the
    /// last session's choice).
    @Published private(set) var isStabilizationEnabled = true

    /// The near-focus lens position actually used for the whole yaw
    /// calibration flow -- see `settleAndLockNearFieldFocus`. Added by
    /// request 2026-08-20: the empirically-swept `defaultNearFocusLensPosition`
    /// (0.35) started producing visibly soft frames on a real device even
    /// though the lock itself was confirmed landing exactly there (not a
    /// focus-race bug -- ruled out via log/frame comparison against the
    /// original sharp reference capture). Rather than re-run the sweep test
    /// blind, this lets the user dial it in directly via the Manual Focus
    /// screen (ContentView) while watching the live zoomed preview, and
    /// PERSISTS it -- unlike most calibration state (which re-verifies every
    /// session on purpose, see isYawVerifiedThisSession's doc comment), this
    /// is a property of the PHYSICAL LENS/MOUNT geometry, not something that
    /// needs re-confirming every drive, same reasoning as yawReferenceNormalizedX
    /// being persisted where most settings aren't.
    @Published private(set) var manualNearFocusLensPosition: Float = CameraManager.defaultNearFocusLensPosition

    /// Whether normal (non-calibration) recording uses 4K instead of the
    /// default 1080p. REACTIVATED 2026-08-18 -- originally added 2026-08-11
    /// for the cone calibration test, then removed 2026-08-15 after a real
    /// walkaround session came back unusable (accidentally recorded at 4K,
    /// see git history) because the old UI toggle had NO confirmation step
    /// -- a stray tap silently switched a real recording to 4K. Reactivated
    /// with a required confirmation dialog (ContentView.toggleFourK/
    /// showFourKConfirmation, same pattern as the high-res ML-inference
    /// toggle) specifically to close that gap, prompted by wanting higher-
    /// fidelity source footage for offline YOLO26x ground-truth labeling.
    /// Defaults off/1080p and is deliberately NOT persisted (see init's
    /// comment) -- always starts back at 1080p on a fresh launch, so a
    /// deliberate choice for one recording can't silently carry over into a
    /// later one. Thermal risk is real and confirmed (see `configure`'s own
    /// comment: 10-15x throttling on a 23-minute 4K/30fps drive) -- that
    /// test predates the Cooler 8 Pro being mount-compatible, so the actual
    /// sustained-4K-with-cooler thermal behavior is still unmeasured.
    @Published private(set) var isFourKEnabled = false

    /// Whether normal (non-calibration) recording captures at 30fps instead
    /// of the default 15fps -- added by request 2026-08-20 specifically to
    /// improve yolo26x reference-pass quality (more temporal samples to
    /// match on-device detections against), NOT to change what the
    /// on-device pipeline itself sees. Safe to do independently:
    /// `InferenceEngine.process`'s existing `isBusy` guard already drops
    /// any frame that arrives while the previous one is still being
    /// processed (see `restrictFrameRate`'s own doc comment, which already
    /// noted this for the 30fps-default/15fps-restricted case) -- so
    /// doubling how fast frames ARRIVE doesn't change how fast inference
    /// actually RUNS, it just means more of the arriving frames get
    /// dropped by that same guard. Recording itself (`appendRecordingFrame`
    /// in `captureOutput`) is unconditional per frame regardless, so it's
    /// the one thing that actually captures at the higher rate. Real cost
    /// is elsewhere: video encoding does genuinely more work at 30fps
    /// (separate from inference), and the file is roughly 2x the size for
    /// the same duration -- same "off by default, confirm before
    /// enabling" treatment as `isFourKEnabled` for that reason, and NOT
    /// persisted for the same "a deliberate choice for one recording
    /// shouldn't silently carry into the next" reasoning.
    @Published private(set) var isThirtyFpsEnabled = false

    /// Normalized [0,1] x-position of the yaw-calibration marker (the
    /// wiper-cowl marker, rigidly part of the car -- see
    /// `defaultYawMarkerNormalizedX`'s doc comment) -- read fresh EVERY
    /// session via `refineYawReference()` (ContentView.enterYawScreen,
    /// unconditional, no skip path), same "don't assume calibration"
    /// discipline as PitchSensor's
    /// referencePitchDegrees/referenceRollDegrees. Persisted only as a
    /// reasonable starting value before that session's own fresh detection
    /// completes, not trusted to stay correct indefinitely on its own.
    @Published private(set) var yawReferenceNormalizedX: CGFloat = CameraManager.defaultYawMarkerNormalizedX

    /// Vertical companion to yawReferenceNormalizedX -- only needed once
    /// the reference switched from the yellow stick (a vertical line,
    /// x-only) to the wiper-cowl marker (a small 2D blob, needs both axes
    /// to place the confirmation crosshair and re-detect it). Same
    /// verify-every-session/persist-as-starting-value discipline as X.
    @Published private(set) var yawReferenceNormalizedY: CGFloat = CameraManager.defaultWiperMarkerNormalizedY

    /// Whether yawReferenceNormalizedX has actually been confirmed THIS
    /// session (auto-detect succeeded, or the user manually dragged/
    /// confirmed the crosshair) -- as opposed to still holding whatever
    /// value was last persisted from a PRIOR session. CONFIRMED 2026-08-18:
    /// auto-detect failing (see detectYawMarkerNormalizedX's threshold
    /// history) left the level screen silently reusing a stale persisted
    /// value with no visible distinction from a freshly-verified one --
    /// the mount-yaw check was technically correct about that stale
    /// number, but looked wrong next to the user's own visual read of the
    /// real stick, since nothing on screen said the number hadn't actually
    /// been checked against reality yet this session. Deliberately NOT
    /// persisted -- resets false every launch, same "every session"
    /// discipline as the fine yaw calibration itself (see
    /// ContentView.beginYawCalibrationFlow's doc comment).
    @Published private(set) var isYawVerifiedThisSession = false

    /// Where a *reasonably mounted* phone's camera should roughly point --
    /// the yaw band's fixed center (see ContentView's YawBand), and the
    /// starting value for yawReferenceNormalizedX before any real
    /// per-session detection has run.
    ///
    /// RETIRED 2026-08-19: this used to reference a loose yellow stick
    /// clipped to the dash, re-measured several times as the mount was
    /// adjusted (see git history for that full lineage -- 2026-08-11
    /// laser dot, 2026-08-15/16 walkaround/drive re-measures, 2026-08-19
    /// post-laser-readjustment manual value). Retired by request: Rick
    /// flagged the stick as not mounted firmly enough to trust as a yaw
    /// reference (see `detectWiperMarkerNormalized`'s doc comment for the
    /// replacement's own rationale) -- a loose object can shift
    /// independently of the mount, defeating the entire point of using it
    /// to measure the mount's own drift. Now points at the wiper-cowl
    /// marker instead, which is rigidly part of the car and can't do that.
    /// See `defaultWiperMarkerNormalizedY` for the y-axis measurement this
    /// pairs with.
    ///
    /// RE-MEASURED 2026-08-19 (later same day): the value above was from
    /// `nearfocus-20260819-132838.mov`, a sweep-test capture that (per the
    /// same day's later mount work) no longer matched the mount's actual
    /// position -- confirmed once the focus-race bug (see
    /// `lockKnownGoodFarFocus`) was fixed and a genuinely clean,
    /// correctly-focused live capture (`wipercal-20260819-183622.mov`)
    /// became available to check against: the real marker sat at pixel
    /// (1647.7, 785.0) of 1920x1080, not the old (1710.4, 835.3). Found via
    /// connected-component analysis offline (a naive whole-ROI centroid was
    /// getting pulled toward a second, larger dark blob -- the wiper cap's
    /// shadowed edge, immediately left of the marker -- see
    /// `detectWiperMarkerNormalized`'s doc comment for how the ROI was
    /// retightened to exclude it).
    ///
    /// RE-MEASURED AGAIN 2026-08-20: two fresh, correctly near-focused
    /// sessions (`wipercal-20260820-104855.mov`, `wipercal-20260820-110040.mov`
    /// -- confirmed genuinely near-focused this time, lensPosition=0.353
    /// landed in 0.2-0.26s per the log, not the focus-race-contaminated
    /// captures earlier fixes were working around) both auto-detected the
    /// marker meaningfully lower (higher normY) than the 2026-08-19 value
    /// above -- 0.7351 and 0.7478 vs the stored 0.7269 -- consistent with
    /// real session-to-session mount drift (this clamp mount isn't
    /// perfectly repeatable), not detector noise in one direction only.
    /// Updated to the FIRST session's value: auto-detect was accepted with
    /// no manual drag correction that time (see the log: "MATCHED" ->
    /// "MATCHED confirmed" with no intervening drag), the closest thing to
    /// a confirmed-accurate reading available, vs. the second session
    /// where the user visibly dragged to correct it. X moved only
    /// slightly and non-monotonically between the two sessions (0.8593,
    /// 0.7269 sample straddles it) -- read as ordinary detector noise, not
    /// drift, so updated to match this same session's own X reading rather
    /// than left inconsistent with the new Y.
    static let defaultYawMarkerNormalizedX: CGFloat = 0.8455

    /// Y-axis companion to `defaultYawMarkerNormalizedX` above -- see that
    /// constant's doc comment for the 2026-08-20 re-measurement.
    static let defaultWiperMarkerNormalizedY: CGFloat = 0.7351

    /// Current system thermal state. Published so the HUD can warn when the device
    /// is under thermal pressure — the ML/capture workload can degrade 10-15x under
    /// sustained heat with no other visible symptom, so this is the only in-the-moment
    /// signal that it's happening. Always logged (`DebugFileLogger`) regardless of
    /// HUD visibility, to confirm after a long drive whether the current capture
    /// settings (1080p/15fps) actually stayed sustainable.
    @Published private(set) var thermalState: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState

    /// Empirical "% of full speed" derived from actual per-frame inference latency
    /// (see `recordInferenceLatency`), not from `thermalState` directly — Apple
    /// doesn't publish a numeric throttle factor for `.fair`/`.serious`/`.critical`,
    /// so this measures the real thing instead: 100% at the baseline latency seen
    /// while `.nominal`, dropping as frames actually start taking longer.
    @Published private(set) var thermalSpeedPercent: Int = 100

    // Baseline inference latency (ms) measured while `.nominal`, and the model/
    // two-pass config it was measured under — see `recordInferenceLatency`. Only
    // ever touched from the main actor (that method is only called from
    // `ContentView`'s `onChange(of: inferenceEngine.lastFrameElapsedMs)`).
    private var thermalBaselineElapsedMs: Double?
    private var thermalBaselineConfigKey: String?
    // Smoothed "current" latency (unlike the baseline, updated every frame regardless
    // of thermal state) so the published percent reflects a stable trend rather than
    // one noisy frame's timing.
    private var thermalCurrentElapsedMs: Double?
    // Gates how often `thermalSpeedPercent` actually changes — every frame was too
    // jumpy to read at a glance, so it now only updates on a real thermal-state
    // transition or every 5s otherwise.
    private var lastPublishedThermalState: ProcessInfo.ThermalState?
    private var lastThermalPercentPublishTime: CFAbsoluteTime = 0
    private static let thermalPercentPublishInterval: CFAbsoluteTime = 5

    /// Nonisolated mirror of `isLowLightBoostEnabled`, readable from `sessionQueue`
    /// while sampling ambient luminance for auto low-light.
    private nonisolated var currentLowLightEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentLowLightEnabled }
        set { overlayLock.lock(); _currentLowLightEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isAutoLowLightEnabled`, readable from `sessionQueue`
    /// while sampling ambient luminance for auto low-light.
    private nonisolated var currentAutoLowLightEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentAutoLowLightEnabled }
        set { overlayLock.lock(); _currentAutoLowLightEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isStabilizationEnabled`, readable from `sessionQueue`
    /// while applying the mode to the capture connection in
    /// `configure()`/`applyStabilizationMode`.
    nonisolated var currentStabilizationEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentStabilizationEnabled }
        set { overlayLock.lock(); _currentStabilizationEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isFourKEnabled`, readable from `sessionQueue`
    /// when choosing the preset in `configure()`/`start(recording:)`.
    private nonisolated var currentFourKEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentFourKEnabled }
        set { overlayLock.lock(); _currentFourKEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `manualNearFocusLensPosition`, readable from
    /// `sessionQueue` in `settleAndLockNearFieldFocus`.
    private nonisolated var currentNearFocusLensPosition: Float {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentNearFocusLensPosition }
        set { overlayLock.lock(); _currentNearFocusLensPosition = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isThirtyFpsEnabled`, readable from
    /// `sessionQueue` in `configure()`/`restrictFrameRate`.
    private nonisolated var currentThirtyFpsEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentThirtyFpsEnabled }
        set { overlayLock.lock(); _currentThirtyFpsEnabled = newValue; overlayLock.unlock() }
    }

    private let overlayLock = NSLock()
    private nonisolated(unsafe) var _currentLowLightEnabled: Bool = false
    private nonisolated(unsafe) var _currentAutoLowLightEnabled: Bool = true
    private nonisolated(unsafe) var _currentStabilizationEnabled: Bool = false
    private nonisolated(unsafe) var _currentFourKEnabled: Bool = false
    private nonisolated(unsafe) var _currentNearFocusLensPosition: Float = CameraManager.defaultNearFocusLensPosition
    private nonisolated(unsafe) var _currentThirtyFpsEnabled: Bool = false

    // nonisolated(unsafe): AVCaptureSession and outputs are internally thread-safe
    // and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()
    private nonisolated(unsafe) var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private nonisolated(unsafe) var rotationObservation: NSKeyValueObservation?
    private nonisolated(unsafe) var captureDevice: AVCaptureDevice?

    // Periodic focus recalibration (see `recalibrateFocusIfDue`): a one-shot
    // lock can drift wrong over a long drive with no way to recover
    // otherwise. Bounding how long a bad lock can last by periodically
    // re-asserting the known-good value, the same principle as the
    // low-light auto-detector re-sampling instead of deciding once and
    // never revisiting it.
    private nonisolated(unsafe) var lastFocusLockTime: CFAbsoluteTime = 0
    private nonisolated static let focusRecalibrationInterval: CFAbsoluteTime = 60

    /// Set for the duration of a `startWalkaroundRecording` session (and
    /// the whole yaw calibration flow -- see `beginYawCalibrationSession`)
    /// -- suppresses `recalibrateFocusIfDue`'s periodic re-lock entirely,
    /// since either session deliberately keeps the lens at a DIFFERENT
    /// fixed position (near-field) than normal driving's far lock, and a
    /// periodic re-assertion of the far value mid-session would fight that.
    private nonisolated(unsafe) var suppressPeriodicFocusRecalibration = false

    /// Empirically confirmed via real far-focus locks across many sessions
    /// -- see `lockKnownGoodFarFocus`, the only place this is used (direct
    /// lock, no AF search -- see that function's doc comment for why
    /// AF-based settling was retired entirely 2026-08-20 after it was
    /// confirmed to poison focus for two full drives).
    private nonisolated static let knownGoodFarLensPosition: Float = 0.76

    /// Filename prefix `startNewRecording` uses -- "recording" for a normal
    /// drive, "calibration" while `startCalibrationRecording` is active, so
    /// a tape-mark reference clip can never be mistaken for real drive
    /// footage.
    private nonisolated(unsafe) var activeRecordingFilenamePrefix = "recording"
    /// Most recent frame captured while `activeRecordingFilenamePrefix ==
    /// "calibration"` -- what `stopCalibrationRecording` runs tape-mark
    /// detection against. Only tracked during calibration recording (see
    /// `captureOutput`), nil otherwise -- no reason to hold a frame
    /// reference during normal driving.
    private nonisolated(unsafe) var latestCalibrationPixelBuffer: CVPixelBuffer?

    /// Unlike latestCalibrationPixelBuffer, updated on EVERY frame regardless
    /// of recording state -- checkYawNudgeNeeded()/refineYawReference()
    /// need a frame to scan on the level/yaw screens, before any recording
    /// (calibration or otherwise) has started.
    private nonisolated(unsafe) var latestPreviewPixelBuffer: CVPixelBuffer?

    /// The exposure bias (in EV) the boost is currently applying, so it can be
    /// restored after a probe. Written and read only on `sessionQueue` — no lock
    /// needed.
    private nonisolated(unsafe) var appliedBiasEV: Float = 0
    private nonisolated(unsafe) var lastLuminanceSampleTime: CFAbsoluteTime = 0

    // Periodic-probe state (see `sampleAutoLowLightIfDue`): while the boost is on,
    // the bias is briefly released to get one uncontaminated brightness reading.
    private nonisolated(unsafe) var isProbingLowLight = false
    private nonisolated(unsafe) var probeStartedAt: CFAbsoluteTime = 0
    private nonisolated(unsafe) var lastProbeTime: CFAbsoluteTime = 0
    // Was 2s -- confirmed (2026-08-05 Obsidian review) this produces a visible ~0.3s
    // brightness dip baked directly into the recording every interval while boost is
    // active (no separate compositing step -- recording reads the same pixel buffer
    // this probe biases). 5s cuts that by more than half with no new calibration risk.
    // The real fix -- estimating scene brightness from device.iso/exposureDuration
    // instead of ever touching bias -- needs its own validated threshold system in
    // different units, which isn't safe to guess without real device data; this is
    // the safe mitigation until that's built and tuned against a real night drive.
    private nonisolated static let probeInterval: CFAbsoluteTime = 5
    private nonisolated static let probeSettleDuration: CFAbsoluteTime = 0.3

    /// 0-255 brightness thresholds (see `sampleAutoLowLightIfDue`). The gap between
    /// them is hysteresis to avoid flicker at the boundary. Starting points — may
    /// need real-world tuning.
    // nonisolated: plain immutable constants, safe to read from the
    // background sessionQueue closures that check luminance thresholds
    // (e.g. beginYawCalibrationSession's torch-on check) without hopping
    // to the main actor first.
    private nonisolated static let autoLowLightOnLuminance: Double = 50
    private nonisolated static let autoLowLightOffLuminance: Double = 90

    /// Consecutive at/below-threshold readings required before the boost actually
    /// turns on -- added after a real false-positive on session
    /// 26_07_30_Day_Hosp_nano_off: a single momentary dip while driving under dense
    /// tree canopy (whole-frame average briefly reads dark even in full daylight,
    /// since the frame is mostly heavy shade) tripped the boost on one sample, and
    /// by the next probe ~2s later the car had moved into a sunlit gap -- boosted
    /// exposure hitting real daylight, causing a severe flare/blowout in the
    /// recording (confirmed by pulling the actual frame). Requiring the low reading
    /// to persist across samples (~1s apart) distinguishes real darkness (tunnels,
    /// dusk) from a momentary shaded patch, which clears before a second sample.
    private static let consecutiveLowSamplesRequired = 2
    private nonisolated(unsafe) var consecutiveLowLuminanceSamples = 0

    // Recording state. Only ever touched on sessionQueue (configure/startNewRecording
    // run there, and the video data output delegate is dispatched to sessionQueue too).
    private nonisolated(unsafe) var assetWriter: AVAssetWriter?
    private nonisolated(unsafe) var assetWriterInput: AVAssetWriterInput?
    private nonisolated(unsafe) var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private nonisolated(unsafe) var currentSegmentURL: URL?
    private nonisolated(unsafe) var observersRegistered = false
    private let backgroundTaskState = OSAllocatedUnfairLock<UIBackgroundTaskIdentifier>(initialState: .invalid)

    /// Below this, the HUD warns so the user can free up space before recording
    /// actually starts failing, instead of only finding out after the fact.
    private let lowStorageThresholdBytes: Int64 = 1_000_000_000

    override init() {
        super.init()
        // Neither isLowLightBoostEnabled/isAutoLowLightEnabled nor
        // isStabilizationEnabled are persisted — every launch starts back at
        // their compiled-in defaults (auto, on) regardless of whatever
        // manual override was last in effect, so a prior drive/test's choice
        // doesn't silently carry over into an unrelated later one.
        // `exitSession` always terminates the app outright (see its own
        // comment) rather than looping back to the level screen, so "every
        // launch" and "every session" are the same thing here -- no separate
        // per-session reset path is needed beyond just not persisting.
        //
        // Keep the nonisolated mirrors in sync with these defaults —
        // `configure()` applies the boost/stabilization to the actual
        // capture device from these once it's available.
        currentAutoLowLightEnabled = isAutoLowLightEnabled
        currentLowLightEnabled = isLowLightBoostEnabled
        currentStabilizationEnabled = isStabilizationEnabled
        currentFourKEnabled = isFourKEnabled
        currentThirtyFpsEnabled = isThirtyFpsEnabled

        // Unlike the settings above, the yaw reference genuinely should
        // persist across launches -- it's a measurement of the physical
        // mount, not a mode that should reset defensively. A session that
        // skips fresh calibration reuses this rather than falling back to
        // defaultYawMarkerNormalizedX.
        if UserDefaults.standard.object(forKey: Self.yawReferenceDefaultsKey) != nil {
            yawReferenceNormalizedX = CGFloat(UserDefaults.standard.double(forKey: Self.yawReferenceDefaultsKey))
        }
        if UserDefaults.standard.object(forKey: Self.yawReferenceYDefaultsKey) != nil {
            yawReferenceNormalizedY = CGFloat(UserDefaults.standard.double(forKey: Self.yawReferenceYDefaultsKey))
        }

        // Same "measures the physical setup, not a mode" persistence
        // reasoning as the yaw reference above.
        if UserDefaults.standard.object(forKey: Self.manualNearFocusLensPositionDefaultsKey) != nil {
            manualNearFocusLensPosition = UserDefaults.standard.float(forKey: Self.manualNearFocusLensPositionDefaultsKey)
        }
        currentNearFocusLensPosition = manualNearFocusLensPosition
    }

    private static let yawReferenceDefaultsKey = "settings.yawReferenceNormalizedX"
    private static let yawReferenceYDefaultsKey = "settings.yawReferenceNormalizedY"
    private static let manualNearFocusLensPositionDefaultsKey = "settings.manualNearFocusLensPosition"

    /// `recording: false` starts the session (live preview + inference feed)
    /// without writing a video file -- lets the configuring screen show a
    /// real camera preview while the user is still picking settings, without
    /// that screen ending up in the recording. Call `beginRecording()` later
    /// to start actually writing, without restarting the session.
    func start(recording: Bool = true) {
        // Requested up front (rather than lazily on the first segment save) so the
        // permission dialog doesn't interrupt an in-progress drive.
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            print("[CameraManager] Photos add-only authorization: \(status.rawValue)")
        }

        Task {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                sessionQueue.async { [weak self] in
                    guard let self else { return }
                    self.configure(startRecording: recording, preset: self.currentFourKEnabled ? .hd4K3840x2160 : .hd1920x1080)
                }
            case .notDetermined:
                let granted = await AVCaptureDevice.requestAccess(for: .video)
                if granted {
                    self.sessionQueue.async { [weak self] in
                        guard let self else { return }
                        self.configure(startRecording: recording, preset: self.currentFourKEnabled ? .hd4K3840x2160 : .hd1920x1080)
                    }
                }
            default:
                break
            }
        }
    }

    /// Starts writing to disk on an already-running preview-only session
    /// (see `start(recording:)`) -- called once the user commits to driving
    /// (Lock Settings/Unlocked), without tearing down and reconfiguring the
    /// session that's already showing the live preview. No-op if a
    /// recording is already in progress, or the session isn't running yet.
    func beginRecording() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.startNewRecording()
        }
    }

    /// `completion` fires on the main queue only after `finishRecording()`
    /// (file finalize + Photos save, both blocking) has actually completed
    /// -- lets a caller that's about to do something irreversible right
    /// after (e.g. terminating the app on exitSession) wait for a clean
    /// finish rather than relying on the mid-write-kill recovery path
    /// (`recoverOrphanedRecordings`) on every normal exit.
    func stop(completion: (@Sendable () -> Void)? = nil) {
        let sessionBox = UncheckedSendableBox(value: session)
        sessionQueue.async { [weak self] in
            self?.finishRecording()
            sessionBox.value.stopRunning()
            if let completion {
                DispatchQueue.main.async { completion() }
            }
        }
    }

    // MARK: - Calibration recording (tape-mark distance calibration)
    //
    // A short, one-off, high-resolution reference clip for the tape-mark
    // distance calibration (DistanceEstimator.fit() / the following-distance
    // measurement plan) -- NOT part of the normal session lifecycle, and
    // deliberately not exposed anywhere in the main UI (see ContentView's
    // hidden long-press trigger on the level screen). Two ways this
    // differs from a normal drive recording:
    // - 4K instead of 1080p: thermal throttling (the reason 1080p was
    //   chosen) isn't a concern for a clip this short, and more rows
    //   between camera and horizon means less rounding error reading off
    //   which row a tape mark lands on.
    // - Locked, settled far-field focus instead of continuous autofocus:
    //   reintroduces (scoped to just this path) the settle-then-lock
    //   approach that was removed from the normal startup flow for causing
    //   a multi-second freeze there -- acceptable here since this is a
    //   rare, deliberate action the user is already waiting on, not
    //   something that silently blocks the normal driving flow. Confirmed
    //   real problem this solves: continuous AF locking onto the dash
    //   (near, high-contrast, filling the bottom of frame) instead of the
    //   distant tape marks.
    //
    // Saves to calibration-<timestamp>.mov (not recording-<timestamp>.mov)
    // so it can never be confused with real drive footage, and is
    // deliberately NOT picked up by `recoverOrphanedRecordings`'s
    // "recording-" prefix filter. Fully tears the session down when
    // stopped (see `teardownSession`) so the next normal `start()` re-runs
    // the full setup path (1080p, continuous AF) instead of inheriting
    // this mode's settings.

    /// Reachable from the level screen (before ever driving anywhere) OR
    /// the configuring screen (its session is always already running in
    /// preview-only mode by the time you get there, since the roll-
    /// calibration flow passes through it first -- e.g. calibrate roll in
    /// a known-level parking spot, drive to a street with room for tape
    /// marks, then start this once there, all in one continuous session
    /// without ever pressing Lock Settings/Unlocked). Either way this stops
    /// and fully tears down whatever the session currently is (stopped, or
    /// running preview-only) so the fresh 4K configure below always takes
    /// the full setup path, not `configure`'s quick-restart shortcut.
    ///
    /// `completion` reports whether the recording actually started (false
    /// if a normal drive recording was already in progress -- can't switch
    /// mid-recording -- or the device/session setup failed) -- fires on the
    /// main queue.
    func startCalibrationRecording(completion: @escaping @Sendable (Bool) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self, self.assetWriter == nil else {
                // CONFIRMED 2026-08-11: this guard and the captureDevice one
                // below used to fail silently (no log at all before
                // completion(false)) -- exactly the two paths that could
                // explain "distance-cal: round=X FAILED to start" with zero
                // other diagnostic in the pulled debug log. Logging which
                // one actually fired, if this recurs.
                DebugFileLogger.log("calibration-recording: FAILED to start, assetWriter still non-nil (previous recording not fully torn down)")
                DispatchQueue.main.async { completion(false) }
                return
            }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.teardownSession()
            self.activeRecordingFilenamePrefix = "calibration"
            self.configure(startRecording: false, preset: .hd4K3840x2160, settleFocus: false)
            guard let device = self.captureDevice else {
                self.activeRecordingFilenamePrefix = "recording"
                DebugFileLogger.log("calibration-recording: FAILED to start, captureDevice nil after configure()")
                DispatchQueue.main.async { completion(false) }
                return
            }
            // CONFIRMED 2026-08-20: settleAndLockFarFieldFocus's AF-based
            // approach can be fooled into a false-instant "settled" report
            // (isAdjustingFocus flips false almost immediately, ~0.19s, not
            // a genuine search) landing on whatever value the lens already
            // happened to be at -- if that value clears minPlausibleFarLensPosition
            // (0.5) by even a little, it's wrongly accepted and cached as
            // lastGoodLensPosition, then every later timed-out recalibration
            // falls back to that SAME wrong value for the rest of the
            // session. Real drive evidence: two full sessions where every
            // single periodic recalibration attempt (every 60s, for 70+
            // minutes) timed out and reused a lensPosition=0.588 poisoned by
            // exactly this failure mode at calibration end -- the AF search
            // never once actually completed, so it was providing zero real
            // value while carrying this risk. `lockKnownGoodFarFocus`
            // (already used successfully at launch) sidesteps the whole
            // failure class by never running an AF search at all.
            self.lockKnownGoodFarFocus(device: device)
            self.startNewRecording()
            DispatchQueue.main.async { completion(true) }
        }
    }

    /// Finishes and saves the calibration clip (to Photos, same as a normal
    /// recording), runs tape-mark detection against the last captured
    /// frame, then fully tears the session down -- see this section's
    /// file-level comment for why the teardown matters. `completion` fires
    /// on the main queue with the detected tape-mark count (0 if no frame
    /// was captured at all) -- as of 2026-08-11 the caller (`ContentView
    /// .recordCalibrationRound`) no longer branches on this at all (tape
    /// marks are read off the recorded footage by hand afterward instead),
    /// so it's logged for reference only now. `completion` firing is still
    /// used as "teardown is complete, safe to restart the normal preview-
    /// only session" (needed on the configuring screen, which would
    /// otherwise be left with a dead camera preview).
    func stopCalibrationRecording(completion: @escaping @Sendable (Int) -> Void) {
        let sessionBox = UncheckedSendableBox(value: session)
        sessionQueue.async { [weak self] in
            guard let self else { return }
            let detectedCount = self.latestCalibrationPixelBuffer.map { self.countRedTapeMarks(in: $0) } ?? 0
            self.latestCalibrationPixelBuffer = nil
            self.finishRecording()
            sessionBox.value.stopRunning()
            self.teardownSession()
            self.activeRecordingFilenamePrefix = "recording"
            // Restarted here, synchronously, in the SAME sessionQueue block
            // -- not left for the caller to restart later with a separate
            // async call. CONFIRMED bug 2026-08-09: ContentView used to call
            // `cameraManager.start(recording: false)` afterward to restore
            // the configuring screen's preview; when the next calibration
            // round started quickly, that separate call raced with the next
            // `startCalibrationRecording`'s own teardown/reconfigure,
            // corrupting the session (a plain "recording-" prefixed file
            // got created mid-calibration, and the round got stuck with no
            // completion ever firing). Doing it here instead makes the
            // whole "finish this round, get back to a normal preview"
            // sequence one atomic chain -- nothing else can interleave.
            self.configure(startRecording: false, settleFocus: true)
            DispatchQueue.main.async { completion(detectedCount) }
        }
    }

    // MARK: - Near-focus test capture (wiper-cowl marker measurement)
    //
    // A DIAGNOSTIC tool, not a finished production calibration path -- added
    // 2026-08-19 to get one real near-focused sample of the wiper-cowl
    // circle (see the yaw-calibration design discussion: this rigid,
    // always-in-frame car feature is meant to replace the loose yellow
    // stick, which the user flagged as at risk of shifting independently
    // of the mount). Every prior color-threshold guess in this file (the
    // yaw marker's own detectYawMarkerNormalizedX, most recently) turned
    // out wrong until measured against real footage -- this exists so the
    // eventual near-focus detector gets built from real pixels too, not
    // another guess. Reuses settleAndLockFarFieldFocus's real settle-
    // detection mechanism (see settleAndLockNearFieldFocus) rather than a
    // blind fixed delay -- CONFIRMED 2026-08-19 that a blind delay doesn't
    // work: the first attempt never actually racked focus near at all.
    // Also turns the torch on for the capture (added same day, for night
    // visibility of the marker) and off again when stopped.
    //
    // Saves to nearfocus-<timestamp>.mov (not recording-/calibration-) so
    // it's unambiguous in Photos and never picked up by
    // recoverOrphanedRecordings's "recording-" filter. Same full
    // teardown-then-fresh-configure / full-teardown-afterward pattern as
    // startCalibrationRecording for the same reason: guarantees this
    // mode's focus override can never leak into a later normal drive.

    /// `completion` reports whether the capture actually started -- same
    /// failure semantics as `startCalibrationRecording`.
    func startNearFocusTestCapture(completion: @escaping @Sendable (Bool) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async { completion(false) }
                return
            }
            // CONFIRMED 2026-08-19: a real drive recording already in
            // progress (Lock Settings/Unlocked tapped) used to make this
            // silently fail -- "can't switch mid-recording" is the right
            // call for the tape-marks/calibration flows (an intentional,
            // rare action worth protecting from an accidental interrupt),
            // but this is a quick diagnostic tool reachable from the same
            // menu during a normal drive -- by request, finish and save
            // whatever's currently recording instead of refusing.
            if self.assetWriter != nil {
                DebugFileLogger.log("nearfocus-capture: a recording was already in progress -- finishing it first")
                self.finishRecording()
            }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.teardownSession()
            self.activeRecordingFilenamePrefix = "nearfocus"
            self.configure(startRecording: false, preset: .hd1920x1080, settleFocus: false)
            guard let device = self.captureDevice else {
                self.activeRecordingFilenamePrefix = "recording"
                DebugFileLogger.log("nearfocus-capture: FAILED to start, captureDevice nil after configure()")
                DispatchQueue.main.async { completion(false) }
                return
            }
            guard device.isFocusModeSupported(.locked) else {
                DebugFileLogger.log("nearfocus-capture: locked focus unsupported on this device, recording at whatever focus is current")
                self.startNewRecording()
                DispatchQueue.main.async { completion(true) }
                return
            }
            // CONFIRMED bug 2026-08-19: an early attempt here just set
            // .near + .continuousAutoFocus and waited a blind fixed delay,
            // which never actually racked focus near at all -- a real
            // captured clip stayed far-focused start to finish. Now uses
            // `settleAndLockNearFieldFocus`'s direct lock (no AF at all,
            // see that function's doc comment) instead.
            self.suppressPeriodicFocusRecalibration = true
            if device.hasTorch, device.isTorchModeSupported(.on) {
                do {
                    try device.lockForConfiguration()
                    device.torchMode = .on
                    device.unlockForConfiguration()
                } catch {
                    DebugFileLogger.log("nearfocus-capture: torch lockForConfiguration failed (\(error))")
                }
            }
            self.settleAndLockNearFieldFocus(device: device) { [weak self] in
                guard let self else { return }
                self.startNewRecording()
                DispatchQueue.main.async { completion(true) }
            }
        }
    }

    /// Empirically measured 2026-08-19: a 6-step sweep (0.05 through 0.55)
    /// captured in one recording, reviewed frame-by-frame against the real
    /// wiper-cowl marker at its actual ~15in distance from the camera.
    /// 0.05/0.15 were still visibly soft (too close), 0.55 had drifted back
    /// toward far-blurry, and 0.35 was unambiguously the sharpest --
    /// individual hex-mesh grille holes and the marker's circular edge both
    /// crisply resolved, with embossed text on the wiper blade itself also
    /// legible in that same frame. Same empirical-measurement discipline as
    /// `knownGoodFarLensPosition` (0.76, confirmed via real far settles),
    /// just for the near end instead.
    ///
    /// RENAMED from `knownGoodNearLensPosition` 2026-08-20: this value alone
    /// started producing visibly soft frames on a real device (confirmed via
    /// direct comparison against the original sharp reference frame -- not
    /// a focus-lock bug, the lens genuinely landed here, it's just no longer
    /// sharp) -- now only the STARTING value for `manualNearFocusLensPosition`
    /// (used the very first time, before any manual adjustment is ever
    /// persisted), not blindly trusted as permanently correct.
    private nonisolated static let defaultNearFocusLensPosition: Float = 0.35

    /// Directly drives the lens to the measured near-focus position, rather
    /// than asking any flavor of autofocus to find it. CONFIRMED
    /// 2026-08-19: TWO different autofocus approaches both failed here,
    /// identically -- `.continuousAutoFocus` + `.near` restriction, and
    /// single-shot `.autoFocus` + `.near` restriction, both reported
    /// settling in ~0.19s at essentially the SAME lensPosition the device
    /// was already far-locked at (real far settles in this same log take
    /// 2.3-2.4s, so 0.19s is not a genuine scan completing) -- range
    /// restriction only constrains where AF is ALLOWED to land if it
    /// decides to search, and on a static, unchanging scene neither AF mode
    /// judged a fresh search necessary, restriction or not. Bypassing AF
    /// entirely and driving the lens directly via `setFocusModeLocked`'s
    /// own completion handler (which only fires once the lens has
    /// genuinely arrived -- a real hardware-driven signal, not an AF
    /// heuristic) is what actually works, combined with the real measured
    /// `knownGoodNearLensPosition` above (0.0, the naive "nearest extreme"
    /// guess, itself overshot the real ~15in distance).
    private nonisolated func settleAndLockNearFieldFocus(
        device: AVCaptureDevice,
        completion: @escaping @Sendable () -> Void
    ) {
        guard device.isFocusModeSupported(.locked) else {
            completion()
            return
        }
        // A prior version of this function had to defensively guard against
        // racing an in-flight AF-based far settle here (settleAndLockFarFieldFocus,
        // removed 2026-08-20) -- moot now that far focus is also a direct,
        // synchronous lock with nothing async left to race.
        DebugFileLogger.log("nearfocus-capture: driving lens to measured near position")
        let settleStart = CFAbsoluteTimeGetCurrent()
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.setFocusModeLocked(lensPosition: self.currentNearFocusLensPosition) { [weak self] _ in
                DebugFileLogger.log(String(
                    format: "nearfocus-capture: lens arrived after %.2fs at lensPosition=%.3f",
                    CFAbsoluteTimeGetCurrent() - settleStart, device.lensPosition
                ))
                self?.sessionQueue.async { completion() }
            }
        } catch {
            DebugFileLogger.log("nearfocus-capture: lockForConfiguration failed (\(error))")
            completion()
        }
    }

    /// Ends the near-focus test capture, saves to Photos (same as a normal
    /// recording), and fully tears the session down so the next normal
    /// `start()` re-runs the full setup path (1080p, far/continuous AF)
    /// instead of inheriting the near-focus override. `completion` fires on
    /// the main queue once teardown is complete.
    func stopNearFocusTestCapture(completion: (@Sendable () -> Void)? = nil) {
        let sessionBox = UncheckedSendableBox(value: session)
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if let device = self.captureDevice, device.hasTorch, device.torchMode == .on {
                do {
                    try device.lockForConfiguration()
                    device.torchMode = .off
                    device.unlockForConfiguration()
                } catch {
                    // Torch will still turn off when the session/device is
                    // torn down just below regardless.
                }
            }
            self.finishRecording()
            sessionBox.value.stopRunning()
            self.teardownSession()
            self.activeRecordingFilenamePrefix = "recording"
            self.suppressPeriodicFocusRecalibration = false
            self.configure(startRecording: false, settleFocus: true)
            if let completion {
                DispatchQueue.main.async { completion() }
            }
        }
    }

    // MARK: - Walkaround recording (tethered-distance distance-cal test)
    //
    // A long-form, real drive-style recording (1080p, "recording-" prefixed,
    // recoverable like any other drive) for the walkaround distance-cal
    // test: the tester walks to several tethered distances in front of the
    // parked car while a tape-measure marker stays in near-field view the
    // whole time. Two ways this differs from a normal drive recording:
    // - Focus is explicitly settled and LOCKED far/infinity before the
    //   first frame is written, same mechanism `startCalibrationRecording`
    //   uses -- necessary because the near-field marker is a permanent
    //   fixture of this test (unlike the tape-marks flow, it can't just be
    //   framed out).
    // - `recalibrateFocusIfDue`'s periodic re-settle is suppressed for the
    //   whole recording (`suppressPeriodicFocusRecalibration`) -- CONFIRMED
    //   2026-08-15 via a real session (data/26_08_15_Walkaround) that even
    //   with the same settle-then-lock-far machinery normal driving uses,
    //   focus drifted inconsistently, most plausibly during one of the
    //   periodic re-settles' brief continuous-AF windows.
    //
    // Unlike the tape-marks clip, there's no fixed duration or auto-stop --
    // `stopWalkaroundRecording` ends it whenever the tester is done.

    /// `completion` reports whether the recording actually started -- same
    /// failure semantics as `startCalibrationRecording`.
    func startWalkaroundRecording(completion: @escaping @Sendable (Bool) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self, self.assetWriter == nil else {
                DebugFileLogger.log("walkaround-recording: FAILED to start, assetWriter still non-nil (previous recording not fully torn down)")
                DispatchQueue.main.async { completion(false) }
                return
            }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.teardownSession()
            self.suppressPeriodicFocusRecalibration = true
            self.configure(startRecording: false, preset: .hd1920x1080, settleFocus: false)
            guard let device = self.captureDevice else {
                self.suppressPeriodicFocusRecalibration = false
                DebugFileLogger.log("walkaround-recording: FAILED to start, captureDevice nil after configure()")
                DispatchQueue.main.async { completion(false) }
                return
            }
            // See startCalibrationRecording's matching comment -- same
            // AF-based false-instant-settle risk, same fix.
            self.lockKnownGoodFarFocus(device: device)
            self.startNewRecording()
            DispatchQueue.main.async { completion(true) }
        }
    }

    /// Ends a walkaround recording, saving to Photos same as a normal
    /// `stop()` -- also tears the session down (see this section's own
    /// comment on why `stopCalibrationRecording` does the same) and clears
    /// `suppressPeriodicFocusRecalibration`, so neither the locked-far focus
    /// nor the suppressed periodic recalibration leaks into whatever session
    /// starts next. `completion` fires on the main queue once finalize +
    /// teardown are done.
    func stopWalkaroundRecording(completion: (@Sendable () -> Void)? = nil) {
        let sessionBox = UncheckedSendableBox(value: session)
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.finishRecording()
            sessionBox.value.stopRunning()
            self.teardownSession()
            self.suppressPeriodicFocusRecalibration = false
            self.configure(startRecording: false, settleFocus: true)
            if let completion {
                DispatchQueue.main.async { completion() }
            }
        }
    }

    /// Counts distinct horizontal red bands crossing a central column
    /// range of the frame -- tape marks are laid perpendicular to the
    /// direction of travel, so each one should appear as its own roughly-
    /// horizontal band at a different row, distinguishable from road/
    /// asphalt by hue alone (red tape against gray/black road is a strong,
    /// simple color signal). Deliberately NOT a general object detector --
    /// scoped tightly to "distinct red band count" since that's exactly
    /// the check `recordCalibrationClip`'s caller needs.
    ///
    /// *** THRESHOLDS NOT YET VALIDATED against a real photo of the actual
    /// tape on the actual street -- reasoned from "red tape, ~4in wide,
    /// daylight" but not tuned against real footage. Expect to loosen/
    /// tighten `redPixel`'s brightness/dominance thresholds or
    /// `minRedRowFraction` after the first real attempt, the same way
    /// every other threshold in this project got corrected once real data
    /// existed to check it against.
    ///
    /// Samples on a coarse stride (not every pixel) for speed -- this runs
    /// once per calibration round on a still frame, not per-frame in a
    /// real-time loop, so a full-resolution scan isn't needed for accuracy,
    /// just enough samples per row to estimate red coverage reliably.
    private nonisolated func countRedTapeMarks(in pixelBuffer: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0 }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Tape marks span the lane, roughly centered in frame -- sampling
        // only the central half of columns avoids shoulder/curb clutter
        // and roughly halves the work.
        let colStart = width / 4
        let colEnd = width - width / 4
        let sampleCount = 200
        let colStride = max(1, (colEnd - colStart) / sampleCount)
        let rowStride = 2

        func redPixel(atByteOffset offset: Int) -> Bool {
            // kCVPixelFormatType_32BGRA byte order: B, G, R, A.
            let b = Double(bytes[offset])
            let g = Double(bytes[offset + 1])
            let r = Double(bytes[offset + 2])
            return r > 100 && r > g * 1.5 && r > b * 1.5
        }

        var redFractionByRow: [Int: Double] = [:]
        var row = 0
        while row < height {
            var redCount = 0
            var total = 0
            var col = colStart
            while col < colEnd {
                redPixel(atByteOffset: row * bytesPerRow + col * 4) ? (redCount += 1) : ()
                total += 1
                col += colStride
            }
            if total > 0 {
                redFractionByRow[row] = Double(redCount) / Double(total)
            }
            row += rowStride
        }

        let minRedRowFraction = 0.4
        let redRows = redFractionByRow.filter { $0.value >= minRedRowFraction }.keys.sorted()

        // A single tape mark spans several consecutive qualifying rows
        // (it has real width in the image, not just one row) -- merge rows
        // within this gap into the same band instead of over-counting one
        // mark as several.
        let mergeGapRows = 20
        var bandCount = 0
        var previousRow: Int?
        for r in redRows {
            if let previous = previousRow, r - previous <= mergeGapRows {
                // Still the same band.
            } else {
                bandCount += 1
            }
            previousRow = r
        }
        return bandCount
    }

    /// Detects the wiper-cowl marker -- a small, DARKER-than-its-
    /// surroundings circular feature on the wiper arm cover, rigidly part
    /// of the car -- and returns its centroid as normalized [0,1] (x,y),
    /// or nil if nothing confidently matched. Replaces the retired yellow-
    /// stick detector (see `defaultYawMarkerNormalizedX`'s doc comment for
    /// why): a loose clipped-on object can shift independently of the
    /// mount, defeating the whole point of using it to measure the
    /// mount's own drift; this can't.
    ///
    /// Only meaningful against a NEAR-focused frame -- see
    /// `settleAndLockNearFieldFocus`/`knownGoodNearLensPosition`, and
    /// `calibrateWiperMarker` (this function's only caller) for why the
    /// whole point of that near-focus detour exists. Under the normal
    /// far-focused driving view this region is soft.
    ///
    /// ROI and threshold RE-MEASURED 2026-08-19 (later same day) against
    /// `wipercal-20260819-183622.mov`, the first capture where the marker
    /// was both correctly near-focused AND from the mount's actual current
    /// position -- every earlier attempt had one or the other wrong (see
    /// `defaultYawMarkerNormalizedX`'s doc comment for the stale-frame
    /// history, and `lockKnownGoodFarFocus` for the focus-race bug). Offline
    /// connected-component analysis of that frame found the true marker as
    /// a compact ~21x24px blob centered at pixel (1647.7, 785.0) of
    /// 1920x1080 -- and, critically, a SECOND, larger dark blob immediately
    /// to its left (the wiper cap's own shadowed edge, bbox roughly
    /// 1555-1581 x 765-830, over 3x the marker's pixel count) that the old
    /// ROI (0.85-0.94 x, 0.68-0.80 y) was wide enough to catch, biasing a
    /// naive whole-ROI centroid rightward toward it -- which is exactly the
    /// rightward bias real on-device sessions kept showing, and why a prior
    /// same-day tightening pass (reasoned from log evidence rather than a
    /// fresh pixel measurement) didn't fix it: it narrowed the ROI but
    /// didn't move it, so the shadow blob was still inside. This pass
    /// re-centers the ROI tightly enough around the real marker to exclude
    /// that shadow blob by x-range alone, and re-derives the threshold
    /// multiplier (0.6 -> 0.75) against the new, correctly-centered ROI's
    /// own mean brightness -- 0.6 was tuned for the old, wider ROI and
    /// produces zero matches in this tighter one. Threshold stays relative
    /// to the ROI's own mean, not a fixed absolute value -- same "ratio
    /// holds across lighting, absolute doesn't" lesson the retired yellow-
    /// stick detector's color thresholds already had to learn twice.
    private nonisolated func detectWiperMarkerNormalized(in pixelBuffer: CVPixelBuffer) -> (x: CGFloat, y: CGFloat)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Tight around the re-measured (0.8582, 0.7269) center -- see this
        // function's doc comment for why the previous, wider ROI reached
        // far enough left to catch the wiper cap's own shadow blob. Still
        // has margin for real session-to-session mount drift (~15-30px
        // each side), just not enough to reach that shadow region again.
        let roiXRange = Int(0.83 * Double(width))..<Int(0.89 * Double(width))
        let roiYRange = Int(0.70 * Double(height))..<Int(0.76 * Double(height))
        let stride = 2

        func brightness(atByteOffset offset: Int) -> Double {
            // kCVPixelFormatType_32BGRA byte order: B, G, R, A.
            let b = Double(bytes[offset])
            let g = Double(bytes[offset + 1])
            let r = Double(bytes[offset + 2])
            return (r + g + b) / 3
        }

        var total = 0.0
        var sampleCount = 0
        var row = roiYRange.lowerBound
        while row < roiYRange.upperBound {
            var col = roiXRange.lowerBound
            while col < roiXRange.upperBound {
                total += brightness(atByteOffset: row * bytesPerRow + col * 4)
                sampleCount += 1
                col += stride
            }
            row += stride
        }
        guard sampleCount > 0 else { return nil }
        let threshold = (total / Double(sampleCount)) * 0.75

        var sumX = 0.0
        var sumY = 0.0
        var darkCount = 0
        row = roiYRange.lowerBound
        while row < roiYRange.upperBound {
            var col = roiXRange.lowerBound
            while col < roiXRange.upperBound {
                if brightness(atByteOffset: row * bytesPerRow + col * 4) < threshold {
                    sumX += Double(col)
                    sumY += Double(row)
                    darkCount += 1
                }
                col += stride
            }
            row += stride
        }

        // Require enough matching samples to trust this as the real
        // marker, not scattered dark noise (shadow, dirt, a gap in the
        // grille texture).
        let minSamples = 30
        guard darkCount >= minSamples else { return nil }
        return (
            CGFloat(sumX / Double(darkCount)) / CGFloat(width),
            CGFloat(sumY / Double(darkCount)) / CGFloat(height)
        )
    }

    /// Remembers what to restore once `endYawCalibrationSession` runs --
    /// only meaningful between a `beginYawCalibrationSession` call and its
    /// matching `end`. Spans BOTH the main (pitch/roll) calibration screen
    /// and the zoomed yaw screen -- see `beginYawCalibrationSession`'s doc
    /// comment for why this moved up from just the zoomed screen.
    private nonisolated(unsafe) var wiperMarkerCalibrationRestoreStabilization = false
    private nonisolated(unsafe) var wiperMarkerCalibrationTorchEngaged = false

    /// Begins a yaw calibration session -- called once, when the MAIN
    /// (pitch/roll) calibration screen first appears, not just when the
    /// user reaches the zoomed yaw screen. Moved up from a prior
    /// yaw-screen-only `beginWiperMarkerCalibration` by request 2026-08-19:
    /// the main screen now shows a live rectangle previewing the yaw
    /// screen's crop area and warns if the marker isn't in it (see
    /// `checkYawNudgeNeeded`), which needs a genuinely near-focused,
    /// correctly-exposed view to check against reliably -- the same
    /// detour the yaw screen itself needs, just started earlier so it's
    /// already settled by the time either screen needs to read a frame
    /// off it. Both screens together are one continuous near-focus
    /// session now:
    ///  1. Stabilization off (a stabilization crop shifts the whole frame,
    ///     which already once invalidated a real measurement for the old
    ///     yellow-stick system -- see git history, "a stabilization-crop
    ///     mismatch, not a real mount change" -- avoided here by never
    ///     letting it happen instead of re-diagnosing it later).
    ///  2. Torch on, but only if the scene actually reads dark -- reuses
    ///     `averageLuminance`/`autoLowLightOnLuminance`, the SAME signal
    ///     and threshold the existing auto-low-light system already
    ///     trusts, rather than a second guessed threshold.
    ///  3. Lock focus near (`settleAndLockNearFieldFocus`).
    ///
    /// Deliberately does NOT restore stabilization/torch/focus here --
    /// that's `endYawCalibrationSession`'s job, called once the user
    /// confirms the zoomed yaw screen. Also does NOT run detection or
    /// suppress the periodic far-focus recalibration by itself -- callers
    /// that need either should also see `checkYawNudgeNeeded`/
    /// `refineYawReference` and the `suppressPeriodicFocusRecalibration`
    /// toggle below, since this session can now legitimately sit open for
    /// a while (the user reading pitch/roll, maybe physically nudging the
    /// mount) rather than the few seconds the old yaw-screen-only version
    /// typically lasted -- CONFIRMED this matters:
    /// `recalibrateFocusIfDue` fires every 60s and would otherwise yank
    /// focus back to far mid-session.
    func beginYawCalibrationSession(completion: @escaping @Sendable () -> Void) {
        beginYawCalibrationSession(retriesRemaining: 25, completion: completion)
    }

    /// CONFIRMED 2026-08-20 on-device: the level screen's `.onAppear` (see
    /// ContentView.beginYawCalibrationFlow) can fire before `configure()`
    /// has actually set `captureDevice` -- `configure()`'s own device setup
    /// runs on `sessionQueue` and isn't guaranteed to have completed by the
    /// time SwiftUI's first layout pass fires `.onAppear`. The OLD version
    /// of this function silently gave up in that case (guard-else straight
    /// to `completion()`, no retry) -- a real session showed ZERO
    /// "nearfocus-capture" log lines despite going through the wiper-marker
    /// flow multiple times, meaning near-focus/torch/stabilization-off/the
    /// "wipercal-" recording never engaged AT ALL that session (confirmed:
    /// no wipercal-*.mov file exists from it either) -- `refineYawReference`
    /// still ran later and still found "a" match, just against a blurry,
    /// still-far-focused frame, which is what actually caused the reported
    /// "near-field focus seemed blurry" symptom. Retrying (rather than
    /// failing open) closes that gap -- 25 attempts * 0.2s = 5s ceiling,
    /// generous relative to how fast `configure()` normally finishes, with
    /// a clear log line if it's ever genuinely exhausted rather than an
    /// indistinguishable silent no-op.
    private nonisolated func beginYawCalibrationSession(retriesRemaining: Int, completion: @escaping @Sendable () -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async { completion() }
                return
            }
            guard let device = self.captureDevice else {
                guard retriesRemaining > 0 else {
                    DebugFileLogger.log("wiper-marker-calibrate: FAILED captureDevice never became available, giving up")
                    DispatchQueue.main.async { completion() }
                    return
                }
                self.sessionQueue.asyncAfter(deadline: .now() + 0.2) { [weak self] in
                    self?.beginYawCalibrationSession(retriesRemaining: retriesRemaining - 1, completion: completion)
                }
                return
            }
            self.suppressPeriodicFocusRecalibration = true
            // Records this whole calibration session to Photos
            // ("wipercal-" prefixed), from the very start (before
            // stabilization/torch/focus even change) -- added by request
            // 2026-08-19: comparing live detection against a SEPARATE,
            // earlier near-focus-test-capture kept coming up stale (real
            // finding: the mount moved between that capture and later live
            // sessions, making the reference itself wrong, not the
            // detector). Recording the ACTUAL calibration session instead
            // means the reference always matches whatever the user was
            // really looking at, no separate capture step needed -- and
            // starting immediately (not waiting for near-focus lock) means
            // the clip also shows the far-to-near transition itself, useful
            // for debugging the focus mechanism, not just the end state.
            self.activeRecordingFilenamePrefix = "wipercal"
            self.startNewRecording()
            self.wiperMarkerCalibrationRestoreStabilization = self.currentStabilizationEnabled
            if self.wiperMarkerCalibrationRestoreStabilization {
                self.applyStabilizationMode(enabled: false)
            }
            self.wiperMarkerCalibrationTorchEngaged = false
            // No longer racing the app's own launch-time far settle here --
            // `configure()` now direct-locks far focus at launch instead of
            // running a multi-second AF sweep (see `lockKnownGoodFarFocus`),
            // so there's nothing slow left in flight for this near-lock to
            // collide with.
            self.settleAndLockNearFieldFocus(device: device) { [weak self] in
                guard let self else {
                    DispatchQueue.main.async { completion() }
                    return
                }
                // CONFIRMED 2026-08-20, real night sessions: this used to
                // run BEFORE settleAndLockNearFieldFocus, checking
                // latestPreviewPixelBuffer at the very moment captureDevice
                // first became available -- often before the session had
                // delivered its first frame at all, so the `if let
                // pixelBuffer = ...` chain silently failed (no frame yet)
                // and torch never engaged, with no log line to explain why.
                // Two full night sessions both went dark-scene-confirmed
                // (first logged luminance 13.8, well under the 50
                // threshold) with zero "torch on" lines. Moved to run here,
                // after the near-focus lock has genuinely completed, by
                // which point real frames are guaranteed to have been
                // flowing for a while.
                if let pixelBuffer = self.latestPreviewPixelBuffer,
                   let luminance = self.averageLuminance(of: pixelBuffer) {
                    if luminance <= Self.autoLowLightOnLuminance, device.hasTorch, device.isTorchModeSupported(.on) {
                        do {
                            try device.lockForConfiguration()
                            device.torchMode = .on
                            device.unlockForConfiguration()
                            self.wiperMarkerCalibrationTorchEngaged = true
                            DebugFileLogger.log("wiper-marker-calibrate: torch on (luminance=\(luminance))")
                        } catch {
                            DebugFileLogger.log("wiper-marker-calibrate: torch lockForConfiguration failed (\(error))")
                        }
                    } else {
                        DebugFileLogger.log("wiper-marker-calibrate: torch not engaged (luminance=\(luminance), hasTorch=\(device.hasTorch))")
                    }
                } else {
                    DebugFileLogger.log("wiper-marker-calibrate: torch check SKIPPED, still no preview frame available")
                }
                DispatchQueue.main.async { completion() }
            }
        }
    }

    /// Re-runs the precise, tightly-centered detector
    /// (`detectWiperMarkerNormalized`) and, on a confident match, commits
    /// it as the new yaw reference -- used once, when the user actually
    /// opens the zoomed yaw screen (to seed its crosshair), NOT by the
    /// main screen's live nudge polling (see `checkYawNudgeNeeded` for
    /// that -- deliberately a separate, wider-search function, since this
    /// one's whole point is precision within a small ROI, not tolerance
    /// for a still-misaligned mount). `completion` reports the detected
    /// (x,y), or nil if nothing confidently matched -- yawReferenceNormalizedX/Y
    /// are left unchanged on failure (same "fall back to the last-known/
    /// persisted value" contract `detectYawMarker` used to have) so the
    /// yaw screen still has a starting position to show and let the user
    /// manually adjust.
    func refineYawReference(completion: @escaping @Sendable ((x: CGFloat, y: CGFloat)?) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let detected = self.latestPreviewPixelBuffer.flatMap { self.detectWiperMarkerNormalized(in: $0) }
            if let detected {
                DebugFileLogger.log("wiper-marker-calibrate: MATCHED x=\(detected.x) y=\(detected.y)")
            } else {
                DebugFileLogger.log("wiper-marker-calibrate: FAILED no confident marker found")
            }
            // setYawReferenceNormalized mutates @Published state and must
            // run on the main actor -- moved into this same main.async
            // hop (rather than calling it directly from this background
            // closure) instead of adding a separate Task, so the commit
            // and the completion callback stay in a fixed, deterministic
            // order on the same thread.
            DispatchQueue.main.async {
                if let detected {
                    self.setYawReferenceNormalized(x: detected.x, y: detected.y)
                }
                completion(detected)
            }
        }
    }

    /// Wide-search variant of `detectWiperMarkerNormalized`, used for the
    /// main calibration screen's live "yaw nudge" check -- that screen
    /// shows a rectangle previewing where the yaw screen will zoom to, and
    /// needs to know, well before the user ever opens that zoomed screen,
    /// whether the marker is inside it, outside it, or not visible at
    /// all, so it can warn the user to nudge the mount (see ContentView's
    /// yaw-nudge polling).
    ///
    /// A plain widened version of `detectWiperMarkerNormalized`'s ROI/
    /// threshold approach doesn't work here -- CONFIRMED 2026-08-19, same
    /// session: a wide net over a naive whole-ROI centroid gets dragged
    /// toward whichever dark region is BIGGEST, not necessarily the
    /// marker itself (the wiper cap's own shadowed edge is ~3x the
    /// marker's pixel count -- see `detectWiperMarkerNormalized`'s doc
    /// comment for the full story of how that fooled the original
    /// detector). This searches a wide window but finds CONNECTED dark
    /// blobs and filters by the marker's real measured size, rather than
    /// trusting a single whole-region centroid -- the shadow blob and any
    /// other incidental dark region get rejected as too big (or too
    /// small, for noise), not blended in.
    ///
    /// Returns the largest accepted blob's centroid (there's normally at
    /// most one real candidate in range; size is just the simplest
    /// tiebreak if something else nearby happens to also qualify).
    private nonisolated func detectWiperMarkerCoarse(in pixelBuffer: CVPixelBuffer) -> (x: CGFloat, y: CGFloat)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Comfortably contains the yaw-screen rectangle (roughly 0.715-
        // 0.882 x, 0.606-0.772 y at the current default center/zoom) with
        // margin on every side for a realistic nudge-range mount
        // misalignment, without spanning far enough to reach unrelated
        // dark regions of the dash/hood.
        let roiXRange = Int(0.70 * Double(width))..<Int(0.95 * Double(width))
        let roiYRange = Int(0.60 * Double(height))..<Int(0.85 * Double(height))
        let stride = 2

        func brightness(atByteOffset offset: Int) -> Double {
            let b = Double(bytes[offset])
            let g = Double(bytes[offset + 1])
            let r = Double(bytes[offset + 2])
            return (r + g + b) / 3
        }

        var total = 0.0
        var sampleCount = 0
        var row = roiYRange.lowerBound
        while row < roiYRange.upperBound {
            var col = roiXRange.lowerBound
            while col < roiXRange.upperBound {
                total += brightness(atByteOffset: row * bytesPerRow + col * 4)
                sampleCount += 1
                col += stride
            }
            row += stride
        }
        guard sampleCount > 0 else { return nil }
        let threshold = (total / Double(sampleCount)) * 0.7

        let gridCols = (roiXRange.upperBound - roiXRange.lowerBound) / stride
        let gridRows = (roiYRange.upperBound - roiYRange.lowerBound) / stride
        guard gridCols > 0, gridRows > 0 else { return nil }
        var dark = [Bool](repeating: false, count: gridCols * gridRows)
        for gy in 0..<gridRows {
            let row = roiYRange.lowerBound + gy * stride
            for gx in 0..<gridCols {
                let col = roiXRange.lowerBound + gx * stride
                if brightness(atByteOffset: row * bytesPerRow + col * 4) < threshold {
                    dark[gy * gridCols + gx] = true
                }
            }
        }

        // Connected-component flood fill (4-connectivity) over the grid,
        // sized in stride-2 SAMPLE count -- calibrated against
        // `detectWiperMarkerNormalized`'s own real measurements of the
        // genuine marker (89-129 samples at stride 2 within a tightly-
        // centered ROI) vs. the shadow blob it was confused with (~325
        // equivalent) -- see that function's doc comment.
        var visited = [Bool](repeating: false, count: gridCols * gridRows)
        var best: (size: Int, sumGX: Int, sumGY: Int)?
        let minBlobSize = 25
        let maxBlobSize = 260
        for start in 0..<(gridCols * gridRows) where dark[start] && !visited[start] {
            var stack = [start]
            visited[start] = true
            var size = 0
            var sumGX = 0
            var sumGY = 0
            while let idx = stack.popLast() {
                size += 1
                let gx = idx % gridCols
                let gy = idx / gridCols
                sumGX += gx
                sumGY += gy
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = gx + dx, ny = gy + dy
                    guard nx >= 0, nx < gridCols, ny >= 0, ny < gridRows else { continue }
                    let neighborIdx = ny * gridCols + nx
                    if dark[neighborIdx], !visited[neighborIdx] {
                        visited[neighborIdx] = true
                        stack.append(neighborIdx)
                    }
                }
            }
            guard (minBlobSize...maxBlobSize).contains(size) else { continue }
            if best == nil || size > best!.size {
                best = (size, sumGX, sumGY)
            }
        }
        guard let match = best else { return nil }
        let centerGX = Double(match.sumGX) / Double(match.size)
        let centerGY = Double(match.sumGY) / Double(match.size)
        let pixelX = Double(roiXRange.lowerBound) + centerGX * Double(stride)
        let pixelY = Double(roiYRange.lowerBound) + centerGY * Double(stride)
        return (CGFloat(pixelX / Double(width)), CGFloat(pixelY / Double(height)))
    }

    /// Public wrapper around `detectWiperMarkerCoarse` for the main
    /// screen's live yaw-nudge polling -- a pure check, does NOT commit
    /// anything to `yawReferenceNormalizedX/Y` (that only happens via
    /// `refineYawReference`, on the yaw screen itself). Returns the
    /// detected (x,y), or nil if nothing in the expected size range was
    /// found anywhere in the wide search window -- ContentView classifies
    /// that result against the yaw-screen rectangle's bounds to decide
    /// in-band / needs-nudge / not-detected.
    func checkYawNudgeNeeded(completion: @escaping @Sendable ((x: CGFloat, y: CGFloat)?) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let result = self.latestPreviewPixelBuffer.flatMap { self.detectWiperMarkerCoarse(in: $0) }
            DispatchQueue.main.async { completion(result) }
        }
    }

    /// Unwinds everything `beginYawCalibrationSession` changed, back to
    /// normal driving state -- recording finished and saved to Photos as
    /// its own separate "wipercal-" clip (a pre-roll, distinct from the
    /// "recording-" drive file that starts later when the user actually
    /// locks in and drives -- NOT the same segment), torch off,
    /// stabilization restored to whatever it actually was (the user's real
    /// setting, not assumed on/off), focus back to far. Called once the
    /// user confirms the zoomed yaw screen -- NOT on that screen's Cancel
    /// (which returns to the main screen with the session still active, so
    /// polling can resume) -- see ContentView's yaw-calibration flow.
    /// `completion` fires once far focus is settled again.
    func endYawCalibrationSession(completion: @escaping @Sendable () -> Void) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.captureDevice else {
                DispatchQueue.main.async { completion() }
                return
            }
            self.suppressPeriodicFocusRecalibration = false
            self.finishRecording()
            self.activeRecordingFilenamePrefix = "recording"
            if self.wiperMarkerCalibrationTorchEngaged {
                do {
                    try device.lockForConfiguration()
                    device.torchMode = .off
                    device.unlockForConfiguration()
                } catch {
                    DebugFileLogger.log("wiper-marker-calibrate: torch-off lockForConfiguration failed (\(error))")
                }
                self.wiperMarkerCalibrationTorchEngaged = false
            }
            if self.wiperMarkerCalibrationRestoreStabilization {
                self.applyStabilizationMode(enabled: true)
                self.wiperMarkerCalibrationRestoreStabilization = false
            }
            // CONFIRMED 2026-08-20, real drive evidence: this is exactly
            // where the "focus stayed near-field for the whole drive" bug
            // originated -- see startCalibrationRecording's matching
            // comment for the full failure mechanism. The Manual Focus
            // screen (added same day) can leave the lens at ANY position
            // when the user backs out, not just the fixed near-focus
            // constant, which made settleAndLockFarFieldFocus's AF search
            // much more likely to falsely report an instant "settled" at
            // whatever near-ish position it started from. Direct lock
            // sidesteps the whole failure class.
            self.lockKnownGoodFarFocus(device: device)
            DispatchQueue.main.async { completion() }
        }
    }

    /// Commits a yaw reference position (from `calibrateWiperMarker`'s
    /// auto-detect, or the confirmation screen's manual drag-adjust) and
    /// persists both axes -- see `yawReferenceNormalizedX`'s doc comment
    /// for why this, unlike the settings above, IS persisted across
    /// launches.
    func setYawReferenceNormalized(x: CGFloat, y: CGFloat) {
        yawReferenceNormalizedX = x
        yawReferenceNormalizedY = y
        isYawVerifiedThisSession = true
        UserDefaults.standard.set(Double(x), forKey: Self.yawReferenceDefaultsKey)
        UserDefaults.standard.set(Double(y), forKey: Self.yawReferenceYDefaultsKey)
    }

    /// Live focus preview for the Manual Focus screen (ContentView) -- drives
    /// the lens directly as the user drags the slider, WITHOUT persisting
    /// anything, so they can see the actual sharpness change in real time
    /// before committing. Fire-and-forget (nil completion) since the slider
    /// can call this many times a second while dragging -- unlike
    /// `settleAndLockNearFieldFocus`, there's no completion to chain off of
    /// and no need for one.
    func previewFocusLensPosition(_ lensPosition: Float) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.captureDevice, device.isFocusModeSupported(.locked) else { return }
            do {
                try device.lockForConfiguration()
                device.setFocusModeLocked(lensPosition: lensPosition, completionHandler: nil)
                device.unlockForConfiguration()
            } catch {
                DebugFileLogger.log("manual-focus: lockForConfiguration failed (\(error))")
            }
        }
    }

    /// Commits the Manual Focus screen's selected lens position -- called
    /// once, on "Done", not on every slider move (see `previewFocusLensPosition`
    /// for that). Persists across launches, same reasoning as
    /// `setYawReferenceNormalized`: this measures the physical lens/mount,
    /// not a mode. Also re-applies it to the device rather than trusting
    /// the last preview call already left it there -- cheap, and correct
    /// even if this is ever called without a preceding preview.
    func commitManualNearFocusLensPosition(_ lensPosition: Float) {
        manualNearFocusLensPosition = lensPosition
        currentNearFocusLensPosition = lensPosition
        UserDefaults.standard.set(lensPosition, forKey: Self.manualNearFocusLensPositionDefaultsKey)
        previewFocusLensPosition(lensPosition)
    }

    /// Removes the session's inputs/outputs (not just stopping it) so the
    /// next `configure()` call takes the full setup path again -- the only
    /// way to guarantee a calibration recording's 4K preset and locked
    /// focus don't leak into a subsequent normal drive recording, since
    /// `configure`'s quick-restart path (inputs still attached) skips
    /// re-applying the preset/focus mode entirely.
    private nonisolated func teardownSession() {
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        session.commitConfiguration()
        captureDevice = nil
        rotationCoordinator = nil
        rotationObservation = nil
    }

    /// Direct-locks to `knownGoodFarLensPosition` with no AF search --
    /// the ONLY way far focus is ever set now (see `knownGoodFarLensPosition`'s
    /// doc comment for why the AF-based settle-then-lock approach this
    /// replaced -- `settleAndLockFarFieldFocus`, removed 2026-08-20 -- was
    /// retired entirely rather than kept as a fallback). Used at launch,
    /// at the end of a yaw calibration session, at the end of a
    /// calibration/walkaround recording's start, and periodically during
    /// driving (see `recalibrateFocusIfDue`) -- one mechanism for every
    /// far-focus call site now, matching `settleAndLockNearFieldFocus`'s
    /// own already-direct approach for the near side.
    private nonisolated func lockKnownGoodFarFocus(device: AVCaptureDevice) {
        guard device.isFocusModeSupported(.locked) else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.setFocusModeLocked(lensPosition: Self.knownGoodFarLensPosition, completionHandler: nil)
            lastFocusLockTime = CFAbsoluteTimeGetCurrent()
            DebugFileLogger.log("focus: direct-locked at known-good far lensPosition=\(Self.knownGoodFarLensPosition), no AF settle")
        } catch {
            DebugFileLogger.log("focus: direct far-lock failed (\(error))")
        }
    }

    /// Re-asserts the known-good far lock roughly every
    /// `focusRecalibrationInterval` -- called from every captured frame,
    /// matching how `sampleAutoLowLightIfDue` self-throttles instead of
    /// needing its own timer.
    ///
    /// SWITCHED 2026-08-20 from the AF-based `settleAndLockFarFieldFocus`
    /// to the direct `lockKnownGoodFarFocus` -- real drive evidence showed
    /// the AF search never once actually completed across two full
    /// sessions (every single attempt, every 60s, for 70+ minutes each,
    /// hit the 5s timeout), so the "correct for real drift" premise this
    /// was originally built for was providing zero real value while
    /// carrying a genuine, confirmed risk: a false-instant "settled"
    /// report landing on a not-quite-implausible-enough value poisons
    /// `lastGoodLensPosition` for every later timed-out attempt to fall
    /// back to, for the rest of the drive (see `endYawCalibrationSession`'s
    /// matching comment for the exact mechanism that caused two spoiled
    /// recordings). If real focus drift over a long drive turns out to
    /// matter, that's a reason to fix the AF-based approach's reliability
    /// (or measure drift some other way), not a reason to keep running a
    /// mechanism that's demonstrably not working as designed.
    private nonisolated func recalibrateFocusIfDue() {
        guard
            !suppressPeriodicFocusRecalibration,
            let device = captureDevice,
            CFAbsoluteTimeGetCurrent() - lastFocusLockTime >= Self.focusRecalibrationInterval
        else { return }
        lockKnownGoodFarFocus(device: device)
    }

    /// Sets the low-light exposure boost to an explicit state (voice commands "low
    /// light on"/"off"). Suspends auto-detection until `enableAutoLowLight()` is
    /// called again, so the two don't immediately fight.
    func setLowLightBoost(_ enabled: Bool) {
        isAutoLowLightEnabled = false
        currentAutoLowLightEnabled = false
        applyLowLightState(enabled)
    }

    /// Hands control back to the auto-detector (voice: "low light auto"). The next
    /// captured frame will re-evaluate the scene (see `sampleAutoLowLightIfDue`).
    func enableAutoLowLight() {
        isAutoLowLightEnabled = true
        currentAutoLowLightEnabled = true
    }

    /// Sets `.standard` video stabilization on/off (voice: "stabilization on"/"off").
    /// `.standard` rather than `.cinematic`/`.cinematicExtended` deliberately — those
    /// crop in further and add multi-frame look-ahead latency, both undesirable here.
    func setStabilizationEnabled(_ enabled: Bool) {
        isStabilizationEnabled = enabled
        currentStabilizationEnabled = enabled
        sessionQueue.async { [weak self] in self?.applyStabilizationMode(enabled: enabled) }
    }

    /// Toggles the normal-drive recording preset between 1080p (default) and
    /// 4K -- see `isFourKEnabled`'s doc comment for the reactivation
    /// history/reasoning. If the session is already running but NOT
    /// actively writing a real recording (level/configuring screens, or
    /// driving before Lock Settings), this takes effect immediately via the
    /// same full teardown-then-fresh-configure sequence
    /// `stopCalibrationRecording` already uses -- safe specifically because
    /// nothing real is being written yet, so there's no in-progress file to
    /// race against. If a real recording IS already in progress, the new
    /// value is only saved -- it takes effect on the next session start
    /// rather than interrupting footage already being written. NOT
    /// persisted to UserDefaults -- see init's comment; this always starts
    /// back at 1080p on a fresh launch.
    func setFourKEnabled(_ enabled: Bool) {
        isFourKEnabled = enabled
        currentFourKEnabled = enabled
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning, self.assetWriter == nil else { return }
            self.session.stopRunning()
            self.teardownSession()
            self.configure(startRecording: false, preset: enabled ? .hd4K3840x2160 : .hd1920x1080, settleFocus: true)
        }
    }

    /// Toggles normal-drive recording between 15fps (default) and 30fps --
    /// see `isThirtyFpsEnabled`'s doc comment for why. Unlike
    /// `setFourKEnabled`, this doesn't need a session teardown/preset
    /// change to apply -- `restrictFrameRate` just sets
    /// activeVideoMin/MaxFrameDuration directly on the already-configured
    /// device, which AVFoundation allows on a running session -- so this
    /// takes effect immediately regardless of whether a recording is
    /// already in progress, rather than waiting for the next session
    /// start.
    func setThirtyFpsEnabled(_ enabled: Bool) {
        isThirtyFpsEnabled = enabled
        currentThirtyFpsEnabled = enabled
        sessionQueue.async { [weak self] in
            guard let self, let device = self.captureDevice else { return }
            self.restrictFrameRate(for: device, to: enabled ? 30 : 15)
        }
    }

    private nonisolated func applyStabilizationMode(enabled: Bool) {
        guard
            let connection = videoOutput.connection(with: .video),
            connection.isVideoStabilizationSupported
        else { return }
        connection.preferredVideoStabilizationMode = enabled ? .standard : .off
    }

    private func applyLowLightState(_ enabled: Bool) {
        isLowLightBoostEnabled = enabled
        currentLowLightEnabled = enabled
        sessionQueue.async { [weak self] in self?.applyExposureBias(enabled: enabled) }
    }

    private nonisolated func applyExposureBias(enabled: Bool) {
        let bias: Float = enabled ? min(3.0, captureDevice?.maxExposureTargetBias ?? 0) : 0
        appliedBiasEV = bias
        setDeviceExposureBias(bias)
    }

    private nonisolated func setDeviceExposureBias(_ bias: Float) {
        guard let device = captureDevice else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.setExposureTargetBias(bias, completionHandler: nil)
        } catch {
            // Device may be mid-reconfiguration; the bias just won't apply this time.
        }
    }

    /// AE is free to reach its brightness target either by raising ISO/gain or
    /// by slowing the shutter. At night -- especially with the low-light bias
    /// below pushing the target brighter still -- a slower shutter is the
    /// cheaper way for AE to get there, but its also the one that blurs a
    /// moving vehicle (and everything else on the road) worse than sensor
    /// noise ever would for YOLO26s purposes. Capping the max exposure
    /// duration removes that option, forcing AE to compensate with ISO/gain
    /// instead.
    private nonisolated func restrictMaxExposureDuration(for device: AVCaptureDevice) {
        // 1/60s is a conservative starting point: every iPhone video format
        // supports at least 1/30s per Apples baseline requirements, so this
        // is safe across devices. Tune against real night-driving footage --
        // slower (e.g. 1/30s) trades some blur for less ISO noise/darkness;
        // faster (e.g. 1/120s) does the opposite.
        let maxDuration = CMTime(value: 1, timescale: 60)
        guard device.activeFormat.maxExposureDuration > maxDuration else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeMaxExposureDuration = maxDuration
        } catch {
            // Device may be mid-reconfiguration; AE keeps its wider default range this time.
        }
    }

    /// Locks capture to a fixed frame rate -- 15fps by default (half the
    /// ISP/encode work of the device's own default 30fps, on top of the 4x
    /// cut from dropping to 1080p, see `configure`), or 30fps when
    /// `isThirtyFpsEnabled` is on. Either way, `InferenceEngine.process`'s
    /// own `isBusy` gate already discards any frame that arrives while the
    /// previous one is still being processed -- inference's effective rate
    /// is governed by how long inference itself takes, not by how fast
    /// frames arrive, so requesting 30fps here doesn't add real-time
    /// inference load, only more ISP/encode work (see
    /// `isThirtyFpsEnabled`'s doc comment).
    private nonisolated func restrictFrameRate(for device: AVCaptureDevice, to fps: Double) {
        let desiredDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
        guard device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
            desiredDuration >= $0.minFrameDuration && desiredDuration <= $0.maxFrameDuration
        }) else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeVideoMinFrameDuration = desiredDuration
            device.activeVideoMaxFrameDuration = desiredDuration
        } catch {
            // Device may be mid-reconfiguration; frame rate stays at its previous default.
        }
    }

    /// Estimates scene brightness from the actual captured frame and, while
    /// auto-detection is enabled, engages or releases the low-light boost.
    ///
    /// While the boost is off, the raw frame is an uncontaminated signal, sampled
    /// once a second. While the boost is on, the frame is biased brighter — and in
    /// anything but genuine darkness, that pushes it into clipping (pixels pinned
    /// near white) rather than scaling linearly, so a bright scene still reads as
    /// "dark" after simply dividing by 2^bias. (That was the previous approach here;
    /// confirmed on-device it kept the boost stuck on even pointed at daylight, for
    /// exactly this reason — same underlying problem as the ISO-based attempt before
    /// it, which also measured a signal the bias itself was driving.) So instead,
    /// periodically release the bias just long enough to get one clean, unbiased
    /// reading, then restore it if still needed.
    private nonisolated func sampleAutoLowLightIfDue(_ pixelBuffer: CVPixelBuffer) {
        let now = CFAbsoluteTimeGetCurrent()

        guard currentLowLightEnabled else {
            guard now - lastLuminanceSampleTime >= 1.0 else { return }
            lastLuminanceSampleTime = now
            if let luminance = averageLuminance(of: pixelBuffer) {
                evaluateAutoLowLight(luminance: luminance)
            }
            return
        }

        if isProbingLowLight {
            guard now - probeStartedAt >= Self.probeSettleDuration else { return }
            isProbingLowLight = false
            lastProbeTime = now
            if let luminance = averageLuminance(of: pixelBuffer) {
                evaluateAutoLowLight(luminance: luminance)
            }
            // Restore the boost's bias unless evaluateAutoLowLight just released it
            // (in which case applyLowLightState is already zeroing it independently).
            if currentLowLightEnabled {
                setDeviceExposureBias(appliedBiasEV)
            }
        } else if now - lastProbeTime >= Self.probeInterval {
            isProbingLowLight = true
            probeStartedAt = now
            setDeviceExposureBias(0)
        }
    }

    /// Rough average brightness (0-255) of a BGRA buffer, subsampled for speed —
    /// this only needs to be a consistent relative signal for thresholding, not a
    /// precise photometric measurement.
    private nonisolated func averageLuminance(of pixelBuffer: CVPixelBuffer) -> Double? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width       = CVPixelBufferGetWidth(pixelBuffer)
        let height      = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer      = base.assumingMemoryBound(to: UInt8.self)

        let stride: Int = 8
        var total = 0
        var count = 0
        var y = 0
        while y < height {
            let rowStart = y * bytesPerRow
            var x = 0
            while x < width {
                // BGRA byte order in memory (premultipliedFirst + byteOrder32Little) —
                // matches the pixel format requested for the video output.
                let offset = rowStart + x * 4
                total += Int(buffer[offset]) + Int(buffer[offset + 1]) + Int(buffer[offset + 2])
                count += 1
                x += stride
            }
            y += stride
        }
        guard count > 0 else { return nil }
        return Double(total) / Double(count * 3)
    }

    private nonisolated func evaluateAutoLowLight(luminance: Double) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            DebugFileLogger.log(
                "auto-lowlight: luminance=\(luminance) auto=\(self.isAutoLowLightEnabled) "
                + "boost=\(self.isLowLightBoostEnabled) consecutiveLow=\(self.consecutiveLowLuminanceSamples)"
            )
            guard self.isAutoLowLightEnabled else { return }
            if !self.isLowLightBoostEnabled {
                guard luminance <= Self.autoLowLightOnLuminance else {
                    self.consecutiveLowLuminanceSamples = 0
                    return
                }
                self.consecutiveLowLuminanceSamples += 1
                if self.consecutiveLowLuminanceSamples >= Self.consecutiveLowSamplesRequired {
                    self.applyLowLightState(true)
                    self.consecutiveLowLuminanceSamples = 0
                }
            } else if luminance >= Self.autoLowLightOffLuminance {
                self.applyLowLightState(false)
            }
        }
    }

    /// `preset` defaults to the normal drive preset (1080p, see the thermal
    /// comment below) -- `startCalibrationRecording` passes 4K instead, for
    /// a short one-off clip where thermal isn't a concern but row precision
    /// is. Only meaningful on a fresh (non-quick-restart) configure -- see
    /// that path's own comment for why a stale preset can't leak from a
    /// calibration recording into a normal one.
    /// `settleFocus: false` skips the automatic focus-settle below --
    /// `startCalibrationRecording` uses this, since it always does its own
    /// explicit settle-then-lock afterward (it needs the completion to
    /// know when to actually start writing frames, which the automatic
    /// fire-and-forget call here can't provide). CONFIRMED bug 2026-08-09:
    /// without this, both calls ran concurrently and raced over the same
    /// shared focus-tracking state (`focusSettleObservation` etc.) -- the
    /// explicit call's completion (the one `startCalibrationRecording`
    /// actually depends on) never fired, so `startNewRecording()` never
    /// ran and the caller's "recording" state stayed stuck forever. Debug
    /// log showed two `focus: settling` lines ~150ms apart per attempt,
    /// same signature as the earlier double-tap race, but this one was a
    /// straightforward logic bug, not a UI timing issue.
    private nonisolated func configure(startRecording: Bool, preset: AVCaptureSession.Preset = .hd1920x1080, settleFocus: Bool = true) {
        guard !session.isRunning else { return }

        // After stop(), inputs/outputs are still attached — just restart.
        // Only reachable for a normal (non-calibration) start: `preset` is
        // ignored here, so `startCalibrationRecording`/`stopCalibrationRecording`
        // deliberately tear the session fully down (removing inputs/outputs)
        // rather than just stopping it, forcing the full setup path below to
        // run again next time -- otherwise a 4K calibration preset could
        // silently persist into the next normal drive recording.
        if !session.inputs.isEmpty {
            session.startRunning()
            if startRecording { startNewRecording() }
            return
        }

        registerSessionObservers()
        recoverOrphanedRecordings()

        // 4K/30fps was confirmed on a real 23-minute drive to thermally throttle the
        // device 10-15x — even at the cheapest inference settings (nano, two-pass
        // off), meaning ML load wasn't the dominant heat source. The sustained 4K
        // ISP+encode pipeline runs regardless of model choice, so that's the actual
        // target: 1080p is 1/4 the pixels of 4K and 15fps is half of 30fps, an ~8x
        // cut to that fixed cost. See `logThermalState` for the telemetry that
        // confirms whether this is actually sustainable on a real long drive.
        session.beginConfiguration()
        session.sessionPreset = session.canSetSessionPreset(preset) ? preset : .hd1280x720

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input  = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        captureDevice = device
        // Fire-and-forget: doesn't block this synchronous configure() from
        // completing/starting the session, matching how it ran before.
        // recalibrateFocusIfDue() (from captureOutput) keeps it fresh for
        // the rest of the drive. Skipped when the caller (e.g.
        // startCalibrationRecording) is going to do its own explicit
        // settle-then-lock right after -- see this function's doc comment.
        // Direct lock to the empirically-known far position instead of
        // running the continuousAutoFocus-based settle here -- CONFIRMED
        // 2026-08-19: that settle takes 2-5s and, being fire-and-forget at
        // launch, was still in flight when the wiper-marker calibration
        // screen's own near-focus lock ran moments later (level screen
        // .onAppear -> runYawAutoDetect after a 0.4s delay), so both calls
        // raced the same physical lens actuator and whichever's completion
        // fired last won -- real captures came back locked far (0.812/0.831)
        // instead of the requested near position despite the near-lock's
        // own completion having already fired. We already know the correct
        // far value (`knownGoodFarLensPosition`, same empirical-measurement
        // discipline as `knownGoodNearLensPosition`), so there's nothing an
        // AF sweep would discover here -- skipping it removes the race at
        // its root instead of trying to out-race it after the fact.
        // `recalibrateFocusIfDue()` still uses the AF-based settle
        // periodically during an actual drive, to correct for real drift
        // (temperature, mechanical settling) -- that's a different, later
        // concern this doesn't touch.
        if settleFocus {
            lockKnownGoodFarFocus(device: device)
        }
        restrictMaxExposureDuration(for: device)
        restrictFrameRate(for: device, to: currentThirtyFpsEnabled ? 30 : 15)

        // Apply a persisted (non-auto) low-light boost to the newly configured
        // device — loaded into the published/mirrored state in init(), before a
        // device existed to apply it to.
        if !currentAutoLowLightEnabled, currentLowLightEnabled {
            applyExposureBias(enabled: true)
        }

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        // The connection this applies to only exists once the output is attached to
        // the session above.
        applyStabilizationMode(enabled: currentStabilizationEnabled)

        configureRotationCoordinator(for: device)

        session.commitConfiguration()
        session.startRunning()
        if startRecording { startNewRecording() }
    }

    /// Watches for the session being interrupted (e.g. another app takes the camera,
    /// or a phone call) or hitting a runtime error, so recording state reflects
    /// reality instead of silently going stale, and the session self-recovers where
    /// possible instead of leaving the camera dark for the rest of the drive.
    private nonisolated func registerSessionObservers() {
        guard !observersRegistered else { return }
        observersRegistered = true

        NotificationCenter.default.addObserver(
            forName: AVCaptureSession.wasInterruptedNotification, object: session, queue: nil
        ) { [weak self] notification in
            let reasonValue = (notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as? NSNumber)?.intValue
            let reason = reasonValue.flatMap(AVCaptureSession.InterruptionReason.init(rawValue:))
            // CONFIRMED 2026-08-11: these three handlers used `print` only,
            // never `DebugFileLogger.log` -- meaning a session interruption
            // or runtime error (e.g. .videoDeviceNotAvailableDueToSystemPressure,
            // plausible here given `logThermalState`'s "serious" readings were
            // already showing throughout that session) was completely invisible
            // outside a live-attached Xcode console. Confirmed as the actual
            // cause of "distance-cal: round=2 FAILED to start" repeating 5x
            // with no other diagnostic in the pulled debug log -- not
            // literally unknowable, just never captured. Fixed by also
            // logging here.
            DebugFileLogger.log("session: MATCHED interrupted reason=\(String(describing: reason))")
            print("[CameraManager] session interrupted: \(String(describing: reason))")
            self?.setRecordingActive(false)
        }

        NotificationCenter.default.addObserver(
            forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil
        ) { _ in
            DebugFileLogger.log("session: MATCHED interruption ended")
            print("[CameraManager] session interruption ended")
        }

        NotificationCenter.default.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil
        ) { [weak self] notification in
            guard let self else { return }
            let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
            DebugFileLogger.log("session: MATCHED runtimeError \(error?.localizedDescription ?? "unknown") code=\(error?.code ?? -1) domain=\(error?.domain ?? "?")")
            print("[CameraManager] session runtime error: \(error?.localizedDescription ?? "unknown")")
            self.sessionQueue.async {
                guard !self.session.isRunning else { return }
                self.session.startRunning()
            }
        }

        // Backgrounding — not a clean stop() — is how virtually every real session
        // actually ends (screen lock, switching to Maps, etc.). Force-finish the
        // current segment and save it to Photos here rather than leaving it to the
        // next-launch recovery sweep, since we have a brief guaranteed window before
        // iOS suspends the process.
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.beginBackgroundFlush()
        }

        // Rarely fires for a suspended app, but costs nothing to also handle for the
        // cases where it does (e.g. a foreground termination).
        NotificationCenter.default.addObserver(
            forName: UIApplication.willTerminateNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.beginBackgroundFlush()
        }

        // No automatic response — this is what actually confirms whether 1080p/15fps
        // holds in `.nominal`/`.fair` for a whole real drive, or still climbs into
        // `.serious`/`.critical` the way 4K/30fps did. Also drives the on-screen/baked
        // HUD warning so a climbing state is visible in the moment, not just after
        // the fact in the debug log.
        handleThermalStateChange(ProcessInfo.processInfo.thermalState, context: "initial")
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.handleThermalStateChange(ProcessInfo.processInfo.thermalState, context: "changed")
        }
    }

    private nonisolated func handleThermalStateChange(_ state: ProcessInfo.ThermalState, context: String) {
        DebugFileLogger.log("thermal: \(context) state=\(Self.thermalStateName(state))")
        Task { @MainActor [weak self] in
            self?.thermalState = state
        }
    }

    /// Feeds a just-completed frame's inference latency into the "% of full speed"
    /// metric. `configKey` (model + two-pass state) resets the baseline whenever it
    /// changes, since switching models/two-pass changes latency on its own and
    /// would otherwise get misread as thermal throttling.
    func recordInferenceLatency(_ elapsedMs: Double, configKey: String) {
        guard elapsedMs > 0 else { return }

        if thermalBaselineConfigKey != configKey {
            thermalBaselineConfigKey = configKey
            // Seed immediately instead of clearing to nil — a config change made
            // while already non-nominal must not strand the baseline forever,
            // since it can only refill below while `.nominal`, and a real drive
            // proved thermal state can climb straight to `.critical` and never
            // come back down. That drive's percent froze at a stale, falsely
            // reassuring reading for the rest of the trip because of exactly
            // this gap. A baseline seeded mid-throttle is provisional (it'll
            // read closer to 100% than it should until real `.nominal` samples
            // refine it below), but a provisional number that keeps updating
            // beats one that's frozen and wrong.
            thermalBaselineElapsedMs = elapsedMs
            thermalCurrentElapsedMs = elapsedMs
            // Force the next available reading to publish immediately rather than
            // waiting up to 5s under what's now a stale reading for the old config.
            lastThermalPercentPublishTime = 0
        }

        if thermalState == .nominal {
            // EMA so a handful of noisy frames don't yank the baseline around.
            if let existing = thermalBaselineElapsedMs {
                thermalBaselineElapsedMs = existing * 0.9 + elapsedMs * 0.1
            } else {
                thermalBaselineElapsedMs = elapsedMs
            }
        }

        // Separate, faster-decaying EMA of the *current* latency (tracked regardless
        // of thermal state) so the displayed percent reflects a stable trend instead
        // of one noisy frame's timing.
        if let existing = thermalCurrentElapsedMs {
            thermalCurrentElapsedMs = existing * 0.8 + elapsedMs * 0.2
        } else {
            thermalCurrentElapsedMs = elapsedMs
        }

        guard let baseline = thermalBaselineElapsedMs, baseline > 0, let current = thermalCurrentElapsedMs else {
            return
        }

        // Only actually update the published value on a real thermal-state
        // transition, or every `thermalPercentPublishInterval` seconds otherwise —
        // computing it every frame was accurate but too jumpy to read at a glance.
        let now = CFAbsoluteTimeGetCurrent()
        let stateChanged = lastPublishedThermalState != thermalState
        guard stateChanged || now - lastThermalPercentPublishTime >= Self.thermalPercentPublishInterval else {
            return
        }
        lastPublishedThermalState = thermalState
        lastThermalPercentPublishTime = now

        let percent = min(100, max(0, Int((baseline / current * 100).rounded())))
        thermalSpeedPercent = percent

        // Logged at the same throttled cadence as the publish itself (not every
        // frame) so a multi-minute drive doesn't flood the file — but with the raw
        // inputs (not just the rounded percent) so a suspiciously-flat reading can
        // actually be checked against real numbers afterward, instead of just
        // re-trusting the same computed value.
        DebugFileLogger.log(String(
            format: "thermal-speed: state=%@ elapsedMs=%.1f smoothedMs=%.1f baselineMs=%.1f percent=%d",
            Self.thermalStateName(thermalState), elapsedMs, current, baseline, percent
        ))
    }

    nonisolated static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown(\(state.rawValue))"
        }
    }

    /// Toggles roughly twice a second, off wall-clock time rather than a running
    /// timer, so live HUD and baked-recording overlay (rendered on separate
    /// pipelines/frame cadences) blink in sync without needing to share state.
    nonisolated static func thermalBlinkOn(at time: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()) -> Bool {
        Int(time / 0.5) % 2 == 0
    }

    /// Color for the always-on thermal readout: nominal reads the same as other
    /// HUD text, fair is a yellow caution (not yet a real problem), serious is a
    /// steady red, and critical blinks red/dim since that's the state where the
    /// 10-15x thermal slowdown is actually happening.
    nonisolated static func thermalColor(for state: ProcessInfo.ThermalState, blinkOn: Bool) -> UIColor {
        switch state {
        case .nominal:  return UIColor.white.withAlphaComponent(0.75)
        case .fair:     return UIColor.systemYellow
        case .serious:  return UIColor.systemRed
        case .critical: return blinkOn ? UIColor.systemRed : UIColor.white.withAlphaComponent(0.3)
        @unknown default: return UIColor.systemRed
        }
    }

    /// Requests extra runtime from iOS to force-finish and save the current segment
    /// before the app is suspended, then immediately starts a fresh segment so
    /// recording resumes seamlessly if/when the app returns to the foreground.
    private nonisolated func beginBackgroundFlush() {
        let alreadyFlushing = backgroundTaskState.withLock { $0 != .invalid }
        guard !alreadyFlushing else { return }

        Task { @MainActor [weak self] in
            guard let self else { return }
            let taskID = UIApplication.shared.beginBackgroundTask(withName: "DriverAssist.flushRecording") { [weak self] in
                // Expiration handler: out of extra time. End the task so the OS doesn't
                // kill the whole process for overrunning it — the fragmented file on
                // disk is still valid either way and will be picked up by
                // recoverOrphanedRecordings on next launch.
                self?.endBackgroundTaskIfNeeded()
            }
            guard taskID != .invalid else { return }

            backgroundTaskState.withLock { $0 = taskID }

            sessionQueue.async { [weak self] in
                self?.finishRecording()
                self?.startNewRecording()
                self?.endBackgroundTaskIfNeeded()
            }
        }
    }

    private nonisolated func endBackgroundTaskIfNeeded() {
        let taskID = backgroundTaskState.withLock { id -> UIBackgroundTaskIdentifier in
            defer { id = .invalid }
            return id
        }

        guard taskID != .invalid else { return }
        DispatchQueue.main.async {
            UIApplication.shared.endBackgroundTask(taskID)
        }
    }

    /// Sweeps any `recording-*.mov` left behind by a previous run (e.g. the process
    /// was killed mid-segment, which is the normal case — see `startNewRecording`)
    /// into the Photos library. `movieFragmentInterval` means a file killed *after*
    /// its first fragment interval has a valid `moov` for everything up to that
    /// point, but PhotoKit's asset-resource validation still rejects it as-is with
    /// `PHPhotosErrorInvalidResource` — passthrough-exporting first (see
    /// `finalizeFragmentedMovie`) rewrites the container header without
    /// re-encoding, which PhotoKit does accept. A file killed *before* its first
    /// fragment interval (or recorded before `movieFragmentInterval` existed at
    /// all) never got a `moov` written at all; that footage is genuinely gone, so
    /// `finalizeFragmentedMovie` discards those instead of retrying them forever.
    private nonisolated func recoverOrphanedRecordings() {
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let files = try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)
        else { return }

        let orphans = files.filter { $0.lastPathComponent.hasPrefix("recording-") && $0.pathExtension == "mov" }
        recoverNextOrphan(orphans, at: 0)
    }

    /// Recovers one orphan at a time rather than firing every export concurrently —
    /// iOS caps how many `AVAssetExportSession`/media-service sessions can run at
    /// once, and launching a dozen at once at app startup made every one of them
    /// fail immediately instead of queuing.
    private nonisolated func recoverNextOrphan(_ files: [URL], at index: Int) {
        guard index < files.count else { return }
        let file = files[index]

        finalizeFragmentedMovie(at: file) { [weak self] finalizedURL in
            guard let self else { return }
            guard let finalizedURL else {
                self.recoverNextOrphan(files, at: index + 1)
                return
            }
            self.saveToPhotoLibrary(finalizedURL) { success in
                if success {
                    try? FileManager.default.removeItem(at: file)
                } else {
                    // Save failed (e.g. permission denial) — drop the redundant
                    // finalized copy and leave the original as the single
                    // manually-recoverable file, per `saveToPhotoLibrary`.
                    try? FileManager.default.removeItem(at: finalizedURL)
                }
                self.recoverNextOrphan(files, at: index + 1)
            }
        }
    }

    /// Rewrites a fragmented, never-finalized `.mov` (see `recoverOrphanedRecordings`)
    /// into a properly closed one via passthrough export — same audio/video data,
    /// no re-encoding, just a corrected container header — so PhotoKit will accept
    /// it. Calls back with the finalized file's URL, or nil if there's nothing to
    /// save: either the original has no `moov` at all (checked up front via
    /// `isReadable`, since asking `AVAssetExportSession` to open one of these
    /// directly fails with an opaque, unhelpful error), in which case it's deleted
    /// here since that footage is unrecoverable, or the export itself failed for
    /// some other reason, in which case the original is left alone for a later
    /// retry.
    private nonisolated func finalizeFragmentedMovie(at url: URL, completion: @escaping (URL?) -> Void) {
        let asset = AVURLAsset(url: url)
        Task {
            do {
                _ = try await asset.load(.isReadable)
            } catch {
                print("[CameraManager] discarding unrecoverable orphaned recording \(url.lastPathComponent) (no moov — killed before its first fragment interval): \(error.localizedDescription)")
                try? FileManager.default.removeItem(at: url)
                completion(nil)
                return
            }

            let finalizedURL = url.deletingLastPathComponent()
                .appendingPathComponent(url.deletingPathExtension().lastPathComponent + "-finalized")
                .appendingPathExtension("mov")
            try? FileManager.default.removeItem(at: finalizedURL)

            guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
                completion(nil)
                return
            }
            do {
                try await export.export(to: finalizedURL, as: .mov)
                completion(finalizedURL)
            } catch {
                print("[CameraManager] failed to finalize orphaned recording \(url.lastPathComponent): \(error.localizedDescription)")
                completion(nil)
            }
        }
    }

    /// Keeps the capture outputs' video rotation in sync with the device's physical
    /// orientation, since the app-level video orientation APIs (`AVCaptureConnection
    /// .videoOrientation`) are deprecated in favor of rotation angles as of iOS 17.
    private nonisolated func configureRotationCoordinator(for device: AVCaptureDevice) {
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        rotationCoordinator = coordinator

        // `...Capture` is Apple's angle for one-shot still-photo capture; our video
        // data output is a continuously-live connection (feeds both the recording
        // and inference), which is what `...Preview` is for. Using `...Capture` here
        // meant the angle it read at startup didn't match the value the coordinator
        // settled on moments later, so the very first recorded frame (which sizes
        // the recording's writer for the whole session) locked in different
        // dimensions than every frame after it — producing a permanently
        // letterboxed recording.
        applyCaptureRotation(coordinator.videoRotationAngleForHorizonLevelPreview)
        rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.new]) { [weak self] _, change in
            guard let angle = change.newValue else { return }
            self?.applyCaptureRotation(angle)
        }
    }

    private nonisolated func applyCaptureRotation(_ angle: CGFloat) {
        guard
            let connection = videoOutput.connection(with: .video),
            connection.isVideoRotationAngleSupported(angle)
        else { return }
        connection.videoRotationAngle = angle
    }

    /// Creates the asset writer for a new recording. The writer's video input/adaptor
    /// are added lazily once the first frame arrives, since only then do we know the
    /// actual (post-rotation) pixel buffer dimensions. One writer covers the whole
    /// session — from launch until the app backgrounds, stops, or a write fails —
    /// rather than splitting into periodic segments.
    ///
    /// If creation fails (e.g. transient low storage), the failure is not silent:
    /// it's logged and a retry is scheduled, rather than leaving this run's
    /// recording permanently and invisibly dead — which is what happened before.
    private nonisolated func startNewRecording() {
        guard
            assetWriter == nil,
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }

        updateStorageStatus(documentsDirectory: documents)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let url = documents.appendingPathComponent("\(activeRecordingFilenamePrefix)-\(formatter.string(from: Date())).mov")

        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            // Protects against the process being killed mid-recording — fragmented
            // .mov files stay structurally valid and playable without ever calling
            // finishWriting(), which is what lets `recoverOrphanedRecordings` import
            // them on the next launch.
            writer.movieFragmentInterval = CMTime(seconds: 10, preferredTimescale: 600)
            assetWriter = writer
            currentSegmentURL = url
        } catch {
            print("[CameraManager] failed to create asset writer: \(error.localizedDescription)")
            setRecordingActive(false)
            scheduleRecordingRetry()
        }
    }

    /// Checked whenever a new recording starts (app launch, retry after failure, or
    /// resuming after backgrounding) rather than per-frame, since it's just an early
    /// warning and doesn't need finer granularity.
    private nonisolated func updateStorageStatus(documentsDirectory: URL) {
        let values = try? documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let available = values?.volumeAvailableCapacityForImportantUsage
        let low = (available ?? .max) < lowStorageThresholdBytes
        if low {
            print("[CameraManager] low storage: \(available.map(String.init) ?? "unknown") bytes available")
        }
        Task { @MainActor [weak self] in
            self?.isStorageLow = low
        }
    }

    /// Retries starting a new segment after a delay. Called whenever recording
    /// setup fails, so a transient issue (temporary low storage, momentary I/O
    /// contention) doesn't permanently end recording for the rest of the session.
    private nonisolated func scheduleRecordingRetry() {
        sessionQueue.asyncAfter(deadline: .now() + 5) { [weak self] in
            guard let self, self.session.isRunning, self.assetWriter == nil else { return }
            self.startNewRecording()
        }
    }

    /// Tears down a recording that failed mid-write (e.g. the writer transitioned to
    /// `.failed`, commonly from disk space running out) and schedules a retry so the
    /// rest of the drive still gets recorded once/if the underlying issue clears.
    private nonisolated func failCurrentRecording() {
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
        currentSegmentURL = nil
        setRecordingActive(false)
        scheduleRecordingRetry()
    }

    private nonisolated func setRecordingActive(_ active: Bool) {
        Task { @MainActor [weak self] in
            self?.isRecording = active
        }
    }

    /// Finishes the current recording, blocking until both the write and the Photos
    /// save complete. Used on a clean stop() and on backgrounding (see
    /// `beginBackgroundFlush`, which calls this then immediately starts a fresh
    /// recording so things continue seamlessly if the app returns to the foreground)
    /// — safe to block here since no more frames are coming until then.
    private nonisolated func finishRecording() {
        guard let writer = assetWriter, let input = assetWriterInput else {
            assetWriter = nil
            assetWriterInput = nil
            pixelBufferAdaptor = nil
            currentSegmentURL = nil
            return
        }
        let url = currentSegmentURL
        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
        currentSegmentURL = nil

        guard writer.status == .completed, let url else { return }
        let saveSemaphore = DispatchSemaphore(value: 0)
        saveToPhotoLibrary(url) { _ in saveSemaphore.signal() }
        saveSemaphore.wait()
    }

    /// Imports a finished recording into the Photos library and removes the local
    /// copy on success, so completed footage doesn't sit invisibly in app storage
    /// (where the last incident's evidence was found) and doesn't grow unbounded.
    /// Leaves the file in place on failure (including permission denial) so it can
    /// still be recovered manually via Files, and so a later sweep can retry it.
    private nonisolated func saveToPhotoLibrary(_ url: URL, completion: ((Bool) -> Void)? = nil) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                print("[CameraManager] Photos permission not granted (\(status.rawValue)); leaving \(url.lastPathComponent) in app storage")
                completion?(false)
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetCreationRequest.forAsset().addResource(with: .video, fileURL: url, options: nil)
            }) { success, error in
                if success {
                    try? FileManager.default.removeItem(at: url)
                } else {
                    print("[CameraManager] failed to save \(url.lastPathComponent) to Photos: \(error?.localizedDescription ?? "unknown")")
                }
                completion?(success)
            }
        }
    }

    /// Composites the current detections onto `pixelBuffer` and appends the result
    /// to the in-progress recording, so the saved file matches what's on screen.
    private nonisolated func appendRecordingFrame(_ pixelBuffer: CVPixelBuffer, presentationTime: CMTime) {
        guard let writer = assetWriter else { return }

        // Most commonly caused by storage filling up mid-drive. Recover instead of
        // silently going dark for the remainder of the session.
        if writer.status == .failed {
            print("[CameraManager] writer failed mid-recording: \(writer.error?.localizedDescription ?? "unknown")")
            failCurrentRecording()
            return
        }

        if assetWriterInput == nil {
            let width  = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let outputSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
            input.expectsMediaDataInRealTime = true

            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: width,
                    kCVPixelBufferHeightKey as String: height
                ]
            )

            guard writer.canAdd(input) else {
                print("[CameraManager] writer cannot accept video input")
                failCurrentRecording()
                return
            }
            writer.add(input)
            assetWriterInput = input
            pixelBufferAdaptor = adaptor

            guard writer.startWriting() else {
                print("[CameraManager] startWriting failed: \(writer.error?.localizedDescription ?? "unknown")")
                failCurrentRecording()
                return
            }
            writer.startSession(atSourceTime: presentationTime)
            setRecordingActive(true)
            // Precise anchor between this recording's internal (session-relative)
            // frame timestamps and wall-clock epoch time — `presentationTime`
            // becomes the movie's own t=0, so any frame's later PTS (as read back
            // from the file) plus this logged epoch gives that frame's real epoch
            // time. Offline reconstruction uses this instead of the recording
            // filename's whole-second precision, and instead of assuming a
            // constant frame rate (which would drift under dropped frames).
            DebugFileLogger.log("recording-start: file=\(writer.outputURL.lastPathComponent) epoch=\(Date().timeIntervalSince1970)")
        }

        guard
            let input = assetWriterInput,
            let adaptor = pixelBufferAdaptor,
            input.isReadyForMoreMediaData
        else { return }

        // Recorded straight from the camera with no compositing step — the overlay
        // is now reconstructed offline (from `detections.jsonl` + the debug log)
        // rather than baked in, so recordings can be compared directly against a
        // reference model without the drawn boxes/HUD contaminating that frame.
        // This also means one fewer pixel-buffer-pool allocation and CGContext
        // draw per frame versus the old bake-in-place approach.
        adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
    }
}

// MARK: — Sample buffer delegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        sampleAutoLowLightIfDue(pixelBuffer)
        recalibrateFocusIfDue()
        appendRecordingFrame(pixelBuffer, presentationTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        latestPreviewPixelBuffer = pixelBuffer
        if activeRecordingFilenamePrefix == "calibration" {
            latestCalibrationPixelBuffer = pixelBuffer
        }

        let box = UncheckedSendableBox(value: pixelBuffer)
        Task { @MainActor [weak self] in
            self?.onFrame?(box.value)
        }
    }
}

// MARK: — SwiftUI camera preview

/// UIKit-backed view that hosts an AVCaptureVideoPreviewLayer.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        PreviewUIView(session: session)
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
        private var rotationObservation: NSKeyValueObservation?
        private var sessionStartObserver: NSObjectProtocol?

        init(session: AVCaptureSession) {
            super.init(frame: .zero)
            previewLayer.session      = session
            previewLayer.videoGravity = .resizeAspectFill
            configureRotation(for: session)
        }

        required init?(coder: NSCoder) { fatalError("not used") }

        deinit {
            if let sessionStartObserver {
                NotificationCenter.default.removeObserver(sessionStartObserver)
            }
        }

        /// The video device isn't attached to the session yet when this view is created
        /// (CameraManager configures it asynchronously), so wait for the session to start
        /// running before wiring up the rotation coordinator.
        private func configureRotation(for session: AVCaptureSession) {
            if let device = videoDevice(in: session) {
                startRotationCoordinator(device: device)
                return
            }

            sessionStartObserver = NotificationCenter.default.addObserver(
                forName: AVCaptureSession.didStartRunningNotification,
                object: session,
                queue: .main
            ) { [weak self] _ in
                guard let self, let device = self.videoDevice(in: session) else { return }
                self.startRotationCoordinator(device: device)
                if let observer = self.sessionStartObserver {
                    NotificationCenter.default.removeObserver(observer)
                    self.sessionStartObserver = nil
                }
            }
        }

        private func videoDevice(in session: AVCaptureSession) -> AVCaptureDevice? {
            session.inputs
                .compactMap { $0 as? AVCaptureDeviceInput }
                .first { $0.device.hasMediaType(.video) }?
                .device
        }

        private func startRotationCoordinator(device: AVCaptureDevice) {
            let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
            rotationCoordinator = coordinator

            applyPreviewRotation(coordinator.videoRotationAngleForHorizonLevelPreview)
            rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.new]) { [weak self] _, change in
                guard let angle = change.newValue else { return }
                self?.applyPreviewRotation(angle)
            }
        }

        private func applyPreviewRotation(_ angle: CGFloat) {
            guard
                let connection = previewLayer.connection,
                connection.isVideoRotationAngleSupported(angle)
            else { return }
            connection.videoRotationAngle = angle
        }
    }
}
