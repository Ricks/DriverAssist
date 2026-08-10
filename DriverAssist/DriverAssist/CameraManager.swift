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
    /// below for applying the mode to the capture connection.
    @Published private(set) var isStabilizationEnabled = false

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

    private let overlayLock = NSLock()
    private nonisolated(unsafe) var _currentLowLightEnabled: Bool = false
    private nonisolated(unsafe) var _currentAutoLowLightEnabled: Bool = true
    private nonisolated(unsafe) var _currentStabilizationEnabled: Bool = false

    // nonisolated(unsafe): AVCaptureSession and outputs are internally thread-safe
    // and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()
    private nonisolated(unsafe) var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private nonisolated(unsafe) var rotationObservation: NSKeyValueObservation?
    private nonisolated(unsafe) var captureDevice: AVCaptureDevice?
    private nonisolated(unsafe) var focusSettleObservation: NSKeyValueObservation?
    /// Identifies which `settleAndLockFarFieldFocus` call is the CURRENT
    /// one -- see that function's doc comment for why `focusSettleObservation
    /// != nil` alone isn't a sufficient guard against a stale/delayed
    /// notification from a PREVIOUS call re-firing after a newer one has
    /// already started.
    private nonisolated(unsafe) var currentFocusSettleToken: UUID?

    // Periodic focus recalibration (see `recalibrateFocusIfDue`): a one-shot
    // settle-then-lock can catch a bad moment -- a dark scene, a close
    // object, still sitting in a driveway -- and then stay wrong for an
    // entire drive with no way to recover, which is what happened on a real
    // night drive. Bounding how long a bad lock can last by periodically
    // re-running the same cycle, the same principle as the low-light
    // auto-detector re-sampling instead of deciding once and never
    // revisiting it. RESTORED 2026-08-09 alongside the settle-then-lock this
    // depends on -- see `settleAndLockFarFieldFocus`'s doc comment for why
    // both are safe to bring back now.
    private nonisolated(unsafe) var lastFocusLockTime: CFAbsoluteTime = 0
    private nonisolated static let focusRecalibrationInterval: CFAbsoluteTime = 60
    private nonisolated static let maxFocusSettleWait: CFAbsoluteTime = 5

    /// The last lens position a lock actually settled on with confidence. A
    /// timed-out (unsettled) recalibration re-locks to this instead of
    /// committing to whatever uncertain position the lens happened to be
    /// hunting through -- otherwise a dark/close-range recalibration with
    /// nothing confident to focus on progressively drifts the lock worse
    /// with every retry.
    private nonisolated(unsafe) var lastGoodLensPosition: Float?
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
    private static let autoLowLightOnLuminance: Double = 50
    private static let autoLowLightOffLuminance: Double = 90

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

    private static let lowLightBoostDefaultsKey = "settings.lowLightBoostEnabled"
    private static let stabilizationDefaultsKey = "settings.stabilizationEnabled"

    override init() {
        super.init()
        let defaults = UserDefaults.standard
        // isAutoLowLightEnabled is deliberately NOT persisted — every launch starts
        // back in auto regardless of whatever manual override was last in effect, so
        // a prior drive's "low light off" doesn't silently carry over and suppress
        // auto-detection on a new one.
        if defaults.object(forKey: Self.lowLightBoostDefaultsKey) != nil {
            isLowLightBoostEnabled = defaults.bool(forKey: Self.lowLightBoostDefaultsKey)
        }
        if defaults.object(forKey: Self.stabilizationDefaultsKey) != nil {
            isStabilizationEnabled = defaults.bool(forKey: Self.stabilizationDefaultsKey)
        }
        // Keep the nonisolated mirrors in sync with what was just loaded — `configure()`
        // applies the boost/stabilization to the actual capture device from these once
        // it's available.
        currentAutoLowLightEnabled = isAutoLowLightEnabled
        currentLowLightEnabled = isLowLightBoostEnabled
        currentStabilizationEnabled = isStabilizationEnabled
    }

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
                sessionQueue.async { [weak self] in self?.configure(startRecording: recording) }
            case .notDetermined:
                let granted = await AVCaptureDevice.requestAccess(for: .video)
                if granted { self.sessionQueue.async { [weak self] in self?.configure(startRecording: recording) } }
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
                DispatchQueue.main.async { completion(false) }
                return
            }
            self.settleAndLockFarFieldFocus(device: device) { [weak self] in
                guard let self else { return }
                self.startNewRecording()
                DispatchQueue.main.async { completion(true) }
            }
        }
    }

    /// Finishes and saves the calibration clip (to Photos, same as a normal
    /// recording), runs tape-mark detection against the last captured
    /// frame, then fully tears the session down -- see this section's
    /// file-level comment for why the teardown matters. `completion` fires
    /// on the main queue with the detected tape-mark count (0 if no frame
    /// was captured at all) -- the caller compares it against however many
    /// were actually placed and decides pass/retry, and can also use the
    /// call landing as "teardown is complete, safe to restart the normal
    /// preview-only session" (needed on the configuring screen, which would
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

    /// Restricts autofocus to the far range, then waits for it to settle on
    /// a genuinely sharp position before locking there. Shared by two
    /// callers: the driving session's start-of-drive focus (re-run
    /// periodically, see `recalibrateFocusIfDue`) and
    /// `startCalibrationRecording`'s one-off lock.
    ///
    /// HISTORY: used to run unconditionally at drive start, then got
    /// removed 2026-08-08 after appearing to cause a multi-second freeze on
    /// Lock Settings -- the on-device debug log showed `captureOutput`
    /// itself stopped firing for the whole settle window, which looked
    /// damning. CONFIRMED 2026-08-09 that diagnosis was wrong: even after
    /// fully removing this machinery, the freeze persisted unchanged. The
    /// real cause was `configuringScreen`/`drivingScreen` each creating
    /// their own `CameraPreviewView`, tearing down and re-attaching a
    /// second `AVCaptureVideoPreviewLayer` to the same running session on
    /// the phase transition -- fixed separately by hoisting
    /// `CameraPreviewView` to the top-level view, shared across that
    /// transition. Restoring this is therefore safe: the thing it was
    /// blamed for has a different, already-fixed cause. Its own real
    /// downside is unrelated and separately documented: confirmed on a
    /// real night drive to sometimes lock onto a bad moment (dark scene,
    /// still in the driveway) and stay wrong for the rest of the drive --
    /// see `recalibrateFocusIfDue` for how that's bounded. Also confirmed
    /// 2026-08-09 (tape-mark calibration clip): the plain `.far`-restriction
    /// -only approach this replaced doesn't reliably keep continuous AF off
    /// a large, close, high-contrast object (e.g. the dash) -- locking
    /// removes that risk instead of just leaving it transient.
    ///
    /// `completion` fires exactly once per call, whether focus settled,
    /// timed out, or the device doesn't support the required modes at all.
    ///
    /// *** Guards against stale re-firing with a unique token, not just
    /// "is focusSettleObservation non-nil" -- CONFIRMED bug 2026-08-09:
    /// during the 3-round distance calibration, round 0's completion
    /// re-fired minutes after round 1 had already started its own settle,
    /// re-running round 0's whole chain a second time (a second, stale
    /// `stopCalibrationRecording` call for a round the UI had already moved
    /// past). Root cause: `setFocusModeLocked` inside `finish` can itself
    /// briefly perturb `isAdjustingFocus`, and/or AVFoundation's KVO
    /// delivery isn't guaranteed to happen on `sessionQueue` -- either way,
    /// a notification tied to THIS call could still arrive after
    /// `focusSettleObservation` had already been reassigned to a NEWER
    /// call's observation, so the old `!= nil` check passed incorrectly
    /// (it was checking "is *some* settle active", not "is *this* one still
    /// current"). A per-call UUID token closes that gap: a stale
    /// notification's captured token can never match a newer call's token.
    private nonisolated func settleAndLockFarFieldFocus(
        device: AVCaptureDevice,
        completion: @escaping @Sendable () -> Void
    ) {
        guard
            device.isFocusModeSupported(.continuousAutoFocus),
            device.isAutoFocusRangeRestrictionSupported,
            device.isFocusModeSupported(.locked)
        else {
            completion()
            return
        }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.autoFocusRangeRestriction = .far
            device.focusMode = .continuousAutoFocus
        } catch {
            completion()
            return
        }

        let token = UUID()
        currentFocusSettleToken = token
        DebugFileLogger.log("focus: settling")
        let settleStart = CFAbsoluteTimeGetCurrent()
        let finish: @Sendable (Bool) -> Void = { [weak self] settled in
            guard let self else {
                completion()
                return
            }
            guard self.currentFocusSettleToken == token else {
                // Stale -- a newer settle call has already superseded this
                // one. Not this call's completion to fire.
                return
            }
            self.focusSettleObservation = nil
            self.currentFocusSettleToken = nil
            self.lastFocusLockTime = CFAbsoluteTimeGetCurrent()

            let lockPosition: Float
            if settled {
                lockPosition = device.lensPosition
                self.lastGoodLensPosition = lockPosition
                DebugFileLogger.log(String(
                    format: "focus: settled after %.2fs at lensPosition=%.3f",
                    CFAbsoluteTimeGetCurrent() - settleStart, lockPosition
                ))
            } else if let goodPosition = self.lastGoodLensPosition {
                lockPosition = goodPosition
                DebugFileLogger.log(String(
                    format: "focus: settle timed out after %.0fs, keeping previous good lensPosition=%.3f instead of unconfirmed=%.3f",
                    Self.maxFocusSettleWait, goodPosition, device.lensPosition
                ))
            } else {
                lockPosition = device.lensPosition
                DebugFileLogger.log(String(
                    format: "focus: settle timed out after %.0fs, no prior good lock, locking at lensPosition=%.3f anyway",
                    Self.maxFocusSettleWait, lockPosition
                ))
            }
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.setFocusModeLocked(lensPosition: lockPosition, completionHandler: nil)
            } catch {
                // Device may be mid-reconfiguration; focus stays continuous
                // (still far-restricted), not ideal but not broken either.
            }
            completion()
        }

        focusSettleObservation = device.observe(\.isAdjustingFocus, options: [.new]) { [weak self] _, change in
            guard change.newValue == false, self?.currentFocusSettleToken == token else { return }
            finish(true)
        }
        sessionQueue.asyncAfter(deadline: .now() + Self.maxFocusSettleWait) { [weak self] in
            guard self?.currentFocusSettleToken == token else { return }
            finish(false)
        }
    }

    /// Re-runs the settle-then-lock focus roughly every
    /// `focusRecalibrationInterval` -- called from every captured frame,
    /// matching how `sampleAutoLowLightIfDue` self-throttles instead of
    /// needing its own timer. Bounds how long a bad one-shot lock can stay
    /// wrong for -- see `settleAndLockFarFieldFocus`'s doc comment.
    private nonisolated func recalibrateFocusIfDue() {
        guard
            let device = captureDevice,
            focusSettleObservation == nil,
            CFAbsoluteTimeGetCurrent() - lastFocusLockTime >= Self.focusRecalibrationInterval
        else { return }
        settleAndLockFarFieldFocus(device: device) {}
    }

    /// Sets the low-light exposure boost to an explicit state (voice commands "low
    /// light on"/"off"). Suspends auto-detection until `enableAutoLowLight()` is
    /// called again, so the two don't immediately fight.
    func setLowLightBoost(_ enabled: Bool) {
        isAutoLowLightEnabled = false
        currentAutoLowLightEnabled = false
        applyLowLightState(enabled)
        UserDefaults.standard.set(enabled, forKey: Self.lowLightBoostDefaultsKey)
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
        UserDefaults.standard.set(enabled, forKey: Self.stabilizationDefaultsKey)
        sessionQueue.async { [weak self] in self?.applyStabilizationMode(enabled: enabled) }
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

    /// Locks capture to a fixed, lower frame rate — half the ISP/encode work of the
    /// default 30fps, on top of the 4x cut from dropping to 1080p (see `configure`).
    /// The `isBusy` gate already discards most frames at 30fps anyway once inference
    /// can't keep up, so this mostly stops paying full capture/encode cost for frames
    /// that would've been dropped regardless.
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
        if settleFocus {
            settleAndLockFarFieldFocus(device: device) {}
        }
        restrictMaxExposureDuration(for: device)
        restrictFrameRate(for: device, to: 15)

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
            print("[CameraManager] session interrupted: \(String(describing: reason))")
            self?.setRecordingActive(false)
        }

        NotificationCenter.default.addObserver(
            forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil
        ) { _ in
            print("[CameraManager] session interruption ended")
        }

        NotificationCenter.default.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil
        ) { [weak self] notification in
            guard let self else { return }
            let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
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
