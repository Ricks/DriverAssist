//
//  CameraManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import AVFoundation
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
    /// manual swipe/voice override suspends this until `enableAutoLowLight()` is
    /// called again (voice: "low light auto"), so auto-detection doesn't immediately
    /// fight an explicit override.
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
    /// below for the recording bake.
    @Published private(set) var isStabilizationEnabled = false

    /// Latest detections to bake into recorded frames. Written from the main actor
    /// (as inference results arrive) and read from `sessionQueue` (as frames are
    /// appended to the recording), so access is guarded by `overlayLock`.
    nonisolated var currentDetections: [Detection] {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentDetections }
        set { overlayLock.lock(); _currentDetections = newValue; overlayLock.unlock() }
    }

    /// Current model label ("small"/"nano") to bake into recorded frames, mirroring
    /// `ModelManager.selectedModel` (owned there, synced in from the main actor).
    nonisolated var currentModelLabel: String {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentModelLabel }
        set { overlayLock.lock(); _currentModelLabel = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isLowLightBoostEnabled`, readable from `sessionQueue`
    /// while baking the recording overlay.
    private nonisolated var currentLowLightEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentLowLightEnabled }
        set { overlayLock.lock(); _currentLowLightEnabled = newValue; overlayLock.unlock() }
    }

    /// Current detection-smoothing state ("smoothing on/off") to bake into recorded
    /// frames, mirroring `InferenceEngine.isSmoothingEnabled` (owned there, synced in
    /// from the main actor).
    nonisolated var currentSmoothingEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentSmoothingEnabled }
        set { overlayLock.lock(); _currentSmoothingEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isAutoLowLightEnabled`, readable from `sessionQueue`
    /// while baking the recording overlay.
    private nonisolated var currentAutoLowLightEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentAutoLowLightEnabled }
        set { overlayLock.lock(); _currentAutoLowLightEnabled = newValue; overlayLock.unlock() }
    }

    /// Current two-pass-detection state ("two-pass on/off") to bake into recorded
    /// frames, mirroring `InferenceEngine.isTwoPassEnabled` (owned there, synced in
    /// from the main actor).
    nonisolated var currentTwoPassEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentTwoPassEnabled }
        set { overlayLock.lock(); _currentTwoPassEnabled = newValue; overlayLock.unlock() }
    }

    /// Nonisolated mirror of `isStabilizationEnabled`, readable from `sessionQueue`
    /// while baking the recording overlay (and while applying the mode to the
    /// capture connection in `configure()`/`applyStabilizationMode`).
    nonisolated var currentStabilizationEnabled: Bool {
        get { overlayLock.lock(); defer { overlayLock.unlock() }; return _currentStabilizationEnabled }
        set { overlayLock.lock(); _currentStabilizationEnabled = newValue; overlayLock.unlock() }
    }

    private let overlayLock = NSLock()
    private nonisolated(unsafe) var _currentDetections: [Detection] = []
    private nonisolated(unsafe) var _currentModelLabel: String = ""
    private nonisolated(unsafe) var _currentLowLightEnabled: Bool = false
    private nonisolated(unsafe) var _currentSmoothingEnabled: Bool = false
    private nonisolated(unsafe) var _currentAutoLowLightEnabled: Bool = true
    private nonisolated(unsafe) var _currentTwoPassEnabled: Bool = true
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
    private nonisolated static let probeInterval: CFAbsoluteTime = 2
    private nonisolated static let probeSettleDuration: CFAbsoluteTime = 0.3

    /// 0-255 brightness thresholds (see `sampleAutoLowLightIfDue`). The gap between
    /// them is hysteresis to avoid flicker at the boundary. Starting points — may
    /// need real-world tuning.
    private static let autoLowLightOnLuminance: Double = 50
    private static let autoLowLightOffLuminance: Double = 90

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

    func start() {
        // Requested up front (rather than lazily on the first segment save) so the
        // permission dialog doesn't interrupt an in-progress drive.
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            print("[CameraManager] Photos add-only authorization: \(status.rawValue)")
        }

        Task {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                sessionQueue.async { [weak self] in self?.configure() }
            case .notDetermined:
                let granted = await AVCaptureDevice.requestAccess(for: .video)
                if granted { self.sessionQueue.async { [weak self] in self?.configure() } }
            default:
                break
            }
        }
    }

    func stop() {
        let sessionBox = UncheckedSendableBox(value: session)
        sessionQueue.async { [weak self] in
            self?.finishRecording()
            sessionBox.value.stopRunning()
        }
    }

    /// Toggles the low-light exposure boost, biasing auto-exposure brighter to help
    /// detection in dark scenes (at the cost of more sensor noise). A manual action,
    /// so this suspends auto-detection — see `setLowLightBoost`.
    func toggleLowLightBoost() {
        setLowLightBoost(!isLowLightBoostEnabled)
    }

    /// Sets the low-light exposure boost to an explicit state (swipe gesture, or
    /// voice commands "low light on"/"off"). Suspends auto-detection until
    /// `enableAutoLowLight()` is called again, so the two don't immediately fight.
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

    /// The road ahead is always far away, so autofocus hunting near objects (the
    /// dashboard, a windshield reflection) is pure downside here. Restricting the
    /// *range* autofocus searches to `.far` (while leaving continuous autofocus
    /// running) stops it hunting near, but confirmed on-device it still visibly
    /// hunts *within* the far range — refocusing between, say, a car 15m ahead and
    /// a building 200m ahead — which reads as "going in and out of focus" over a
    /// real drive. So: let it run just long enough to settle on a genuinely sharp
    /// far-field position, then lock there. A blind `lensPosition` guess isn't
    /// guaranteed to land at true infinity focus (a lens's mechanical far stop and
    /// true infinity focus aren't always the same point) — that's what produced
    /// permanently blurry recordings before — but the position autofocus itself
    /// converges on has actually been verified sharp.
    private nonisolated func restrictFocusToFarRange(for device: AVCaptureDevice) {
        guard
            device.isFocusModeSupported(.continuousAutoFocus),
            device.isAutoFocusRangeRestrictionSupported,
            device.isFocusModeSupported(.locked)
        else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.autoFocusRangeRestriction = .far
            device.focusMode = .continuousAutoFocus
        } catch {
            return
        }

        focusSettleObservation = device.observe(\.isAdjustingFocus, options: [.new]) { [weak self] device, change in
            guard change.newValue == false else { return }
            self?.focusSettleObservation = nil
            self?.lockFocus(at: device)
        }
    }

    private nonisolated func lockFocus(at device: AVCaptureDevice) {
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.setFocusModeLocked(lensPosition: device.lensPosition, completionHandler: nil)
        } catch {
            // Device may be mid-reconfiguration; focus stays continuous (still far-restricted).
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
                // BGRA byte order in memory (premultipliedFirst + byteOrder32Little),
                // matching the convention used throughout compositeOverlay.
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
            DebugFileLogger.log("auto-lowlight: luminance=\(luminance) auto=\(self.isAutoLowLightEnabled) boost=\(self.isLowLightBoostEnabled)")
            guard self.isAutoLowLightEnabled else { return }
            if !self.isLowLightBoostEnabled, luminance <= Self.autoLowLightOnLuminance {
                self.applyLowLightState(true)
            } else if self.isLowLightBoostEnabled, luminance >= Self.autoLowLightOffLuminance {
                self.applyLowLightState(false)
            }
        }
    }

    private nonisolated func configure() {
        guard !session.isRunning else { return }

        // After stop(), inputs/outputs are still attached — just restart.
        if !session.inputs.isEmpty {
            session.startRunning()
            startNewRecording()
            return
        }

        registerSessionObservers()
        recoverOrphanedRecordings()

        session.beginConfiguration()
        session.sessionPreset = session.canSetSessionPreset(.hd4K3840x2160) ? .hd4K3840x2160 : .hd1280x720

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
        restrictFocusToFarRange(for: device)

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
        startNewRecording()
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
        let url = documents.appendingPathComponent("recording-\(formatter.string(from: Date())).mov")

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
        }

        guard
            let input = assetWriterInput,
            let adaptor = pixelBufferAdaptor,
            input.isReadyForMoreMediaData,
            let pool = adaptor.pixelBufferPool
        else { return }

        var outputBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBuffer)
        guard let outputBuffer else { return }

        compositeOverlay(
            from: pixelBuffer,
            into: outputBuffer,
            detections: currentDetections,
            modelLabel: currentModelLabel,
            lowLightEnabled: currentLowLightEnabled,
            autoLowLightEnabled: currentAutoLowLightEnabled,
            smoothingEnabled: currentSmoothingEnabled,
            twoPassEnabled: currentTwoPassEnabled,
            stabilizationEnabled: currentStabilizationEnabled
        )
        adaptor.append(outputBuffer, withPresentationTime: presentationTime)
    }

    private nonisolated func compositeOverlay(
        from source: CVPixelBuffer,
        into destination: CVPixelBuffer,
        detections: [Detection],
        modelLabel: String,
        lowLightEnabled: Bool,
        autoLowLightEnabled: Bool,
        smoothingEnabled: Bool,
        twoPassEnabled: Bool,
        stabilizationEnabled: Bool
    ) {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
            CVPixelBufferUnlockBaseAddress(destination, [])
        }

        guard
            let srcBase = CVPixelBufferGetBaseAddress(source),
            let dstBase = CVPixelBufferGetBaseAddress(destination)
        else { return }

        let width          = CVPixelBufferGetWidth(destination)
        let dstHeight      = CVPixelBufferGetHeight(destination)
        let srcHeight      = CVPixelBufferGetHeight(source)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(source)
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(destination)

        // `source` and `destination` come from independent allocations (the live
        // camera frame vs. the writer's pixel buffer pool) and aren't guaranteed to
        // match — e.g. right after backgrounding starts a fresh writer/pool, its
        // buffer can differ slightly from the incoming frame. When they don't match,
        // clear the destination first: `destination` comes from a recycled pool, so
        // otherwise the copy below (clamped to the smaller of the two, to avoid
        // reading past the end of whichever buffer is shorter) only overwrites part
        // of it and leaves a previous frame's pixels — HUD and all — sitting in the
        // rest, which is what made a recording look like it had split into two
        // frames stacked on top of each other.
        if srcHeight != dstHeight || srcBytesPerRow != dstBytesPerRow {
            memset(dstBase, 0, dstBytesPerRow * dstHeight)
        }

        let height   = min(srcHeight, dstHeight)
        let rowBytes = min(srcBytesPerRow, dstBytesPerRow)

        for row in 0..<height {
            memcpy(dstBase.advanced(by: row * dstBytesPerRow), srcBase.advanced(by: row * srcBytesPerRow), rowBytes)
        }

        guard let context = CGContext(
            data: dstBase,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: dstBytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }

        // Flip to a top-left origin so it matches Detection.boundingBox's convention
        // (and matches UIKit's coordinate space for the label text drawn below).
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)

        let size = CGSize(width: width, height: height)
        OverlayRenderer.draw(detections, in: context, size: size)
        OverlayRenderer.drawHUD(
            modelLabel: modelLabel,
            lowLightEnabled: lowLightEnabled,
            autoLowLightEnabled: autoLowLightEnabled,
            smoothingEnabled: smoothingEnabled,
            twoPassEnabled: twoPassEnabled,
            stabilizationEnabled: stabilizationEnabled,
            in: context,
            size: size
        )
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
        appendRecordingFrame(pixelBuffer, presentationTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

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
