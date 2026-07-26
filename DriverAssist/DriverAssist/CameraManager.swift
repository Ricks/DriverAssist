//
//  CameraManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import AVFoundation
import CoreVideo
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

    private let overlayLock = NSLock()
    private nonisolated(unsafe) var _currentDetections: [Detection] = []
    private nonisolated(unsafe) var _currentModelLabel: String = ""
    private nonisolated(unsafe) var _currentLowLightEnabled: Bool = false
    private nonisolated(unsafe) var _currentSmoothingEnabled: Bool = true
    private nonisolated(unsafe) var _currentAutoLowLightEnabled: Bool = true

    // nonisolated(unsafe): AVCaptureSession and outputs are internally thread-safe
    // and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()
    private nonisolated(unsafe) var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private nonisolated(unsafe) var rotationObservation: NSKeyValueObservation?
    private nonisolated(unsafe) var captureDevice: AVCaptureDevice?

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
    private static let probeInterval: CFAbsoluteTime = 2
    private static let probeSettleDuration: CFAbsoluteTime = 0.3

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
    private nonisolated(unsafe) var segmentStartedAt: Date?
    private nonisolated(unsafe) var observersRegistered = false
    private let backgroundTaskLock = NSLock()
    private nonisolated(unsafe) var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid

    /// Segments are rotated out to the Photos library at this interval so a long
    /// session's footage is durably saved well before the app might get killed,
    /// rather than living only as one giant file in the app's own sandbox. Kept
    /// short (rather than, say, 10+ minutes) specifically to shrink the one gap that
    /// can't be closed outright: if the app is deleted or the device wiped before its
    /// next launch, whatever segment hasn't yet rotated out is lost for good.
    private let segmentDuration: TimeInterval = 120

    /// Below this, the HUD warns so the user can free up space before recording
    /// actually starts failing, instead of only finding out after the fact.
    private let lowStorageThresholdBytes: Int64 = 1_000_000_000

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
    }

    /// Hands control back to the auto-detector (voice: "low light auto"). The next
    /// captured frame will re-evaluate the scene (see `sampleAutoLowLightIfDue`).
    func enableAutoLowLight() {
        isAutoLowLightEnabled = true
        currentAutoLowLightEnabled = true
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
        session.sessionPreset = .hd1280x720

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

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

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
        backgroundTaskLock.lock()
        guard backgroundTaskID == .invalid else {
            backgroundTaskLock.unlock()
            return // a flush is already in flight
        }
        backgroundTaskLock.unlock()

        let taskID = UIApplication.shared.beginBackgroundTask(withName: "DriverAssist.flushRecording") { [weak self] in
            // Expiration handler: out of extra time. End the task so the OS doesn't
            // kill the whole process for overrunning it — the fragmented file on
            // disk is still valid either way and will be picked up by
            // recoverOrphanedRecordings on next launch.
            self?.endBackgroundTaskIfNeeded()
        }
        guard taskID != .invalid else { return }

        backgroundTaskLock.lock()
        backgroundTaskID = taskID
        backgroundTaskLock.unlock()

        sessionQueue.async { [weak self] in
            self?.finishRecording()
            self?.startNewRecording()
            self?.endBackgroundTaskIfNeeded()
        }
    }

    private nonisolated func endBackgroundTaskIfNeeded() {
        backgroundTaskLock.lock()
        let taskID = backgroundTaskID
        backgroundTaskID = .invalid
        backgroundTaskLock.unlock()

        guard taskID != .invalid else { return }
        UIApplication.shared.endBackgroundTask(taskID)
    }

    /// Sweeps any `recording-*.mov` left behind by a previous run (e.g. the process
    /// was killed mid-segment, which is the normal case — see `startNewRecording`)
    /// into the Photos library. Fragmented `.mov` files stay structurally valid
    /// without a clean `finishWriting()`, so these are safe to import as-is; this is
    /// what guarantees a session's footage ends up saved regardless of how the app
    /// was terminated, not just on a clean stop.
    private nonisolated func recoverOrphanedRecordings() {
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let files = try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: nil)
        else { return }

        for file in files where file.lastPathComponent.hasPrefix("recording-") && file.pathExtension == "mov" {
            saveToPhotoLibrary(file)
        }
    }

    /// Keeps the capture outputs' video rotation in sync with the device's physical
    /// orientation, since the app-level video orientation APIs (`AVCaptureConnection
    /// .videoOrientation`) are deprecated in favor of rotation angles as of iOS 17.
    private nonisolated func configureRotationCoordinator(for device: AVCaptureDevice) {
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
        rotationCoordinator = coordinator

        applyCaptureRotation(coordinator.videoRotationAngleForHorizonLevelCapture)
        rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.new]) { [weak self] _, change in
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

    /// Creates the asset writer for a new recording segment. The writer's video
    /// input/adaptor are added lazily once the first frame arrives, since only then
    /// do we know the actual (post-rotation) pixel buffer dimensions.
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
            // Segments are capped at `segmentDuration` and rotated out to Photos, so
            // this fragment interval only needs to protect against the process being
            // killed mid-segment (fragmented .mov files stay playable without a
            // clean finishWriting()).
            writer.movieFragmentInterval = CMTime(seconds: 10, preferredTimescale: 600)
            assetWriter = writer
            currentSegmentURL = url
            segmentStartedAt = Date()
        } catch {
            print("[CameraManager] failed to create asset writer: \(error.localizedDescription)")
            setRecordingActive(false)
            scheduleRecordingRetry()
        }
    }

    /// Checked once per segment (roughly every 2 minutes) rather than per-frame,
    /// since it's just an early warning and doesn't need finer granularity.
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

    /// Tears down a segment that failed mid-write (e.g. the writer transitioned to
    /// `.failed`, commonly from disk space running out) and schedules a retry so the
    /// rest of the drive still gets recorded once/if the underlying issue clears.
    private nonisolated func failCurrentRecording() {
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
        currentSegmentURL = nil
        segmentStartedAt = nil
        setRecordingActive(false)
        scheduleRecordingRetry()
    }

    private nonisolated func setRecordingActive(_ active: Bool) {
        Task { @MainActor [weak self] in
            self?.isRecording = active
        }
    }

    /// Ends the current segment and starts a fresh one, keeping recording continuous
    /// across the rotation (see `appendRecordingFrame`, which calls this then falls
    /// straight through into setting up the new segment with the same frame).
    private nonisolated func rotateSegment() {
        finishCurrentSegmentAndSave()
        startNewRecording()
    }

    /// Finishes the current segment without blocking `sessionQueue` (frames keep
    /// arriving on this same queue during a routine rotation) and saves it to Photos
    /// once finalized.
    private nonisolated func finishCurrentSegmentAndSave() {
        guard let writer = assetWriter, let input = assetWriterInput else {
            assetWriter = nil
            assetWriterInput = nil
            pixelBufferAdaptor = nil
            currentSegmentURL = nil
            segmentStartedAt = nil
            return
        }
        let url = currentSegmentURL
        input.markAsFinished()
        writer.finishWriting { [weak self] in
            guard writer.status == .completed else {
                print("[CameraManager] segment finish failed: \(writer.error?.localizedDescription ?? "unknown")")
                return
            }
            if let url { self?.saveToPhotoLibrary(url) }
        }
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
        currentSegmentURL = nil
        segmentStartedAt = nil
    }

    /// Finishes the final segment on a clean stop, blocking until both the write and
    /// the Photos save complete — safe here since no more frames are coming.
    private nonisolated func finishRecording() {
        guard let writer = assetWriter, let input = assetWriterInput else {
            assetWriter = nil
            assetWriterInput = nil
            pixelBufferAdaptor = nil
            currentSegmentURL = nil
            segmentStartedAt = nil
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
        segmentStartedAt = nil

        guard writer.status == .completed, let url else { return }
        let saveSemaphore = DispatchSemaphore(value: 0)
        saveToPhotoLibrary(url) { saveSemaphore.signal() }
        saveSemaphore.wait()
    }

    /// Imports a finished recording into the Photos library and removes the local
    /// copy on success, so completed footage doesn't sit invisibly in app storage
    /// (where the last incident's evidence was found) and doesn't grow unbounded.
    /// Leaves the file in place on failure (including permission denial) so it can
    /// still be recovered manually via Files, and so a later sweep can retry it.
    private nonisolated func saveToPhotoLibrary(_ url: URL, completion: (() -> Void)? = nil) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                print("[CameraManager] Photos permission not granted (\(status.rawValue)); leaving \(url.lastPathComponent) in app storage")
                completion?()
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
                completion?()
            }
        }
    }

    /// Composites the current detections onto `pixelBuffer` and appends the result
    /// to the in-progress recording, so the saved file matches what's on screen.
    private nonisolated func appendRecordingFrame(_ pixelBuffer: CVPixelBuffer, presentationTime: CMTime) {
        if let segmentStartedAt, Date().timeIntervalSince(segmentStartedAt) >= segmentDuration {
            rotateSegment()
        }

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
            smoothingEnabled: currentSmoothingEnabled
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
        smoothingEnabled: Bool
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
        // `source` and `destination` come from independent allocations (the live
        // camera frame vs. the writer's pixel buffer pool) and aren't guaranteed to
        // match — a segment-rotation boundary in particular can hand this a freshly
        // (re)created pool whose buffer differs slightly from the incoming frame.
        // Clamping to the smaller of the two avoids reading past the end of
        // whichever buffer is shorter, which previously crashed the whole session.
        let height         = min(CVPixelBufferGetHeight(source), CVPixelBufferGetHeight(destination))
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(source)
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(destination)
        let rowBytes       = min(srcBytesPerRow, dstBytesPerRow)

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
