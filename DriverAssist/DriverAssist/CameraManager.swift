//
//  CameraManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import AVFoundation
import CoreVideo
import SwiftUI

// MARK: — CameraManager

@MainActor
final class CameraManager: NSObject, ObservableObject {

    /// Called on the main actor for every captured frame.
    var onFrame: ((CVPixelBuffer) -> Void)?

    /// Latest detections to bake into recorded frames. Written from the main actor
    /// (as inference results arrive) and read from `sessionQueue` (as frames are
    /// appended to the recording), so access is guarded by `detectionsLock`.
    nonisolated var currentDetections: [Detection] {
        get { detectionsLock.lock(); defer { detectionsLock.unlock() }; return _currentDetections }
        set { detectionsLock.lock(); _currentDetections = newValue; detectionsLock.unlock() }
    }

    private let detectionsLock = NSLock()
    private nonisolated(unsafe) var _currentDetections: [Detection] = []

    // nonisolated(unsafe): AVCaptureSession and outputs are internally thread-safe
    // and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()
    private nonisolated(unsafe) var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private nonisolated(unsafe) var rotationObservation: NSKeyValueObservation?

    // Recording state. Only ever touched on sessionQueue (configure/startNewRecording
    // run there, and the video data output delegate is dispatched to sessionQueue too).
    private nonisolated(unsafe) var assetWriter: AVAssetWriter?
    private nonisolated(unsafe) var assetWriterInput: AVAssetWriterInput?
    private nonisolated(unsafe) var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?

    func start() {
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

    private nonisolated func configure() {
        guard !session.isRunning else { return }

        // After stop(), inputs/outputs are still attached — just restart.
        if !session.inputs.isEmpty {
            session.startRunning()
            startNewRecording()
            return
        }

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

    /// Creates the asset writer for a new recording. The writer's video input/adaptor
    /// are added lazily once the first frame arrives, since only then do we know the
    /// actual (post-rotation) pixel buffer dimensions.
    private nonisolated func startNewRecording() {
        guard
            assetWriter == nil,
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let url = documents.appendingPathComponent("recording-\(formatter.string(from: Date())).mov")
        let writer = try? AVAssetWriter(outputURL: url, fileType: .mov)
        // This app has no reliable stop point in normal use (the root view never
        // disappears), so the process is typically killed rather than cleanly
        // stopped. Periodic fragments keep the file's moov atom written incrementally
        // so a recording stays playable even if finishWriting() never runs.
        writer?.movieFragmentInterval = CMTime(seconds: 10, preferredTimescale: 600)
        assetWriter = writer
    }

    private nonisolated func finishRecording() {
        guard let writer = assetWriter, let input = assetWriterInput else {
            assetWriter = nil
            assetWriterInput = nil
            pixelBufferAdaptor = nil
            return
        }
        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
    }

    /// Composites the current detections onto `pixelBuffer` and appends the result
    /// to the in-progress recording, so the saved file matches what's on screen.
    private nonisolated func appendRecordingFrame(_ pixelBuffer: CVPixelBuffer, presentationTime: CMTime) {
        guard let writer = assetWriter else { return }

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

            guard writer.canAdd(input) else { return }
            writer.add(input)
            assetWriterInput = input
            pixelBufferAdaptor = adaptor

            writer.startWriting()
            writer.startSession(atSourceTime: presentationTime)
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

        compositeOverlay(from: pixelBuffer, into: outputBuffer, detections: currentDetections)
        adaptor.append(outputBuffer, withPresentationTime: presentationTime)
    }

    private nonisolated func compositeOverlay(
        from source: CVPixelBuffer,
        into destination: CVPixelBuffer,
        detections: [Detection]
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
        let height         = CVPixelBufferGetHeight(destination)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(source)
        let dstBytesPerRow = CVPixelBufferGetBytesPerRow(destination)

        if srcBytesPerRow == dstBytesPerRow {
            memcpy(dstBase, srcBase, srcBytesPerRow * height)
        } else {
            let rowBytes = min(srcBytesPerRow, dstBytesPerRow)
            for row in 0..<height {
                memcpy(dstBase.advanced(by: row * dstBytesPerRow), srcBase.advanced(by: row * srcBytesPerRow), rowBytes)
            }
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

        OverlayRenderer.draw(detections, in: context, size: CGSize(width: width, height: height))
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
                forName: .AVCaptureSessionDidStartRunning,
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
