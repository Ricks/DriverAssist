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

    // nonisolated(unsafe): AVCaptureSession and outputs are internally thread-safe
    // and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()
    private nonisolated(unsafe) let movieOutput = AVCaptureMovieFileOutput()
    private nonisolated(unsafe) var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private nonisolated(unsafe) var rotationObservation: NSKeyValueObservation?

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
        let movieBox   = UncheckedSendableBox(value: movieOutput)
        sessionQueue.async {
            if movieBox.value.isRecording { movieBox.value.stopRecording() }
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

        if session.canAddOutput(movieOutput) {
            session.addOutput(movieOutput)
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
        for output in [videoOutput as AVCaptureOutput, movieOutput as AVCaptureOutput] {
            guard
                let connection = output.connection(with: .video),
                connection.isVideoRotationAngleSupported(angle)
            else { continue }
            connection.videoRotationAngle = angle
        }
    }

    private nonisolated func startNewRecording() {
        guard
            !movieOutput.isRecording,
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let url = documents.appendingPathComponent("recording-\(formatter.string(from: Date())).mov")
        movieOutput.startRecording(to: url, recordingDelegate: self)
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
        let box = UncheckedSendableBox(value: pixelBuffer)
        Task { @MainActor [weak self] in
            self?.onFrame?(box.value)
        }
    }
}

// MARK: — Recording delegate

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didStartRecordingTo fileURL: URL,
        from connections: [AVCaptureConnection]
    ) {}

    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {}
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
