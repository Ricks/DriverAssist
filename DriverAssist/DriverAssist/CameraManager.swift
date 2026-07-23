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

    // nonisolated(unsafe): AVCaptureSession and AVCaptureVideoDataOutput are internally
    // thread-safe and are always accessed on sessionQueue or before the session starts.
    nonisolated(unsafe) let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "CameraManager.session", qos: .userInitiated)
    private nonisolated(unsafe) let videoOutput = AVCaptureVideoDataOutput()

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
        let box = UncheckedSendableBox(value: session)
        sessionQueue.async {
            box.value.stopRunning()
        }
    }

    private nonisolated func configure() {
        guard !session.isRunning else { return }

        // After stop(), inputs/outputs are still attached — just restart.
        if !session.inputs.isEmpty {
            session.startRunning()
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

        session.commitConfiguration()
        session.startRunning()
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

        init(session: AVCaptureSession) {
            super.init(frame: .zero)
            previewLayer.session      = session
            previewLayer.videoGravity = .resizeAspectFill
        }

        required init?(coder: NSCoder) { fatalError("not used") }
    }
}
