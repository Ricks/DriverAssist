//
//  VoiceCommandManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/25/26.
//

import AVFoundation
import Speech

/// Listens continuously for a fixed set of spoken commands ("small", "nano",
/// "low light on", "low light off") and reports them via `onCommand`.
@MainActor
final class VoiceCommandManager: NSObject, ObservableObject {
    enum Command: Equatable {
        case selectModel(DetectorModel)
        case lowLight(Bool)
    }

    var onCommand: ((Command) -> Void)?

    @Published private(set) var isListening = false
    @Published private(set) var authorizationDenied = false

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()

    // Read from the input node's tap callback (an audio-engine thread), written
    // from the main actor — SFSpeechAudioBufferRecognitionRequest.append(_:) is
    // documented as safe to call concurrently with request setup.
    private nonisolated(unsafe) var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?

    private var recognitionTask: SFSpeechRecognitionTask?
    private var tapInstalled = false

    /// Bumped on every (re)start so stale completion callbacks from a just-cancelled
    /// task — which fire asynchronously — can recognize they're superseded and no-op.
    private var listeningGeneration = 0

    func start() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard status == .authorized else {
                    self.authorizationDenied = true
                    return
                }
                self.requestMicrophoneAndBeginSession()
            }
        }
    }

    private func requestMicrophoneAndBeginSession() {
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard granted else {
                    self.authorizationDenied = true
                    return
                }
                self.beginSession()
            }
        }
    }

    func stop() {
        listeningGeneration += 1
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.stop()
        isListening = false
    }

    private func beginSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            return
        }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        tapInstalled = true

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            return
        }

        isListening = true
        startRecognitionTask()
    }

    /// Speech recognition tasks are single-shot (they end after ~1 minute or a
    /// pause), so a fresh one is started after every result — including right
    /// after a command fires, which also clears the accumulated transcript so
    /// the same phrase can't immediately re-trigger.
    private func startRecognitionTask() {
        guard let speechRecognizer, speechRecognizer.isAvailable else { return }

        listeningGeneration += 1
        let generation = listeningGeneration

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if speechRecognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        recognitionRequest = request

        recognitionTask = speechRecognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self, generation == self.listeningGeneration else { return }
                if let result {
                    self.handle(transcript: result.bestTranscription.formattedString)
                }
                if error != nil {
                    self.startRecognitionTask()
                }
            }
        }
    }

    private func handle(transcript: String) {
        let lowered = transcript.lowercased()
        let command: Command?
        if lowered.contains("low light on") {
            command = .lowLight(true)
        } else if lowered.contains("low light off") {
            command = .lowLight(false)
        } else if lowered.contains("nano") {
            command = .selectModel(.nano)
        } else if lowered.contains("small") {
            command = .selectModel(.small)
        } else {
            command = nil
        }
        guard let command else { return }

        onCommand?(command)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        startRecognitionTask()
    }
}
