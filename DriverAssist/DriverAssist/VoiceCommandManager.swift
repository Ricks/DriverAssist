//
//  VoiceCommandManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/25/26.
//

import AVFoundation
import Speech
import UIKit

/// Listens continuously for a fixed set of spoken commands ("small", "nano", "medium",
/// "low light on", "low light off", "low light auto", "two pass on", "two pass off",
/// "stabilization on", "stabilization off") and reports them via `onCommand`.
///
/// No tracking-level command (e.g. "tracking normal"/"tracking high") for now --
/// TrackingLevel.appearance is a no-op stub until a real on-device ReID model is
/// integrated (see AppearanceEmbedder.swift), so exposing a choice between it and
/// .recovery would present a fake distinction as a real one. The app always runs
/// .recovery; TrackingManager's setTrackingLevel(_:) stays available in code for
/// whenever that changes.
@MainActor
final class VoiceCommandManager: NSObject, ObservableObject {
    enum Command: Equatable {
        case selectModel(DetectorModel)
        case lowLight(Bool)
        case lowLightAuto
        case twoPass(Bool)
        case stabilization(Bool)
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

    private var observersRegistered = false
    /// True while paused for an audio interruption or backgrounding, so foreground/
    /// interruption-ended notifications know whether to resume — as opposed to the
    /// user (or the app) having called `stop()` deliberately.
    private var wasInterrupted = false

    func start() {
        registerLifecycleObservers()
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
        wasInterrupted = false
    }

    /// Recognition silently and permanently stops on any mic interruption (a call,
    /// Siri, Control Center) or backgrounding (screen lock, switching apps) unless
    /// explicitly restarted — `AVAudioEngine`/`SFSpeechRecognitionTask` don't resume
    /// on their own. This restarts listening once the interruption clears or the app
    /// returns to the foreground, mirroring `CameraManager`'s session-recovery observers.
    private func registerLifecycleObservers() {
        guard !observersRegistered else { return }
        observersRegistered = true

        NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: nil
        ) { [weak self] notification in
            guard
                let typeValue = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                let type = AVAudioSession.InterruptionType(rawValue: typeValue)
            else { return }
            switch type {
            case .began:
                Task { @MainActor in self?.pauseForInterruption() }
            case .ended:
                let optionsValue = notification.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
                guard AVAudioSession.InterruptionOptions(rawValue: optionsValue).contains(.shouldResume) else { return }
                Task { @MainActor in self?.resumeAfterInterruption() }
            @unknown default:
                break
            }
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil
        ) { [weak self] _ in
            Task { @MainActor in self?.pauseForInterruption() }
        }

        NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification, object: nil, queue: nil
        ) { [weak self] _ in
            Task { @MainActor in self?.resumeAfterInterruption() }
        }
    }

    private func pauseForInterruption() {
        guard isListening else { return }
        stop()
        wasInterrupted = true
    }

    private func resumeAfterInterruption() {
        guard wasInterrupted else { return }
        wasInterrupted = false
        beginSession()
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
                // Tasks are single-shot — they always end via an error or a final
                // result, and either way something needs to start the next one or
                // listening silently stops for good. `handle` above already restarts
                // when a command matched; the generation check skips a double
                // restart in that case (it will have already moved on).
                if generation == self.listeningGeneration, error != nil || (result?.isFinal ?? false) {
                    self.startRecognitionTask()
                }
            }
        }
    }

    private func handle(transcript: String) {
        let lowered = transcript.lowercased()
        let command: Command?
        if lowered.contains("low light auto") {
            command = .lowLightAuto
        } else if lowered.contains("low light on") {
            command = .lowLight(true)
        } else if lowered.contains("low light off") {
            command = .lowLight(false)
        // "two" reliably transcribes as "to"/"too" here (confirmed on-device: saying
        // "two-pass on" settles as "to pass on", never "two pass on") — accept all
        // three homophones for both directions.
        } else if isTwoPassCommand(lowered, state: "on") {
            command = .twoPass(true)
        } else if isTwoPassCommand(lowered, state: "off") {
            command = .twoPass(false)
        } else if lowered.contains("stabilization on") {
            command = .stabilization(true)
        } else if lowered.contains("stabilization off") {
            command = .stabilization(false)
        } else if lowered.contains("nano") {
            command = .selectModel(.nano)
        } else if lowered.contains("medium") {
            command = .selectModel(.medium)
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

    private func isTwoPassCommand(_ lowered: String, state: String) -> Bool {
        ["two", "to", "too"].contains { lowered.contains("\($0) pass \(state)") }
    }
}
