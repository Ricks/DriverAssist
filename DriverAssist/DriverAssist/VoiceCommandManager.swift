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
/// "stabilization on", "stabilization off", "high resolution", "low resolution",
/// "calibrate pitch") and reports them via `onCommand`.
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
        case calibratePitch
        case highRes(Bool)
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

    /// How much of the current task's accumulated transcript (by Character count)
    /// has already been matched against -- SFSpeechRecognitionTask's transcript only
    /// grows within one task's lifetime, so without tracking this, an old word still
    /// sitting in the string (e.g. "nano" from a minute ago) would keep re-matching
    /// forever and block later commands earlier in the if/else chain (e.g. "small")
    /// from ever being reached again. Reset to 0 whenever a genuinely new task starts
    /// (see `startRecognitionTask`).
    private var matchedTranscriptLength: Int = 0

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
            // .record (not .playAndRecord) silently drops .duckOthers -- Apple only
            // honors that option for .ambient/.soloAmbient/.playback/.playAndRecord --
            // so this was fully interrupting other apps' audio (e.g. Audible over
            // CarPlay) instead of ducking it, for as long as voice listening was
            // active, which in practice means the whole drive.
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
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
    /// pause), so a fresh one is started once the current one ends. Deliberately
    /// NOT restarted just because a command matched anymore -- see the comment in
    /// `handle` on why that was actually the source of a multi-second lag.
    private func startRecognitionTask() {
        guard let speechRecognizer, speechRecognizer.isAvailable else { return }

        listeningGeneration += 1
        let generation = listeningGeneration
        matchedTranscriptLength = 0

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
                    // Logged raw (partial and final alike) so a slow-to-match command
                    // like the recent multi-second "low light auto" delay can actually
                    // be diagnosed from real timing/transcript data next time, instead
                    // of guessed at -- same reasoning as this app's other debug logging.
                    DebugFileLogger.log(
                        "voice: isFinal=\(result.isFinal) transcript=\"\(result.bestTranscription.formattedString)\""
                    )
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
        // Only match against the portion of the transcript not already matched --
        // see `matchedTranscriptLength`'s doc comment. A transcript shorter than
        // what's on record means a new task actually started without going through
        // `startRecognitionTask` resetting it first (shouldn't normally happen, but
        // fail safe rather than crash on the negative-range dropFirst below).
        guard transcript.count >= matchedTranscriptLength else {
            matchedTranscriptLength = 0
            return
        }
        let newPortion = String(transcript.dropFirst(matchedTranscriptLength))
        // Hyphens normalized to spaces -- Apple's dictation renders common compound
        // terms like "low-light" hyphenated (matching this app's own HUD text), which
        // would otherwise never match the space-separated phrases checked below.
        let lowered = newPortion.lowercased().replacingOccurrences(of: "-", with: " ")
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
        // Voice equivalent of the vertical swipe (see ContentView). "resolution"
        // rather than "res" -- confirmed from real on-device logs that "res" as a
        // bare word is acoustically ambiguous enough that the recognizer never once
        // transcribed it correctly (got "Rez"/"IRS"/"Iris"/"Eris" instead, repeatedly,
        // across a dozen+ tries); "res" kept as a fallback in case it's ever heard
        // correctly, but "resolution" is the reliable phrase.
        } else if lowered.contains("high resolution") || lowered.contains("high res") {
            command = .highRes(true)
        } else if lowered.contains("low resolution") || lowered.contains("low res") {
            command = .highRes(false)
        } else if lowered.contains("calibrate pitch") {
            command = .calibratePitch
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

        // Deliberately NOT tearing down/restarting the recognition task here anymore
        // -- confirmed from real on-device logs that doing so cost 5-12 seconds of
        // dead air before the new task produced its next partial result (on-device
        // SFSpeechRecognitionTask cold-start), which was the actual source of the
        // "took several seconds" lag. The task keeps running; `matchedTranscriptLength`
        // above is what prevents this same phrase from re-firing on every subsequent
        // partial result instead.
        matchedTranscriptLength = transcript.count
        DebugFileLogger.log("voice: MATCHED \(command)")
        onCommand?(command)
    }

    private func isTwoPassCommand(_ lowered: String, state: String) -> Bool {
        ["two", "to", "too"].contains { lowered.contains("\($0) pass \(state)") }
    }
}
