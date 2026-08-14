//
//  VoiceTestView.swift
//  DriverAssist
//
//  Standalone scratch view for auditioning AVSpeechSynthesizer voices/accents
//  for the verbal-warnings feature -- not wired into the real app flow.
//  DriverAssistApp.swift's WindowGroup is temporarily pointed at this instead
//  of ContentView() to run it; swap that back once done experimenting.
//

import SwiftUI
import AVFoundation

/// One AVSpeechSynthesisVoice, plus the human-readable accent name derived
/// from its language code -- Apple's `AVSpeechSynthesisVoice.language` is a
/// BCP-47 code (e.g. "en-IE"), not a friendly name, so this maps the ones
/// Apple actually ships English voices for.
private struct VoiceOption: Identifiable {
    let voice: AVSpeechSynthesisVoice
    var id: String { voice.identifier }
    var accent: String { Self.accentName(for: voice.language) }
    var genderLabel: String {
        switch voice.gender {
        case .male: return "Male"
        case .female: return "Female"
        default: return "Unspecified"
        }
    }
    var qualityLabel: String {
        switch voice.quality {
        case .premium: return "Premium"
        case .enhanced: return "Enhanced"
        default: return "Standard"
        }
    }
    var isNovelty: Bool {
        if #available(iOS 17.0, *) {
            return voice.voiceTraits.contains(.isNoveltyVoice)
        }
        return false
    }

    private static func accentName(for language: String) -> String {
        let known: [String: String] = [
            "en-US": "American", "en-GB": "British", "en-AU": "Australian",
            "en-IE": "Irish", "en-ZA": "South African", "en-IN": "Indian",
            "en-SG": "Singaporean", "en-PH": "Filipino", "en-NZ": "New Zealand",
            "en-CA": "Canadian",
        ]
        return known[language] ?? language
    }
}

@MainActor
private final class VoiceTestModel: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published var accents: [String] = []
    @Published var selectedAccent: String = "" { didSet { rebuildGenderOptions() } }
    @Published var genders: [String] = []
    @Published var selectedGender: String = "" { didSet { rebuildVoiceOptions() } }
    @Published var voicesForSelection: [VoiceOption] = []
    @Published var selectedVoiceID: String = ""
    @Published var rate: Float = AVSpeechUtteranceDefaultSpeechRate
    @Published var pitch: Float = 1.0
    @Published var isSpeaking = false

    private var allVoices: [VoiceOption] = []
    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
        refreshVoices()
    }

    /// Re-scans installed voices -- `AVSpeechSynthesisVoice.speechVoices()`
    /// only reflects what's downloaded AT CALL TIME, so a voice downloaded
    /// via Settings while this view is already open won't appear until this
    /// runs again. Preserves the current accent/gender/SPECIFIC VOICE
    /// selection where still valid, rather than resetting to defaults.
    ///
    /// CONFIRMED bug 2026-08-12: the specific-voice restoration was missing
    /// entirely -- `selectedAccent`'s didSet cascades into
    /// `rebuildGenderOptions()`, which unconditionally resets
    /// `selectedGender` to "Female", whose own didSet cascades into
    /// `rebuildVoiceOptions()`, which unconditionally resets
    /// `selectedVoiceID` to the alphabetically-first match. Since this runs
    /// automatically on every return from Settings (see `scenePhase`
    /// handling in the view), downloading a Premium voice and switching
    /// back silently reverted the selection to whatever sorted first for
    /// that name -- typically the Standard-quality version of the SAME
    /// voice, which is exactly why it sounded like Premium made things
    /// *more* robotic instead of less: the app was quietly still playing
    /// Standard the whole time.
    func refreshVoices() {
        let previousAccent = selectedAccent
        let previousGender = selectedGender
        let previousVoiceID = selectedVoiceID
        allVoices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix("en") }
            .map { VoiceOption(voice: $0) }
        accents = Array(Set(allVoices.map { $0.accent })).sorted()
        selectedAccent = accents.contains(previousAccent) ? previousAccent : (accents.first(where: { $0 == "American" }) ?? accents.first ?? "")
        if genders.contains(previousGender) {
            selectedGender = previousGender
        }
        // Explicit final restore, after the didSet cascade above has already
        // run and reset things to defaults -- this is the step that was
        // missing before.
        if voicesForSelection.contains(where: { $0.id == previousVoiceID }) {
            selectedVoiceID = previousVoiceID
        }
    }

    private func rebuildGenderOptions() {
        let matches = allVoices.filter { $0.accent == selectedAccent }
        genders = Array(Set(matches.map { $0.genderLabel })).sorted()
        selectedGender = genders.first(where: { $0 == "Female" }) ?? genders.first ?? ""
    }

    private func rebuildVoiceOptions() {
        voicesForSelection = allVoices
            .filter { $0.accent == selectedAccent && $0.genderLabel == selectedGender }
            .sorted { $0.voice.name < $1.voice.name }
        selectedVoiceID = voicesForSelection.first?.id ?? ""
    }

    /// `.voicePrompt` mode + `.duckOthers` mirrors exactly how turn-by-turn
    /// navigation apps interject over an audiobook/podcast/music -- lower
    /// (or, with `.interruptSpokenAudioAndMixWithOthers`, pause) other audio
    /// only while this utterance plays, then hand the session back.
    func speak(_ text: String) {
        guard !text.isEmpty, let option = voicesForSelection.first(where: { $0.id == selectedVoiceID }) else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playback, mode: .voicePrompt, options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers])
            try session.setActive(true)
        } catch {
            print("[VoiceTest] audio session error: \(error)")
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = option.voice
        utterance.rate = rate
        utterance.pitchMultiplier = pitch
        synthesizer.speak(utterance)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor in self.isSpeaking = true }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        }
    }
}

struct VoiceTestView: View {
    @StateObject private var model = VoiceTestModel()
    @State private var customText = ""
    @Environment(\.scenePhase) private var scenePhase

    private let presetPhrases = [
        "Too bloody close mate",
        "Ped crossing from right, 30 meters",
        "Cyclist ahead",
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Voice warning test bench")
                    .font(.title2.bold())

                groupBox("Preset phrases") {
                    ForEach(presetPhrases, id: \.self) { phrase in
                        Button {
                            model.speak(phrase)
                        } label: {
                            HStack {
                                Text(phrase)
                                Spacer()
                                Image(systemName: "speaker.wave.2.fill")
                            }
                        }
                        .buttonStyle(.bordered)
                    }
                }

                groupBox("Custom phrase") {
                    TextField("Type anything to speak", text: $customText)
                        .textFieldStyle(.roundedBorder)
                    Button("Speak custom text") { model.speak(customText) }
                        .buttonStyle(.bordered)
                        .disabled(customText.isEmpty)
                }

                groupBox("Accent") {
                    Picker("Accent", selection: $model.selectedAccent) {
                        ForEach(model.accents, id: \.self) { Text($0).tag($0) }
                    }
                    .pickerStyle(.segmented)
                }

                groupBox("Voice") {
                    Button {
                        model.refreshVoices()
                    } label: {
                        Label("Refresh installed voices", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    Picker("Gender", selection: $model.selectedGender) {
                        ForEach(model.genders, id: \.self) { Text($0).tag($0) }
                    }
                    .pickerStyle(.segmented)

                    if model.voicesForSelection.isEmpty {
                        Text("No voices installed for this combination -- check Settings > Accessibility > Spoken Content > Voices to download more.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("Specific voice", selection: $model.selectedVoiceID) {
                            ForEach(model.voicesForSelection) { option in
                                Text("\(option.voice.name) (\(option.qualityLabel)\(option.isNovelty ? ", novelty" : ""))")
                                    .tag(option.id)
                            }
                        }
                    }
                }

                groupBox("Delivery") {
                    VStack(alignment: .leading) {
                        Text("Rate: \(model.rate, specifier: "%.2f")")
                        Slider(value: $model.rate, in: 0.3...0.65)
                    }
                    VStack(alignment: .leading) {
                        Text("Pitch: \(model.pitch, specifier: "%.2f")")
                        Slider(value: $model.pitch, in: 0.6...1.6)
                    }
                }

                if model.isSpeaking {
                    Label("Speaking...", systemImage: "waveform")
                        .foregroundStyle(.orange)
                }
            }
            .padding()
        }
        .onChange(of: scenePhase) { _, newPhase in
            // Catches the common flow -- background this app, download a
            // voice in Settings, switch back -- without needing the manual
            // refresh button.
            if newPhase == .active { model.refreshVoices() }
        }
    }

    @ViewBuilder
    private func groupBox<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.headline)
            content()
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
    }
}
