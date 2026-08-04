import Foundation
import CoreML

// nano/small are gesture-reachable (horizontal swipe -- see ContentView); medium
// is voice-only (VoiceCommandManager's .selectModel command), not on that grid --
// it has no high-res export (see ModelManager.highResTiers).
enum DetectorModel: String, CaseIterable, Codable {
    case nano   = "yolo26n"
    case small  = "yolo26s"
    case medium = "yolo26m"
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var selectedModel: DetectorModel
    @Published private(set) var isLoaded = false
    @Published private(set) var lastError: String?

    // Gesture-driven alternative input resolution (1920x1088, matching the
    // camera's own 1080p capture, vs. the standard 1152x640 export) -- see
    // the offline resolution_comparison.py experiment this came out of,
    // which found a real small-object recall gain from feeding the model
    // more of the already-captured detail. Not yet on-device validated for
    // latency/thermal cost (that's the point of exposing it as a manual,
    // deliberate swipe rather than a silent default).
    //
    // Only nano and small have a high-res export (see highResTiers) --
    // medium was deliberately excluded, being the most expensive tier
    // already, stacking it with the more expensive input too compounds two
    // untested costs at once. Toggling high-res while medium is selected is
    // a harmless no-op.
    @Published private(set) var isHighResEnabled = false

    // Both models are kept resident so switching is O(1).
    private var models: [DetectorModel: MLModel] = [:]
    private var highResModels: [DetectorModel: MLModel] = [:]
    private let configuration: MLModelConfiguration
    private var loadTasks: [DetectorModel: Task<Void, Never>] = [:]
    private var highResLoadTasks: [DetectorModel: Task<Void, Never>] = [:]

    private static let highResTiers: Set<DetectorModel> = [.nano, .small]

    // Actual model input dimensions, for HUD display -- see the resolution
    // comparison this came out of (isHighResEnabled's doc comment above).
    private static let baseResolutionLabel = "1152x640"
    private static let highResResolutionLabel = "1920x1088"

    private static func highResResourceName(for tier: DetectorModel) -> String {
        "\(tier.rawValue)-\(highResResolutionLabel)"
    }

    /// What the HUD shows in place of the old two-pass indicator -- the actual
    /// input resolution the selected model is running at right now, not just
    /// "low res"/"high res", since the numbers themselves are what a real
    /// drive's latency/thermal readings need to be interpreted against.
    var currentResolutionLabel: String {
        (isHighResEnabled && Self.highResTiers.contains(selectedModel))
            ? Self.highResResolutionLabel
            : Self.baseResolutionLabel
    }

    /// The high-res variant is substituted in only once it's actually
    /// finished loading -- falling back to the standard resolution rather
    /// than nil during that window means a swipe to high-res doesn't
    /// briefly stall inference while the (already resident, just not yet
    /// toggled-to) alternate model finishes loading.
    var model: MLModel? {
        if isHighResEnabled, let highRes = highResModels[selectedModel] {
            return highRes
        }
        return models[selectedModel]
    }

    private static let selectedModelDefaultsKey = "settings.selectedModel"

    init(defaultModel: DetectorModel = .small) {
        let config = MLModelConfiguration()
        // .all routes through the GPU/MPSGraph backend, which crashes compiling this
        // model ("MLIR pass manager failed"). CPU + Neural Engine works.
        config.computeUnits = .cpuAndNeuralEngine
        self.configuration = config
        if let raw = UserDefaults.standard.string(forKey: Self.selectedModelDefaultsKey),
           let persisted = DetectorModel(rawValue: raw) {
            self.selectedModel = persisted
        } else {
            self.selectedModel = defaultModel
        }
    }

    func loadInitialModel() {
        guard models.isEmpty, loadTasks.isEmpty else { return }
        for detectorModel in DetectorModel.allCases {
            startLoading(detectorModel)
        }
        for tier in Self.highResTiers {
            startLoadingHighRes(tier)
        }
    }

    func switchModel(to newModel: DetectorModel) {
        guard newModel != selectedModel else { return }
        selectedModel = newModel
        isLoaded = models[newModel] != nil
        lastError = nil
        UserDefaults.standard.set(newModel.rawValue, forKey: Self.selectedModelDefaultsKey)
    }

    func setHighResEnabled(_ enabled: Bool) {
        isHighResEnabled = enabled
    }

    private func startLoading(_ newModel: DetectorModel) {
        guard loadTasks[newModel] == nil else { return }
        loadTasks[newModel] = Task { await loadAsync(newModel) }
    }

    private func loadAsync(_ newModel: DetectorModel) async {
        guard let url = Bundle.main.url(forResource: newModel.rawValue, withExtension: "mlmodelc") else {
            print("[ModelManager] \(newModel.rawValue): .mlmodelc not found in bundle")
            if newModel == selectedModel {
                lastError = ModelManagerError.modelNotFound(newModel.rawValue).localizedDescription
            }
            return
        }

        do {
            let loaded = try await MLModel.load(contentsOf: url, configuration: configuration)
            guard !Task.isCancelled else { return }
            models[newModel] = loaded
            print("[ModelManager] \(newModel.rawValue): loaded")
            if newModel == selectedModel {
                isLoaded = true
                lastError = nil
            }
        } catch {
            guard !Task.isCancelled else { return }
            print("[ModelManager] \(newModel.rawValue): load failed: \(error)")
            if newModel == selectedModel {
                lastError = error.localizedDescription
            }
        }
    }

    private func startLoadingHighRes(_ tier: DetectorModel) {
        guard highResLoadTasks[tier] == nil else { return }
        let resourceName = Self.highResResourceName(for: tier)
        highResLoadTasks[tier] = Task {
            guard let url = Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") else {
                print("[ModelManager] \(resourceName): .mlmodelc not found in bundle")
                return
            }
            do {
                let loaded = try await MLModel.load(contentsOf: url, configuration: configuration)
                guard !Task.isCancelled else { return }
                highResModels[tier] = loaded
                print("[ModelManager] \(resourceName): loaded")
            } catch {
                guard !Task.isCancelled else { return }
                print("[ModelManager] \(resourceName): load failed: \(error)")
            }
        }
    }

    deinit {
        loadTasks.values.forEach { $0.cancel() }
        highResLoadTasks.values.forEach { $0.cancel() }
    }
}

enum ModelManagerError: LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Could not find \(name).mlmodelc in app bundle."
        }
    }
}
