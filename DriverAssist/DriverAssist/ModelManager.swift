import Foundation
import CoreML

enum DetectorModel: String, CaseIterable, Codable {
    case small = "yolo26s"
    case nano  = "yolo26n"
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var selectedModel: DetectorModel
    @Published private(set) var isLoaded = false
    @Published private(set) var lastError: String?

    // Both models are kept resident so switching is O(1).
    private var models: [DetectorModel: MLModel] = [:]
    private let configuration: MLModelConfiguration
    private var loadTasks: [DetectorModel: Task<Void, Never>] = [:]

    var model: MLModel? { models[selectedModel] }

    init(defaultModel: DetectorModel = .small) {
        let config = MLModelConfiguration()
        // .all routes through the GPU/MPSGraph backend, which crashes compiling this
        // model ("MLIR pass manager failed"). CPU + Neural Engine works.
        config.computeUnits = .cpuAndNeuralEngine
        self.configuration = config
        self.selectedModel = defaultModel
    }

    func loadInitialModel() {
        guard models.isEmpty, loadTasks.isEmpty else { return }
        for detectorModel in DetectorModel.allCases {
            startLoading(detectorModel)
        }
    }

    func switchModel(to newModel: DetectorModel) {
        guard newModel != selectedModel else { return }
        selectedModel = newModel
        isLoaded = models[newModel] != nil
        lastError = nil
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

    deinit {
        loadTasks.values.forEach { $0.cancel() }
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
