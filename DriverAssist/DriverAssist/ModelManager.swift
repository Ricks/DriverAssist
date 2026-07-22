import Foundation
import CoreML

enum DetectorModel: String, CaseIterable, Codable {
    case small = "yolo26s"
    case nano = "yolo26n"
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var selectedModel: DetectorModel
    @Published private(set) var isLoaded = false
    @Published private(set) var lastError: String?

    private(set) var model: MLModel?
    private let configuration: MLModelConfiguration
    private var loadTask: Task<Void, Never>?

    init(defaultModel: DetectorModel = .small) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.configuration = config
        self.selectedModel = defaultModel
    }

    func loadInitialModel() {
        guard model == nil else { return }
        startLoading(model: selectedModel)
    }

    func switchModel(to newModel: DetectorModel) {
        guard newModel != selectedModel else { return }
        startLoading(model: newModel)
    }

    private func startLoading(model newModel: DetectorModel) {
        loadTask?.cancel()
        lastError = nil
        loadTask = Task {
            await loadAsync(model: newModel)
        }
    }
    
    private func loadAsync(model newModel: DetectorModel) async {
        lastError = nil
        isLoaded = false

        guard let url = Bundle.main.url(forResource: newModel.rawValue, withExtension: "mlmodelc") else {
            lastError = ModelManagerError.modelNotFound(newModel.rawValue).localizedDescription
            isLoaded = model != nil   // keep any previously loaded model working
            return
        }

        do {
            let loadedModel = try await MLModel.load(contentsOf: url, configuration: configuration)
            guard !Task.isCancelled else { return }

            model = loadedModel
            selectedModel = newModel
            isLoaded = true
        } catch {
            guard !Task.isCancelled else { return }

            lastError = error.localizedDescription
            isLoaded = model != nil   // keep any previously loaded model working
        }
    }

    deinit {
        loadTask?.cancel()
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
