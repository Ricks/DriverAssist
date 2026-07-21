//
//  ModelManager.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//
//  The ModelManager is the app’s single source of truth for model lifecycle and selection: it knows
//  which YOLO model is currently active, loads the chosen Core ML package from the app bundle using
//  a configured MLModel, exposes whether the model is ready or failed to load, and gives the rest of
//  the app—especially the InferenceEngine—a clean way to switch between bundled detectors like
//  yolo26s and yolo26n without scattering model-loading logic across the UI or camera pipeline.

import Foundation
import CoreML

enum DetectorModel: String, CaseIterable, Codable {
    case small = "yolo26s"
    case nano = "yolo26n"
}

@MainActor
final class ModelManager: ObservableObject {
    @Published private(set) var selectedModel: DetectorModel = .small
    @Published private(set) var isLoaded: Bool = false
    @Published private(set) var lastError: String?

    private(set) var model: MLModel?
    private let configuration: MLModelConfiguration

    init(defaultModel: DetectorModel = .small) {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.configuration = config
        self.selectedModel = defaultModel
    }

    func loadInitialModel() {
        do {
            try load(model: selectedModel)
        } catch {
            lastError = error.localizedDescription
            isLoaded = false
        }
    }

    func switchModel(to newModel: DetectorModel) {
        guard newModel != selectedModel else { return }

        do {
            try load(model: newModel)
        } catch {
            lastError = error.localizedDescription
            isLoaded = false
        }
    }

    private func load(model newModel: DetectorModel) throws {
        lastError = nil
        isLoaded = false

        guard let url = Bundle.main.url(forResource: newModel.rawValue, withExtension: "mlpackage") else {
            throw ModelManagerError.modelNotFound(newModel.rawValue)
        }

        let loadedModel = try MLModel(contentsOf: url, configuration: configuration)

        self.model = loadedModel
        self.selectedModel = newModel
        self.isLoaded = true
    }
}

enum ModelManagerError: LocalizedError {
    case modelNotFound(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Could not find \(name).mlpackage in app bundle."
        }
    }
}
