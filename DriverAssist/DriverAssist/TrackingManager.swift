import CoreVideo
import Foundation

/// Owns the persisted `TrackingLevel` setting and the live `ByteTracker`
/// instance, mirroring `ModelManager`'s pattern for `DetectorModel`. No
/// automatic/thermal-driven switching -- this is a plain manual setting
/// today, same as model tier and two-pass are (see ContentView's thermal HUD,
/// which is display-only; nothing in the app auto-adjusts settings from it).
///
/// Defaults to (and for MVP, always runs as) .recovery -- .appearance is a
/// no-op stub until a real on-device ReID model is integrated (see
/// AppearanceEmbedder.swift), so it's not exposed as a user choice right now
/// (no voice command selects it). setTrackingLevel(_:) still works
/// programmatically for whenever that changes.
@MainActor
final class TrackingManager: ObservableObject {
    @Published private(set) var trackingLevel: TrackingLevel

    private var tracker: ByteTracker
    private static let trackingLevelDefaultsKey = "settings.trackingLevel"

    init(defaultLevel: TrackingLevel = .recovery) {
        let level: TrackingLevel
        if let raw = UserDefaults.standard.string(forKey: Self.trackingLevelDefaultsKey),
           let persisted = TrackingLevel(rawValue: raw) {
            level = persisted
        } else {
            level = defaultLevel
        }
        trackingLevel = level
        tracker = Self.makeTracker(for: level)
    }

    func setTrackingLevel(_ level: TrackingLevel) {
        guard level != trackingLevel else { return }
        trackingLevel = level
        UserDefaults.standard.set(level.rawValue, forKey: Self.trackingLevelDefaultsKey)
        tracker = Self.makeTracker(for: level)
    }

    /// Runs GMC's optical-flow motion estimate for `pixelBuffer` -- see
    /// `ByteTracker.computeGMC`'s doc comment. Call this CONCURRENTLY with
    /// this frame's inference (see InferenceEngine.process's `gmcTask`),
    /// then pass the result into `track(...)`'s `gmcResult` parameter once
    /// both are ready.
    func computeGMC(pixelBuffer: CVPixelBuffer) -> GMCResult? {
        tracker.computeGMC(pixelBuffer: pixelBuffer)
    }

    /// Returns detections with `trackID` populated, plus this frame's GMC
    /// stats. Call once per completed inference, in chronological order --
    /// the tracker is stateful. Pass `computeGMC`'s result as `gmcResult`
    /// and `PitchSensor.smoothedYawRateDegreesPerSecond` as
    /// `yawRateDegreesPerSecond` -- see `ByteTracker.update`'s doc comment
    /// for exactly how (and how narrowly) each is used.
    func track(
        _ detections: [Detection],
        pixelBuffer: CVPixelBuffer?,
        gmcResult: GMCResult?,
        yawRateDegreesPerSecond: Double? = nil
    ) -> TrackingResult {
        tracker.update(
            detections,
            pixelBuffer: pixelBuffer,
            gmcResult: gmcResult,
            yawRateDegreesPerSecond: yawRateDegreesPerSecond
        )
    }

    private static func makeTracker(for level: TrackingLevel) -> ByteTracker {
        switch level {
        case .recovery:
            return ByteTracker(useGMC: true, embedder: nil)
        case .appearance:
            // See AppearanceEmbedder.swift -- UnavailableAppearanceEmbedder
            // always returns nil, so this currently behaves identically to
            // .recovery until a real on-device embedder replaces it.
            return ByteTracker(useGMC: true, embedder: UnavailableAppearanceEmbedder())
        }
    }
}
