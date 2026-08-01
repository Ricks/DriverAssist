//
//  Track.swift
//  DriverAssist
//
//  A single tracked object's persistent state -- ports tracker.py's Track
//  class.
//

import Foundation

final class Track {
    let id: Int
    let label: String
    let kf: KalmanBoxTracker
    var hits: Int = 1
    var timeSinceUpdate: Int = 0
    var age: Int = 0
    /// L2-normalized appearance embedding, exponentially smoothed across
    /// updates -- nil until/unless a real on-device embedder is wired in
    /// (see AppearanceEmbedder.swift).
    private(set) var feat: [Double]?

    init(id: Int, label: String, box: TrackedBox, feat: [Double]? = nil) {
        self.id = id
        self.label = label
        self.kf = KalmanBoxTracker(box: box)
        if let feat {
            updateFeat(feat)
        }
    }

    /// Exponential moving average of this track's appearance embedding
    /// (mirrors BoT-SORT's BOTrack.update_features, ported via tracker.py) --
    /// smooths out a single bad crop rather than snapping to it.
    func updateFeat(_ newFeat: [Double], alpha: Double = 0.9) {
        let normalized = Self.normalize(newFeat)
        if let existing = feat {
            let blended = zip(existing, normalized).map { alpha * $0 + (1 - alpha) * $1 }
            feat = Self.normalize(blended)
        } else {
            feat = normalized
        }
    }

    private static func normalize(_ v: [Double]) -> [Double] {
        let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot() + 1e-12
        return v.map { $0 / norm }
    }
}
