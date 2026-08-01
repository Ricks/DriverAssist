//
//  AppearanceEmbedder.swift
//  DriverAssist
//
//  Appearance/ReID embedding provider for ByteTracker's Appearance tier --
//  ports tools/tracker.py's build_reid_encoder() + ultralytics ReID usage.
//
//  IMPORTANT: no on-device CoreML ReID model has been converted/integrated
//  yet -- UnavailableAppearanceEmbedder is the only implementation right
//  now, and it always returns nil embeddings. That makes TrackingLevel
//  .appearance behave *identically* to .recovery in practice today (nothing
//  for the eligibility-gated appearance fusion in ByteTracker to compare),
//  even though the enum case and all the matching logic exist. Do not
//  present Appearance as a functioning, distinct choice to the user until:
//    1. yolo26n-reid.onnx (or equivalent) is converted to Core ML and
//       bundled, and a real embedder is wired in here, and
//    2. its on-device inference latency is actually measured (Mac-side
//       ONNX Runtime timings, which is all that's been validated so far,
//       say nothing about real Core ML / Neural Engine performance).
//

import CoreVideo
import Foundation

protocol AppearanceEmbedder {
    /// Returns one embedding per input box, same order as `boxes`; nil for
    /// any box whose crop couldn't be embedded (e.g. degenerate size).
    /// Embeddings need not be pre-normalized -- Track.updateFeat handles that.
    func embed(boxes: [TrackedBox], pixelBuffer: CVPixelBuffer) -> [[Double]?]
}

/// See file-level note above -- always nil, making Appearance-tier matching
/// degrade to exactly Recovery's behavior until a real embedder replaces
/// this.
struct UnavailableAppearanceEmbedder: AppearanceEmbedder {
    func embed(boxes: [TrackedBox], pixelBuffer: CVPixelBuffer) -> [[Double]?] {
        Array(repeating: nil, count: boxes.count)
    }
}
