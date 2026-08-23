//
//  ByteTracker.swift
//  DriverAssist
//
//  Multi-object tracker (ByteTrack-style, with optional appearance/ReID
//  matching and camera-motion compensation) -- ports tools/tracker.py's
//  ByteTracker class. See that file's module docstring for the full
//  rationale behind every piece of this; this mirrors it rather than
//  re-deriving it:
//
//  - Two confidence tiers: high-confidence detections match first and are
//    the only ones allowed to spawn a new track; a track still unmatched
//    after that gets a second chance against this frame's low-confidence
//    detections (letting it survive a bad frame instead of losing its ID).
//    NOTE: the on-device detector currently only surfaces detections at or
//    above `highConfThreshold` (see YOLODecoder's confidenceThreshold) --
//    until that changes, this low-confidence recovery stage never has
//    anything to match against, same caveat as tracker.py's on Python-side
//    data.
//  - Appearance fusion (see AppearanceEmbedder.swift): cosine distance
//    between embeddings, fused with IoU via `min(iouCost, embeddingCost)`.
//    Eligibility to attempt an appearance match is gated by a size-scaled
//    *distance* (not IoU -- IoU is exactly 0 for any non-overlapping pair,
//    which can't distinguish a track that drifted just out of matching
//    range from one that's nowhere near a detection), while the final
//    acceptance bar stays `iouThreshold` regardless.
//  - Camera motion compensation (see GMC.swift): each track's predicted
//    position/velocity is corrected for the camera's own estimated motion
//    before matching.
//
//  Validated defaults (via tools/track_benchmark.py sweeps against 5 real
//  driving sessions): iouThreshold=0.2, maxAge=5 (increasing this made
//  every session's tracking measurably worse, not better -- a longer max
//  age keeps more "zombie" tracks around to wrongly latch onto a new,
//  unrelated detection more often than it successfully bridges a real gap),
//  appearanceThresh=0.7.
//

import CoreGraphics
import CoreVideo
import Foundation

struct TrackingResult {
    let detections: [Detection]
    /// nil if GMC is disabled, there were no live tracks to correct, or this
    /// was the very first frame seen -- see GMCStats.
    let gmcStats: GMCStats?
}

final class ByteTracker {
    let iouThreshold: Double
    let maxAge: Int
    let minHits: Int
    let highConfThreshold: Double
    let lowConfThreshold: Double
    let appearanceThresh: Double
    let appearanceEligibilityScale: Double

    private var tracks: [Track] = []
    private var nextID = 1
    private let gmc: GMC?
    private let embedder: AppearanceEmbedder?

    /// Label pairs whose real-world class boundary this detector can lose
    /// entirely from a single viewpoint -- added 2026-08-22 in response to a
    /// real, reported gap: a cyclist viewed head-on collapses to almost no
    /// bicycle-discriminative signal (no side profile, wheels overlap into a
    /// sliver), so the detector can genuinely stop emitting "bicycle" boxes
    /// for that whole stretch, sometimes surfacing the same physical object
    /// as "person" instead. Per-label matching (see `update` below) means a
    /// "person" detection can NEVER match a "bicycle" track on its own --
    /// without this, that track either dies once `maxAge` passes (losing its
    /// distance/velocity history right as the object gets closest) or the
    /// object briefly becomes a brand-new "person" track. See the
    /// confusable-label rescue pass at the end of `update` for how this is
    /// used -- deliberately NOT a general "any label may rescue any label"
    /// mechanism (that would risk exactly the kind of false continuation
    /// `maxAge`'s own tuning history already warned against), just this one
    /// evidenced pair. Symmetric (either label may rescue the other) since
    /// there's no reason to think the confusion only runs one direction.
    private static let confusableLabels: [String: Set<String>] = [
        "bicycle": ["person"],
        "person": ["bicycle"],
    ]

    /// Wall-clock time of the previous `update` call -- used only to size the
    /// yaw-rate fallback transform below (needs an actual elapsed-seconds dt,
    /// not an assumed fixed frame interval, since inference cadence isn't
    /// perfectly regular). nil before the first call.
    private var lastUpdateTime: CFAbsoluteTime?

    init(
        iouThreshold: Double = 0.2,
        maxAge: Int = 5,
        minHits: Int = 1,
        highConfThreshold: Double = 0.25,
        lowConfThreshold: Double = 0.1,
        appearanceThresh: Double = 0.7,
        appearanceEligibilityScale: Double = 3.0,
        useGMC: Bool = true,
        embedder: AppearanceEmbedder? = nil
    ) {
        self.iouThreshold = iouThreshold
        self.maxAge = maxAge
        self.minHits = minHits
        self.highConfThreshold = highConfThreshold
        self.lowConfThreshold = lowConfThreshold
        self.appearanceThresh = appearanceThresh
        self.appearanceEligibilityScale = appearanceEligibilityScale
        self.gmc = useGMC ? GMC() : nil
        self.embedder = embedder
    }

    /// Runs GMC's optical-flow motion estimate against this frame's pixel
    /// buffer -- split out from `update` so it can be kicked off
    /// CONCURRENTLY with this frame's CoreML inference (see
    /// InferenceEngine.process's `gmcTask`), instead of only starting once
    /// `update` is called with the finished detections. Safe to call
    /// independently: unlike `update`, this never reads or mutates `tracks`
    /// -- it only checks whether any exist yet (skip the optical-flow work
    /// if there's nothing to correct) and touches GMC's own private state,
    /// nothing `update` concurrently relies on.
    ///
    /// CONFIRMED 2026-08-16 against real drive data: before this split, GMC
    /// only ever started after inference's detections already existed,
    /// despite never needing them -- pure unforced serialization (mean
    /// combined per-frame latency 80.9ms measured vs. ~46.5ms estimated
    /// once overlapped). Pass the result into `update`'s `gmcResult`
    /// parameter.
    func computeGMC(pixelBuffer: CVPixelBuffer) -> GMCResult? {
        guard let gmc, !tracks.isEmpty else { return nil }
        return gmc.apply(to: pixelBuffer)
    }

    /// Returns detections with `trackID` populated (nil where nothing
    /// matched -- only possible for a low-confidence detection that didn't
    /// extend an existing track), plus this frame's GMC quality stats.
    /// `gmcResult`, if given, should come from a PRIOR call to `computeGMC`
    /// for this same frame -- see that method's doc comment for why this is
    /// split out rather than computed here. `pixelBuffer` is still needed
    /// here for its width/height (to apply GMC's transform in pixel space)
    /// and for appearance embedding -- omit both (or pass useGMC:false /
    /// embedder:nil at init) to run geometry-only.
    /// `yawRateDegreesPerSecond`, if given, should be
    /// `PitchSensor.smoothedYawRateDegreesPerSecond` -- see
    /// `yawFallbackTransform` below for how (and how narrowly) it's used.
    func update(
        _ detections: [Detection],
        pixelBuffer: CVPixelBuffer?,
        gmcResult: GMCResult?,
        yawRateDegreesPerSecond: Double? = nil
    ) -> TrackingResult {
        for t in tracks {
            t.kf.predict()
            t.age += 1
        }

        let now = CFAbsoluteTimeGetCurrent()
        let dt = lastUpdateTime.map { now - $0 }
        lastUpdateTime = now

        var gmcStats: GMCStats?
        if let pixelBuffer, let result = gmcResult {
            gmcStats = result.stats
            let w = Double(CVPixelBufferGetWidth(pixelBuffer))
            let h = Double(CVPixelBufferGetHeight(pixelBuffer))

            // GMC reports transform == .identity (no compensation at all)
            // whenever it didn't trust its own optical-flow fit (see
            // GMC.minTrustedInlierCount) -- that's exactly the gap the
            // CoreMotion-derived fallback below fills, rather than silently
            // matching against un-compensated predictions during a real turn.
            let gmcUntrusted = result.stats == nil || result.stats!.inlierCount < GMC.minTrustedInlierCount
            var transform = result.transform
            if gmcUntrusted, let yawRateDegreesPerSecond, let dt, dt > 0, dt < 1.0 {
                transform = Self.yawFallbackTransform(
                    yawRateDegreesPerSecond: yawRateDegreesPerSecond,
                    dt: dt,
                    frameHeightPx: h
                )
            }

            for t in tracks {
                t.kf.applyGMC(transform, frameWidth: w, frameHeight: h)
            }
        }

        var embeddings: [[Double]?] = Array(repeating: nil, count: detections.count)
        if let embedder, let pixelBuffer, !detections.isEmpty {
            let boxes = detections.map { TrackedBox($0.boundingBox) }
            embeddings = embedder.embed(boxes: boxes, pixelBuffer: pixelBuffer)
        }

        var trackIDsOut: [Int?] = Array(repeating: nil, count: detections.count)
        let highIdxs = detections.indices.filter { Double(detections[$0].confidence) >= highConfThreshold }
        let lowIdxs = detections.indices.filter {
            let c = Double(detections[$0].confidence)
            return c >= lowConfThreshold && c < highConfThreshold
        }

        var labels = Set(tracks.map(\.label))
        for i in highIdxs { labels.insert(detections[i].label) }
        for i in lowIdxs { labels.insert(detections[i].label) }

        for label in labels {
            let trackIdxs = tracks.indices.filter { tracks[$0].label == label }
            let labelHighIdxs = highIdxs.filter { detections[$0].label == label }

            let trackBoxes = trackIdxs.map { tracks[$0].kf.box }
            let trackFeats = trackIdxs.map { tracks[$0].feat }
            let detBoxes = labelHighIdxs.map { TrackedBox(detections[$0].boundingBox) }
            let detFeats = labelHighIdxs.map { embeddings[$0] }

            let stage1 = match(trackBoxes: trackBoxes, detBoxes: detBoxes, trackFeats: trackFeats, detFeats: detFeats)

            for (tLocal, dLocal) in stage1.matches {
                let track = tracks[trackIdxs[tLocal]]
                let detIdx = labelHighIdxs[dLocal]
                track.kf.update(with: TrackedBox(detections[detIdx].boundingBox))
                if let feat = embeddings[detIdx] { track.updateFeat(feat) }
                track.hits += 1
                track.timeSinceUpdate = 0
                trackIDsOut[detIdx] = track.id
            }

            // Second stage: remaining unmatched tracks of this label vs.
            // this frame's low-confidence detections of this label.
            let remainingTrackIdxs = stage1.unmatchedTracks.map { trackIdxs[$0] }
            let labelLowIdxs = lowIdxs.filter { detections[$0].label == label }
            let trackBoxes2 = remainingTrackIdxs.map { tracks[$0].kf.box }
            let trackFeats2 = remainingTrackIdxs.map { tracks[$0].feat }
            let detBoxes2 = labelLowIdxs.map { TrackedBox(detections[$0].boundingBox) }
            let detFeats2 = labelLowIdxs.map { embeddings[$0] }

            let stage2 = match(trackBoxes: trackBoxes2, detBoxes: detBoxes2, trackFeats: trackFeats2, detFeats: detFeats2)

            var stillUnmatchedLocal = Set(0..<remainingTrackIdxs.count)
            for (tLocal, dLocal) in stage2.matches {
                let track = tracks[remainingTrackIdxs[tLocal]]
                let detIdx = labelLowIdxs[dLocal]
                track.kf.update(with: TrackedBox(detections[detIdx].boundingBox))
                if let feat = embeddings[detIdx] { track.updateFeat(feat) }
                track.timeSinceUpdate = 0
                trackIDsOut[detIdx] = track.id
                stillUnmatchedLocal.remove(tLocal)
            }

            for tLocal in stillUnmatchedLocal {
                tracks[remainingTrackIdxs[tLocal]].timeSinceUpdate += 1
            }

            // Unmatched high-confidence detections spawn new tracks.
            // Low-confidence detections never do (they're the less reliable
            // signal -- see module docstring).
            for dLocal in stage1.unmatchedDets {
                let detIdx = labelHighIdxs[dLocal]
                let newTrack = Track(
                    id: nextID, label: label,
                    box: TrackedBox(detections[detIdx].boundingBox), feat: embeddings[detIdx]
                )
                nextID += 1
                tracks.append(newTrack)
                trackIDsOut[detIdx] = newTrack.id
            }
        }

        // Confusable-label rescue -- see `confusableLabels`'s doc comment.
        // Runs after every label's normal matching above, so it only sees
        // tracks genuinely unmatched this frame (`timeSinceUpdate` was just
        // incremented, not reset) and detections no other label's pass
        // already claimed. On a match, geometry/appearance update as
        // normal, but `track.label` is deliberately left untouched -- the
        // track's own established identity (built from its real,
        // confidently-classified hits) is trusted over one ambiguous
        // frame's guess. Only high-confidence detections may rescue, same
        // "less reliable signal" reasoning as new-track spawning above.
        for (fromLabel, toLabels) in Self.confusableLabels {
            let rescueTrackIdxs = tracks.indices.filter { tracks[$0].label == fromLabel && tracks[$0].timeSinceUpdate > 0 }
            guard !rescueTrackIdxs.isEmpty else { continue }
            let rescueDetIdxs = detections.indices.filter {
                trackIDsOut[$0] == nil
                    && toLabels.contains(detections[$0].label)
                    && Double(detections[$0].confidence) >= highConfThreshold
            }
            guard !rescueDetIdxs.isEmpty else { continue }

            let trackBoxes = rescueTrackIdxs.map { tracks[$0].kf.box }
            let trackFeats = rescueTrackIdxs.map { tracks[$0].feat }
            let detBoxes = rescueDetIdxs.map { TrackedBox(detections[$0].boundingBox) }
            let detFeats = rescueDetIdxs.map { embeddings[$0] }

            let rescue = match(trackBoxes: trackBoxes, detBoxes: detBoxes, trackFeats: trackFeats, detFeats: detFeats)
            for (tLocal, dLocal) in rescue.matches {
                let track = tracks[rescueTrackIdxs[tLocal]]
                let detIdx = rescueDetIdxs[dLocal]
                track.kf.update(with: TrackedBox(detections[detIdx].boundingBox))
                if let feat = embeddings[detIdx] { track.updateFeat(feat) }
                track.hits += 1
                track.timeSinceUpdate = 0
                trackIDsOut[detIdx] = track.id
            }
        }

        tracks.removeAll { $0.timeSinceUpdate > maxAge }

        for i in trackIDsOut.indices {
            guard let tid = trackIDsOut[i] else { continue }
            if let track = tracks.first(where: { $0.id == tid }), track.hits < minHits {
                trackIDsOut[i] = nil  // not yet confirmed
            }
        }

        let out = zip(detections, trackIDsOut).map { detection, trackID -> Detection in
            var updated = detection
            updated.trackID = trackID
            return updated
        }
        return TrackingResult(detections: out, gmcStats: gmcStats)
    }

    /// Hungarian assignment minimizing 1-IoU distance, gated at
    /// iouThreshold. If feats are given, appearance distance is fused in
    /// exactly like BoT-SORT's get_dists(): `combined = min(iouDist,
    /// embeddingDist)`, with embedding distance only able to help pairs
    /// within appearanceEligibilityScale (a size-scaled *distance*, not
    /// IoU), and only when it itself clears appearanceThresh.
    private func match(
        trackBoxes: [TrackedBox], detBoxes: [TrackedBox],
        trackFeats: [[Double]?], detFeats: [[Double]?]
    ) -> (matches: [(Int, Int)], unmatchedTracks: [Int], unmatchedDets: [Int]) {
        guard !trackBoxes.isEmpty, !detBoxes.isEmpty else {
            return ([], Array(trackBoxes.indices), Array(detBoxes.indices))
        }

        var cost = [[Double]](repeating: [Double](repeating: 0, count: detBoxes.count), count: trackBoxes.count)
        for i in trackBoxes.indices {
            for j in detBoxes.indices {
                cost[i][j] = 1.0 - Double(DetectionFilter.iou(trackBoxes[i].cgRect, detBoxes[j].cgRect))
            }
        }

        let hasAnyFeat = trackFeats.contains(where: { $0 != nil }) || detFeats.contains(where: { $0 != nil })
        if hasAnyFeat {
            var embCost = [[Double]](repeating: [Double](repeating: 1.0, count: detBoxes.count), count: trackBoxes.count)
            for i in trackBoxes.indices {
                guard let tf = trackFeats[i] else { continue }
                for j in detBoxes.indices {
                    guard let df = detFeats[j] else { continue }
                    let maxDiag = max(diagonal(trackBoxes[i]), diagonal(detBoxes[j]))
                    guard centerDistance(trackBoxes[i], detBoxes[j]) <= appearanceEligibilityScale * maxDiag else { continue }
                    embCost[i][j] = 1.0 - dot(tf, df)
                }
            }
            for i in trackBoxes.indices {
                for j in detBoxes.indices {
                    if embCost[i][j] > (1.0 - appearanceThresh) { embCost[i][j] = 1.0 }
                    cost[i][j] = min(cost[i][j], embCost[i][j])
                }
            }
        }

        let assignment = HungarianAlgorithm.solve(cost: cost)
        var matches: [(Int, Int)] = []
        var matchedTracks = Set<Int>()
        var matchedDets = Set<Int>()
        for (row, colOpt) in assignment.enumerated() {
            guard let col = colOpt else { continue }
            if 1.0 - cost[row][col] >= iouThreshold {
                matches.append((row, col))
                matchedTracks.insert(row)
                matchedDets.insert(col)
            }
        }

        let unmatchedTracks = trackBoxes.indices.filter { !matchedTracks.contains($0) }
        let unmatchedDets = detBoxes.indices.filter { !matchedDets.contains($0) }
        return (matches, unmatchedTracks, unmatchedDets)
    }

    /// Rotation-only camera-pan estimate built from gyro yaw rate instead of
    /// GMC's own optical flow -- used ONLY when GMC didn't trust its fit for
    /// this frame (see the `gmcUntrusted` check above), never to override a
    /// trusted vision-based result. Vision-based GMC stays the primary
    /// source because it observes the ACTUAL pixel motion (real lens/crop/
    /// stabilization behavior included); this fallback is a coarser model
    /// that only exists so a low-texture or fast-optical-flow-failure frame
    /// doesn't fall all the way back to *no* compensation (.identity) during
    /// a genuine turn, which is exactly when GMC is most likely to strain
    /// (large frame-to-frame motion) and a missed compensation would most
    /// hurt matching.
    ///
    /// Deliberately rotation-only (no translation/scale term), matching
    /// GMC's own file-level scope note ("Only rotation/pan/tilt is corrected
    /// this way -- forward-motion parallax needs per-object depth to
    /// compensate properly and isn't attempted here"): the vehicle's own
    /// forward speed and the lever-arm's translational yaw component (see
    /// DistanceEstimator.LeverArm) both shift each background point by an
    /// amount that depends on that point's own depth, so neither can be
    /// expressed as a single frame-wide pixel transform the way a pure
    /// rotation can -- only the rotational term is used here, deliberately
    /// dropping the (depth-dependent, unmodelable this way) translational
    /// ones rather than approximating them badly.
    ///
    /// Derivation: a camera yaw of dTheta radians (positive = turning left,
    /// matching PitchSensor's convention) rotates every world point's
    /// bearing by -dTheta in the camera's own frame, which -- for the
    /// small per-frame angles this is actually evaluated at -- shifts a
    /// point's image column by approximately +pixelFocalLength * dTheta
    /// (turning left => the background shifts right in frame, same as
    /// turning your own head left makes the world appear to slide right).
    /// `pixelFocalLength` reuses DistanceEstimator's calibrated
    /// (row-normalized) focal length directly against the actual pixel
    /// buffer height -- valid for the column axis too under the same
    /// square-pixel assumption DistanceEstimator itself already makes
    /// (see its `focalLengthColumnNormalized` comment).
    private static func yawFallbackTransform(
        yawRateDegreesPerSecond: Double,
        dt: Double,
        frameHeightPx: Double
    ) -> SimilarityTransform {
        let dThetaRadians = yawRateDegreesPerSecond * .pi / 180 * dt
        let pixelFocalLength = DistanceEstimator.calibrated.focalLengthNormalized * frameHeightPx
        return SimilarityTransform(a: 1, b: 0, tx: pixelFocalLength * dThetaRadians, ty: 0)
    }

    private func diagonal(_ box: TrackedBox) -> Double {
        (box.w * box.w + box.h * box.h).squareRoot()
    }

    private func centerDistance(_ a: TrackedBox, _ b: TrackedBox) -> Double {
        let acx = a.x + a.w / 2, acy = a.y + a.h / 2
        let bcx = b.x + b.w / 2, bcy = b.y + b.h / 2
        return ((acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy)).squareRoot()
    }

    private func dot(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
