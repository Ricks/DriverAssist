//
//  WidthDistanceOverride.swift
//  DriverAssist
//
//  Overrides DistanceEstimator's row-based (ground-plane) distance with a
//  width-based reading when the width-based estimate is substantially
//  closer. One-directional by design: nearly every row-based failure mode
//  (misjudged pitch, hood truncation, ground-plane violations) biases it
//  toward reading TOO FAR, not too close (downhill grade is the one
//  exception), so only ever trusting width-based when it reads CLOSER is
//  the safe direction -- a width-based reading that comes out FARTHER than
//  row-based is more likely a bad box or misclassification than a real
//  correction, and this never applies it.
//
//  CONFIRMED 2026-08-15 (real walkaround-test data, data/26_08_15_Walkaround)
//  this is a real, not theoretical, problem: row-based read 3m/5m tethered
//  distances as ~5.7m/~6.0m (+92%/+19% error) -- both below
//  DistanceEstimator.hoodCutoffAngleDegrees, where the box bottom is pinned
//  to the ego hood's edge rather than the object's real ground contact.
//

import Foundation

/// Real per-class object width mean/std, computed from actual KITTI
/// training labels (7,481 images, 28,742 Car / 4,487 Pedestrian instances --
/// not estimated or guessed). Only classes with a direct, well-matched
/// width prior are included:
///   - KITTI "Car" -> YOLO "car".
///   - KITTI "Pedestrian" -> YOLO "person".
/// Deliberately NOT extended to:
///   - truck/bus: no separate KITTI stats were computed for these, and
///     approximating them as "car" width would UNDERESTIMATE their
///     distance -- a wide truck box would be mistaken for a close car,
///     exactly the wrong direction for a safety-relevant override.
///   - bicycle/cyclist: KITTI's "Cyclist" width is a person-ON-a-bicycle
///     combined box, but the raw YOLO "bicycle" label -- the only one
///     available at the point WidthDistanceOverrideManager runs, before
///     ComboManager's rider-pairing -- is just the bike alone. Wiring this
///     into the combo layer instead (post-pairing, where a real
///     rider+bicycle union box exists) is future work, not done here.
enum ObjectWidthPriors {
    struct Width {
        let meanMeters: Double
        let stdMeters: Double
    }

    static let byLabel: [String: Width] = [
        "car": Width(meanMeters: 1.629, stdMeters: 0.102),
        "person": Width(meanMeters: 0.660, stdMeters: 0.143),
    ]
}

/// Per-trackID hysteresis + the actual override decision, mirroring
/// ComboManager's confirm/grace-frame pattern (see that file's doc comment
/// for the original rationale) -- a one-shot "width is closer this frame"
/// reading can be a fluke (a misclassified box, a momentary bad detection),
/// so the override only actually takes effect after `confirmFrames`
/// consecutive frames agree, and stays active through up to `graceFrames`
/// frames of disagreement before dropping back to row-based, rather than
/// flickering every frame.
@MainActor
final class WidthDistanceOverrideManager {
    private struct OverrideState {
        var streak: Int = 0
        var active: Bool = false
        var framesSinceAgreed: Int = 0
    }

    /// Base fraction closer than row-based that width-based must read
    /// before being considered at all -- e.g. 0.15 means at least 15%
    /// closer. The ACTUAL required fraction is this plus a per-class term
    /// derived from `ObjectWidthPriors`' real relative width variance (see
    /// `widthUncertaintyMultiplier`), not this value alone.
    private let minCloserFraction: Double
    /// Scales each class's relative width uncertainty (stdMeters/meanMeters)
    /// into additional required closer-fraction margin -- a class with more
    /// variable real-world width (pedestrian, ~22% relative std) needs a
    /// bigger margin before a single width-based reading is trustworthy
    /// than a class with tightly-clustered width (car, ~6% relative std).
    /// At the defaults: car needs ~21% closer, pedestrian needs ~37%.
    private let widthUncertaintyMultiplier: Double
    /// Detector confidence floor for the "normal" regime (row-based reading
    /// well above `DistanceEstimator.hoodCutoffAngleDegrees`).
    private let normalConfidenceFloor: Float
    /// Looser confidence floor used only in the near-minimum-distance
    /// regime (row-based reading within `nearCutoffMarginDegrees` of the
    /// hood-cutoff angle) -- row-based is already known-unreliable there,
    /// so a width-based correction is worth trusting at a lower bar than it
    /// would need elsewhere.
    ///
    /// PROVISIONAL: both this and `normalConfidenceFloor` are Rick's own
    /// proposed starting values (2026-08-15: "90% confidence to override a
    /// normal row-based distance calculation, and 50% to override one near
    /// the minimum distance threshold"), not yet validated against real
    /// confidence distributions from false_positive-labeled ground truth --
    /// see the project's width-based-distance-override design notes for how
    /// to validate/retune these once real driving data exists.
    private let nearCutoffConfidenceFloor: Float
    /// Degrees of margin above `DistanceEstimator.hoodCutoffAngleDegrees`
    /// that still counts as "near minimum" -- that constant is a measured
    /// MEAN (std 0.771°), not already padded, so this margin is what
    /// actually provides the buffer against the real measured scatter.
    private let nearCutoffMarginDegrees: Double
    /// Hard cap on how much closer than row-based the FINAL applied
    /// override is allowed to report, as a fraction of the row-based
    /// reading -- a safety-critical backstop against a misclassification-
    /// or bad-box-driven false "collision imminent" alert, independent of
    /// the trigger-threshold check above. Deliberately generous (not tied
    /// to width-variance the way the trigger threshold is): the whole
    /// point of this feature is correcting row-based errors that were
    /// measured as large as +92% (see file header), so a tight cap here
    /// would neuter the feature's actual purpose. This just guards against
    /// an absurd claim (e.g. "0.3m away"), not ordinary large corrections.
    private let maxOverrideCloserFraction: Double
    /// Absolute plausibility ceiling (meters) for EITHER candidate distance
    /// before the override is even considered -- see the CONFIRMED comment
    /// in `evaluateGate`. This feature exists for near-field correction; a
    /// candidate this far out is never something the override should be
    /// deciding between, regardless of which formula produced it.
    private static let maxPlausibleDistanceMeters: Double = 150
    private let confirmFrames: Int
    private let graceFrames: Int

    private var states: [Int: OverrideState] = [:]

    init(
        minCloserFraction: Double = 0.15,
        widthUncertaintyMultiplier: Double = 1.0,
        normalConfidenceFloor: Float = 0.9,
        nearCutoffConfidenceFloor: Float = 0.5,
        nearCutoffMarginDegrees: Double = 2.0,
        maxOverrideCloserFraction: Double = 0.8,
        confirmFrames: Int = 3,
        graceFrames: Int = 5
    ) {
        self.minCloserFraction = minCloserFraction
        self.widthUncertaintyMultiplier = widthUncertaintyMultiplier
        self.normalConfidenceFloor = normalConfidenceFloor
        self.nearCutoffConfidenceFloor = nearCutoffConfidenceFloor
        self.nearCutoffMarginDegrees = nearCutoffMarginDegrees
        self.maxOverrideCloserFraction = maxOverrideCloserFraction
        self.confirmFrames = confirmFrames
        self.graceFrames = graceFrames
    }

    /// Runs the override over one frame's detections -- call after
    /// DistanceEstimator's row-based `distanceMeters` has already been
    /// attached (see InferenceEngine.attachDistances), before
    /// DetectionLogger.log, so the override's effect is itself logged (the
    /// same "log everything for later validation" discipline as everything
    /// else in this pipeline -- see DetectionLogEntry.Box's
    /// widthDistanceMeters/distanceMetersIsWidthOverridden fields).
    ///
    /// `cameraHeightMeters`/`aspectRatio` match `DistanceEstimator
    /// .distanceMeters`'s own parameters -- pass the same values used to
    /// produce this frame's row-based readings.
    func apply(
        to detections: [Detection],
        cameraHeightMeters: Double,
        aspectRatio: Double
    ) -> [Detection] {
        var seenTrackIDs = Set<Int>()

        let result = detections.map { detection -> Detection in
            var detection = detection
            guard let trackID = detection.trackID else { return detection }
            seenTrackIDs.insert(trackID)

            var state = states[trackID] ?? OverrideState()
            let gatePassed = evaluateGate(
                &detection,
                cameraHeightMeters: cameraHeightMeters,
                aspectRatio: aspectRatio
            )

            if gatePassed {
                state.streak += 1
                state.framesSinceAgreed = 0
                if state.streak >= confirmFrames {
                    state.active = true
                }
            } else {
                state.framesSinceAgreed += 1
                state.streak = 0
                if state.active && state.framesSinceAgreed > graceFrames {
                    state.active = false
                }
            }
            states[trackID] = state

            if state.active, let widthDistance = detection.widthDistanceMeters, let rowDistance = detection.distanceMeters {
                let flooredDistance = rowDistance * (1 - maxOverrideCloserFraction)
                detection.distanceMeters = max(widthDistance, flooredDistance)
                detection.distanceMetersIsWidthOverridden = true
            }
            return detection
        }

        // Evict tracks not seen at all this frame (departed/lost) so the
        // dictionary doesn't grow unboundedly over a long drive -- a track
        // still present but failing the gate goes through the normal
        // grace-period decay above instead, this only clears out ones that
        // vanished from the detector entirely.
        for trackID in states.keys where !seenTrackIDs.contains(trackID) {
            states.removeValue(forKey: trackID)
        }

        return result
    }

    /// Computes and attaches `widthDistanceMeters` (always, if applicable),
    /// and returns whether this frame's reading passes every trigger
    /// condition on its own (confidence floor for the current pitch regime,
    /// class-relative-variance-adjusted closer-fraction requirement) --
    /// hysteresis in `apply` decides whether a pass/fail actually flips the
    /// override, this only judges the single frame.
    private func evaluateGate(
        _ detection: inout Detection,
        cameraHeightMeters: Double,
        aspectRatio: Double
    ) -> Bool {
        guard
            let rowDistance = detection.distanceMeters,
            let prior = ObjectWidthPriors.byLabel[detection.label]
        else { return false }

        let boxWidth = Double(detection.boundingBox.width)
        guard let widthDistance = DistanceEstimator.calibrated.widthBasedDistanceMeters(
            boxWidthNormalized: boxWidth,
            realWidthMeters: prior.meanMeters,
            aspectRatio: aspectRatio
        ) else { return false }
        detection.widthDistanceMeters = widthDistance

        // CONFIRMED 2026-08-16 (real test drive): the trigger below is a pure
        // RATIO comparison (is width closer than row) -- it has no absolute
        // sense of "plausible". `widthBasedDistanceMeters`'s own floor
        // catches a degenerate box width, but row-based can independently
        // degenerate too (phi near zero, from a detection near the modeled
        // horizon), and a "closer than an already-absurd row reading" result
        // is still absurd. This is a near-field correction tool -- neither
        // candidate belongs anywhere near this range regardless of which
        // formula produced it, so reject the frame outright rather than let
        // the ratio check compare two nonsense numbers and pick a winner.
        guard rowDistance <= Self.maxPlausibleDistanceMeters, widthDistance <= Self.maxPlausibleDistanceMeters else {
            return false
        }

        let phiDegrees = atan(cameraHeightMeters / rowDistance) * 180 / .pi
        let isNearCutoff = phiDegrees <= DistanceEstimator.hoodCutoffAngleDegrees + nearCutoffMarginDegrees
        let confidenceFloor = isNearCutoff ? nearCutoffConfidenceFloor : normalConfidenceFloor
        guard detection.confidence >= confidenceFloor else { return false }

        let relativeWidthStd = prior.stdMeters / prior.meanMeters
        let requiredCloserFraction = minCloserFraction + widthUncertaintyMultiplier * relativeWidthStd
        return widthDistance <= rowDistance * (1 - requiredCloserFraction)
    }
}
