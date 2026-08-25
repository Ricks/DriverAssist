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

/// Real per-class object width/height mean/std, computed from actual KITTI
/// training labels (7,481 images, 28,742 Car / 4,487 Pedestrian instances --
/// not estimated or guessed; height added 2026-08-23 from the same label
/// set's own 3D `dimensions` field, alongside the original width --
/// re-extracting width from that same field reproduced the original
/// 1.629/0.102 car and 0.660/0.143 person numbers exactly, confirming this
/// is the same source data, not a second guess). Only classes with a
/// direct, well-matched prior are included:
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
        /// Real per-class HEIGHT mean, meters -- added alongside
        /// `meanMeters`/`stdMeters` specifically so `expectedHeadOnAspect`
        /// below has a real width/height ratio to compare a box's own
        /// aspect ratio against (see WidthDistanceOverrideManager's
        /// obliqueness gate for why: pixel aspect ratio = realWidth/
        /// realHeight is only true for a genuine head-on/tail-on view, a
        /// CONFIRMED 2026-08-23 real-drive failure mode, see
        /// project_width_based_distance_override memory).
        let heightMeanMeters: Double
        let heightStdMeters: Double
        /// How much wider than `expectedHeadOnAspect` a real box may be
        /// before WidthDistanceOverrideManager's obliqueness gate rejects
        /// it -- PER-CLASS because a rigid, elongated body (car) departs
        /// from its head-on aspect ratio far less from ordinary detector
        /// noise than a walking human does (gait/arm-swing/clothing).
        /// CONFIRMED 2026-08-23 empirically for "person": replaying every
        /// already-known-good override activation from the real
        /// data/26_08_15_Walkaround validation session's 3m window (61
        /// frames, the same window documented in this file's own history
        /// as correctly activating 27/29 + 10/14 real frames) against
        /// candidate tolerances -- 1.4 preserved only 64% of that
        /// genuinely-good real data, 1.8 preserves 98% (60/61), 1.9+
        /// preserves all 61 but stops rejecting anything meaningfully.
        /// 1.8 is the chosen balance. "car" has NO equivalent real
        /// known-good head-on validation session yet (the 08-15 walkaround
        /// test used a person) -- 1.4 for car is validated only against
        /// the CONFIRMED-BAD case (track #37, data/26_08_21_Day_Small,
        /// observed aspect 1.75-2.85 vs expected 1.067, comfortably
        /// rejected at any tolerance under ~1.6), not yet checked for a
        /// false-positive rejection rate the way person's is -- real
        /// head-on car validation data would let this be tightened or
        /// loosened with actual evidence instead of a guess.
        let aspectTolerance: Double

        /// The pixel-space box aspect ratio (width/height) a genuine
        /// head-on/tail-on view of this class should show, independent of
        /// distance -- see WidthDistanceOverrideManager's obliqueness gate
        /// doc comment for the full derivation (both box dimensions share
        /// the same per-pixel focal length in this camera model, so the
        /// distance term cancels out of the ratio).
        var expectedHeadOnAspect: Double { meanMeters / heightMeanMeters }
    }

    static let byLabel: [String: Width] = [
        "car": Width(meanMeters: 1.629, stdMeters: 0.102, heightMeanMeters: 1.526, heightStdMeters: 0.137, aspectTolerance: 1.4),
        "person": Width(meanMeters: 0.660, stdMeters: 0.143, heightMeanMeters: 1.761, heightStdMeters: 0.113, aspectTolerance: 1.8),
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
    /// Minimum |smoothedYawRateDegreesPerSecond| before the obliqueness
    /// check in `evaluateGate` even runs -- see that check's own NEW
    /// 2026-08-23 comment for the two real cases (a confirmed bug and a
    /// confirmed false positive) this threshold was reasoned from.
    private static let minYawRateForObliquenessCheckDegS: Double = 1.0
    /// How close to the LEFT/RIGHT frame edge (normalized [0,1] fraction)
    /// a box may sit before it's treated as edge-truncated and rejected --
    /// see the CONFIRMED comment in `evaluateGate`. Not an exact 0.0/1.0
    /// equality check since a genuinely fully-in-frame box can still land
    /// a hair off either boundary from ordinary detector noise/rounding;
    /// a real truncated box in the confirming data sat pinned at exactly
    /// 0.000 or 1.000 (x+w), so this margin is deliberately tight -- a
    /// reasoned starting point, not yet validated against a wider set of
    /// edge-crossing real detections.
    private static let edgeTruncationMarginNormalized: Double = 0.01
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
    /// `aspectRatio` matches `DistanceEstimator.distanceMeters`'s own
    /// parameter -- pass the same value used to produce this frame's
    /// row-based readings. `yawRateDegreesPerSecond` gates the obliqueness
    /// check specifically -- see evaluateGate's own doc comment on why a
    /// stationary/non-turning ego can't be the cause of the viewing-angle
    /// distortion that check exists to catch. Relies on each detection's
    /// `groundContactAngleDegrees` already being attached (by the same
    /// InferenceEngine.attachDistances call that attaches `distanceMeters`)
    /// for the near-hood-cutoff check -- no longer takes cameraHeightMeters
    /// itself, since that angle can no longer be backed out of
    /// `distanceMeters` alone (see that field's own doc comment).
    func apply(
        to detections: [Detection],
        aspectRatio: Double,
        yawRateDegreesPerSecond: Double?
    ) -> [Detection] {
        var seenTrackIDs = Set<Int>()

        let result = detections.map { detection -> Detection in
            var detection = detection
            guard let trackID = detection.trackID else { return detection }
            seenTrackIDs.insert(trackID)

            var state = states[trackID] ?? OverrideState()
            let gatePassed = evaluateGate(
                &detection,
                aspectRatio: aspectRatio,
                yawRateDegreesPerSecond: yawRateDegreesPerSecond
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
        aspectRatio: Double,
        yawRateDegreesPerSecond: Double?
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

        // CONFIRMED 2026-08-22 (real drive, data/26_08_21_Day_Small, ~2:03):
        // a box truncated by the LEFT or RIGHT frame edge (a parked car the
        // vehicle is passing, partially cropped out of frame) has an
        // apparent width narrower than the real object's, for a reason that
        // has nothing to do with distance -- part of it is simply off-
        // screen. widthBasedDistanceMeters has no way to tell that apart
        // from a genuinely narrow/far object, so it reads a rapidly
        // inflating distance as the crop worsens: two real, stationary
        // parked cars in this session showed distanceMeters balloon from
        // 6.2m to 20.9m and 4.6m to 6.2m in under a second, purely from
        // edge cropping, not real motion -- surfaced by a downstream optic-
        // flow visualization whose arrows depend on distance accuracy, but
        // this is a real distance-estimation bug independent of that tool.
        // This is the LEFT/RIGHT counterpart to the file's existing hood-
        // truncation handling (bottom edge, near-field) -- that one is
        // about the box bottom being pinned to the ego hood instead of the
        // object's real ground contact; this one is about the box's own
        // width being an underestimate of the real object's width. widthDistanceMeters
        // stays attached/logged either way (same "log everything" discipline
        // as the rest of this file) -- only the trigger below is gated.
        let isEdgeTruncated = Double(detection.boundingBox.minX) <= Self.edgeTruncationMarginNormalized
            || Double(detection.boundingBox.maxX) >= 1.0 - Self.edgeTruncationMarginNormalized
        guard !isEdgeTruncated else { return false }

        // NEW 2026-08-23, CONFIRMED via real drive (data/26_08_21_Day_Small,
        // track #37 in this session's raw log -- rendered as #24 in the
        // offline flow-arrow video, which reruns its own separate tracker
        // offline and assigns its own IDs -- ~107-110s): `widthBasedDistanceMeters` assumes the
        // box's pixel width IS the class's real width (ObjectWidthPriors),
        // which only holds for a genuine head-on/tail-on view. A parked
        // car swept across frame by the ego vehicle's own yaw during a
        // slow turn was viewed at a persistent oblique/quartering angle
        // instead, and its box picked up foreshortened LENGTH on top of
        // width -- the override read it as ~2x closer than a real,
        // independently-recomputed row-based/ground-contact distance,
        // consistent the entire ~1.5s window, not a one-frame fluke. Fed
        // into the offline flow-arrow visualization's 1/z term, this
        // produced a predicted flow ~1.9x too large and a large spurious
        // "independent motion" residual on an object later confirmed
        // (via the source video) to never have moved at all.
        //
        // Gate: pixel-space box aspect ratio (width/height) equals
        // realWidth/realHeight ONLY for a head-on/tail-on view, because
        // both box dimensions share the same per-pixel focal length in
        // this camera model (DistanceEstimator's column focal length is
        // derived from the row focal length via `aspectRatio`, which
        // cancels out of the ratio once both dimensions are converted to
        // pixels) -- confirmed against track #37's own numbers: observed
        // pixel aspect ~2.85 vs car's expected head-on ~1.07, a ~2.7x
        // inflation, matching the ~2x distance understatement closely
        // enough to be the same effect, not a coincidence. As viewing
        // angle rotates away from head-on toward broadside, foreshortened
        // length can only ADD to the visible footprint width (height is
        // unaffected by a yaw rotation of the object), so pixel aspect
        // ratio can only grow above the head-on value -- one-directional,
        // same design as every other gate in this file.
        //
        // NEW 2026-08-23, CONFIRMED FALSE POSITIVE via the same real
        // session (data/26_08_21_Day_Small, track #13 in the raw log,
        // t~0s): a genuinely PARKED, dead-ahead car (confirmed via the
        // source frame -- squarely facing the camera) was rejected by this
        // check anyway (observed pixel aspect 2.36 vs a ~1.49 threshold),
        // because its box height came out short for a reason that has
        // NOTHING to do with viewing angle: it sat right at the edge of
        // the near-field regime (row-based phi 8.97 degrees, just 0.93
        // degrees under `DistanceEstimator.hoodCutoffAngleDegrees` =
        // 9.899), close enough that the ego's own hood/dash edge was
        // already encroaching on its visible bottom without (yet) crossing
        // the HARD cutoff the hood-truncation case above catches.
        // Rejecting it substituted a WORSE row-based reading (6.45m) for a
        // width-based one (3.03m) that was correct -- confirmed against a
        // real user report that the car was right in front of the
        // vehicle, not ~6m away.
        //
        // The oblique-view mechanism above REQUIRES the camera to be
        // rotating relative to the object -- a foreshortened, more-square
        // box only appears as viewing angle sweeps away from head-on. At
        // this false positive's own timestamp, smoothedYawRateDegreesPer
        // Second was ~0.003 deg/s and egoSpeedMps was 0 (parked, not
        // turning) -- nothing could have been rotating the view. The
        // confirmed-BAD case (track #37 above) only ever showed this
        // failure mode while yaw rate was ramping through 2.6-6.2 deg/s.
        // A yaw-rate floor separates the two real cases with wide margin,
        // so the check below only runs when the ego is actually turning
        // meaningfully -- PROVISIONAL threshold, reasoned from these two
        // real cases, not yet validated against a wider set.
        //
        // KNOWN REMAINING GAP: this doesn't catch a car viewed at a
        // genuinely constant oblique angle from a STATIONARY or
        // straight-driving ego (no yaw at all, but the object itself sits
        // at an angle to the road) -- untested territory, left for
        // whenever real data surfaces it, same "don't fix what isn't
        // confirmed yet" discipline as the rest of this file.
        if let yawRate = yawRateDegreesPerSecond, abs(yawRate) >= Self.minYawRateForObliquenessCheckDegS {
            let boxHeightNormalized = Double(detection.boundingBox.height)
            guard boxHeightNormalized > 0 else { return false }
            // (width/height in NORMALIZED units) * aspectRatio ==
            // width/height in actual PIXELS -- normalizing divides width
            // by frame width and height by frame height, so their ratio is
            // off by exactly frameWidthPx/frameHeightPx (== aspectRatio)
            // from the true pixel ratio; multiplying back by aspectRatio
            // here undoes that.
            let observedPixelAspect = (boxWidth / boxHeightNormalized) * aspectRatio
            guard observedPixelAspect <= prior.expectedHeadOnAspect * prior.aspectTolerance else {
                return false
            }
        }

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

        guard let phiDegrees = detection.groundContactAngleDegrees else { return false }
        let isNearCutoff = phiDegrees <= DistanceEstimator.hoodCutoffAngleDegrees + nearCutoffMarginDegrees
        let confidenceFloor = isNearCutoff ? nearCutoffConfidenceFloor : normalConfidenceFloor
        guard detection.confidence >= confidenceFloor else { return false }

        let relativeWidthStd = prior.stdMeters / prior.meanMeters
        let requiredCloserFraction = minCloserFraction + widthUncertaintyMultiplier * relativeWidthStd
        return widthDistance <= rowDistance * (1 - requiredCloserFraction)
    }
}
