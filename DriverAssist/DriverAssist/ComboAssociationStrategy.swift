//
//  ComboAssociationStrategy.swift
//  DriverAssist
//
//  Strategy seam for combining a "vehicle"-class detection (bicycle,
//  motorcycle, ...) with the person(s) riding it into a single synthesized
//  detection (cyclist, motorcyclist, ...). One conformance per combo type --
//  see ComboManager.swift for the shared Hungarian-assignment/hysteresis
//  machinery every conformance rides on top of, and DriverAssist's
//  project memory for the full architecture discussion this implements.
//
//  Swapping which concrete type backs a given combo type during development
//  is a one-line change at ComboManager's construction site (in
//  InferenceEngine.init) -- no changes needed here or in ComboManager
//  itself. That's the whole point of this being a protocol.
//

import CoreGraphics
import Foundation

protocol ComboAssociationStrategy {
    /// Detection.label this strategy pairs against "person" -- e.g. "bicycle".
    var vehicleLabel: String { get }

    /// Detection.label this strategy synthesizes -- e.g. "cyclist".
    var outputLabel: String { get }

    /// How many people this strategy will attach to one vehicle -- 1 for
    /// bicycle/skateboard/horse, 2 for motorcycle (driver + passenger).
    /// This is the entire mechanism for the motorcycle-passenger case:
    /// declarative data here, not special-cased code in ComboManager.
    var maxRidersPerVehicle: Int { get }

    /// Minimum score to accept a pairing outside of an already-confirmed
    /// pairing's grace period. Hungarian assignment always finds SOME
    /// match when candidates exist, so below-threshold pairs must be
    /// discarded post-hoc by ComboManager, not encoded as an infeasible
    /// cost (which would just make a *different*, still-bad pairing win).
    var acceptanceThreshold: Double { get }

    /// Higher = better match; nil = geometrically impossible, excluded
    /// from the assignment problem entirely rather than scored low.
    func affinityScore(person: Detection, vehicle: Detection) -> Double?

    /// Synthesizes the single output Detection from a matched vehicle plus
    /// its 1+ riders -- owns box/confidence/trackID policy. Deliberately
    /// not a fixed union-box formula shared across all strategies: a
    /// motorcyclist's true ground-contact point may need different
    /// handling than a cyclist's once this feeds distance estimation.
    func combine(vehicle: Detection, riders: [Detection]) -> Detection
}

/// Shared geometry for the "rider's box sits above and overlaps the
/// vehicle's box" family (bicycle, motorcycle) -- NOT a universal fit.
/// Skateboard (rider's feet sit ON TOP of/behind a low board, minimal
/// vertical separation) and horse (rider sits mostly WITHIN the horse's
/// own box, not above it) need their own geometric relationship, not just
/// different tuning of this one -- see project memory. Kept here, not
/// duplicated per-strategy, since BicycleComboStrategy and
/// MotorcycleComboStrategy are meant to share the same formula and differ
/// only in their own threshold/weight constants.
///
/// *** NOT YET TUNED against real labeled data. *** The labeling tool
/// (tools/label_leading_vehicle_frontend.html, cyclist/motorcyclist modes)
/// exists and was verified working, but no real cyclist/motorcyclist
/// ground truth has been collected yet -- these weights/thresholds are a
/// reasoned starting point, not a fitted one. Revisit with a grid search
/// once labeled data exists, same discipline as classify_leading's gates.
enum RiderAboveVehicleAffinity {
    static func score(person: Detection, vehicle: Detection) -> Double? {
        let p = person.boundingBox
        let v = vehicle.boundingBox

        let intersection = p.intersection(v)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return nil }
        guard v.width > 0, v.height > 0, p.width > 0, p.height > 0 else { return nil }

        // Overlap relative to the SMALLER (vehicle) box, not symmetric IoU --
        // a genuine rider's box only partially overlaps the vehicle's, so
        // plain IoU reads misleadingly low for real pairs.
        let overlapFraction = (intersection.width * intersection.height) / (v.width * v.height)

        // Vertical alignment: the vehicle's vertical center should sit at
        // or below the person's (rider's torso above the bike/motorcycle
        // frame), normalized by the person's own height so this scales
        // with the detection's size in frame rather than a fixed pixel
        // amount -- near/far detections have very different absolute
        // extents, the same lesson already learned in DistanceEstimator's
        // calibration work.
        let verticalOffsetFraction = (v.midY - p.midY) / p.height
        let verticalScore = verticalOffsetFraction >= 0
            ? 1.0
            : max(0, 1.0 + verticalOffsetFraction * 2)

        // Horizontal centering: how far the vehicle's horizontal center is
        // from the person's, normalized by the person's own width.
        let horizontalOffsetFraction = abs(v.midX - p.midX) / p.width
        let horizontalScore = max(0, 1.0 - horizontalOffsetFraction)

        return overlapFraction * 0.5 + verticalScore * 0.3 + horizontalScore * 0.2
    }

    /// Union bounding box + conservative (min) confidence -- shared
    /// synthesis policy for the bicycle/motorcycle family. `trackID` is
    /// always the vehicle's, the more stable of the two boxes in practice.
    static func combine(vehicle: Detection, riders: [Detection], outputLabel: String) -> Detection {
        var box = vehicle.boundingBox
        for rider in riders { box = box.union(rider.boundingBox) }
        let confidence = ([vehicle.confidence] + riders.map { $0.confidence }).min() ?? vehicle.confidence
        var combo = Detection(label: outputLabel, confidence: confidence, boundingBox: box)
        combo.trackID = vehicle.trackID
        return combo
    }
}
