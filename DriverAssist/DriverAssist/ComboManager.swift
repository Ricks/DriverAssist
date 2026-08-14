//
//  ComboManager.swift
//  DriverAssist
//
//  Orchestrator for combining person + "vehicle" detections (bicycle,
//  motorcycle, ...) into synthesized rider detections (cyclist,
//  motorcyclist, ...), per the registered ComboAssociationStrategy list.
//  Mirrors TrackingManager's ownership pattern: owned/constructed by
//  InferenceEngine, plain @MainActor class (not ObservableObject --
//  InferenceEngine already republishes the resulting `detections`).
//
//  PURELY A DISPLAY/CONSUMPTION-LAYER TRANSFORM. DetectionLogger must keep
//  logging the RAW, pre-combine detections -- several offline tools
//  (tune_leading_vehicle.py etc.) and the cyclist/motorcyclist labeling
//  tool read detections.jsonl directly and depend on seeing the original
//  separate person/vehicle boxes. Call combine(_:) only on the copy fed to
//  the overlay/UI, after DetectionLogger has already logged the input.
//
//  Cross-cutting logic lives here once instead of being duplicated per
//  strategy: builds a person x vehicle cost matrix per strategy and solves
//  it via the existing HungarianAlgorithm.solve(cost:) (a free-standing,
//  generically-typed solver -- no ByteTracker coupling, confirmed reusable
//  as originally hoped when this was designed). A second, simple greedy
//  pass then handles any strategy with maxRidersPerVehicle > 1 (the
//  motorcycle-passenger case) -- one shared code path parameterized by the
//  strategy's declared max, not a motorcycle-specific fork.
//
//  Hysteresis mirrors leading_vehicle.py's confirm_frames/grace_frames
//  pattern (defaults 3/5, matching that file's DEFAULT_CONFIRM_FRAMES/
//  DEFAULT_GRACE_FRAMES) rather than inventing a new scheme, adapted to
//  this file's different data shape (multiple simultaneous pairings, keyed
//  by vehicle trackID -- globally unique across classes per
//  TrackingManager/ByteTracker, so no per-strategy key is needed) instead
//  of leading_vehicle.py's single locked candidate. Simplified relative to
//  that file in one respect, flagged explicitly: a miss before
//  confirmation resets progress to zero here, where leading_vehicle.py
//  tolerates brief gaps even pre-confirmation. Revisit if that turns out
//  to matter once real data/tuning exists -- this is a reasoned first
//  pass, not a byte-for-byte port.
//
//  A confirmed pairing's grace period ALSO loosens acceptance (any
//  non-nil affinityScore, not just one clearing the strategy's
//  acceptanceThreshold) rather than replaying a stale box -- every frame's
//  output always uses that frame's live detections, never a frozen
//  position from an earlier frame.

import Foundation

@MainActor
final class ComboManager {
    private let strategies: [ComboAssociationStrategy]
    private let confirmFrames: Int
    private let graceFrames: Int

    private struct PairState {
        /// Consecutive accepted frames while not yet confirmed.
        var streak: Int = 0
        var confirmed: Bool = false
        /// Consecutive frames since the last accepted (incl. lenient
        /// grace-period) match -- confirmed pairings are dropped once this
        /// exceeds graceFrames.
        var framesSinceSeen: Int = 0
    }

    /// Keyed by the vehicle's trackID.
    private var states: [Int: PairState] = [:]

    init(strategies: [ComboAssociationStrategy], confirmFrames: Int = 3, graceFrames: Int = 5) {
        self.strategies = strategies
        self.confirmFrames = confirmFrames
        self.graceFrames = graceFrames
    }

    func combine(_ detections: [Detection]) -> [Detection] {
        let people = detections.enumerated().filter { $0.element.label == "person" && $0.element.trackID != nil }

        var consumedPersonOffsets = Set<Int>()
        var touchedVehicleTrackIDs = Set<Int>()
        var output = detections
        var riderOffsetsToRemove = Set<Int>()

        for strategy in strategies {
            let vehicles = detections.enumerated().filter { $0.element.label == strategy.vehicleLabel && $0.element.trackID != nil }
            guard !vehicles.isEmpty else { continue }

            let availablePeople = people.filter { !consumedPersonOffsets.contains($0.offset) }
            let scoreMatrix: [[Double?]] = vehicles.map { vehicle in
                availablePeople.map { person in strategy.affinityScore(person: person.element, vehicle: vehicle.element) }
            }
            let costMatrix: [[Double]] = scoreMatrix.map { row in row.map { $0.map { -$0 } ?? 1e6 } }
            let assignment = availablePeople.isEmpty ? [] : HungarianAlgorithm.solve(cost: costMatrix)

            for (vIdx, vehicleEntry) in vehicles.enumerated() {
                let vehicleTrackID = vehicleEntry.element.trackID!
                touchedVehicleTrackIDs.insert(vehicleTrackID)
                var state = states[vehicleTrackID] ?? PairState()
                let inGracePeriod = state.confirmed

                var matchedRiders: [(offset: Int, element: Detection)] = []
                if !assignment.isEmpty, vIdx < assignment.count, let pIdx = assignment[vIdx],
                   let score = scoreMatrix[vIdx][pIdx] {
                    let accept = inGracePeriod ? true : score >= strategy.acceptanceThreshold
                    if accept { matchedRiders.append(availablePeople[pIdx]) }
                }

                // Passenger relaxation: only once the primary pairing
                // succeeded, and only for strategies that allow it.
                if !matchedRiders.isEmpty, strategy.maxRidersPerVehicle > 1 {
                    let claimedOffsets = Set(matchedRiders.map { $0.offset })
                    let remaining = availablePeople.filter { !claimedOffsets.contains($0.offset) && !consumedPersonOffsets.contains($0.offset) }
                    let scored = remaining.compactMap { person -> (score: Double, entry: (offset: Int, element: Detection))? in
                        guard let score = strategy.affinityScore(person: person.element, vehicle: vehicleEntry.element),
                              score >= strategy.acceptanceThreshold else { return nil }
                        return (score, person)
                    }.sorted { $0.score > $1.score }
                    for candidate in scored {
                        if matchedRiders.count >= strategy.maxRidersPerVehicle { break }
                        matchedRiders.append(candidate.entry)
                    }
                }

                if !matchedRiders.isEmpty {
                    state.framesSinceSeen = 0
                    if !state.confirmed {
                        state.streak += 1
                        if state.streak >= confirmFrames { state.confirmed = true }
                    }
                    for r in matchedRiders { consumedPersonOffsets.insert(r.offset) }
                } else {
                    state.streak = 0
                    state.framesSinceSeen += 1
                }

                if state.confirmed, !matchedRiders.isEmpty {
                    let riderDetections = matchedRiders.map { $0.element }
                    output[vehicleEntry.offset] = strategy.combine(vehicle: vehicleEntry.element, riders: riderDetections)
                    for r in matchedRiders { riderOffsetsToRemove.insert(r.offset) }
                }

                if state.framesSinceSeen > graceFrames {
                    states.removeValue(forKey: vehicleTrackID)
                } else {
                    states[vehicleTrackID] = state
                }
            }
        }

        // Decay/evict state for any previously-tracked vehicle not seen by
        // ANY strategy this frame (e.g. it briefly dropped out of
        // detection entirely). Snapshot keys first -- removing entries
        // while iterating states.keys directly is unsafe.
        for trackID in Array(states.keys) where !touchedVehicleTrackIDs.contains(trackID) {
            states[trackID]?.framesSinceSeen += 1
            if let framesSinceSeen = states[trackID]?.framesSinceSeen, framesSinceSeen > graceFrames {
                states.removeValue(forKey: trackID)
            }
        }

        guard !riderOffsetsToRemove.isEmpty else { return output }
        return output.enumerated()
            .filter { !riderOffsetsToRemove.contains($0.offset) }
            .map { $0.element }
    }
}
