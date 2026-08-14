//
//  MotorcycleComboStrategy.swift
//  DriverAssist
//
//  person(s) + motorcycle -> "motorcyclist". Same geometry family as
//  BicycleComboStrategy (see ComboAssociationStrategy.swift and its NOT
//  YET TUNED caveat), but maxRidersPerVehicle=2 for a passenger -- that's
//  the whole mechanism for the passenger case, handled generically by
//  ComboManager's passenger-relaxation pass, not special-cased here.
//  Threshold kept identical to bicycle's for now as a starting point;
//  project memory flags motorcycle/bicycle thresholds as likely needing
//  separate tuning once real data exists (motorcycles are generally
//  larger/taller than bicycles), not yet done.
//

import Foundation

struct MotorcycleComboStrategy: ComboAssociationStrategy {
    let vehicleLabel = "motorcycle"
    let outputLabel = "motorcyclist"
    let maxRidersPerVehicle = 2
    let acceptanceThreshold = 0.35

    func affinityScore(person: Detection, vehicle: Detection) -> Double? {
        RiderAboveVehicleAffinity.score(person: person, vehicle: vehicle)
    }

    func combine(vehicle: Detection, riders: [Detection]) -> Detection {
        RiderAboveVehicleAffinity.combine(vehicle: vehicle, riders: riders, outputLabel: outputLabel)
    }
}
