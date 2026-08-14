//
//  BicycleComboStrategy.swift
//  DriverAssist
//
//  person + bicycle -> "cyclist". See ComboAssociationStrategy.swift for
//  the shared geometry this uses and its NOT YET TUNED caveat -- threshold
//  below is a reasoned starting point, not a fitted one.
//

import Foundation

struct BicycleComboStrategy: ComboAssociationStrategy {
    let vehicleLabel = "bicycle"
    let outputLabel = "cyclist"
    let maxRidersPerVehicle = 1
    let acceptanceThreshold = 0.35

    func affinityScore(person: Detection, vehicle: Detection) -> Double? {
        RiderAboveVehicleAffinity.score(person: person, vehicle: vehicle)
    }

    func combine(vehicle: Detection, riders: [Detection]) -> Detection {
        RiderAboveVehicleAffinity.combine(vehicle: vehicle, riders: riders, outputLabel: outputLabel)
    }
}
