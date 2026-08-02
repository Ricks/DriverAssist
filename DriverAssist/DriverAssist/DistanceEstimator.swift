//
//  DistanceEstimator.swift
//  DriverAssist
//
//  Ground-plane / bottom-of-box distance estimation (see the distance/
//  velocity/rotation plan) -- a vehicle's real-world distance follows from
//  the image row where its box's bottom (ground-contact point) appears,
//  given a flat-road assumption. Modeled as v = horizonRow + k/D (a point at
//  distance D projects to row v; horizonRow is where D -> infinity projects;
//  k folds together focal length and camera height), so D = k / (v -
//  horizonRow).
//
//  Deliberately NOT the physically-separated form (camera height, focal
//  length, and pitch as three independently measured constants) --
//  horizonRow/k are meant to be fit directly from real tape-mark reference
//  points (laser-measured height + marks at known distances), which
//  sidesteps needing to trust the phone's nominal field of view or any
//  stabilization/crop factor. There are no valid constants yet -- this type
//  can't be constructed without supplying them, deliberately, so a
//  not-yet-calibrated instance can't silently produce numbers that look
//  real (same reasoning as EgoSpeedManager/PitchSensor logging nil instead
//  of a placeholder 0).
//
//  Known limitation worth validating explicitly once real calibration data
//  exists: v = horizonRow + k/D is the small-angle approximation of the
//  true (tan-based) ground-plane relation. It's exact only in the
//  level-camera, distant-point limit; with real camera pitch it drifts most
//  at *close* range -- exactly the range this app's eventual "following too
//  closely" warning cares about most. Hold at least one calibration point
//  out of the fit and check the model against it (same discipline as the
//  leading-vehicle classifier's train/held-out tuning split) rather than
//  trusting a fit through all points blindly.
//

import Foundation

struct DistanceEstimator {
    /// Row (top-left-origin normalized, 0...1 -- same convention as
    /// `Detection.boundingBox`) the ground plane's horizon projects to, i.e.
    /// what a point at D -> infinity would appear at. Fit from real
    /// reference points, not assumed.
    var horizonRowNormalized: Double

    /// Scale constant folding together focal length and camera height, in
    /// units of (normalized-row * meters) so that `k / (v - horizonRow)`
    /// comes out in meters. Fit alongside `horizonRowNormalized` from the
    /// same reference points -- the two numbers aren't independently
    /// meaningful outside that fit.
    var kMeters: Double

    /// Distance in meters to the ground-contact point at normalized row
    /// `bottomY` (0 = top of frame, 1 = bottom -- pass a detection's
    /// `boundingBox.maxY`). Returns nil for a row at or above the horizon
    /// (v <= horizonRow): geometrically that's not a point on the ground
    /// plane in front of the camera -- an artifact, a mismeasured box, or
    /// the flat-road assumption not holding -- not a far-but-finite distance.
    func distanceMeters(bottomY: Double) -> Double? {
        let denom = bottomY - horizonRowNormalized
        guard denom > 0 else { return nil }
        return kMeters / denom
    }

    func distanceFeet(bottomY: Double) -> Double? {
        distanceMeters(bottomY: bottomY).map { $0 * 3.28084 }
    }

    /// Solves the 2-parameter model exactly from two (row, distance)
    /// reference points -- e.g. two tape-mark measurements. With 3+ points,
    /// fit on two and check the model's prediction against the held-out
    /// one(s) rather than averaging all of them into a single fit -- the
    /// whole point of the extra point is to catch this model being wrong,
    /// which an all-points fit would hide.
    static func fit(row1: Double, distance1Meters: Double, row2: Double, distance2Meters: Double) -> DistanceEstimator? {
        // v = horizonRow + k/D  =>  k = (v1 - v2) / (1/D1 - 1/D2)
        let invD1 = 1 / distance1Meters
        let invD2 = 1 / distance2Meters
        let denom = invD1 - invD2
        guard denom != 0 else { return nil }
        let k = (row1 - row2) / denom
        let horizonRow = row1 - k / distance1Meters
        return DistanceEstimator(horizonRowNormalized: horizonRow, kMeters: k)
    }
}
