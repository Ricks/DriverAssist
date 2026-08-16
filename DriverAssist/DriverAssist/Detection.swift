//
//  Detection.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import Foundation
import CoreGraphics

struct Detection: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let confidence: Float
    // Normalized [0, 1] in top-left origin form: (x, y, width, height)
    let boundingBox: CGRect
    /// Persistent identity assigned by ByteTracker -- nil until tracking has
    /// run (or for a low-confidence detection that matched nothing). Not the
    /// same as `id`: `id` is a fresh UUID every detection, purely for
    /// SwiftUI's Identifiable; `trackID` is what stays stable across frames
    /// for the same real-world object.
    var trackID: Int? = nil
    /// Estimated ground-plane distance to this detection, meters -- see
    /// DistanceEstimator.swift. nil until InferenceEngine attaches it (not
    /// computed here), and stays nil whenever the estimator itself returns
    /// nil (no reference pitch captured yet, or invalid geometry). This is
    /// the single authoritative "best estimate" -- if WidthDistanceOverride
    /// activates for this detection, THIS value becomes the (sanity-capped)
    /// width-based reading, not the original row-based one; see
    /// `distanceMetersIsWidthOverridden` to tell which is currently here.
    var distanceMeters: Double? = nil
    /// Distance implied by this detection's box WIDTH (see
    /// DistanceEstimator.widthBasedDistanceMeters) -- informational, always
    /// computed when the class has a real-world width prior
    /// (ObjectWidthPriors) and a row-based reading exists to compare
    /// against, regardless of whether the override actually activated.
    /// nil for classes with no width prior, or before WidthDistanceOverride
    /// has run.
    var widthDistanceMeters: Double? = nil
    /// True if `distanceMeters` currently holds the width-based reading
    /// (WidthDistanceOverrideManager's hysteresis-confirmed override),
    /// false if it's still the original row-based `DistanceEstimator`
    /// reading.
    var distanceMetersIsWidthOverridden: Bool = false
}
