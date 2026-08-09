//
//  DrivingSide.swift
//  DriverAssist
//
//  Which side of the road this vehicle drives on -- right-hand traffic (US
//  and most of the world) vs left-hand traffic (UK, Japan, Australia, etc.).
//  Default is .right, matching where this app is actually being developed
//  and test-driven.
//
//  A plain persisted setting, not a live sensor reading -- there's no UI
//  control for it (no button/voice command) since it isn't expected to
//  change mid-drive, or often at all. Read directly wherever needed rather
//  than threaded through as a parameter, same reasoning as other simple
//  static settings in this codebase.
//
//  Not consumed by anything yet -- logged per detection (see
//  DetectionLogger.swift) for the same reason model/resolution are logged
//  per-frame despite usually being stable: keeps every log entry self-
//  contained without needing separate session metadata. Intended future use
//  is any lane-position/path heuristic (path_awareness.py's curve-awareness
//  work, "what's in my path") that currently implicitly assumes right-hand
//  traffic without being explicit about it -- logging this now means real
//  drive data exists tagged with the right answer whenever that need shows
//  up, rather than a wasted drive later.
//

import Foundation

enum DrivingSide: String, Codable {
    case right
    case left
}

enum DrivingSideSetting {
    private static let defaultsKey = "settings.drivingSide"

    static var current: DrivingSide {
        get {
            guard
                let raw = UserDefaults.standard.string(forKey: defaultsKey),
                let side = DrivingSide(rawValue: raw)
            else { return .right }
            return side
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: defaultsKey)
        }
    }
}
