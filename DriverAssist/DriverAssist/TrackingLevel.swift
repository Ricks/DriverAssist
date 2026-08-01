import Foundation

/// Tracking-quality tier -- mirrors `DetectorModel`'s persisted-setting
/// pattern (see ModelManager). Deliberately just two cases, no "off"/"basic":
/// tracking itself is foundational (not an optional extra), and Recovery's
/// two-confidence-tier matching runs at effectively the same cost as no
/// tracking at all, so there's no reason to offer a cheaper tier below it.
///
/// - recovery: motion (Kalman) + two-confidence-tier ByteTrack-style
///   matching + camera-motion compensation. Fully implemented and validated
///   offline (see tools/track_benchmark.py sweeps).
/// - appearance: same, plus ReID embedding-based re-identification for
///   pairs geometry alone can't resolve. Dropped for MVP: no on-device
///   embedding model is wired in (see AppearanceEmbedder.swift), and getting
///   one there hit real, open-ended conversion difficulty -- not worth it
///   for the modest validated gain over .recovery (+GMC) alone. This case
///   stays in the enum/ByteTracker's matching logic for whenever it's
///   revisited, but isn't reachable from any UI or voice command right now
///   -- the app always runs .recovery.
enum TrackingLevel: String, CaseIterable, Codable {
    case recovery
    case appearance
}
