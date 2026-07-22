//
//  Utilities.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/21/26.
//

/// Wraps a non-Sendable value so it can cross a `@Sendable` closure boundary.
///
/// `CVPixelBuffer` is a Core Foundation type that is not marked `Sendable`,
/// but its retain/release and read access are safe across threads as long as
/// nobody mutates it concurrently. Here the buffer is handed off once to the
/// background queue and not touched afterwards on the calling side, so
/// asserting `@unchecked Sendable` is safe.
struct UncheckedSendableBox<Value>: @unchecked Sendable {
    let value: Value
}
