//
//  DebugFileLogger.swift
//  DriverAssist
//
//  TEMPORARY diagnostic aid — not for production use. Appends timestamped lines to
//  a file in Documents so behavior can be inspected after a run, bypassing the
//  unreliable wireless console relay.
//

import Foundation

enum DebugFileLogger {
    private static let url: URL = {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return dir.appendingPathComponent("overlay-debug.log")
    }()

    private static let queue = DispatchQueue(label: "DebugFileLogger")

    static func reset() {
        queue.async { try? FileManager.default.removeItem(at: url) }
    }

    static func log(_ message: String) {
        let line = "\(Date().timeIntervalSince1970) \(message)\n"
        queue.async {
            guard let data = line.data(using: .utf8) else { return }
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            } else {
                try? data.write(to: url)
            }
        }
    }
}
