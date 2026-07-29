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

    /// Archives (never deletes) the previous run's log before starting a fresh one.
    /// Deleting outright is exactly what lost a real drive's log this session — a
    /// later, unrelated app launch (e.g. installing a test build) reset the file
    /// before the drive's data had been pulled off the device. Archived files are
    /// timestamped so a specific drive's log can still be found afterward.
    static func reset() {
        queue.async {
            let fm = FileManager.default
            guard fm.fileExists(atPath: url.path) else { return }
            let archiveURL = url.deletingLastPathComponent()
                .appendingPathComponent("overlay-debug-\(Int(Date().timeIntervalSince1970)).log")
            try? fm.moveItem(at: url, to: archiveURL)
        }
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
