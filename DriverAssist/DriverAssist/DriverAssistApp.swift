//
//  DriverAssistApp.swift
//  DriverAssist
//
//  Created by Rick Clark on 7/20/26.
//

import SwiftUI
import UIKit

private class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        .landscape
    }
}

@main
struct DriverAssistApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            // TEMPORARY 2026-08-12: voice-warning test bench, swapped in for
            // ContentView() while auditioning AVSpeechSynthesizer voices/
            // accents for the verbal-warnings feature. Swap back to
            // ContentView() before resuming normal app testing.
            VoiceTestView()
        }
    }
}
