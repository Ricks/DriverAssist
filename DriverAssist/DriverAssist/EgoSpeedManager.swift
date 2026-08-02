//
//  EgoSpeedManager.swift
//  DriverAssist
//
//  Publishes the vehicle's own ground speed from GPS (CLLocation.speed, in
//  m/s) -- deliberately NOT derived from CoreMotion/accelerometer, which
//  requires integrating acceleration and drifts too fast for sustained speed
//  estimation on a phone that isn't rigidly fixed to the chassis (see the
//  ego-speed design discussion). GPS gives an already-filtered speed
//  directly, no integration.
//
//  This is groundwork, not yet consumed: the intended first use is
//  disambiguating "following a vehicle at matched speed" from "both vehicles
//  stationary" in the leading-vehicle classifier (currently indistinguishable
//  -- see tools/leading_vehicle.py's design notes on trackID #13), and later
//  the closing-rate/following-distance-warning math. For now this only logs
//  speed (see DetectionLogger.swift) so real drive data exists to validate
//  against once the classifier/warning work actually consumes it.
//

import CoreLocation
import Foundation

final class EgoSpeedManager: NSObject, ObservableObject {
    /// nil until the first valid fix -- CLLocation.speed is negative when
    /// there isn't one yet, which callers should treat as "no data", not 0.
    @Published private(set) var speedMps: Double?
    @Published private(set) var speedAccuracyMps: Double?
    @Published private(set) var authorizationDenied = false

    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        // Tunes GPS filtering for driving specifically (vs. e.g. pedestrian/
        // fitness), matching how this data will actually be used.
        manager.activityType = .automotiveNavigation
    }

    func start() {
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            manager.startUpdatingLocation()
        case .notDetermined:
            manager.requestWhenInUseAuthorization()  // resumes in locationManagerDidChangeAuthorization
        default:
            authorizationDenied = true
        }
    }

    func stop() {
        manager.stopUpdatingLocation()
    }
}

extension EgoSpeedManager: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        Task { @MainActor [weak self] in
            guard let self else { return }
            switch status {
            case .authorizedWhenInUse, .authorizedAlways:
                self.manager.startUpdatingLocation()
            case .denied, .restricted:
                self.authorizationDenied = true
            default:
                break
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let latest = locations.last, latest.speed >= 0 else { return }
        let speed = latest.speed
        let accuracy = latest.speedAccuracy >= 0 ? latest.speedAccuracy : nil
        Task { @MainActor [weak self] in
            self?.speedMps = speed
            self?.speedAccuracyMps = accuracy
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("[EgoSpeedManager] location update failed: \(error)")
    }
}
