//
//  LensCalibration.swift
//  DriverAssist
//
//  Real per-device factory lens calibration (AVCameraCalibrationData) --
//  built 2026-08-24 after a real tethered walkaround test showed
//  DistanceEstimator's plain pinhole model understating distance by up to
//  ~25% for detections far off the frame's horizontal center, in a way
//  that survived fixing two other confirmed bugs in that session (see
//  row_based_distance_meters' and corrected_distance_meters' own doc
//  comments in tools/reconstruct_annotated.py). What's left after those
//  fixes is exactly what an IDEAL pinhole model can't represent: real lens
//  distortion, worst toward the frame edges.
//
//  Rather than a manual checkerboard calibration, this uses Apple's own
//  factory-measured per-device calibration data -- a real intrinsic matrix
//  plus a radial distortion lookup table, delivered via AVCapturePhotoOutput
//  (NOT the AVCaptureVideoDataOutput this app's live pipeline already uses --
//  calibration data delivery is a photo-capture-only API). Manufacturing
//  tolerances genuinely vary lens to lens, which is why this is captured
//  fresh from the actual device rather than hardcoded from a spec sheet --
//  see LensCalibrationCapture below.
//
//  *** UNVALIDATED ON A REAL DEVICE as of this writing. The capture
//  *** mechanics and the lookup-table interpolation math both follow
//  *** Apple's own documented approach as closely as this could be written
//  *** without a physical device/Xcode build in hand -- but coordinate-
//  *** space assumptions (pixel vs normalized, which resolution a table's
//  *** radii are relative to) are exactly the kind of thing that took real
//  *** device data to get right for `distanceMeters` itself earlier this
//  *** same day. Confirm `isCameraCalibrationDataDeliverySupported` returns
//  *** true and sanity-check a few corrected points against known-straight
//  *** real-world lines before trusting this for anything safety-relevant.

import AVFoundation
import CoreGraphics
import Foundation

/// Everything DistanceEstimator needs to correct a detection's raw
/// (distorted) pixel position to where it would appear in an ideal pinhole
/// image, extracted once from `AVCameraCalibrationData` at session start.
struct LensCalibrationData: Codable, Sendable {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    /// Resolution this calibration data's intrinsics/distortion tables are
    /// relative to -- NOT necessarily the same as whatever resolution a
    /// given detection's own box is normalized against (see
    /// `correctedNormalizedPoint`'s own doc comment).
    let referenceDimensionWidth: Double
    let referenceDimensionHeight: Double
    let distortionCenterX: Double
    let distortionCenterY: Double
    /// Distorted -> undistorted radial magnification table (see
    /// `lensDistortionPoint` below) -- this is the direction needed here,
    /// since real captured detections start out distorted and this project's
    /// whole ground-plane model assumes an ideal (undistorted) pinhole.
    let lensDistortionLookupTable: [Float]
    /// Undistorted -> distorted -- kept for completeness/symmetry (e.g. if
    /// something ever needs to draw an ideal-model prediction back onto the
    /// real distorted preview) but NOT used by `correctedNormalizedPoint`.
    let inverseLensDistortionLookupTable: [Float]

    init?(_ calibrationData: AVCameraCalibrationData) {
        let matrix = calibrationData.intrinsicMatrix
        self.fx = Double(matrix.columns.0.x)
        self.fy = Double(matrix.columns.1.y)
        self.cx = Double(matrix.columns.2.x)
        self.cy = Double(matrix.columns.2.y)
        let dims = calibrationData.intrinsicMatrixReferenceDimensions
        self.referenceDimensionWidth = Double(dims.width)
        self.referenceDimensionHeight = Double(dims.height)
        let center = calibrationData.lensDistortionCenter
        self.distortionCenterX = Double(center.x)
        self.distortionCenterY = Double(center.y)
        guard let table = calibrationData.lensDistortionLookupTable,
              let inverseTable = calibrationData.inverseLensDistortionLookupTable
        else { return nil }
        self.lensDistortionLookupTable = table.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        self.inverseLensDistortionLookupTable = inverseTable.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private init(
        fx: Double, fy: Double, cx: Double, cy: Double,
        referenceDimensionWidth: Double, referenceDimensionHeight: Double,
        distortionCenterX: Double, distortionCenterY: Double,
        lensDistortionLookupTable: [Float], inverseLensDistortionLookupTable: [Float]
    ) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.referenceDimensionWidth = referenceDimensionWidth
        self.referenceDimensionHeight = referenceDimensionHeight
        self.distortionCenterX = distortionCenterX
        self.distortionCenterY = distortionCenterY
        self.lensDistortionLookupTable = lensDistortionLookupTable
        self.inverseLensDistortionLookupTable = inverseLensDistortionLookupTable
    }

    /// Real capture, 2026-08-24, from this exact physical device (iPhone 17
    /// Pro Max) via IsolatedLensCalibrationCapture -- see that type's own
    /// doc comment for the full capture recipe (GDC disabled, 2+
    /// constituent devices required in settings, virtual-constituent-
    /// delivery enabled before startRunning). Cross-checked against the
    /// independent cone-calibration fit (DistanceEstimator.calibrated
    /// .focalLengthNormalized = 1.322673): converted to the same column-
    /// normalized units (fx / referenceDimensionWidth = 0.7116 vs
    /// 1.322673 / (16/9) = 0.7440), these agree to within 4.4% -- a
    /// reasonable match given this data comes from a full still-photo
    /// capture (4224x2376) rather than the cropped/stabilized video
    /// pipeline (1920x1080) the cone test used, not a red flag. Confirmed
    /// independently via BOTH .builtInTripleCamera (fx=3005.68) and
    /// .builtInDualWideCamera (fx=3005.46, used here) -- the two agree to
    /// within 0.01%. A fixed hardware/lens property, not expected to
    /// change unless this phone's camera module itself is repaired or
    /// replaced. Raw capture backed up at
    /// calibration/lens_calibration_20260824.json.
    static let factoryMeasured = LensCalibrationData(
        fx: 3005.462158203125, fy: 3005.462158203125,
        cx: 2103.8779296875, cy: 1188.294921875,
        referenceDimensionWidth: 4224, referenceDimensionHeight: 2376,
        distortionCenterX: 2103.962646484375, distortionCenterY: 1188.3253173828125,
        lensDistortionLookupTable: [
            0, 9.703804e-05, 0.0003860671, 0.000860824, 0.0015108482, 0.002321466, 0.0032738035,
            0.004344853, 0.005507639, 0.0067315097, 0.007982596, 0.00922447, 0.010419019,
            0.011527535, 0.012512018, 0.013336645, 0.01396934, 0.014383375, 0.014558875,
            0.014484134, 0.014156586, 0.013583343, 0.012781186, 0.011775943, 0.010601236,
            0.009296611, 0.00790514, 0.006470612, 0.0050345063, 0.0036329643, 0.0022940412,
            0.0010355223, -0.00013637631, -0.0012271275, -0.0022525825, -0.0032360968,
            -0.004204166, -0.005180965, -0.006182665, -0.0072130556, -0.008262857, -0.009316202,
        ],
        inverseLensDistortionLookupTable: [
            0, -9.757848e-05, -0.0003879658, -0.0008641425, -0.0015144957, -0.0023229565,
            -0.0032692095, -0.004328987, -0.0054744636, -0.00667476, -0.007896571, -0.009104917,
            -0.010264015, -0.011338261, -0.012293302, -0.013097172, -0.013721441, -0.014142365,
            -0.014341936, -0.014308819, -0.014039072, -0.01353662, -0.012813373, -0.011888959,
            -0.010790021, -0.009549018, -0.008202583, -0.0067894403, -0.0053480244, -0.003913922,
            -0.0025173787, -0.0011811305, 8.115734e-05, 0.0012655599, 0.0023774581, 0.0034299153,
            0.0044409954, 0.005430471, 0.006416822, 0.007416047, 0.008444657, 0.009530303,
        ]
    )

    /// Corrects a normalized [0,1], top-left-origin point (same convention
    /// as `Detection.boundingBox`) from the real, distorted camera image to
    /// where it would appear in an ideal pinhole image.
    ///
    /// A normalized coordinate is resolution-independent WITHIN one field of
    /// view/crop, so this scales directly by `referenceDimensionWidth/
    /// Height` -- deliberately NOT by whatever resolution the detection's
    /// own box happens to be normalized against (see
    /// `InferenceEngine.decodeDetections`'s `sourceWidth`/`sourceHeight`).
    ///
    /// *** UNCONFIRMED ASSUMPTION, flag before trusting this: that the
    /// *** detection's own normalization and this calibration data's
    /// *** `referenceDimensionWidth/Height` describe the SAME field of view/
    /// *** crop, just possibly at different pixel counts -- true if both
    /// *** come from the same physical camera/format, which AVCapturePhoto
    /// *** Output and AVCaptureVideoDataOutput attached to the same
    /// *** AVCaptureDevice normally do, but NOT guaranteed if either output
    /// *** is using a different active format/crop. A per-axis pixel-count
    /// *** ratio can't correct for a genuine crop-region mismatch (different
    /// *** offset, not just different scale) -- if `referenceDimensionWidth
    /// *** / referenceDimensionHeight` doesn't match the video pipeline's
    /// *** own source aspect ratio once real calibration data comes back,
    /// *** that's the signal this assumption broke and needs a real fix,
    /// *** not just a ratio patched on top.
    func correctedNormalizedPoint(_ point: CGPoint) -> CGPoint {
        let pixelPoint = CGPoint(x: point.x * referenceDimensionWidth, y: point.y * referenceDimensionHeight)
        let opticalCenter = CGPoint(x: distortionCenterX, y: distortionCenterY)
        let imageSize = CGSize(width: referenceDimensionWidth, height: referenceDimensionHeight)
        let corrected = Self.lensDistortionPoint(
            for: pixelPoint, lookupTable: lensDistortionLookupTable,
            distortionOpticalCenter: opticalCenter, imageSize: imageSize
        )
        return CGPoint(x: corrected.x / referenceDimensionWidth, y: corrected.y / referenceDimensionHeight)
    }

    /// Apple's own documented algorithm for consuming an
    /// AVCameraCalibrationData lookup table: the table holds the relative
    /// radial magnification at `lookupTable.count` linearly spaced radii
    /// from 0 to `r_max` (the distance from the optical center to the
    /// FARTHEST image corner) -- NOT independently derived here, ported as
    /// closely as possible without a device in hand to check it against.
    private static func lensDistortionPoint(
        for point: CGPoint, lookupTable: [Float], distortionOpticalCenter opticalCenter: CGPoint, imageSize: CGSize
    ) -> CGPoint {
        let deltaOcxMax = Float(max(opticalCenter.x, imageSize.width - opticalCenter.x))
        let deltaOcyMax = Float(max(opticalCenter.y, imageSize.height - opticalCenter.y))
        let rMax = (deltaOcxMax * deltaOcxMax + deltaOcyMax * deltaOcyMax).squareRoot()

        let vx = Float(point.x - opticalCenter.x)
        let vy = Float(point.y - opticalCenter.y)
        let rPoint = (vx * vx + vy * vy).squareRoot()

        let lastIndex = lookupTable.count - 1
        guard rPoint != 0, rMax != 0, lastIndex >= 0 else { return point }

        let scaledR = min(Float(lastIndex), (rPoint / rMax) * Float(lastIndex))
        let lowerIndex = Int(scaledR)
        let t = scaledR - Float(lowerIndex)
        let lowerValue = lookupTable[lowerIndex]
        let upperValue = lookupTable[min(lowerIndex + 1, lastIndex)]
        let magnification = lowerValue + t * (upperValue - lowerValue)

        let newX = vx + magnification * vx
        let newY = vy + magnification * vy
        return CGPoint(x: opticalCenter.x + CGFloat(newX), y: opticalCenter.y + CGFloat(newY))
    }
}

/// Runs the one-shot AVCameraCalibrationData capture -- adds a temporary
/// AVCapturePhotoOutput to the live session, captures exactly one photo
/// with calibration data delivery enabled, then removes the output again.
/// Must run right after the session starts, before real driving begins --
/// see CameraManager.configure's call site. Reconfiguring a running
/// session's outputs briefly interrupts ALL of them, including the live
/// video feed this app's real-time detection depends on, so this can't
/// safely run mid-drive.
final class LensCalibrationCapture: NSObject, AVCapturePhotoCaptureDelegate {
    private var completion: ((LensCalibrationData?) -> Void)?
    private let photoOutput = AVCapturePhotoOutput()
    // AVCaptureOutput doesn't expose a back-reference to its session, so the
    // delegate callback (which only receives the output) needs its own way
    // to find the session again for cleanup -- held weakly since this
    // capture object shouldn't be what keeps the session alive.
    private weak var activeSession: AVCaptureSession?

    /// `session` must already be configured and running with a video input
    /// attached. Calls `completion` (on an arbitrary queue -- the session
    /// queue, in practice) with nil if calibration data delivery isn't
    /// supported on this device/configuration, or the capture fails.
    func capture(on session: AVCaptureSession, completion: @escaping (LensCalibrationData?) -> Void) {
        self.completion = completion
        self.activeSession = session

        DebugFileLogger.log("[LensCalibrationCapture] starting one-shot capture")
        session.beginConfiguration()
        guard session.canAddOutput(photoOutput) else {
            session.commitConfiguration()
            DebugFileLogger.log("[LensCalibrationCapture] session refused the temporary photo output")
            completion(nil)
            self.completion = nil
            return
        }
        session.addOutput(photoOutput)
        session.commitConfiguration()

        guard photoOutput.isCameraCalibrationDataDeliverySupported else {
            DebugFileLogger.log("[LensCalibrationCapture] isCameraCalibrationDataDeliverySupported == false")
            session.beginConfiguration()
            session.removeOutput(photoOutput)
            session.commitConfiguration()
            completion(nil)
            self.completion = nil
            return
        }

        let settings = AVCapturePhotoSettings()
        settings.isCameraCalibrationDataDeliveryEnabled = true
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let session = activeSession
        defer {
            session?.beginConfiguration()
            session?.removeOutput(output)
            session?.commitConfiguration()
            activeSession = nil
        }
        guard error == nil else {
            DebugFileLogger.log("[LensCalibrationCapture] capture failed: \(error!)")
            completion?(nil)
            completion = nil
            return
        }
        guard let raw = photo.cameraCalibrationData else {
            DebugFileLogger.log("[LensCalibrationCapture] AVCapturePhoto.cameraCalibrationData was nil")
            completion?(nil)
            completion = nil
            return
        }
        guard let data = LensCalibrationData(raw) else {
            DebugFileLogger.log(
                "[LensCalibrationCapture] got AVCameraCalibrationData but lookup tables were nil -- " +
                "intrinsicMatrix fx=\(raw.intrinsicMatrix.columns.0.x) referenceDims=\(raw.intrinsicMatrixReferenceDimensions)"
            )
            completion?(nil)
            completion = nil
            return
        }
        DebugFileLogger.log(
            "[LensCalibrationCapture] SUCCESS fx=\(data.fx) fy=\(data.fy) cx=\(data.cx) cy=\(data.cy) " +
            "referenceDims=\(data.referenceDimensionWidth)x\(data.referenceDimensionHeight) " +
            "distortionCenter=(\(data.distortionCenterX),\(data.distortionCenterY)) " +
            "lookupTableCount=\(data.lensDistortionLookupTable.count)"
        )
        completion?(data)
        completion = nil
    }
}

/// Writes the session's captured lens calibration data to its own file,
/// once, alongside detections.jsonl/overlay-debug.log -- deliberately NOT
/// folded into every DetectionLogEntry line the way pitch/roll are.
/// Unlike those, this is a hardware constant that cannot change mid-session
/// (or, realistically, between sessions on the same physical device) --
/// repeating a multi-element distortion lookup table on every single
/// logged frame would cost real file size for zero benefit, unlike the
/// small scalar fields DetectionLogEntry already repeats per-entry for
/// self-containedness.
enum LensCalibrationLogger {
    private static let url: URL = {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return dir.appendingPathComponent("lens_calibration.json")
    }()

    static func reset() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return }
        let archiveURL = url.deletingLastPathComponent()
            .appendingPathComponent("lens_calibration-\(Int(Date().timeIntervalSince1970)).json")
        try? fm.moveItem(at: url, to: archiveURL)
    }

    static func log(_ data: LensCalibrationData) {
        guard let encoded = try? JSONEncoder().encode(data) else { return }
        try? encoded.write(to: url)
    }
}

/// DIAGNOSTIC ONLY, 2026-08-24 -- answers one question: is
/// AVCameraCalibrationData delivery actually available on THIS device at
/// all, just not through `.builtInWideAngleCamera`? CONFIRMED via a real
/// device log that the live pipeline's own device type reports
/// `isCameraCalibrationDataDeliverySupported == false` -- Apple's own docs
/// scope that property to multi-camera systems (dual/dual-wide/triple) or a
/// depth-capable single camera, so a plain single-lens wide-angle device
/// may simply never qualify. This tries the virtual multi-camera device
/// types instead, which wrap the SAME physical lenses (including the same
/// wide lens `.builtInWideAngleCamera` addresses directly -- see
/// AVCaptureDevice.constituentDevices) through an abstraction Apple's docs
/// DO list as supported.
///
/// Deliberately uses a completely SEPARATE, throwaway AVCaptureSession +
/// AVCaptureDeviceInput, never touching CameraManager's live session or
/// device -- this is purely a probe to answer the yes/no support question
/// and sanity-check what comes back, NOT (yet) wired into
/// InferenceEngine.attachDistances. Running two AVCaptureSessions
/// concurrently against overlapping physical camera hardware in one app
/// isn't something AVFoundation explicitly documents as safe -- if it
/// upsets the live session, CameraManager's existing wasInterrupted/
/// interruptionEnded handlers (registerSessionObservers) should self-
/// recover it, same safety net a real interruption (phone call, another
/// app) already relies on. Kept to the same session-start timing the live
/// capture used, before real driving begins, for the same reason.
///
/// Does NOT lock the virtual device to its wide-lens constituent (see
/// AVCaptureDevice.setPrimaryConstituentDeviceSwitchingBehavior) -- that
/// matters for a PRODUCTION integration (an automatic low-light switch to
/// the ultra-wide sensor would silently hand back the wrong lens's
/// calibration), but this is a single, brief, one-shot probe, and the
/// logged fx/fy is cross-checked below against the independently-measured
/// `focalLengthNormalized` (1.322673, from the real cone-calibration fit)
/// as a self-contained sanity check that whatever lens answered is at
/// least plausibly the wide one -- add real locking before ever trusting
/// this for actual distance correction.
enum IsolatedLensCalibrationCapture {
    private static let candidateDeviceTypes: [AVCaptureDevice.DeviceType] = [
        .builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera,
    ]

    /// Known-good normalized focal length from the real cone-calibration
    /// fit (DistanceEstimator.calibrated.focalLengthNormalized) -- NOT
    /// re-imported from there directly, to keep this throwaway diagnostic
    /// fully self-contained and easy to delete later without touching
    /// production code.
    private static let knownFocalLengthNormalized = 1.322673

    static func runDiagnostic() {
        DispatchQueue.global(qos: .utility).async {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] starting diagnostic across \(candidateDeviceTypes.count) candidate device types")
            for deviceType in candidateDeviceTypes {
                guard let device = AVCaptureDevice.default(deviceType, for: .video, position: .back) else {
                    DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): not available on this hardware")
                    continue
                }
                tryCapture(device: device, deviceType: deviceType)
            }
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] diagnostic complete")
        }
    }

    /// Runs entirely on the calling (background) queue -- blocks on a
    /// semaphore instead of a callback chain so `runDiagnostic` can just
    /// loop over candidates sequentially. Always tears its own throwaway
    /// session down before returning, regardless of outcome.
    private static func tryCapture(device: AVCaptureDevice, deviceType: AVCaptureDevice.DeviceType) {
        let session = AVCaptureSession()
        session.beginConfiguration()

        guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): couldn't create/add device input")
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let photoOutput = AVCapturePhotoOutput()
        guard session.canAddOutput(photoOutput) else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): couldn't add photo output")
            session.commitConfiguration()
            return
        }
        session.addOutput(photoOutput)

        // CONFIRMED via Apple's own AVCaptureDevice.h/AVCapturePhotoOutput.h
        // headers: AVCapturePhotoOutput.maxPhotoDimensions has no documented
        // fallback if left unset (unlike AVCapturePhotoSettings' own
        // maxPhotoDimensions, which explicitly defaults to the smallest
        // supported size) -- "Changing this property may trigger a lengthy
        // reconfiguration of the capture render pipeline so it is
        // recommended that this is set before calling -[AVCaptureSession
        // startRunning]." A virtual multi-camera device may not have a
        // sane implicit default the way a plain single-lens device does.
        if let maxDims = device.activeFormat.supportedMaxPhotoDimensions.max(by: { $0.width * $0.height < $1.width * $1.height }) {
            photoOutput.maxPhotoDimensions = maxDims
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): set maxPhotoDimensions=\(maxDims.width)x\(maxDims.height) (before startRunning)")
        } else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): activeFormat reported no supportedMaxPhotoDimensions")
        }

        // CONFIRMED via Apple's own AVCapturePhotoOutput.h header (this
        // property's doc comment, quoted verbatim): "Virtual device
        // constituent photo delivery requires a lengthy reconfiguration of
        // the capture render pipeline, so if you intend to do any
        // constituent photo delivery captures, you should set this
        // property to YES BEFORE calling -[AVCaptureSession startRunning]."
        // CONFIRMED CRASH, 2026-08-24, via two real device crash reports:
        // setting this AFTER startRunning() (as an earlier version of this
        // function did) crashed synchronously inside capturePhotoWithSettings
        // :delegate: (SIGABRT, uncaught NSException) EVERY time, even after
        // separately confirming the video connection was active+enabled --
        // the pipeline reconfiguration this property triggers apparently
        // never actually completed correctly when requested after the
        // session was already running, leaving capturePhoto in an invalid
        // internal state. Also see cameraCalibrationDataDeliverySupported's
        // own doc comment for why this needs to be YES at all: calibration
        // data delivery requires virtualDeviceConstituentPhotoDeliveryEnabled
        // == YES, contentAwareDistortionCorrectionEnabled == NO, and the
        // device's own geometricDistortionCorrectionEnabled == NO -- GDC is
        // very likely ON by default on a modern iPhone, which is why the
        // FIRST diagnostic pass (before either of these existed) got a hard
        // false across every device type.
        if photoOutput.isVirtualDeviceConstituentPhotoDeliverySupported {
            photoOutput.isVirtualDeviceConstituentPhotoDeliveryEnabled = true
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): enabled isVirtualDeviceConstituentPhotoDeliveryEnabled (before startRunning)")
        } else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): isVirtualDeviceConstituentPhotoDeliverySupported == false (expected for a non-virtual device type)")
        }

        session.commitConfiguration()
        session.startRunning()
        defer { session.stopRunning() }

        if device.isGeometricDistortionCorrectionSupported {
            do {
                try device.lockForConfiguration()
                device.isGeometricDistortionCorrectionEnabled = false
                device.unlockForConfiguration()
                DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): disabled isGeometricDistortionCorrectionEnabled")
            } catch {
                DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): couldn't lock device to disable GDC: \(error)")
            }
        } else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): isGeometricDistortionCorrectionSupported == false")
        }

        guard photoOutput.isCameraCalibrationDataDeliverySupported else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): isCameraCalibrationDataDeliverySupported == false")
            return
        }

        // CONFIRMED CRASH, 2026-08-24, via a real device crash report:
        // -[AVCapturePhotoOutput capturePhotoWithSettings:delegate:] threw
        // an uncaught NSException synchronously (SIGABRT) when called
        // immediately after session.startRunning() + the GDC/virtual-
        // constituent reconfiguration above -- a well-documented AVFoundation
        // failure mode ("No active and enabled video connection") from
        // calling capturePhoto before the photo output's connection has
        // actually settled into an active+enabled state, which reconfiguring
        // the session/device (as just happened) can briefly disrupt even
        // after startRunning() has already returned. Poll for the
        // connection to genuinely settle instead of assuming it's ready the
        // instant configuration calls return.
        let connectionReady = waitForActiveEnabledConnection(on: photoOutput, timeout: 2.0)
        DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): video connection ready=\(connectionReady)")
        guard connectionReady else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): connection never became active+enabled, skipping capture to avoid the known crash")
            return
        }

        // CONFIRMED CRASH REASON, 2026-08-24, via a LIVE console attached
        // with `devicectl device process launch --console` (the .ips crash
        // report alone never captured this -- it only has the backtrace,
        // not the NSException's actual reason string): "settings.camera
        // CalibrationDataDeliveryEnabled may not be set to YES unless 2 or
        // more AVCaptureDevices are added to settings.virtualDevice
        // ConstituentPhotoDeliveryEnabledDevices". Confirmed by the header
        // too (AVCapturePhotoSettings.cameraCalibrationDataDeliveryEnabled's
        // own doc comment says the same thing, just not as an obvious
        // "required" until you hit the actual crash). Calibration data is
        // fundamentally about the relationship between lenses (built for
        // stereo depth), so Apple requires an actual simultaneous multi-
        // lens capture to ask for it -- a single "1x" shot was never valid
        // no matter how correctly everything else was configured.
        let constituentDevices = device.constituentDevices
        guard constituentDevices.count >= 2 else {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): only \(constituentDevices.count) constituent device(s), need 2+ for calibration data")
            return
        }
        DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): calibration data delivery IS supported -- capturing across \(constituentDevices.count) constituents (zoomFactor=\(device.videoZoomFactor))")

        let semaphore = DispatchSemaphore(value: 0)
        let delegate = OneShotPhotoDelegate { photo, error in
            if let error {
                DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): capture failed: \(error)")
            } else if let photo {
                // sourceDeviceType tells us which constituent lens actually
                // produced THIS callback -- capturePhoto fires once per
                // device in virtualDeviceConstituentPhotoDeliveryEnabledDevices,
                // and we only care about the plain wide lens's own
                // calibration (the one .builtInWideAngleCamera addresses
                // directly elsewhere in this app).
                let source = photo.sourceDeviceType?.rawValue ?? "unknown"
                if let raw = photo.cameraCalibrationData, let data = LensCalibrationData(raw) {
                    // Sanity check, not a substitute for real per-lens
                    // locking: a wildly different normalized focal length
                    // means whatever constituent answered isn't plausibly
                    // the same wide lens the app's existing cone
                    // calibration already measured.
                    let normalizedFx = data.fx / data.referenceDimensionWidth
                    let percentOffFromKnown = (normalizedFx / knownFocalLengthNormalized - 1) * 100
                    DebugFileLogger.log(
                        "[IsolatedLensCalibrationCapture] \(deviceType.rawValue): SUCCESS source=\(source) fx=\(data.fx) fy=\(data.fy) " +
                        "cx=\(data.cx) cy=\(data.cy) referenceDims=\(data.referenceDimensionWidth)x\(data.referenceDimensionHeight) " +
                        "distortionCenter=(\(data.distortionCenterX),\(data.distortionCenterY)) " +
                        "lookupTableCount=\(data.lensDistortionLookupTable.count) " +
                        "normalizedFx=\(normalizedFx) (known cone-cal focalLengthNormalized=\(knownFocalLengthNormalized), " +
                        "\(percentOffFromKnown)% off -- near 0% means this is plausibly the same wide lens)"
                    )
                    if source == AVCaptureDevice.DeviceType.builtInWideAngleCamera.rawValue {
                        LensCalibrationLogger.log(data)
                        DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): wrote lens_calibration.json from the WIDE lens's result")
                    }
                } else {
                    DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): source=\(source) no usable calibration data in this constituent's result")
                }
            }
            // photoCount is 1-based; expectedPhotoCount is the total for
            // this request (== constituentDevices.count here) -- only
            // release the semaphore once every constituent has reported in,
            // so runDiagnostic() doesn't move to the next candidate device
            // type while callbacks for this one are still in flight.
            if photo == nil || photo!.photoCount >= photo!.resolvedSettings.expectedPhotoCount {
                semaphore.signal()
            }
        }
        let settings = AVCapturePhotoSettings()
        settings.isCameraCalibrationDataDeliveryEnabled = true
        settings.virtualDeviceConstituentPhotoDeliveryEnabledDevices = constituentDevices
        photoOutput.capturePhoto(with: settings, delegate: delegate)
        let waitResult = semaphore.wait(timeout: .now() + 5)
        if waitResult == .timedOut {
            DebugFileLogger.log("[IsolatedLensCalibrationCapture] \(deviceType.rawValue): capture timed out after 5s")
        }
    }

    /// Polls (every 50ms, up to `timeout` seconds) for `photoOutput`'s video
    /// connection to be both active and enabled -- see this call site's own
    /// doc comment for why: a real crash confirmed calling capturePhoto
    /// right after session/device reconfiguration, before the connection
    /// had actually settled, is fatal (an uncaught NSException), not just
    /// unreliable.
    private static func waitForActiveEnabledConnection(on photoOutput: AVCapturePhotoOutput, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let connection = photoOutput.connection(with: .video), connection.isActive, connection.isEnabled {
                return true
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        return false
    }
}

/// AVCapturePhotoCaptureDelegate requires a persistent object -- capturePhoto
/// doesn't retain its delegate parameter -- so this minimal adapter exists
/// purely to let IsolatedLensCalibrationCapture's static functions hand it a
/// plain closure instead of conforming itself.
private final class OneShotPhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let handler: (AVCapturePhoto?, Error?) -> Void
    init(handler: @escaping (AVCapturePhoto?, Error?) -> Void) { self.handler = handler }
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        handler(photo, error)
    }
}
