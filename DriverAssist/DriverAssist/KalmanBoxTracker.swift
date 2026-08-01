//
//  KalmanBoxTracker.swift
//  DriverAssist
//
//  Constant-velocity Kalman filter over an 8-dim state
//  [cx, cy, w, h, vcx, vcy, vw, vh] (box center, width, height, and their
//  velocities), observing the 4-dim [cx, cy, w, h] -- ports tools/tracker.py's
//  KalmanBoxTracker exactly, including the size-velocity damping fix and GMC
//  application. See that file's docstring for the full rationale; this
//  mirrors it rather than re-deriving it.
//

import CoreGraphics
import Foundation

/// Damps how strongly a fitted width/height velocity extrapolates forward
/// during an unmatched ("coasting") stretch. Without this, a handful of
/// noisy early observations can fit a spurious growth trend that then
/// compounds linearly for as long as the track goes unmatched, inflating the
/// predicted box far past its real size -- measured in the Python tracker: a
/// predicted width ~2.6x the real one after 14 coasted steps, alone enough
/// to fail the IoU gate on re-detection. Position velocity feeds forward at
/// full strength; only size velocity is damped.
let sizeVelocityDamping = 0.3

/// Normalized [0, 1], top-left-origin, matching `Detection.boundingBox`.
struct TrackedBox {
    var x: Double
    var y: Double
    var w: Double
    var h: Double
}

final class KalmanBoxTracker {
    /// [cx, cy, w, h, vcx, vcy, vw, vh]
    private(set) var x: [Double]
    private var F: Matrix
    private var H: Matrix
    private var R: Matrix
    private var Q: Matrix
    private var P: Matrix

    init(box: TrackedBox) {
        let cx = box.x + box.w / 2
        let cy = box.y + box.h / 2
        x = [cx, cy, box.w, box.h, 0, 0, 0, 0]

        var f = Matrix.identity(8)
        for i in 0..<4 {
            f[i, i + 4] = i < 2 ? 1.0 : sizeVelocityDamping
        }
        F = f

        var h = Matrix(rows: 4, cols: 8)
        for i in 0..<4 { h[i, i] = 1.0 }
        H = h

        // Modest measurement noise; larger process noise on the velocity
        // terms since we have no real prior on how fast a box's size/position
        // changes frame-to-frame.
        R = Matrix.diagonal([0.01, 0.01, 0.01, 0.01])
        var q = Matrix.diagonal(Array(repeating: 0.01, count: 8))
        for i in 4..<8 { q[i, i] *= 5.0 }
        Q = q

        // Start with high uncertainty on velocity (unknown at init) and
        // moderate uncertainty on position (we just observed it).
        var p = Matrix.identity(8)
        for i in 4..<8 { p[i, i] *= 100.0 }
        P = p
    }

    @discardableResult
    func predict() -> TrackedBox {
        x = F * x
        P = F * P * F.transposed + Q
        return boxFromState()
    }

    func update(with box: TrackedBox) {
        let z = [box.x + box.w / 2, box.y + box.h / 2, box.w, box.h]
        let predictedZ = H * x
        let y = zip(z, predictedZ).map { $0 - $1 }

        let S = H * P * H.transposed + R
        guard let sInv = S.inverse else { return }  // shouldn't happen, see Matrix.inverse
        let K = P * H.transposed * sInv

        let Ky = K * y
        for i in 0..<8 { x[i] += Ky[i] }
        P = (Matrix.identity(8) - K * H) * P
    }

    /// Corrects this track's just-predicted state for the camera's own
    /// estimated motion (see GMC.swift) -- call after predict(), before
    /// matching. `warp` is a similarity transform (rotation + uniform scale +
    /// translation) in *pixel space*; applied in pixel space here too (not
    /// directly on our normalized [0,1] state) because a non-square frame
    /// normalizes x and y by different factors, which would distort a
    /// rotation if applied in normalized coordinates directly. Position is a
    /// point (full transform, including translation); velocity is a
    /// direction (rotation/scale only, no translation) -- size and its own
    /// velocity scale uniformly with the transform's scale factor. Only the
    /// mean state is corrected, not the covariance P -- a simplification,
    /// but the mean is what matching is actually scored against.
    func applyGMC(_ warp: SimilarityTransform, frameWidth: Double, frameHeight: Double) {
        let posPx = (x[0] * frameWidth, x[1] * frameHeight)
        let newPosPx = warp.apply(to: posPx)
        x[0] = newPosPx.0 / frameWidth
        x[1] = newPosPx.1 / frameHeight

        let velPx = (x[4] * frameWidth, x[5] * frameHeight)
        let newVelPx = warp.applyLinearOnly(to: velPx)
        x[4] = newVelPx.0 / frameWidth
        x[5] = newVelPx.1 / frameHeight

        let scale = warp.scale
        x[2] *= scale
        x[3] *= scale
        x[6] *= scale
        x[7] *= scale
    }

    var box: TrackedBox { boxFromState() }

    private func boxFromState() -> TrackedBox {
        TrackedBox(x: x[0] - x[2] / 2, y: x[1] - x[3] / 2, w: x[2], h: x[3])
    }
}

extension TrackedBox {
    init(_ rect: CGRect) {
        self.init(x: rect.minX, y: rect.minY, w: rect.width, h: rect.height)
    }

    var cgRect: CGRect { CGRect(x: x, y: y, width: w, height: h) }
}
