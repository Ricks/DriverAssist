//
//  Matrix.swift
//  DriverAssist
//
//  Minimal dense-matrix type backing KalmanBoxTracker's covariance math.
//  Swift has no numpy equivalent for this; at the 8x8 scale this runs at
//  (one Kalman filter per active track, once per frame), a plain O(n^3)
//  implementation is nowhere near a bottleneck, so this favors clarity over
//  using e.g. Accelerate/BLAS.
//

import Foundation

struct Matrix {
    let rows: Int
    let cols: Int
    private var storage: [Double]

    init(rows: Int, cols: Int, repeating value: Double = 0) {
        self.rows = rows
        self.cols = cols
        storage = Array(repeating: value, count: rows * cols)
    }

    init(rows: Int, cols: Int, values: [Double]) {
        precondition(values.count == rows * cols, "Matrix values count mismatch")
        self.rows = rows
        self.cols = cols
        storage = values
    }

    static func identity(_ size: Int) -> Matrix {
        var m = Matrix(rows: size, cols: size)
        for i in 0..<size { m[i, i] = 1 }
        return m
    }

    static func diagonal(_ values: [Double]) -> Matrix {
        var m = Matrix(rows: values.count, cols: values.count)
        for (i, v) in values.enumerated() { m[i, i] = v }
        return m
    }

    subscript(row: Int, col: Int) -> Double {
        get { storage[row * cols + col] }
        set { storage[row * cols + col] = newValue }
    }

    var transposed: Matrix {
        var result = Matrix(rows: cols, cols: rows)
        for r in 0..<rows {
            for c in 0..<cols {
                result[c, r] = self[r, c]
            }
        }
        return result
    }

    static func * (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.cols == b.rows, "Matrix dimension mismatch in multiply")
        var result = Matrix(rows: a.rows, cols: b.cols)
        for r in 0..<a.rows {
            for k in 0..<a.cols {
                let aVal = a[r, k]
                guard aVal != 0 else { continue }
                for c in 0..<b.cols {
                    result[r, c] += aVal * b[k, c]
                }
            }
        }
        return result
    }

    static func + (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols, "Matrix dimension mismatch in add")
        var result = Matrix(rows: a.rows, cols: a.cols)
        for i in 0..<a.storage.count { result.storage[i] = a.storage[i] + b.storage[i] }
        return result
    }

    static func - (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.rows == b.rows && a.cols == b.cols, "Matrix dimension mismatch in subtract")
        var result = Matrix(rows: a.rows, cols: a.cols)
        for i in 0..<a.storage.count { result.storage[i] = a.storage[i] - b.storage[i] }
        return result
    }

    /// Column vector (as a rows x 1 Matrix) times this being implicit -- convenience
    /// for multiplying by a plain [Double] treated as a column vector.
    static func * (a: Matrix, v: [Double]) -> [Double] {
        precondition(a.cols == v.count, "Matrix/vector dimension mismatch")
        var result = [Double](repeating: 0, count: a.rows)
        for r in 0..<a.rows {
            var sum = 0.0
            for c in 0..<a.cols { sum += a[r, c] * v[c] }
            result[r] = sum
        }
        return result
    }

    /// Gauss-Jordan inversion with partial pivoting. Returns nil if singular
    /// (shouldn't happen for the well-conditioned small matrices this is used
    /// on -- S = H P H^T + R is always positive-definite with R's diagonal
    /// keeping it non-singular even if P degenerates).
    var inverse: Matrix? {
        precondition(rows == cols, "Only square matrices are invertible")
        let n = rows
        var a = self
        var inv = Matrix.identity(n)

        for col in 0..<n {
            var pivotRow = col
            var maxVal = abs(a[col, col])
            for r in (col + 1)..<n where abs(a[r, col]) > maxVal {
                maxVal = abs(a[r, col])
                pivotRow = r
            }
            guard maxVal > 1e-12 else { return nil }

            if pivotRow != col {
                a.swapRows(col, pivotRow)
                inv.swapRows(col, pivotRow)
            }

            let pivot = a[col, col]
            for c in 0..<n {
                a[col, c] /= pivot
                inv[col, c] /= pivot
            }

            for r in 0..<n where r != col {
                let factor = a[r, col]
                guard factor != 0 else { continue }
                for c in 0..<n {
                    a[r, c] -= factor * a[col, c]
                    inv[r, c] -= factor * inv[col, c]
                }
            }
        }
        return inv
    }

    private mutating func swapRows(_ i: Int, _ j: Int) {
        guard i != j else { return }
        for c in 0..<cols {
            let tmp = self[i, c]
            self[i, c] = self[j, c]
            self[j, c] = tmp
        }
    }
}
