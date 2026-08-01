//
//  HungarianAlgorithm.swift
//  DriverAssist
//
//  Optimal (minimum total cost) bipartite assignment -- Swift has no
//  equivalent to scipy.optimize.linear_sum_assignment (used by tracker.py's
//  ByteTracker), so this ports the classic O(n^3) Kuhn-Munkres algorithm via
//  successive shortest augmenting paths with potentials.
//

import Foundation

enum HungarianAlgorithm {
    /// `cost[row][col]` for a (possibly non-square) rows x cols matrix.
    /// Returns an array of length `rows`, each entry either the assigned
    /// column index or nil (only possible when rows > cols -- some row goes
    /// unassigned). Internally pads to square with a very large cost so
    /// padding rows/columns are never preferred over a real match.
    static func solve(cost: [[Double]]) -> [Int?] {
        let rows = cost.count
        guard rows > 0, let firstRow = cost.first else { return [] }
        let cols = firstRow.count
        guard cols > 0 else { return Array(repeating: nil, count: rows) }

        let n = max(rows, cols)
        let bigCost = 1e9
        // 1-indexed n x n cost matrix, e-maxx-style padded with bigCost for
        // any row/col beyond the real data so a real assignment is always
        // preferred when one exists.
        var a = [[Double]](repeating: [Double](repeating: bigCost, count: n + 1), count: n + 1)
        for r in 0..<rows {
            for c in 0..<cols {
                a[r + 1][c + 1] = cost[r][c]
            }
        }

        var u = [Double](repeating: 0, count: n + 1)
        var v = [Double](repeating: 0, count: n + 1)
        var p = [Int](repeating: 0, count: n + 1)   // p[j] = row currently assigned to column j
        var way = [Int](repeating: 0, count: n + 1)

        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minV = [Double](repeating: .greatestFiniteMagnitude, count: n + 1)
            var used = [Bool](repeating: false, count: n + 1)

            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = Double.greatestFiniteMagnitude
                var j1 = -1
                for j in 1...n where !used[j] {
                    let cur = a[i0][j] - u[i0] - v[j]
                    if cur < minV[j] {
                        minV[j] = cur
                        way[j] = j0
                    }
                    if minV[j] < delta {
                        delta = minV[j]
                        j1 = j
                    }
                }
                for j in 0...n {
                    if used[j] {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minV[j] -= delta
                    }
                }
                j0 = j1
            } while p[j0] != 0

            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }

        var result: [Int?] = Array(repeating: nil, count: rows)
        for j in 1...n {
            let row = p[j]
            guard row >= 1, row <= rows else { continue }
            let col = j - 1
            if col < cols {
                result[row - 1] = col
            }
        }
        return result
    }
}
