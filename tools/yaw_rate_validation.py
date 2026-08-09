#!/usr/bin/env python3
"""
Cross-checks PitchSensor's gyro-derived yawRateDegreesPerSecond (the
attitude-quaternion projection, sign confirmed via bench test 2026-08-06,
magnitude/accuracy not yet validated) against two independent signals
already logged in detections.jsonl -- neither requires anything from the
driver/passenger during the drive, unlike hand-labeling:

  1. GPS course differentiation: d(course)/dt, sign-flipped to match the
     gyro's confirmed left=positive convention (GPS course increases
     clockwise/rightward; the gyro convention is positive=counter-clockwise/
     left, per the bench test). Weakest at low speed, where GPS course
     itself is noisy/undefined -- exactly the tight, slow turns this
     project cares about most, so treat disagreement there with suspicion
     of the GPS signal, not automatically the gyro.

  2. Accelerometer: for a coordinated turn (no lateral tire slip),
     a_lateral ~= v * omega. Deliberately compares MAGNITUDE only, not
     sign -- sign is already confirmed, and there's no reliable way to get
     a signed "lateral" direction from userAcceleration offline without
     knowing the vehicle's true-north heading (GPS course) merged with
     CoreMotion's own arbitrary (not true-north-referenced) world frame,
     which detections.jsonl doesn't have enough logged to resolve safely.
     Instead: remove the gravity-parallel component of userAcceleration
     (gravityX/Y/Z is a unit vector already logged), leaving the horizontal-
     plane acceleration magnitude -- that also includes any longitudinal
     (speeding up/slowing down) acceleration, not just lateral, so this
     check is only meaningful where the car is turning at roughly constant
     speed. Don't trust it during combined braking+turning.

Usage:
    python3 tools/yaw_rate_validation.py <session_dir_or_detections.jsonl> [--output yaw_rate_validation.png] [--smooth-window N]
"""
import argparse
import math
import sys
from pathlib import Path

from driverassist_sync import load_detections

G_MPS2 = 9.80665


def circular_diff_deg(a: float, b: float) -> float:
    """Shortest signed angular difference a - b, wrapped to [-180, 180] --
    naive subtraction breaks at the 0/360 wraparound (e.g. 350 -> 10 is a
    real 20 deg turn, not -340)."""
    return ((a - b + 180) % 360) - 180


def gps_yaw_rate_series(rows: list[dict]) -> list[tuple]:
    """(t, yaw_rate_deg_per_s) from consecutive DISTINCT GPS course fixes.
    Course only updates at GPS's own ~1Hz, while rows are logged at ~15fps,
    so most consecutive rows repeat the same course value -- diffing every
    row directly would mostly compute noise from repeated values (0) plus
    occasional real jumps counted at the wrong dt. Skip repeats and rows
    with no course data (stationary, or no fix yet)."""
    results = []
    prev = None  # (t, course) of the last genuinely new fix
    for r in rows:
        course = r.get("courseDegrees")
        if course is None:
            continue
        t = r["t"]
        if prev is None:
            prev = (t, course)
            continue
        if course == prev[1]:
            continue  # stale/repeated fix, not a new sample
        dt = t - prev[0]
        if not (0 < dt < 5):
            continue  # gap too large (dropped fixes) to trust a rate from
        dcourse = circular_diff_deg(course, prev[1])
        # Sign flip: GPS course increases clockwise (right turn); gyro
        # convention is positive = counter-clockwise = left turn (confirmed
        # via bench test) -- negate to compare on the same axis.
        results.append((t, -dcourse / dt))
        prev = (t, course)
    return results


def accel_lateral_check(rows: list[dict]) -> list[tuple]:
    """(t, predicted_ms2, observed_ms2) -- predicted is |v * omega| from GPS
    speed and gyro yaw rate; observed is the horizontal-plane magnitude of
    userAcceleration (gravity-parallel component removed), converted from g
    to m/s^2. See module docstring for why this only checks magnitude, and
    the constant-speed caveat."""
    results = []
    for r in rows:
        v = r.get("egoSpeedMps")
        omega_deg = r.get("yawRateDegreesPerSecond")
        gx, gy, gz = r.get("gravityX"), r.get("gravityY"), r.get("gravityZ")
        ux, uy, uz = r.get("userAccelerationX"), r.get("userAccelerationY"), r.get("userAccelerationZ")
        if None in (v, omega_deg, gx, gy, gz, ux, uy, uz):
            continue

        omega_rad = omega_deg * math.pi / 180
        predicted = abs(v * omega_rad)

        # Vector projection: remove the component of userAcceleration
        # parallel to gravity (a unit vector), leaving only the horizontal-
        # plane part. Standard u_horizontal = u - (u . g) * g.
        dot = ux * gx + uy * gy + uz * gz
        hx, hy, hz = ux - dot * gx, uy - dot * gy, uz - dot * gz
        observed = math.sqrt(hx * hx + hy * hy + hz * hz) * G_MPS2

        results.append((r["t"], predicted, observed))
    return results


def rolling_mean(series: list[tuple], window: int) -> list[tuple]:
    """Simple centered rolling mean over the second element of each (t, value)
    pair, for smoothing at analysis time only -- deliberately NOT applied
    on-device (see the on-device smoothing discussion: the raw signal needs
    to be seen as-is first before deciding a filter is warranted)."""
    if window <= 1:
        return series
    values = [v for _, v in series]
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return [(t, s) for (t, _), s in zip(series, smoothed)]


def correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / math.sqrt(var_a * var_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detections", type=Path, help="Session's detections.jsonl, or a directory containing one")
    parser.add_argument("--output", type=Path, default=Path("yaw_rate_validation.png"))
    parser.add_argument("--smooth-window", type=int, default=5, help="Rolling-mean window (samples) for the plot only, not the correlation stats")
    args = parser.parse_args()

    rows = load_detections(args.detections)
    t0 = rows[0]["t"]

    gyro_series = [(r["t"], r["yawRateDegreesPerSecond"]) for r in rows if r.get("yawRateDegreesPerSecond") is not None]
    gps_series = gps_yaw_rate_series(rows)
    accel_series = accel_lateral_check(rows)

    print(f"{len(rows)} total rows, {(rows[-1]['t'] - t0) / 60:.1f} min")
    print(f"gyro yaw-rate samples: {len(gyro_series)}")
    print(f"GPS-course-derived samples: {len(gps_series)} (real GPS fixes only, not per-row)")
    print(f"accelerometer-check samples: {len(accel_series)}")

    # Correlate GPS-derived yaw rate against the nearest gyro sample in time
    # (different sample rates -- 15Hz gyro vs ~1Hz GPS -- so pair by nearest
    # timestamp rather than assuming aligned indices).
    gyro_times = [t for t, _ in gyro_series]
    gyro_values = [v for _, v in gyro_series]

    def nearest_gyro(t: float) -> float:
        import bisect
        i = bisect.bisect_left(gyro_times, t)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(gyro_times)]
        best = min(candidates, key=lambda j: abs(gyro_times[j] - t))
        return gyro_values[best]

    if gps_series:
        paired_gyro = [nearest_gyro(t) for t, _ in gps_series]
        paired_gps = [v for _, v in gps_series]
        print(f"\nGPS-course cross-check: correlation r={correlation(paired_gyro, paired_gps):.3f}")
        mae = sum(abs(a - b) for a, b in zip(paired_gyro, paired_gps)) / len(paired_gyro)
        print(f"  mean absolute difference: {mae:.1f} deg/s")
    else:
        print("\nGPS-course cross-check: no valid course samples (stationary the whole session, or no fix)")

    if accel_series:
        predicted = [p for _, p, _ in accel_series]
        observed = [o for _, _, o in accel_series]
        print(f"\nAccelerometer cross-check: correlation r={correlation(predicted, observed):.3f}")
        mae = sum(abs(p - o) for p, o in zip(predicted, observed)) / len(predicted)
        print(f"  mean absolute difference: {mae:.2f} m/s^2 (includes longitudinal accel -- only trust this where speed is ~constant)")
    else:
        print("\nAccelerometer cross-check: no valid samples")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available -- skipping plot, stats above are still valid")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    gyro_plot = rolling_mean(gyro_series, args.smooth_window)
    axes[0].plot([(t - t0) / 60 for t, _ in gyro_plot], [v for _, v in gyro_plot], label="gyro (attitude projection)", color="tab:blue")
    if gps_series:
        axes[0].plot([(t - t0) / 60 for t, _ in gps_series], [v for _, v in gps_series], label="GPS course diff", color="tab:orange", marker="o", markersize=3, linestyle="none")
    axes[0].set_ylabel("yaw rate (deg/s)")
    axes[0].set_title("Gyro yaw rate vs. GPS-course-derived yaw rate")
    axes[0].legend()
    axes[0].axhline(0, color="gray", linewidth=0.5)

    if accel_series:
        pred_plot = rolling_mean([(t, p) for t, p, _ in accel_series], args.smooth_window)
        obs_plot = rolling_mean([(t, o) for t, _, o in accel_series], args.smooth_window)
        axes[1].plot([(t - t0) / 60 for t, _ in pred_plot], [v for _, v in pred_plot], label="predicted |v * omega|", color="tab:blue")
        axes[1].plot([(t - t0) / 60 for t, _ in obs_plot], [v for _, v in obs_plot], label="observed horizontal accel", color="tab:orange")
    axes[1].set_ylabel("lateral accel (m/s^2)")
    axes[1].set_title("Accelerometer cross-check (magnitude only -- see caveat re: constant speed)")
    axes[1].legend()

    speed_series = [(r["t"], r["egoSpeedMps"]) for r in rows if r.get("egoSpeedMps") is not None]
    axes[2].plot([(t - t0) / 60 for t, _ in speed_series], [v for _, v in speed_series], color="tab:green")
    axes[2].set_ylabel("GPS speed (m/s)")
    axes[2].set_xlabel("time (min)")
    axes[2].set_title("Ego speed, for context -- both cross-checks are weakest at low speed")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
