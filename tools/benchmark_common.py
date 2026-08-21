"""
Shared stats/chart/data-file code for benchmark.py and plot_benchmarks.py.
Not a script itself.

Results are saved as one JSON file per benchmarked video (see save_results/
load_results) — readable, and easy to recombine later without re-running the
(slow) reference-model inference. merge_results() is what plot_benchmarks.py
uses to combine several of these files correctly: TP/FP/FN are summed per
(config, class) and precision/recall/F1 recomputed from the pooled totals,
and latency lists are concatenated before recomputing mean/median/p95 — not
just averaging each session's own mean, which would be statistically sloppy
across sessions of different lengths.
"""
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

TARGET_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# yolo26x is ground truth, so its own resolution shouldn't be a limiting
# factor -- always run it at (roughly) native content resolution, regardless
# of which on-device config it's being compared against. This is the same
# (h, w) shape the on-device high-res export uses; for this project's native
# 1920x1080 capture that's essentially zero letterbox padding (not an
# upscale) -- there's no real resolution left to gain by going higher.
# Previously this defaulted to a plain int (1280, square), which let
# ultralytics letterbox a 1920x1080 frame down to an effective ~1280x720 --
# lower than on-device high-res's own ~1920x1080 content, which unfairly
# penalized high-res on-device configs' measured precision (see
# project_reference_model_resolution_fix memory for the full story).
DEFAULT_REFERENCE_IMGSZ = (1088, 1920)


def parse_imgsz(value: str):
    """argparse type for --imgsz: 'HxW' (e.g. '1088x1920') for a rectangular
    shape, or a bare int for a square one."""
    if "x" in value:
        h, w = value.split("x")
        return (int(h), int(w))
    return int(value)

# Validated categorical palette (dataviz skill's default), assigned in a
# FIXED order keyed to config identity — a config keeps its color whether or
# not other configs appear in a given chart, never reassigned by rank.
# CONFIRMED gap 2026-08-20: this used to be model x two-pass only, from
# before the high-res ML-inference toggle (added 2026-08-17) existed --
# config_label (benchmark.py) now includes resolution too, so a chart for a
# model x resolution matrix session needs those combos in here as well or
# `c in stats` below would never match, silently dropping every bar. Two
# resolution values exist in practice (ModelManager.baseResolutionLabel/
# highResResolutionLabel) -- medium has no high-res export, so
# "medium, res 1920x1088, ..." is a harmless never-populated slot, not a bug.
CONFIG_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#8b5fbf", "#c94f4f", "#4fb0af", "#b0a13f", "#5f8bbf", "#bf5f96",
]
CONFIG_ORDER = [
    f"{name}, res {resolution}, two-pass {state}"
    for name in ("nano", "small", "medium")
    for resolution in ("1152x640", "1920x1088")
    for state in ("off", "on")
]
CONFIG_COLOR_BY_LABEL = dict(zip(CONFIG_ORDER, CONFIG_COLORS * (len(CONFIG_ORDER) // len(CONFIG_COLORS) + 1)))


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def print_legend(iou_threshold: float) -> None:
    print("Legend:")
    print("  config       on-device settings active for these entries (model tier, two-pass on/off)")
    print("  class        COCO object category (person, bicycle, car, motorcycle, bus, truck)")
    print("  TP / FP / FN true / false positive, false negative — this run's on-device boxes matched")
    print(f"               against the yolo26x reference model's boxes at IoU >= {iou_threshold}")
    print("  P / R / F1   precision, recall, F1 (harmonic mean of P and R); '-' = undefined")
    print("               (no on-device and/or reference boxes existed for that class)")
    print("  mean/median/p95 ms   inference latency stats from the app's own logged elapsedMs")
    print("                       (this is NOT compared against the reference model)")
    print("  n            number of logged detection entries included in that row")


def print_summary(stats: dict, latencies: dict, iou_threshold: float) -> None:
    def fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "  -  "

    print_legend(iou_threshold)

    print(f"\nAccuracy vs. reference model (IoU >= {iou_threshold}):")
    print(f"{'config':<24} {'class':<12} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for config in sorted(stats):
        for cls in sorted(stats[config]):
            tp, fp, fn = stats[config][cls]
            if tp + fp + fn == 0:
                continue
            p, r, f1 = precision_recall_f1(tp, fp, fn)
            print(f"{config:<24} {cls:<12} {tp:>5} {fp:>5} {fn:>5} {fmt(p):>6} {fmt(r):>6} {fmt(f1):>6}")

    print("\nLatency (from the app's own logged elapsedMs):")
    print(f"{'config':<24} {'mean ms':>8} {'median ms':>10} {'p95 ms':>8} {'n':>6}")
    for config in sorted(latencies):
        arr = np.array(latencies[config])
        print(f"{config:<24} {arr.mean():>8.1f} {np.median(arr):>10.1f} {np.percentile(arr, 95):>8.1f} {len(arr):>6}")


def make_chart(stats: dict, latencies: dict, output: Path, iou_threshold: float, sessions: list[str]) -> None:
    acc_configs = [c for c in CONFIG_ORDER if c in stats]
    lat_configs = [c for c in CONFIG_ORDER if c in latencies]
    classes = sorted(TARGET_CLASSES)

    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(14, 6.5))

    n = max(len(acc_configs), 1)
    bar_width = 0.8 / n
    x = np.arange(len(classes))
    for i, config in enumerate(acc_configs):
        f1_scores = []
        for cls in classes:
            tp, fp, fn = stats[config].get(cls, (0, 0, 0))
            if tp + fp + fn == 0:
                # Never appeared (predicted or reference) for this config —
                # NaN so matplotlib leaves a gap, distinct from a real 0.0
                # F1 (which means it was tested and every detection missed).
                f1_scores.append(np.nan)
                continue
            _, _, f1 = precision_recall_f1(tp, fp, fn)
            f1_scores.append(f1 if f1 is not None else 0.0)
        offset = (i - (n - 1) / 2) * bar_width
        ax_acc.bar(x + offset, f1_scores, bar_width, label=config, color=CONFIG_COLOR_BY_LABEL[config])
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(classes, rotation=20, ha="right")
    ax_acc.set_ylabel("F1 score (vs. yolo26x reference)")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title(f"Detection accuracy by class and config (IoU ≥ {iou_threshold})")
    ax_acc.legend(fontsize=8)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    means = [float(np.mean(latencies[c])) for c in lat_configs]
    stds = [float(np.std(latencies[c])) for c in lat_configs]
    colors = [CONFIG_COLOR_BY_LABEL[c] for c in lat_configs]
    ax_lat.bar(range(len(lat_configs)), means, yerr=stds, color=colors, capsize=4)
    ax_lat.set_xticks(range(len(lat_configs)))
    ax_lat.set_xticklabels(lat_configs, rotation=20, ha="right")
    ax_lat.set_ylabel("Inference latency (ms/frame)")
    ax_lat.set_title("Compute latency by config (mean ± std)")
    ax_lat.spines["top"].set_visible(False)
    ax_lat.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    sessions_text = "Sessions: " + ", ".join(sessions)
    wrapped = "\n".join(textwrap.wrap(sessions_text, width=140))
    fig.text(0.01, 0.01, wrapped, fontsize=7, color="#52514e", va="bottom", ha="left")

    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_results(path: Path, video_name: str, iou_threshold: float, stats: dict, latencies: dict) -> None:
    payload = {
        "video": video_name,
        "iou_threshold": iou_threshold,
        "stats": {config: dict(classes) for config, classes in stats.items()},
        "latencies": dict(latencies),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_results(results: list) -> tuple:
    """Returns (stats, latencies, sessions, iou_thresholds) — stats/latencies
    combined per the module docstring's pooling rules; sessions is the list of
    source video names in the order given; iou_thresholds is the set of
    distinct thresholds seen (callers should warn if this has more than one
    entry, since mixing runs computed at different IoU thresholds is
    misleading)."""
    merged_stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    merged_latencies: dict = defaultdict(list)
    sessions = []
    iou_thresholds = set()

    for r in results:
        sessions.append(r["video"])
        iou_thresholds.add(r["iou_threshold"])
        for config, classes in r["stats"].items():
            for cls, counts in classes.items():
                tp, fp, fn = counts
                merged = merged_stats[config][cls]
                merged[0] += tp
                merged[1] += fp
                merged[2] += fn
        for config, lats in r["latencies"].items():
            merged_latencies[config].extend(lats)

    return merged_stats, merged_latencies, sessions, iou_thresholds
