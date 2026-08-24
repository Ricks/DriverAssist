#!/usr/bin/env python3
"""
Scans every real drive session under data/ for "bicycle"-labeled detection
periods and ranks them by how head-on-likely the encounter looks, purely
from the logged box geometry (no video decoding needed for the ranking
itself) -- a fast first pass to gauge how much usable head-on-cyclist
footage already exists in this project's recordings, before deciding
whether an external dataset (BDD100K/Cityscapes/nuScenes/KITTI, which have
native rider/cyclist classes) is needed to bootstrap a "cyclist" class that
survives head-on views (the case where YOLO's stock "bicycle" class loses
almost all its discriminative signal -- no side profile, wheels collapse
into a sliver -- see tools/tracker.py's CONFUSABLE_LABELS comment).

Distinct from the earlier per-frame visual-triage pass in
data/head_on_cyclist_frames/ (see its own REPORT.md): that one grouped by
trackID and eyeballed sampled frames one at a time. This one groups by
*time* (a ~3s gap tolerance between consecutive bicycle-labeled detections,
same pattern find_disagreement_intervals.py uses to merge nearby
disagreement stretches) into head-on-vs-not "periods", scores each period
as a whole from its box trajectory, and only then extracts short video
clips for the highest-scoring candidates -- letting a human do the final
visual confirmation on a small, pre-ranked shortlist instead of scrolling
through thousands of individual frames.

Scoring (see score_period): three geometric cues, each already well-known
in this project's own calibration/tracking tooling --

  - aspect ratio (w/h): a side-view bicycle box is wide relative to height;
    a head-on view collapses toward a standing person's aspect (roughly
    0.3-0.6). Averaged across the whole period, so it's the least noisy of
    the three cues -- highest weight.
  - lateral drift: a head-on (approaching) cyclist grows in place near the
    image center; a crossing/side-view cyclist sweeps across x. Measured as
    the box center's full x-range over the period (also low-noise, uses
    every frame) -- second-highest weight.
  - height growth: an approaching cyclist's box should grow over the
    period. Measured by comparing the mean height of the period's first and
    last quarter of frames (smoothed, but still just two small subsets, so
    it's the most sensitive of the three to detector jitter/occlusion --
    lowest weight. It also structurally only rewards *approaching*
    (head-on), not *receding* (tail-on) cyclists -- see REPORT.md's
    discussion of this asymmetry, since the earlier per-frame pass found
    tail-on examples valuable too).

Usage:
    python3 tools/scan_head_on_cyclists.py
    python3 tools/scan_head_on_cyclists.py --top-n 20 --skip-clips
"""
import argparse
import statistics
import subprocess
import sys
from pathlib import Path

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "head_on_cyclist_scan_2026-08-22"

VIDEO_EXTENSIONS = {".mov", ".mp4"}
# Per the task spec: skip calibration-purpose recordings by filename prefix,
# not by session directory (several "*Calibration*"-named directories, e.g.
# 26_08_11_YawCalibration, actually contain plain recording-*.MOV clips and
# are kept).
VIDEO_SKIP_PREFIXES = ("wipercal-", "calibration-", "nearfocus-")
# Derived/output clips from other tools in this project (debug overlays,
# review reels) that happen to sit next to the raw recording -- substrings
# rather than prefixes since e.g. "recording-<ts>-yawline-check.mp4" still
# starts with "recording-". These aren't raw footage and have no detections
# file of their own, so counting them as sibling videos would wrongly starve
# a same-directory raw recording of its shared detections.jsonl fallback.
DERIVED_VIDEO_MARKERS = ("-annotated", "-yawline-check", "-disagreement-reel", "-labeled-vs-algorithm")
# Confirmed junk/dev clips (<1 minute, not real drives) by the earlier
# data/head_on_cyclist_frames/REPORT.md pass -- excluded by directory name
# since their video filenames (TEST.MOV etc.) don't match any prefix rule.
EXCLUDE_SESSION_DIRS = {"TEST", "TEST2", "TEST3", "head_on_cyclist_frames"}

GAP_TOLERANCE_SECONDS = 3.0  # matches find_disagreement_intervals.py's --merge-gap default
MIN_DETECTIONS_PER_PERIOD = 5

# Aspect ratio (w/h): at/below ASPECT_LOW scores 1.0 (fully head-on-like),
# at/above ASPECT_HIGH scores 0.0 (fully side-view-like), linear between.
ASPECT_LOW = 0.6
ASPECT_HIGH = 1.5
# Lateral drift: box-center x-range (normalized to frame width) at/above
# this counts as a full side/crossing sweep (score 0.0).
DRIFT_NORM = 0.30
# Height growth: proportional growth (h_end/h_start - 1) at/above this
# counts as maximal approach signal (score 1.0). Shrinking/flat clamps to 0.
GROWTH_CAP = 0.50

SCORE_WEIGHTS = {"aspect": 0.40, "drift": 0.35, "growth": 0.25}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def is_video_candidate(path: Path) -> bool:
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    stem_lower = path.stem.lower()
    if any(marker in stem_lower for marker in DERIVED_VIDEO_MARKERS):
        return False
    return not stem_lower.startswith(VIDEO_SKIP_PREFIXES)


def find_session_video_candidates(session_dir: Path) -> list:
    return sorted(p for p in session_dir.iterdir() if is_video_candidate(p))


def resolve_detections_for_video(session_dir: Path, video: Path, sibling_video_count: int):
    """Per-video-named file first (multi-recording sessions), else the
    shared detections.jsonl -- but only when this video is the only real
    candidate in the session, so a shared file is never silently split
    across multiple videos it might not actually all belong to."""
    per_video = session_dir / f"{video.stem}-detections.jsonl"
    if per_video.exists():
        return per_video
    shared = session_dir / "detections.jsonl"
    if sibling_video_count == 1 and shared.exists():
        return shared
    return None


def resolve_debug_log_for_video(session_dir: Path, video: Path) -> Path:
    per_video = session_dir / f"{video.stem}-overlay-debug.log"
    if per_video.exists():
        return per_video
    shared = session_dir / "overlay-debug.log"
    if shared.exists():
        return shared
    return DEFAULT_LOGS_DIR


def extract_bicycle_periods(detections_path: Path) -> list:
    """Raw bicycle detections from detections_path, grouped into contiguous
    periods (capture-time gap <= GAP_TOLERANCE_SECONDS). Returns every
    period found, regardless of size -- callers filter by
    MIN_DETECTIONS_PER_PERIOD themselves so the "too few, skipped" count in
    the report is accurate."""
    entries = load_detections(detections_path)
    bicycle_dets = []
    for e in entries:
        capture_t = e["t"] - e.get("elapsedMs", 0.0) / 1000.0
        for d in e.get("detections", []):
            if d.get("label") == "bicycle" and d.get("h", 0) > 0:
                bicycle_dets.append({
                    "t": capture_t,
                    "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                    "confidence": d.get("confidence"),
                })
    bicycle_dets.sort(key=lambda r: r["t"])

    periods = []
    current = []
    for r in bicycle_dets:
        if current and r["t"] - current[-1]["t"] > GAP_TOLERANCE_SECONDS:
            periods.append(current)
            current = []
        current.append(r)
    if current:
        periods.append(current)
    return periods


def score_period(period: list) -> dict:
    aspects = [d["w"] / d["h"] for d in period]
    avg_aspect = statistics.mean(aspects)
    aspect_score = clamp(1.0 - (avg_aspect - ASPECT_LOW) / (ASPECT_HIGH - ASPECT_LOW), 0.0, 1.0)

    centers_x = [d["x"] + d["w"] / 2 for d in period]
    x_drift = max(centers_x) - min(centers_x)
    drift_score = clamp(1.0 - x_drift / DRIFT_NORM, 0.0, 1.0)

    n = len(period)
    k = max(1, n // 4)
    h_start = statistics.mean(d["h"] for d in period[:k])
    h_end = statistics.mean(d["h"] for d in period[-k:])
    growth_ratio = (h_end / h_start - 1.0) if h_start > 0 else 0.0
    growth_score = clamp(growth_ratio / GROWTH_CAP, 0.0, 1.0)

    score = (
        SCORE_WEIGHTS["aspect"] * aspect_score
        + SCORE_WEIGHTS["drift"] * drift_score
        + SCORE_WEIGHTS["growth"] * growth_score
    )
    return {
        "score": score,
        "avg_aspect": avg_aspect, "aspect_score": aspect_score,
        "x_drift": x_drift, "drift_score": drift_score,
        "growth_ratio": growth_ratio, "growth_score": growth_score,
        "avg_confidence": statistics.mean(
            d["confidence"] for d in period if d["confidence"] is not None
        ) if any(d["confidence"] is not None for d in period) else None,
    }


def scan_all_sessions(data_dir: Path) -> tuple:
    """Fast pass: no ffprobe/ffmpeg, just detections.jsonl parsing. Returns
    (all_period_records, stats_dict)."""
    all_records = []
    sessions_scanned = 0
    sessions_skipped_no_detections = []
    total_periods = 0
    periods_too_few = 0

    for session_dir in sorted(data_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name in EXCLUDE_SESSION_DIRS:
            continue
        videos = find_session_video_candidates(session_dir)
        if not videos:
            continue

        session_had_usable_data = False
        for video in videos:
            det_path = resolve_detections_for_video(session_dir, video, len(videos))
            if det_path is None:
                continue
            session_had_usable_data = True
            periods = extract_bicycle_periods(det_path)
            total_periods += len(periods)
            for period in periods:
                if len(period) < MIN_DETECTIONS_PER_PERIOD:
                    periods_too_few += 1
                    continue
                scored = score_period(period)
                all_records.append({
                    "session": session_dir.name,
                    "session_dir": session_dir,
                    "video": video,
                    "n_detections": len(period),
                    "start_t": period[0]["t"],
                    "end_t": period[-1]["t"],
                    **scored,
                })

        if session_had_usable_data:
            sessions_scanned += 1
        else:
            sessions_skipped_no_detections.append(session_dir.name)

    all_records.sort(key=lambda r: r["score"], reverse=True)
    stats = {
        "sessions_scanned": sessions_scanned,
        "sessions_skipped_no_detections": sessions_skipped_no_detections,
        "total_periods": total_periods,
        "periods_too_few": periods_too_few,
        "periods_scored": total_periods - periods_too_few,
    }
    return all_records, stats


def select_top_diverse(records: list, top_n: int, max_per_session: int) -> list:
    """Highest score first, capped at max_per_session per session so the
    shortlist isn't dominated by one long encounter -- then backfilled from
    the remaining ranked list if the cap left the shortlist short."""
    selected = []
    session_counts = {}
    for r in records:
        if len(selected) >= top_n:
            break
        if session_counts.get(r["session"], 0) < max_per_session:
            selected.append(r)
            session_counts[r["session"]] = session_counts.get(r["session"], 0) + 1
    if len(selected) < top_n:
        selected_ids = {id(r) for r in selected}
        for r in records:
            if len(selected) >= top_n:
                break
            if id(r) not in selected_ids:
                selected.append(r)
    return selected


def extract_clip(record: dict, rank: int, clips_dir: Path, pad: float) -> dict:
    """Resolves the video-relative window (the only step needing
    ffprobe/resolve_start_epoch) and cuts a padded clip via ffmpeg. Mutates
    nothing on record; returns clip metadata for the report."""
    video = record["video"]
    session_dir = record["session_dir"]
    debug_log = resolve_debug_log_for_video(session_dir, video)
    start_epoch, _ = resolve_start_epoch(video, debug_log)

    rel_start = record["start_t"] - start_epoch
    rel_end = record["end_t"] - start_epoch
    clip_start = max(0.0, rel_start - pad)
    clip_end = rel_end + pad
    duration = clip_end - clip_start

    mm = int(rel_start // 60)
    ss = int(rel_start % 60)
    out_name = f"rank{rank:02d}_{record['session']}__{video.stem}__{mm:02d}-{ss:02d}.mp4"
    out_path = clips_dir / out_name

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(clip_start), "-i", str(video),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-an",
            str(out_path),
        ],
        check=True,
    )
    return {"clip_path": out_path, "rel_start": rel_start, "rel_end": rel_end, "mm_ss": f"{mm:02d}:{ss:02d}"}


def score_histogram(records: list, n_buckets: int = 5) -> list:
    buckets = [0] * n_buckets
    for r in records:
        idx = min(n_buckets - 1, int(r["score"] * n_buckets))
        buckets[idx] += 1
    width = 1.0 / n_buckets
    return [(round(i * width, 2), round((i + 1) * width, 2), c) for i, c in enumerate(buckets)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=18, help="How many highest-scoring periods to extract clips for")
    parser.add_argument("--max-per-session", type=int, default=3, help="Cap on shortlisted periods from any one session, for spread")
    parser.add_argument("--clip-pad", type=float, default=1.0, help="Seconds of padding before/after each period in extracted clips")
    parser.add_argument("--skip-clips", action="store_true", help="Fast pass only -- rank periods but don't run ffmpeg")
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} ...", file=sys.stderr)
    records, stats = scan_all_sessions(args.data_dir)

    print(f"\nSessions scanned (usable detections found): {stats['sessions_scanned']}", file=sys.stderr)
    print(f"Sessions with videos but no usable detections file: {len(stats['sessions_skipped_no_detections'])} "
          f"({', '.join(stats['sessions_skipped_no_detections'])})", file=sys.stderr)
    print(f"Total bicycle periods found: {stats['total_periods']}", file=sys.stderr)
    print(f"  -- too few detections (<{MIN_DETECTIONS_PER_PERIOD}), skipped: {stats['periods_too_few']}", file=sys.stderr)
    print(f"  -- scored: {stats['periods_scored']}", file=sys.stderr)
    print("Score distribution (scored periods):", file=sys.stderr)
    for lo, hi, count in score_histogram(records):
        print(f"  [{lo:.2f}, {hi:.2f}): {count}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    shortlist = select_top_diverse(records, args.top_n, args.max_per_session)

    print(f"\nShortlisted top {len(shortlist)} periods (max {args.max_per_session}/session):", file=sys.stderr)
    clip_results = []
    for i, r in enumerate(shortlist, start=1):
        clip_info = None
        if not args.skip_clips:
            try:
                clip_info = extract_clip(r, i, clips_dir, args.clip_pad)
            except subprocess.CalledProcessError as exc:
                print(f"  [{i:02d}] ffmpeg FAILED for {r['session']}/{r['video'].name}: {exc}", file=sys.stderr)
        clip_results.append({**r, "clip_info": clip_info})
        loc = clip_info["mm_ss"] if clip_info else "(clip skipped)"
        print(f"  [{i:02d}] score={r['score']:.3f} {r['session']}/{r['video'].name} @ {loc} "
              f"(n={r['n_detections']}, aspect={r['avg_aspect']:.2f}, drift={r['x_drift']:.3f}, growth={r['growth_ratio']:+.2f})",
              file=sys.stderr)

    import json
    dump_path = args.output_dir / "scan_results.json"
    dump = {
        "stats": stats,
        "histogram": score_histogram(records),
        "all_periods": [
            {k: v for k, v in r.items() if k not in ("session_dir", "video")}
            | {"video": str(r["video"])}
            for r in records
        ],
        "shortlist": [
            {
                "rank": i + 1,
                "session": r["session"],
                "video": str(r["video"]),
                "score": r["score"],
                "n_detections": r["n_detections"],
                "avg_aspect": r["avg_aspect"],
                "x_drift": r["x_drift"],
                "growth_ratio": r["growth_ratio"],
                "avg_confidence": r["avg_confidence"],
                "clip": str(r["clip_info"]["clip_path"]) if r["clip_info"] else None,
                "mm_ss": r["clip_info"]["mm_ss"] if r["clip_info"] else None,
            }
            for i, r in enumerate(clip_results)
        ],
    }
    dump_path.write_text(json.dumps(dump, indent=2, default=str))
    print(f"\nWrote {dump_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
