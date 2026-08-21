#!/usr/bin/env python3
"""
Packages one or more DriverAssist recording sessions into a single
self-contained directory, instead of the raw video(s) sitting in the repo
root while their detections/thermal data stays mixed into the shared
~/DriverAssist/logs/ directory alongside every other drive.

Assumed workflow (single session):
    mkdir 26_07_30_Day_Hosp_nano_off
    mv 26_07_30_Day_Hosp_nano_off.MOV 26_07_30_Day_Hosp_nano_off/
    python3 tools/package_session.py 26_07_30_Day_Hosp_nano_off

Or multiple short sessions sharing one directory (e.g. a config-comparison
matrix -- one recording per config, all recorded back-to-back the same day):
    mkdir 26_08_17_Matrix_Day
    mv recording-*.MOV 26_08_17_Matrix_Day/
    python3 tools/package_session.py 26_08_17_Matrix_Day

For EACH raw recording found in the directory, this:
  1. Resolves the recording's start time (see driverassist_sync.py) and slices
     just that session's entries out of the shared detections.jsonl/
     overlay-debug*.log data, writing them into the directory -- so the
     directory no longer depends on ~/DriverAssist/logs/ to be useful later.
     With exactly one recording in the directory, these are written as the
     plain `detections.jsonl`/`overlay-debug.log` (unchanged from before,
     so existing single-session directories/tooling keep working). With more
     than one recording, each gets its own `<video>-detections.jsonl`/
     `<video>-overlay-debug.log` instead, since a shared filename can't hold
     more than one session's slice.
  2. Runs reconstruct_annotated.py against those local files, writing
     <video>-annotated.mp4 into the same directory.
  3. With --benchmark, also runs benchmark.py (slow -- loads the yolo26x
     reference model and runs inference per logged frame), writing
     <video>-benchmark.json/.png into the directory too.
  4. Runs efficacy_score.py (slow -- loads the yolo26x reference model and
     runs inference at a dense, near-every-frame interval across the whole
     session, not just at each logged on-device frame), writing
     <video>-efficacy-score.json and <video>-track-cache.pkl into the
     directory. On by default, same as annotation -- pass --skip-efficacy
     to opt out. The tracking-aware (idf1/mota/miss rate/P(warned-in-time))
     counterpart to --benchmark's simpler per-frame precision/recall -- see
     efficacy_score.py's own docstring for the two-gate framework this
     produces. The cache is what a follow-up per-class analysis (e.g.
     "what's the bicycle detection rate vs yolo26x") can load directly
     instead of re-running the expensive dense pass.

The shared logs pool is pulled from the device once per run (not once per
video) -- all recordings in the directory are sliced out of that single
fresh pull.

Safe to re-run: each output is skipped if it already exists (delete it first
to force regeneration).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_debug_log_files, resolve_start_epoch

VIDEO_EXTENSIONS = {".mov", ".mp4"}

# Slack around the video's [start, start+duration] window -- covers logging
# latency (a detection's logged completion time trails the frame's actual
# capture time by up to elapsedMs) and epoch/PTS rounding, without pulling in
# an adjacent, unrelated session.
RANGE_BUFFER_BEFORE_SECONDS = 2.0
RANGE_BUFFER_AFTER_SECONDS = 5.0


def pull_logs(tools_dir: Path) -> None:
    """Runs pull_logs.sh to refresh the shared logs pool from the device before
    extracting this session's slice -- without this, a session packaged after the
    pool went stale (nobody re-ran pull_logs.sh since an earlier drive) silently
    filters against old data and finds nothing for the new time range, which then
    surfaces as a confusing downstream crash in reconstruct_annotated.py instead of
    the real problem. Failure here (device unreachable, wifi flake -- known to
    happen) doesn't abort packaging: the shared pool may still be usable/current
    enough, or the caller may be intentionally working offline from already-pulled
    data, so this only warns and lets write_session_logs's own empty-range check
    below catch it if the local data really is too stale to use.
    """
    script = tools_dir / "pull_logs.sh"
    if not script.exists():
        return
    print("Pulling latest logs from device...")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: pull_logs.sh failed ({result.returncode}) -- continuing with "
              f"whatever's already local, which may be stale:\n{result.stderr.strip()}",
              file=sys.stderr)
    else:
        for line in result.stdout.splitlines():
            print(f"  {line}")


def find_session_videos(session_dir: Path) -> list:
    """Returns every raw recording in session_dir, sorted by filename (which
    -- given the app's recording-YYYYMMDD-HHMMSS.MOV naming -- also sorts
    them chronologically). One recording is the common case; more than one
    is the config-comparison-matrix case (see module docstring)."""
    candidates = [
        p for p in session_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and "-annotated" not in p.stem
    ]
    if not candidates:
        sys.exit(f"No raw recording found in {session_dir} -- move it in first.")
    return sorted(candidates)


def find_session_video(session_dir: Path) -> Path:
    """Singular counterpart to find_session_videos, for callers that need
    exactly one video and should error on a config-comparison-matrix
    directory rather than silently picking one -- e.g. tracker.py's own
    single-video CLI."""
    videos = find_session_videos(session_dir)
    if len(videos) > 1:
        names = ", ".join(p.name for p in videos)
        sys.exit(f"Found more than one candidate raw recording in {session_dir}: {names} -- this tool needs exactly one.")
    return videos[0]


def video_duration_seconds(video: Path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def write_session_logs(video: Path, detections_path: Path, debug_log_path: Path, logs_dir: Path) -> tuple:
    """Filters the shared detections.jsonl/overlay-debug*.log data down to
    just this video's time range and writes self-contained copies to
    detections_path/debug_log_path."""
    start_epoch, _ = resolve_start_epoch(video, logs_dir)
    duration = video_duration_seconds(video)
    range_start = start_epoch - RANGE_BUFFER_BEFORE_SECONDS
    if duration is None:
        print("Warning: couldn't read the video's duration via ffprobe -- not bounding the end of the "
              "session range, so trailing entries from whatever recorded next may be pulled in too.",
              file=sys.stderr)
        range_end = float("inf")
    else:
        range_end = start_epoch + duration + RANGE_BUFFER_AFTER_SECONDS

    all_detections = load_detections(logs_dir)
    session_detections = [e for e in all_detections if range_start <= e["t"] <= range_end]
    with open(detections_path, "w") as f:
        for entry in session_detections:
            f.write(json.dumps(entry) + "\n")

    lines_written = 0
    with open(debug_log_path, "w") as out:
        for log_file in resolve_debug_log_files(logs_dir):
            with open(log_file) as f:
                for line in f:
                    try:
                        t = float(line.split(" ", 1)[0])
                    except ValueError:
                        continue
                    if range_start <= t <= range_end:
                        out.write(line)
                        lines_written += 1

    print(f"Wrote {len(session_detections)} detection entries -> {detections_path}")
    print(f"Wrote {lines_written} debug-log lines -> {debug_log_path}")

    # Zero of both is almost never a real empty drive -- the app logs a line per
    # captured frame (15fps) regardless of whether anything was detected, so a
    # genuine drive has thousands of overlay-debug lines even with zero objects
    # in frame. This combination is the actual symptom of a stale shared logs
    # pool (nobody re-ran pull_logs.sh since an earlier drive), which otherwise
    # surfaces downstream as a confusing reconstruct_annotated.py crash instead
    # of the real, fixable cause.
    if not session_detections and lines_written == 0:
        sys.exit(
            f"Found 0 entries in {logs_dir} for this video's time range "
            f"({range_start:.0f}-{range_end if range_end == float('inf') else f'{range_end:.0f}'}). "
            "This almost certainly means the shared logs pool is stale, not that the "
            "drive had nothing happen -- pull_logs.sh should have refreshed it "
            "automatically unless that failed (check the warning above, if any) or "
            "--skip-pull was passed. Fix the underlying issue and re-run rather than "
            "trusting this output."
        )

    return detections_path, debug_log_path


def existing_logs_cover_video(video: Path, detections_path: Path, debug_log_path: Path, logs_dir: Path) -> bool:
    """True if the ALREADY-PRESENT detections.jsonl/overlay-debug.log's own
    logged time range actually overlaps this video's -- guards against a
    session directory being reused for a re-recorded take (same directory
    name, different physical recording). CONFIRMED 2026-08-15 (a walkaround
    retest): package_session.py saw detections.jsonl/overlay-debug.log
    already present (from the FIRST recording), skipped extraction per its
    documented "safe to re-run" behavior, then happily annotated the NEW
    video against the OLD, temporally unrelated detection data -- no crash,
    no warning, just a plausible-looking but silently wrong reconstruction.
    Only peeks at each file's first/last line (not a full parse) since all
    that's needed is "roughly the right neighborhood in time", not exact
    coverage.
    """
    start_epoch, _ = resolve_start_epoch(video, logs_dir)
    duration = video_duration_seconds(video)
    range_start = start_epoch - RANGE_BUFFER_BEFORE_SECONDS
    range_end = start_epoch + duration + RANGE_BUFFER_AFTER_SECONDS if duration is not None else float("inf")

    def line_epoch(line: str):
        try:
            return float(line.split(" ", 1)[0])
        except ValueError:
            pass
        try:
            return json.loads(line)["t"]
        except Exception:
            return None

    def file_time_range(path: Path):
        times = []
        with open(path) as f:
            first = f.readline()
        t = line_epoch(first)
        if t is not None:
            times.append(t)
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 4096)
            f.seek(-block, 2)
            tail = f.read().decode(errors="ignore")
        lines = tail.strip().splitlines()
        if lines:
            t = line_epoch(lines[-1])
            if t is not None:
                times.append(t)
        return (min(times), max(times)) if times else None

    existing_range = file_time_range(detections_path) or file_time_range(debug_log_path)
    if existing_range is None:
        return True  # Inconclusive (e.g. an empty file) -- don't block on it.
    existing_start, existing_end = existing_range
    return existing_start <= range_end and existing_end >= range_start


def process_video(video: Path, session_dir: Path, args, tools_dir: Path, shared_names: bool) -> None:
    """Packages one recording. shared_names=True uses the plain
    detections.jsonl/overlay-debug.log names (single-recording-per-directory
    case, unchanged from before); False prefixes them with the video's stem
    (multi-recording case, so each session gets its own files)."""
    if shared_names:
        detections_path = session_dir / "detections.jsonl"
        debug_log_path = session_dir / "overlay-debug.log"
    else:
        detections_path = session_dir / f"{video.stem}-detections.jsonl"
        debug_log_path = session_dir / f"{video.stem}-overlay-debug.log"

    if detections_path.exists() and debug_log_path.exists():
        if existing_logs_cover_video(video, detections_path, debug_log_path, args.logs_dir):
            print(f"{detections_path.name}/{debug_log_path.name} already present -- skipping extraction "
                  "(delete them to force re-extraction).")
        else:
            sys.exit(
                f"{detections_path.name}/{debug_log_path.name} already exist in {session_dir}, but their "
                f"logged time range doesn't overlap {video.name}'s at all. This almost certainly means "
                "the directory was reused for a re-recorded take (same name, different session) and "
                "these are stale data from the PREVIOUS recording, not this one -- annotating against "
                "them would silently produce a plausible-looking but wrong reconstruction. Delete "
                f"{detections_path.name}/{debug_log_path.name} (and any stale {video.stem}-annotated.mp4) "
                "and re-run to extract the correct data for this video."
            )
    else:
        detections_path, debug_log_path = write_session_logs(video, detections_path, debug_log_path, args.logs_dir)

    annotated_path = session_dir / f"{video.stem}-annotated.mp4"
    if args.skip_annotate:
        print(f"--skip-annotate passed -- not generating {annotated_path.name}.")
    elif annotated_path.exists() and annotated_path.stat().st_mtime >= debug_log_path.stat().st_mtime:
        print(f"{annotated_path.name} already exists and is newer than its inputs -- skipping (delete it to force regeneration).")
    else:
        print("Generating annotated reconstruction...")
        subprocess.run(
            [sys.executable, str(tools_dir / "reconstruct_annotated.py"), str(video),
             "--detections", str(detections_path), "--debug-log", str(debug_log_path),
             "--output", str(annotated_path)],
            check=True,
        )

    if args.benchmark:
        benchmark_json = session_dir / f"{video.stem}-benchmark.json"
        benchmark_png = session_dir / f"{video.stem}-benchmark.png"
        if benchmark_json.exists() and benchmark_png.exists():
            print(f"{benchmark_json.name}/{benchmark_png.name} already exist -- skipping "
                  "(delete them to force regeneration).")
        else:
            print("Running benchmark against the reference model (slow)...")
            cmd = [sys.executable, str(tools_dir / "benchmark.py"), str(video),
                   "--detections", str(detections_path), "--debug-log", str(debug_log_path),
                   "--output", str(benchmark_png), "--results", str(benchmark_json)]
            if args.reference_model:
                cmd += ["--reference-model", str(args.reference_model)]
            subprocess.run(cmd, check=True)

    if args.skip_efficacy:
        print("--skip-efficacy passed -- not running efficacy_score.py's dense reference pass.")
    else:
        efficacy_json = session_dir / f"{video.stem}-efficacy-score.json"
        efficacy_cache = session_dir / f"{video.stem}-track-cache.pkl"
        if efficacy_json.exists() and efficacy_cache.exists():
            print(f"{efficacy_json.name}/{efficacy_cache.name} already exist -- skipping "
                  "(delete them, or pass --rebuild-efficacy-cache, to force regeneration).")
        else:
            print("Running efficacy_score.py's dense yolo26x reference pass (slow)...")
            cmd = [sys.executable, str(tools_dir / "efficacy_score.py"), str(video),
                   "--detections", str(detections_path), "--debug-log", str(debug_log_path),
                   "--results", str(efficacy_json), "--cache", str(efficacy_cache)]
            if args.reference_model:
                cmd += ["--reference-model", str(args.reference_model)]
            if args.rebuild_efficacy_cache:
                cmd += ["--rebuild-cache"]
            subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Directory containing one or more raw recordings")
    parser.add_argument(
        "--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"Shared logs directory to pull this session's slice from (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Also run benchmark.py (slow -- runs the yolo26x reference model) into this directory",
    )
    parser.add_argument("--reference-model", type=Path, default=None, help="Passed through to benchmark.py/efficacy_score.py if --benchmark is set / --skip-efficacy isn't")
    parser.add_argument(
        "--skip-efficacy", action="store_true",
        help="Skip efficacy_score.py's dense yolo26x reference pass (slow -- runs at a near-every-frame "
             "interval across the whole session) -- runs by default, same as annotation",
    )
    parser.add_argument(
        "--rebuild-efficacy-cache", action="store_true",
        help="Passed through as --rebuild-cache to efficacy_score.py -- forces the dense reference pass "
             "to redo even if efficacy_score.py's own cache signature matches",
    )
    parser.add_argument(
        "--skip-annotate", action="store_true",
        help="Only extract each session's local detections.jsonl/overlay-debug.log -- skip generating the annotated reconstruction",
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="Don't run pull_logs.sh first -- use whatever's already in --logs-dir as-is",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    if not session_dir.is_dir():
        sys.exit(f"{session_dir} is not a directory -- create it and move the raw recording(s) in first.")

    tools_dir = Path(__file__).parent
    if args.skip_pull:
        print("--skip-pull passed -- not refreshing the shared logs pool from the device.")
    else:
        pull_logs(tools_dir)

    videos = find_session_videos(session_dir)
    shared_names = len(videos) == 1
    if shared_names:
        print(f"Session video: {videos[0]}")
    else:
        print(f"{len(videos)} recordings found in {session_dir} -- packaging each separately:")
        for v in videos:
            print(f"  {v.name}")

    for video in videos:
        if not shared_names:
            print(f"\n=== {video.name} ===")
        process_video(video, session_dir, args, tools_dir, shared_names)

    print(f"\nSession{'s' if not shared_names else ''} packaged in {session_dir}")


if __name__ == "__main__":
    main()
