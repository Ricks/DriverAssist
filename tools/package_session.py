#!/usr/bin/env python3
"""
Packages one DriverAssist recording session into a single self-contained
directory, instead of the raw video sitting in the repo root while its
detections/thermal data stays mixed into the shared ~/DriverAssist/logs/
directory alongside every other drive.

Assumed workflow:
    mkdir 26_07_30_Day_Hosp_nano_off
    mv 26_07_30_Day_Hosp_nano_off.MOV 26_07_30_Day_Hosp_nano_off/
    python3 tools/package_session.py 26_07_30_Day_Hosp_nano_off

Given a directory containing exactly one raw recording, this:
  1. Resolves the recording's start time (see driverassist_sync.py) and slices
     just this session's entries out of the shared detections.jsonl/
     overlay-debug*.log data, writing them as local detections.jsonl/
     overlay-debug.log inside the directory -- so the directory no longer
     depends on ~/DriverAssist/logs/ to be useful later.
  2. Runs reconstruct_annotated.py against those local files, writing
     <video>-annotated.mp4 into the same directory.
  3. With --benchmark, also runs benchmark.py (slow -- loads the yolo26x
     reference model and runs inference per logged frame), writing
     <video>-benchmark.json/.png into the directory too.

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


def find_session_video(session_dir: Path) -> Path:
    candidates = [
        p for p in session_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and "-annotated" not in p.stem
    ]
    if not candidates:
        sys.exit(f"No raw recording found in {session_dir} -- move it in first.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        sys.exit(f"Found more than one candidate raw recording in {session_dir}: {names} -- keep just one.")
    return candidates[0]


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


def write_session_logs(video: Path, session_dir: Path, logs_dir: Path) -> tuple:
    """Filters the shared detections.jsonl/overlay-debug*.log data down to
    just this video's time range and writes self-contained copies into
    session_dir."""
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

    detections_path = session_dir / "detections.jsonl"
    all_detections = load_detections(logs_dir)
    session_detections = [e for e in all_detections if range_start <= e["t"] <= range_end]
    with open(detections_path, "w") as f:
        for entry in session_detections:
            f.write(json.dumps(entry) + "\n")

    debug_log_path = session_dir / "overlay-debug.log"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Directory containing exactly one raw recording")
    parser.add_argument(
        "--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"Shared logs directory to pull this session's slice from (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Also run benchmark.py (slow -- runs the yolo26x reference model) into this directory",
    )
    parser.add_argument("--reference-model", type=Path, default=None, help="Passed through to benchmark.py if --benchmark is set")
    parser.add_argument(
        "--skip-annotate", action="store_true",
        help="Only extract this session's local detections.jsonl/overlay-debug.log -- skip generating the annotated reconstruction",
    )
    parser.add_argument(
        "--skip-pull", action="store_true",
        help="Don't run pull_logs.sh first -- use whatever's already in --logs-dir as-is",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    if not session_dir.is_dir():
        sys.exit(f"{session_dir} is not a directory -- create it and move the raw recording in first.")

    tools_dir = Path(__file__).parent
    if args.skip_pull:
        print("--skip-pull passed -- not refreshing the shared logs pool from the device.")
    else:
        pull_logs(tools_dir)

    video = find_session_video(session_dir)
    print(f"Session video: {video}")

    detections_path = session_dir / "detections.jsonl"
    debug_log_path = session_dir / "overlay-debug.log"
    if detections_path.exists() and debug_log_path.exists():
        print(f"{detections_path.name}/{debug_log_path.name} already present -- skipping extraction "
              "(delete them to force re-extraction).")
    else:
        detections_path, debug_log_path = write_session_logs(video, session_dir, args.logs_dir)

    annotated_path = session_dir / f"{video.stem}-annotated.mp4"
    if args.skip_annotate:
        print(f"--skip-annotate passed -- not generating {annotated_path.name}.")
    elif annotated_path.exists():
        print(f"{annotated_path.name} already exists -- skipping (delete it to force regeneration).")
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

    print(f"\nSession packaged in {session_dir}")


if __name__ == "__main__":
    main()
