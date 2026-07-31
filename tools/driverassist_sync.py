"""
Shared sync/log-parsing helpers for DriverAssist's offline tooling
(reconstruct_annotated.py, benchmark.py). Not a script itself.

Frame-to-timestamp sync: a video frame's actual embedded presentation
timestamp (PTS, relative to the start of the recording) should be read
directly from the video rather than assumed from a constant frame rate —
that avoids drift over a long drive if frames were ever dropped or timing
wasn't perfectly steady. That relative PTS, added to a precise wall-clock
anchor (the exact epoch time the app itself logged the moment recording
started — a `recording-start:` line in the debug log), gives a frame's real
epoch time.

Matching a recording to its `recording-start:` line is NOT done by filename
— the .mov gets renamed as part of the normal workflow (e.g. to
27_06_29_1.MOV), and a single debug log can contain *multiple*
`recording-start:` lines if the app was ever backgrounded and resumed
mid-drive (each resume starts a new recording segment). Instead, this reads
the video's own embedded `creation_time` (QuickTime/MP4 container metadata,
UTC, survives any renaming) via ffprobe, and picks whichever
`recording-start:` line is closest to it in time. If ffprobe/creation_time
isn't available, or no candidate is within a sanity-check tolerance, this
falls back to parsing the recording's filename
(recording-YYYYMMDD-HHMMSS.mov — only if it hasn't been renamed), which only
encodes local wall-clock time to one-second precision — a printed warning
flags whenever a fallback is used.

--detections and --debug-log (in callers) both accept a single file or a
directory of them (e.g. the whole ~/DriverAssist/logs/ that pull_logs.sh
fills up) — every matching file in a directory is searched/merged.
"""
import bisect
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_LOGS_DIR = Path.home() / "DriverAssist" / "logs"
MODEL_DISPLAY_NAMES = {"yolo26n": "nano", "yolo26s": "small", "yolo26m": "medium"}

RECORDING_NAME_RE = re.compile(r"recording-(\d{8})-(\d{6})")
THERMAL_LINE_RE = re.compile(
    r"^([\d.]+) thermal-speed: state=(\w+) elapsedMs=[\d.]+ smoothedMs=[\d.]+ baselineMs=[\d.]+ percent=(\d+)"
)
RECORDING_START_LINE_RE = re.compile(r"^[\d.]+ recording-start: file=(\S+) epoch=([\d.]+)")

# How close a video's own creation_time must land to a candidate
# `recording-start:` epoch to trust the match. Generous relative to how far
# apart distinct segments actually are (minutes, from backgrounding/resuming),
# but tight enough to catch "wrong log file entirely" mismatches.
MATCH_TOLERANCE_SECONDS = 10.0


def fallback_filename_epoch(video_path: Path) -> float:
    """One-second precision only — see module docstring. Used only when a
    precise `recording-start:` log line isn't available."""
    match = RECORDING_NAME_RE.search(video_path.stem)
    if not match:
        raise ValueError(
            f"Can't parse a start time out of '{video_path.name}' — expected "
            "a name like recording-20260729-145247.mov"
        )
    dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return dt.timestamp()  # naive datetime -> local-timezone epoch


def video_creation_time_epoch(video_path: Path) -> Optional[float]:
    """The video's own embedded creation time (QuickTime/MP4 container
    metadata), read via ffprobe — UTC, and unaffected by any local renaming.
    None if ffprobe isn't available or the tag is missing."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    if not value:
        return None
    dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def resolve_debug_log_files(path: Optional[Path]) -> list[Path]:
    """--debug-log accepts a single file or a directory of them (e.g. the
    whole ~/DriverAssist/logs/ pull_logs.sh fills up) — expand a directory
    into every overlay-debug*.log inside it. Missing entirely (e.g. the
    default logs dir doesn't exist yet) just means no thermal data — not
    fatal, unlike a missing --detections."""
    if path is None or not path.exists():
        return []
    if path.is_dir():
        return sorted(path.glob("overlay-debug*.log"))
    return [path]


def load_recording_starts(debug_log_path: Optional[Path]) -> list[tuple]:
    """All (epoch, original_device_filename, source_path) triples logged
    across the given log file(s) — there can be more than one per file if the
    app was ever backgrounded and resumed mid-drive (each resume starts a new
    recording segment)."""
    entries = []
    for log_path in resolve_debug_log_files(debug_log_path):
        with open(log_path) as f:
            for line in f:
                m = RECORDING_START_LINE_RE.match(line)
                if m:
                    entries.append((float(m.group(2)), m.group(1), log_path))
    return entries


def precise_start_epoch(video_path: Path, debug_log_path: Optional[Path]) -> tuple:
    """Returns (epoch, source_log_path) — the second so callers can load
    thermal data from the exact log file the match came from — or (None,
    None) if no confident match was found anywhere searched."""
    starts = load_recording_starts(debug_log_path)
    if not starts:
        return None, None

    # Fast path: filename hasn't been renamed away from the device's own
    # convention — an exact match needs no further disambiguation.
    for epoch, filename, source in starts:
        if filename == video_path.name:
            return epoch, source

    # Otherwise, match by the video's own embedded creation time, which
    # survives renaming — pick whichever logged segment start is closest,
    # regardless of which file (across possibly many) it came from.
    creation_epoch = video_creation_time_epoch(video_path)
    if creation_epoch is None:
        return None, None
    closest_epoch, closest_name, closest_source = min(starts, key=lambda e: abs(e[0] - creation_epoch))
    if abs(closest_epoch - creation_epoch) > MATCH_TOLERANCE_SECONDS:
        return None, None
    print(
        f"Matched '{video_path.name}' to logged segment '{closest_name}' "
        f"(in {closest_source.name}) by creation time.",
        file=sys.stderr,
    )
    return closest_epoch, closest_source


def resolve_start_epoch(video_path: Path, debug_log_path: Optional[Path]) -> tuple:
    """Returns (start_epoch, thermal_log_path) — the second is the specific
    log file to pull thermal readings from (None if unknown, e.g. the
    filename fallback was used and we never identified which log matches)."""
    precise, source = precise_start_epoch(video_path, debug_log_path)
    if precise is not None:
        return precise, source
    print(
        "No matching 'recording-start:' line found (checked exact filename and "
        "creation-time match across every log searched) — falling back to the "
        "recording filename's whole-second precision. Sync may be off by up "
        "to ~1s, and this only works if the file still has its original "
        "recording-YYYYMMDD-HHMMSS name.",
        file=sys.stderr,
    )
    try:
        # A single explicit file (not a directory) is still usable for
        # thermal even without a matching recording-start line in it.
        thermal_fallback = debug_log_path if debug_log_path is not None and debug_log_path.is_file() else None
        return fallback_filename_epoch(video_path), thermal_fallback
    except ValueError:
        sys.exit(
            f"Couldn't determine {video_path.name}'s start time by any method: "
            "no matching 'recording-start:' line (check --debug-log points at "
            "the log(s) from the same drive), and the filename isn't the "
            "original recording-YYYYMMDD-HHMMSS.mov (so it can't be parsed "
            "either). Pass the correct --debug-log, or keep a copy of the "
            "file under its original device filename."
        )


def resolve_detections_files(path: Path) -> list[Path]:
    """--detections accepts a single file or a directory of them (e.g. the
    whole ~/DriverAssist/logs/ pull_logs.sh fills up) — expand a directory
    into every detections*.jsonl inside it. No filtering by time range: the
    bisect-based nearest-lookup already only ever picks entries relevant to
    this video's own timestamps, so merging in unrelated sessions' data is
    harmless — simpler than trying to pre-filter."""
    if not path.exists():
        return []
    if path.is_dir():
        return sorted(path.glob("detections*.jsonl"))
    return [path]


def load_detections(path: Path) -> list[dict]:
    files = resolve_detections_files(path)
    if not files:
        sys.exit(
            f"No detections*.jsonl found at {path} — run tools/pull_logs.sh "
            "first, or pass --detections explicitly."
        )
    entries = []
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    if not entries:
        sys.exit(f"Found detections file(s) at {path}, but they were all empty.")
    entries.sort(key=lambda e: e["t"])
    return entries


def load_thermal(path: Optional[Path]) -> list[tuple]:
    if path is None:
        return []
    entries = []
    with open(path) as f:
        for line in f:
            m = THERMAL_LINE_RE.match(line)
            if m:
                entries.append((float(m.group(1)), m.group(2), int(m.group(3))))
    entries.sort(key=lambda e: e[0])
    return entries


def build_key_index(entries: list, key_fn) -> list:
    """Precomputed keys for repeated bisect lookups — recomputing per-frame
    would be O(frames * entries), too slow for a 20-30 minute drive."""
    return [key_fn(e) for e in entries]


def nearest_at_or_before(entries: list, keys: list, t: float):
    """Last entry with key <= t — mirrors what the on-device HUD actually
    showed (the most recently completed reading), not something from the
    future relative to this frame. Returns None if nothing exists at or
    before t (e.g. a frame captured before this session's first completed
    reading) — entries/keys can now span many merged, unrelated sessions
    (--detections/--debug-log accepting a whole logs directory), so falling
    back to entries[0] would risk grabbing data from a totally different
    drive instead of correctly showing "no data yet"."""
    if not entries:
        return None
    i = bisect.bisect_right(keys, t) - 1
    return None if i < 0 else entries[i]
