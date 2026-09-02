#!/usr/bin/env python3
"""
Unified browser-based viewer/labeling tool -- merges what used to be two
separate tools (real request 2026-08-25, once flow_debug_viewer.py's
Part-1 on-device-track-ID work landed):

  - flow_debug_viewer.py's read-only flow-arrow/distance debug view (reads
    a precomputed --flow-debug-json from reconstruct_annotated.py, which
    mirrors ON-DEVICE track identity and the on-device app's own logic
    rather than an independent re-tracking/re-analysis pass -- see that
    file's own doc comments)
  - label_leading_vehicle.py's click-to-label ground-truth modes
    (followed_vehicle, false_positive, hood_truncation, other_truncation,
    cyclist, motorcyclist, skateboarder, equestrian)

A distance-anomaly labeling mode (too far/too close + actual ground-truth
distance + truncation type, added 2026-08-29) slots in as one more entry
in the frontend's MODES registry -- same chassis as the other
'single'-shaped modes, just with three extra fields captured via its trim
menu.

MULTI-SESSION (2026-09-01): one server process now holds a *registry* of
recordings you've pulled up (persisted to ~/.driverassist_super_tool_
sessions.json), and the frontend's left sidebar switches the active one in
place -- the per-recording view state (time, mode, zoom/pan, selection)
lives in the browser so toggling between sessions never loses your place.
Sessions are added/removed from the sidebar (a native macOS file/folder
chooser via osascript). Launching with a positional <video> still works:
it just registers + activates that one.

Ground-truth labeling prefers ON-DEVICE track IDs (from a session's
--flow-debug-json when present) over a live geometry-only ByteTracker
pass, for consistency with what the on-device app itself would show; the
live pass is the fallback when a session has no flow-debug JSON.

Usage:
    python3 super_tool.py [<video>] [--flow-debug-json path] [--port 5050]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from reconstruct_annotated import row_based_distance_meters
from tracker import ByteTracker

FRONTEND_PATH = Path(__file__).parent / "super_tool_frontend.html"

# One ground-truth file per labeling mode -- followed_vehicle keeps its
# original filename since other tools (tune_leading_vehicle.py, ...) read
# <dir>/ground_truth.json directly. flow_debug is READ-ONLY and excluded.
GROUND_TRUTH_MODES = {
    "followed_vehicle", "false_positive", "hood_truncation", "other_truncation",
    "distance_calibration", "distance_anomaly", "cyclist", "motorcyclist",
    "skateboarder", "equestrian",
}
# distance_calibration is hand-marked calibration ground truth, NOT large
# regenerable recording output -- so it lives in a git-tracked top-level dir.
REPO_ROOT = Path(__file__).parent.parent

# Small user-global JSON blobs, persisted server-side so they survive the
# browser/webview dropping localStorage between sessions. NOT in the repo.
SETTINGS_PATH = Path.home() / ".driverassist_super_tool_settings.json"
SESSIONS_PATH = Path.home() / ".driverassist_super_tool_sessions.json"


def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------
# Session registry
# ----------------------------------------------------------------------------
# REG: {"sessions": [ {id, label, video, flow_debug_json|None}, ... ],
#       "active": <id>|None}
# LOADED: id -> {"frames", "videoResolution", "videoFps", "hasFlowData"}
#         (lazy; the flow-debug JSON is only parsed on first activation).
REG = {"sessions": [], "active": None}
LOADED = {}


def _sid(video: Path) -> str:
    return hashlib.sha1(str(Path(video).resolve()).encode()).hexdigest()[:12]


def _find_flow_debug(video: Path):
    cand = video.with_name(video.stem + "-flow-debug.json")
    if cand.exists():
        return cand
    alt = video.parent / "flow_debug.json"
    return alt if alt.exists() else None


def _recordings_in(directory: Path):
    vids = set(directory.glob("recording-*.MOV")) | set(directory.glob("recording-*.mov"))
    return sorted(vids, key=lambda p: p.name.lower())


@lru_cache(maxsize=128)
def _flow_params_cached(video_str: str, mtime: float) -> tuple:
    """(model, inference-resolution) from a video's flow-debug JSON, for
    label disambiguation. Cache-keyed on the JSON's mtime."""
    fj = _find_flow_debug(Path(video_str))
    if not fj:
        return (None, None)
    try:
        p = json.loads(fj.read_text())
        frs = p.get("frames") or []
        f0 = frs[0] if frs else {}
        return (f0.get("model"), f0.get("resolution"))
    except (OSError, ValueError):
        return (None, None)


def _flow_params(video: Path) -> tuple:
    fj = _find_flow_debug(video)
    mt = fj.stat().st_mtime if fj else 0.0
    return _flow_params_cached(str(video), mt)


def derive_label(video: Path) -> str:
    """dir name; + ' #N' when the dir holds several recordings; + '(model,
    res)' when those recordings differ in inference params."""
    video = Path(video)
    dir_name = video.parent.name
    is_std = video.name.lower().startswith("recording-")
    if not is_std:
        stem = video.stem
        return stem if stem.lower() in dir_name.lower() else f"{dir_name} / {stem}"

    sibs = _recordings_in(video.parent)
    if len(sibs) <= 1:
        return dir_name

    try:
        idx = sibs.index(video) + 1
    except ValueError:
        idx = 1
    label = f"{dir_name} #{idx}"

    params = [_flow_params(s) for s in sibs]
    mine = _flow_params(video)
    extra = []
    if len({m for m, _ in params if m}) > 1 and mine[0]:
        extra.append(str(mine[0]))
    if len({str(r) for _, r in params if r}) > 1 and mine[1]:
        extra.append(str(mine[1]))
    if extra:
        label += " (" + ", ".join(extra) + ")"
    return label


def load_registry() -> dict:
    try:
        r = json.loads(SESSIONS_PATH.read_text())
        if isinstance(r, dict) and isinstance(r.get("sessions"), list):
            r.setdefault("active", None)
            return r
    except (OSError, ValueError):
        pass
    return {"sessions": [], "active": None}


def save_registry() -> None:
    SESSIONS_PATH.write_text(json.dumps(REG, indent=2))


def _session_by_id(sid):
    return next((s for s in REG["sessions"] if s["id"] == sid), None)


def _active():
    return _session_by_id(REG.get("active"))


def _relabel_all() -> None:
    for s in REG["sessions"]:
        try:
            s["label"] = derive_label(Path(s["video"]))
        except OSError:
            pass


def _register(video: Path):
    """Add a recording to the registry (idempotent). Returns (session, is_new)."""
    video = Path(video).resolve()
    sid = _sid(video)
    existing = _session_by_id(sid)
    if existing:
        return existing, False
    fj = _find_flow_debug(video)
    s = {
        "id": sid,
        "label": derive_label(video),
        "video": str(video),
        "flow_debug_json": str(fj) if fj else None,
    }
    REG["sessions"].append(s)
    return s, True


def _sessions_public():
    return [
        {"id": s["id"], "label": s["label"], "video": s["video"],
         "hasFlow": bool(s.get("flow_debug_json"))}
        for s in REG["sessions"]
    ]


def _live_track(video: Path) -> dict:
    """Fallback for a session with no flow-debug JSON: a geometry-only
    ByteTracker pass over detections found next to the video (or the shared
    logs dir). No flow vectors / distances in this mode."""
    try:
        dbg = video.parent if any(video.parent.glob("overlay-debug*.log")) else DEFAULT_LOGS_DIR
        det = video.parent if any(video.parent.glob("detections*.jsonl")) else DEFAULT_LOGS_DIR
        start_epoch, _ = resolve_start_epoch(video, dbg if Path(dbg).exists() else DEFAULT_LOGS_DIR)
        entries = load_detections(det)
        tracker = ByteTracker()
        frames = []
        for entry in entries:
            dets = entry["detections"]
            tids = tracker.update(dets)
            frames.append({
                "t": entry["t"] - start_epoch,
                "detections": [
                    {"trackID": t, "label": d["label"], "confidence": d["confidence"],
                     "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]}
                    for d, t in zip(dets, tids) if t is not None
                ],
            })
        print(f"  live-tracked {len(frames)} frames for {video.name} (no flow-debug JSON)", file=sys.stderr)
        return {"frames": frames, "videoResolution": None, "videoFps": None, "hasFlowData": False}
    except Exception as e:  # noqa: BLE001 -- fallback must never take the server down
        print(f"  live-track fallback failed for {video}: {e}", file=sys.stderr)
        return {"frames": [], "videoResolution": None, "videoFps": None, "hasFlowData": False}


def _ensure_loaded(sid):
    if sid in LOADED:
        return LOADED[sid]
    s = _session_by_id(sid)
    if not s:
        return None
    video = Path(s["video"])
    fj = Path(s["flow_debug_json"]) if s.get("flow_debug_json") else None
    if fj and fj.exists():
        payload = json.loads(fj.read_text())
        LOADED[sid] = {
            "frames": payload["frames"],
            "videoResolution": payload.get("videoResolution"),
            "videoFps": payload.get("videoFps"),
            "hasFlowData": True,
        }
    else:
        LOADED[sid] = _live_track(video)
    return LOADED[sid]


def _native_pick():
    """Native macOS chooser. One '+' click -> a Folder/File/Cancel prompt ->
    the matching picker. Returns {'kind','path'} | None (cancel) | {'error'}."""
    scr = r'''
    try
        set b to button returned of (display dialog "Add a session to the list" ¬
            buttons {"Cancel", "Folder…", "File…"} default button "File…" with title "super_tool")
        if b is "Folder…" then
            return "folder:" & POSIX path of (choose folder with prompt "Choose a session folder")
        else
            return "file:" & POSIX path of (choose file with prompt "Choose a recording (.MOV)" ¬
                of type {"mov", "public.movie", "com.apple.quicktime-movie"})
        end if
    on error number -128
        return ""
    end try
    '''
    try:
        r = subprocess.run(["osascript", "-e", scr], capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    out = (r.stdout or "").strip()
    if out.startswith("file:"):
        return {"kind": "file", "path": out[5:]}
    if out.startswith("folder:"):
        return {"kind": "folder", "path": out[7:]}
    return None


# ----------------------------------------------------------------------------
# Flask app -- every data route acts on the ACTIVE session
# ----------------------------------------------------------------------------
def build_app() -> Flask:
    app = Flask(__name__)

    def path_for_mode(mode: str) -> Path:
        if mode not in GROUND_TRUTH_MODES:
            raise ValueError(f"unknown labeling mode: {mode}")
        s = _active()
        if s is None:
            raise ValueError("no active session")
        video = Path(s["video"])
        if mode == "distance_calibration":
            return REPO_ROOT / "calibration" / f"distance_calibration_{video.stem}.json"
        if mode == "followed_vehicle":
            return video.parent / "ground_truth.json"
        return video.parent / f"ground_truth_{mode}.json"

    @app.route("/")
    def index():
        response = send_file(FRONTEND_PATH)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/video")
    def video_route():
        s = _active()
        if s is None:
            return ("no active session", 404)
        return send_file(s["video"], conditional=True)  # Range support

    @app.route("/api/frames")
    def frames_route():
        s = _active()
        if s is None:
            return jsonify({"frames": [], "videoResolution": None, "videoFps": None, "hasFlowData": False})
        return jsonify(_ensure_loaded(s["id"]))

    # --- session registry --------------------------------------------------
    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        resp = jsonify({"sessions": _sessions_public(), "active": REG.get("active")})
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/sessions/activate", methods=["POST"])
    def activate_session():
        sid = (request.get_json() or {}).get("id")
        if not _session_by_id(sid):
            return jsonify({"error": "unknown session"}), 404
        REG["active"] = sid
        save_registry()
        _ensure_loaded(sid)
        s = _session_by_id(sid)
        return jsonify({"ok": True, "active": sid, "label": s["label"]})

    @app.route("/api/sessions/pick_and_add", methods=["POST"])
    def pick_and_add():
        picked = _native_pick()
        if picked is None:
            return jsonify({"cancelled": True})
        if "error" in picked:
            return jsonify({"error": picked["error"]}), 500
        added = []
        if picked["kind"] == "file":
            p = Path(picked["path"])
            if p.suffix.lower() == ".mov" and p.exists():
                s, is_new = _register(p)
                if is_new:
                    added.append(s)
        else:  # folder
            d = Path(picked["path"])
            vids = _recordings_in(d) or sorted(
                set(d.glob("*.MOV")) | set(d.glob("*.mov")), key=lambda p: p.name.lower()
            )
            for v in vids:
                s, is_new = _register(v)
                if is_new:
                    added.append(s)
        if added:
            _relabel_all()
            save_registry()
        return jsonify({
            "added": [{"id": s["id"], "label": s["label"]} for s in added],
            "sessions": _sessions_public(),
        })

    @app.route("/api/sessions/remove", methods=["POST"])
    def remove_session():
        sid = (request.get_json() or {}).get("id")
        REG["sessions"] = [s for s in REG["sessions"] if s["id"] != sid]
        LOADED.pop(sid, None)
        if REG.get("active") == sid:
            REG["active"] = REG["sessions"][0]["id"] if REG["sessions"] else None
        _relabel_all()
        save_registry()
        return jsonify({"active": REG.get("active"), "sessions": _sessions_public()})

    # --- ground truth (active session's directory) -----------------------
    @app.route("/api/ground_truth/<mode>", methods=["GET"])
    def get_ground_truth(mode):
        try:
            path = path_for_mode(mode)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        if path.exists():
            return jsonify(json.loads(path.read_text()))
        return jsonify([])

    @app.route("/api/ground_truth/<mode>", methods=["POST"])
    def save_ground_truth(mode):
        try:
            path = path_for_mode(mode)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request.get_json(), indent=2))
        return jsonify({"ok": True})

    @app.route("/api/compute_distance", methods=["POST"])
    def compute_distance():
        """The EXACT row_based_distance_meters reconstruct_annotated.py / the
        on-device DistanceEstimator use -- imported, not re-implemented."""
        b = request.get_json()
        dist = row_based_distance_meters(
            bottom_y=b["row"], center_x=b["col"], aspect=b["aspect"],
            reference_pitch_deg=b["referencePitchDegrees"],
            reference_roll_deg=b["referenceRollDegrees"],
        )
        return jsonify({"distanceMeters": dist})

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        resp = jsonify(_read_settings())
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/api/settings", methods=["POST"])
    def save_settings():
        merged = _read_settings()
        merged.update(request.get_json() or {})
        SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", type=Path, default=None,
                        help="Optional: a recording to register + activate on startup (the sidebar manages the rest)")
    parser.add_argument("--flow-debug-json", type=Path, default=None,
                        help="Flow-debug JSON for the positional <video> (otherwise auto-found as <stem>-flow-debug.json)")
    parser.add_argument("--port", type=int, default=5050)
    # accepted for back-compat with old invocations; no longer used
    parser.add_argument("--detections", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--debug-log", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ground-truth-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ground-truth-name", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global REG
    REG = load_registry()

    if args.video is not None:
        if not args.video.exists():
            sys.exit(f"{args.video} doesn't exist.")
        s, _ = _register(args.video.resolve())
        if args.flow_debug_json is not None:
            if not args.flow_debug_json.exists():
                sys.exit(f"{args.flow_debug_json} doesn't exist.")
            s["flow_debug_json"] = str(args.flow_debug_json.resolve())
        REG["active"] = s["id"]

    if REG.get("active") is None and REG["sessions"]:
        REG["active"] = REG["sessions"][0]["id"]

    _relabel_all()
    save_registry()

    n = len(REG["sessions"])
    print(f"\nsuper_tool: {n} session{'' if n == 1 else 's'} in the list; "
          f"open http://localhost:{args.port} in a browser.\n")
    build_app().run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
