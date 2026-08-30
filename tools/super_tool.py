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
menu. (An earlier "off by (m)" error-magnitude field was dropped
2026-08-29 in favour of recording the actual distance directly.)

Merged because both halves already shared the exact same chassis (video
serving with HTTP Range support so scrubbing a multi-GB file works, a
canvas overlay synced to video time via binary search, frame-by-frame
stepping, simulated reverse playback) -- maintaining that chassis twice
was pure duplication.

One real behavior change falls out of the merge, not just where the code
lives: ground-truth labeling now prefers ON-DEVICE track IDs (from a
--flow-debug-json, when passed and it exists) over label_leading_
vehicle.py's own live geometry-only ByteTracker pass, for the same reason
reconstruct_annotated.py's Part-1 change did -- consistency with what the
on-device app itself would show. The live geometry-only tracker remains
as the fallback when no --flow-debug-json is given (the common case for a
quick ad-hoc labeling pass over a session with no precompute step ever
run), and for recordings from before on-device trackID logging existed.

CAUTION: ground-truth labels recorded under the OLD (pre-merge,
label_leading_vehicle.py-only) live-tracker IDs for a session that NOW has
a --flow-debug-json will have stale trackID values -- the two tracking
passes assign IDs completely independently and have no reason to agree.
Re-label from scratch for any session where this matters, rather than
trusting an old label file's trackID fields once a flow-debug.json exists
for that session.

Usage:
    python3 super_tool.py <video> [--flow-debug-json path/to/flow-debug.json] [--port 5050]

Then open http://localhost:5050 in a browser. --detections/--debug-log
(only used for the live-tracking fallback, i.e. when --flow-debug-json is
omitted) default to ~/DriverAssist/logs/ and accept a single file or a
directory, same auto-matching-by-creation-time as reconstruct_annotated.py
-- point them at the session's own directory to avoid needlessly loading
every other session ever pulled to that shared directory.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from tracker import ByteTracker

FRONTEND_PATH = Path(__file__).parent / "super_tool_frontend.html"

# One ground-truth file per labeling mode -- followed_vehicle keeps its
# original filename/CLI override (ground_truth_name) since other tools
# (tune_leading_vehicle.py, analyze_leading_vehicle_errors.py) already read
# <ground_truth_dir>/ground_truth.json directly from disk; a renamed
# default here would silently break them. The rest are newer, so their
# filenames are just "ground_truth_<mode>.json". flow_debug is READ-ONLY
# (a viewing mode, not a labeling one) and deliberately excluded here.
GROUND_TRUTH_MODES = {
    "followed_vehicle", "false_positive", "hood_truncation", "other_truncation",
    "distance_anomaly", "cyclist", "motorcyclist", "skateboarder", "equestrian",
}


def build_app(video: Path, detections_path: Path, debug_log: Path, flow_debug_json,
              ground_truth_dir: Path, ground_truth_name: str) -> Flask:
    if flow_debug_json is not None and flow_debug_json.exists():
        payload = json.loads(flow_debug_json.read_text())
        frames = payload["frames"]
        video_resolution = payload.get("videoResolution")
        video_fps = payload.get("videoFps")
        has_flow_data = True
    else:
        start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
        entries = load_detections(detections_path)

        # Same rationale/threshold as reconstruct_annotated.py's own
        # --detections-defaulting-to-DEFAULT_LOGS_DIR warning: this loop
        # burns CPU re-tracking every entry loaded, including any from
        # unrelated earlier sessions if --detections wasn't scoped to this
        # video's own directory.
        n_before = bisect.bisect_left([e["t"] for e in entries], start_epoch)
        if n_before > 2000:
            print(
                f"WARNING: {n_before} of {len(entries)} loaded detections entries are from BEFORE "
                f"this video even starts -- {detections_path} likely wasn't scoped to this session. "
                "Pass --detections/--debug-log pointed at this video's own session directory to avoid "
                "burning through them needlessly.",
                file=sys.stderr,
            )

        print(f"No --flow-debug-json given -- recomputing trackIDs live (geometry-only) over {len(entries)} entries...", file=sys.stderr)
        tracker = ByteTracker()  # geometry-only, no ReID/GMC -- cheap enough to redo on every server start
        frames = []
        for entry in entries:
            dets = entry["detections"]
            track_ids = tracker.update(dets)
            frames.append({
                "t": entry["t"] - start_epoch,  # video-relative seconds, matches <video>.currentTime
                "detections": [
                    {
                        "trackID": tid, "label": d["label"], "confidence": d["confidence"],
                        "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                    }
                    for d, tid in zip(dets, track_ids) if tid is not None
                ],
            })
        print("Done.", file=sys.stderr)
        video_resolution = None
        video_fps = None
        has_flow_data = False

    def path_for_mode(mode: str) -> Path:
        if mode not in GROUND_TRUTH_MODES:
            raise ValueError(f"unknown labeling mode: {mode}")
        if mode == "followed_vehicle":
            return ground_truth_dir / ground_truth_name
        return ground_truth_dir / f"ground_truth_{mode}.json"

    app = Flask(__name__)

    @app.route("/")
    def index():
        # No caching -- this file gets iterated on; a stale cached copy
        # silently running old logic is worse than re-reading a small file
        # on every load.
        response = send_file(FRONTEND_PATH)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/video")
    def video_route():
        return send_file(video, conditional=True)  # conditional=True enables Range support

    @app.route("/api/frames")
    def frames_route():
        return jsonify({
            "frames": frames,
            "videoResolution": video_resolution,
            "videoFps": video_fps,
            "hasFlowData": has_flow_data,
        })

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
        path.write_text(json.dumps(request.get_json(), indent=2))
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to view/label, named however you like")
    parser.add_argument(
        "--detections", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"detections.jsonl, or a directory of them -- only used as the live-tracking fallback when "
             f"--flow-debug-json is omitted (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--debug-log", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"overlay-debug.log, or a directory of them, for sync -- same fallback-only scope as "
             f"--detections (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--flow-debug-json", type=Path, default=None,
        help="Output of reconstruct_annotated.py --flow-debug-json -- on-device track IDs, corrected "
             "distances, and flow vectors. Without this, falls back to a live geometry-only re-tracking "
             "pass over --detections (no flow vectors/HUD/distance data available in that mode; the Flow "
             "Debug view just shows plain boxes).",
    )
    parser.add_argument(
        "--ground-truth-dir", type=Path, default=None,
        help="Where to read/write ground_truth*.json -- defaults to the video's own directory.",
    )
    parser.add_argument(
        "--ground-truth-name", default="ground_truth.json",
        help="Filename (within --ground-truth-dir) for the followed_vehicle mode -- use a different name "
             "to run a separate labeling pass (e.g. a pared-down subset) without touching the original.",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"{args.video} doesn't exist.")
    if args.flow_debug_json is not None and not args.flow_debug_json.exists():
        sys.exit(
            f"{args.flow_debug_json} doesn't exist -- generate it with reconstruct_annotated.py "
            "--flow-debug-json, or omit --flow-debug-json to fall back to live geometry-only tracking."
        )

    ground_truth_dir = args.ground_truth_dir or args.video.parent

    app = build_app(
        args.video, args.detections, args.debug_log, args.flow_debug_json,
        ground_truth_dir, args.ground_truth_name,
    )
    print(f"\nOpen http://localhost:{args.port} in a browser.\n")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
