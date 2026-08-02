#!/usr/bin/env python3
"""
Interactive browser-based ground-truth labeling tool for "which vehicle am I
following" -- lets a human step through a real session's recorded video
(any speed, frame-by-frame, forward or backward), click a detected vehicle
to mark it as the followed one, and click it again to end/delete that
label. Ground truth is saved as a list of {start_t, end_t, trackID} segments
(epoch timestamps, matching detections.jsonl's own "t" field) to
<session_dir>/ground_truth.json.

This exists to move heuristic tuning (leading_vehicle.py's gates: min
confidence/width, max bottom-y, max velocity, band width, hysteresis
confirm/grace frames) from one-at-a-time spot-checks against proxy metrics
(switch counts, flip-flops) to a real accuracy metric against labeled data,
and to eventually support a parameter sweep that optimizes jointly rather
than one threshold at a time.

Scope/limitation worth knowing: this can only label among boxes the
detector actually produced. If the real leading vehicle goes fully
undetected for a stretch, there's nothing to click, and that stretch can
only be labeled "no candidate" even though a human watching the raw footage
could see the car. This ground truth validates classify_leading's
*selection* logic, not the underlying detector's recall.

TrackIDs are recomputed offline via tracker.py's ByteTracker (geometry-only,
matching leading_vehicle.py/track_consistency.py's approach), not read from
on-device logging -- so this works on any already-recorded session,
including ones from before on-device trackID logging existed.

Usage:
    python3 label_leading_vehicle.py <session_dir> [--port 5050]

Then open http://localhost:5050 in a browser. The video is served directly
from disk with HTTP Range support (via Flask/Werkzeug) so seeking in the
<video> element actually works on a multi-GB file -- detection boxes are
drawn as a live canvas overlay synced to video time, not baked into a
pre-rendered file, so no re-render is needed when anything about the
detection data changes.
"""
import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch
from package_session import find_session_video
from tracker import ByteTracker

FRONTEND_PATH = Path(__file__).parent / "label_leading_vehicle_frontend.html"


def build_app(session_dir: Path, video: Path, detections_path: Path, debug_log: Path) -> Flask:
    start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
    entries = load_detections(detections_path)

    print(f"Recomputing trackIDs offline over {len(entries)} entries...", file=sys.stderr)
    tracker = ByteTracker()  # geometry-only -- see module docstring
    frames = []
    for entry in entries:
        dets = entry["detections"]
        track_ids = tracker.update(dets)
        frames.append({
            "t": entry["t"] - start_epoch,  # video-relative seconds, matches <video>.currentTime
            "detections": [
                {
                    "trackID": tid,
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"],
                }
                for d, tid in zip(dets, track_ids)
                if tid is not None
            ],
        })
    print("Done.", file=sys.stderr)

    ground_truth_path = session_dir / "ground_truth.json"

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

    @app.route("/api/detections")
    def detections_route():
        return jsonify({"frames": frames})

    @app.route("/api/ground_truth", methods=["GET"])
    def get_ground_truth():
        if ground_truth_path.exists():
            return jsonify(json.loads(ground_truth_path.read_text()))
        return jsonify([])

    @app.route("/api/ground_truth", methods=["POST"])
    def save_ground_truth():
        ground_truth_path.write_text(json.dumps(request.get_json(), indent=2))
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Packaged session directory (see package_session.py)")
    parser.add_argument("--video", type=Path, default=None, help="Defaults to the one raw recording in <session_dir>")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to <session_dir>/detections.jsonl")
    parser.add_argument("--debug-log", type=Path, default=None, help="Defaults to <session_dir>/overlay-debug.log")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist.")

    video = args.video or find_session_video(args.session_dir)
    debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")

    app = build_app(args.session_dir, video, detections_path, debug_log)
    print(f"\nOpen http://localhost:{args.port} in a browser.\n")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
