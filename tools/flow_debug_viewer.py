#!/usr/bin/env python3
"""
Interactive browser-based viewer for the five per-track flow-arrow vectors
reconstruct_annotated.py's --flow-arrows computes (predicted flow, previous
flow, raw observed rate, raw motion, and the tracker's smoothed motion) --
same video-serving/frame-stepping mechanics as label_leading_vehicle.py
(HTTP Range support so scrubbing a multi-GB file works, a <canvas> overlay
synced to video time via binary search, frame-by-frame stepping), but
read-only (no labeling) and reading a precomputed --flow-debug-json file
instead of computing tracks live -- see that flag's own doc comment in
reconstruct_annotated.py for why: getting good track IDs needs the same
full ReID pass over the whole video --flow-arrows itself needs, which is
too slow to redo on every server start the way label_leading_vehicle.py's
much cheaper geometry-only live tracking can.

Default view draws only each track's smoothed (red) motion arrow, at full
opacity, kept minimal so the scene reads clearly during normal playback --
click a box to select it and switch into "full overlay" mode, drawing all
five vectors at full opacity for that track alone (every other track in
frame keeps showing just its red arrow, for scene context). Click the
selected box again, click empty space, or press Escape to return to the
red-only default. Intended to be paired with frame-by-frame stepping
(already built into the shared frontend chassis) once a track is selected,
not with playback.

Usage:
    python3 flow_debug_viewer.py <video> <flow-debug.json> [--port 5051]
"""
import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, send_file

FRONTEND_PATH = Path(__file__).parent / "flow_debug_viewer_frontend.html"


def build_app(video: Path, flow_debug_json: Path) -> Flask:
    payload = json.loads(flow_debug_json.read_text())
    frames = payload["frames"]
    video_resolution = payload.get("videoResolution")
    video_fps = payload.get("videoFps")

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
        return jsonify({"frames": frames, "videoResolution": video_resolution, "videoFps": video_fps})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="The clean recording flow_debug_json was computed from")
    parser.add_argument("flow_debug_json", type=Path, help="Output of reconstruct_annotated.py --flow-debug-json")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"{args.video} doesn't exist.")
    if not args.flow_debug_json.exists():
        sys.exit(f"{args.flow_debug_json} doesn't exist -- generate it with reconstruct_annotated.py --flow-debug-json first.")

    app = build_app(args.video, args.flow_debug_json)
    print(f"\nOpen http://localhost:{args.port} in a browser.\n")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
