#!/usr/bin/env python3
"""
Interactive browser-based tool for marking real-world reference points
(traffic cone bases, tape marks, etc.) on a still frame at known tape-
measured distances, for fitting/checking DistanceEstimator's row-based
ground-plane calibration (principalRowNormalized/focalLengthNormalized --
see DistanceEstimator.swift's own doc comment on `calibrated` for the
fit this tool exists to support/reproduce).

REBUILT 2026-08-23 -- the original tools/cone_pinpoint_tool.html used for
the 2026-08-15 cone-calibration fit (8 points: 4 near cones at 8/10/12/14m,
4 far cones at 14/16/18/20m, data/26_08_14_ConeCalibration/) was never
actually inside this repo -- git history shows it was never committed,
because (per Rick, who always runs `git add .` from the repo root before
committing) it must have been written and run from OUTSIDE the tracked
tree entirely, so `git add .` never had a chance to see it. Its marked
points were never saved either -- they lived only in browser memory,
hand-copied out once to run a fit script elsewhere. This rebuild fixes
BOTH gaps: it lives in tools/ (so `git add .` catches it like any other
checked-in tool), and every marked point is saved server-side to a real
JSON file the moment "Save" is clicked -- not just held in the page.

Also computes each marked point's CURRENT row-based distance live (same
row_based_distance_meters reconstruct_annotated.py itself uses) so you see
the calibration residual immediately as you mark points, instead of only
finding out after a separate offline fit run.

Usage:
    python3 cone_pinpoint_tool.py <image1> [<image2> ...] [--output points.json] [--port 5052]

Defaults --output to <first image's directory>/cone_calibration_points.json
and pre-loads any points already saved there (so re-running the tool to add
more points, or fix a mismarked one, doesn't lose earlier work).
"""
import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, str(Path(__file__).parent))
from reconstruct_annotated import row_based_distance_meters  # noqa: E402

FRONTEND_PATH = Path(__file__).parent / "cone_pinpoint_tool_frontend.html"


def build_app(images: list, output_path: Path) -> Flask:
    image_by_name = {img.name: img for img in images}
    points = json.loads(output_path.read_text())["points"] if output_path.exists() else []

    app = Flask(__name__)

    @app.route("/")
    def index():
        response = send_file(FRONTEND_PATH)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/images")
    def images_route():
        return jsonify({"images": list(image_by_name.keys())})

    @app.route("/image/<name>")
    def image_route(name):
        if name not in image_by_name:
            return jsonify({"error": "unknown image"}), 404
        return send_file(image_by_name[name])

    @app.route("/api/points", methods=["GET"])
    def get_points():
        return jsonify({"points": points})

    @app.route("/api/points", methods=["POST"])
    def save_points():
        nonlocal points
        points = request.get_json()["points"]
        output_path.write_text(json.dumps({"points": points}, indent=2))
        return jsonify({"ok": True, "path": str(output_path), "count": len(points)})

    @app.route("/api/compute_distance", methods=["POST"])
    def compute_distance():
        """Runs the EXACT same row_based_distance_meters formula
        reconstruct_annotated.py/the on-device DistanceEstimator use --
        single source of truth, so this tool's live residual display can't
        drift from what the real pipeline would compute for the same
        marked point."""
        body = request.get_json()
        dist = row_based_distance_meters(
            bottom_y=body["row"], center_x=body["col"], aspect=body["aspect"],
            reference_pitch_deg=body["referencePitchDegrees"],
            reference_roll_deg=body["referenceRollDegrees"],
        )
        return jsonify({"distanceMeters": dist})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+", help="Still frame(s) to mark points on")
    parser.add_argument("--output", type=Path, default=None, help="Points JSON path (default: <first image's dir>/cone_calibration_points.json)")
    parser.add_argument("--port", type=int, default=5052)
    args = parser.parse_args()

    for img in args.images:
        if not img.exists():
            sys.exit(f"{img} doesn't exist.")
    output_path = args.output or (args.images[0].parent / "cone_calibration_points.json")

    app = build_app(args.images, output_path)
    print(f"\nOpen http://localhost:{args.port} in a browser.")
    print(f"Points save to: {output_path}\n")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
