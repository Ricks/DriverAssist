#!/usr/bin/env python3
"""
Reconstructs an annotated copy of a clean DriverAssist recording by drawing
detection boxes + HUD text (from detections.jsonl, and optionally thermal
state/percent from the app's debug log) onto each frame. The source .mov is
never modified — this always writes a new file.

Usage:
    python3 reconstruct_annotated.py <clean.mov>

--detections and --debug-log both default to ~/DriverAssist/logs/ (the
directory tools/pull_logs.sh keeps filled with everything pulled off the
device), and both accept either a single file or a directory of them — so
the common case is just the video path, and the right data is found
automatically regardless of what the video's been renamed to. Override
either with an explicit file/directory, and --output, if needed:
    python3 reconstruct_annotated.py <clean.mov> \
        [--detections detections.jsonl] [--debug-log overlay-debug.log] \
        [--output annotated.mp4]

See driverassist_sync.py for how frame-to-timestamp sync and log matching
actually work.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

from driverassist_sync import (
    DEFAULT_LOGS_DIR,
    MODEL_DISPLAY_NAMES,
    build_key_index,
    load_detections,
    load_thermal,
    nearest_at_or_before,
    resolve_start_epoch,
)
from leading_vehicle import (
    DEFAULT_BAND_HALF_WIDTH,
    DEFAULT_CENTER_X,
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_GRACE_FRAMES,
    DEFAULT_THRESHOLD,
    LeadingVehicleLock,
    VelocityEstimator,
    classify_leading,
)
from tracker import DEFAULT_REID_MODEL, ByteTracker, build_reid_encoder

# BGR (OpenCV convention) approximations of the on-device box colors — these
# don't need to be pixel-identical to iOS's system colors, just visually
# distinct per class, matching OverlayStyle.color(for:) in the app.
BOX_COLORS_BGR = {
    "person": (0, 255, 255),      # yellow
    "bicycle": (255, 255, 0),     # cyan
    "motorcycle": (255, 255, 0),  # cyan
    "car": (0, 255, 0),           # green
    "bus": (0, 0, 255),           # red
    "truck": (0, 0, 255),         # red
}
DEFAULT_BOX_COLOR_BGR = (255, 255, 255)
HUD_DEFAULT_COLOR_BGR = (191, 191, 191)  # ~white @ 75% opacity, matches on-device HUD
HUD_YELLOW_BGR = (0, 215, 255)
HUD_RED_BGR = (0, 0, 255)

# Magenta -- doesn't collide with any BOX_COLORS_BGR value, so the tint reads
# unambiguously as "this is the followed vehicle" regardless of its class.
LEADING_TINT_BGR = (255, 0, 255)
LEADING_TINT_ALPHA = 0.35

# Top-row text baseline — nudged down from a bare margin so QuickTime
# Player's title bar/menu chrome doesn't sit directly on top of it when
# reviewing a reconstructed clip.
TOP_ROW_Y = 70


def draw_box(frame, det: dict, track_id=None) -> None:
    h, w = frame.shape[:2]
    x, y = int(det["x"] * w), int(det["y"] * h)
    bw, bh = int(det["w"] * w), int(det["h"] * h)
    color = BOX_COLORS_BGR.get(det["label"], DEFAULT_BOX_COLOR_BGR)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    id_prefix = f"#{track_id} " if track_id is not None else ""
    label = f"{id_prefix}{det['label']} {int(det['confidence'] * 100)}%"
    cv2.putText(frame, label, (x + 4, y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)


def draw_leading_tint(frame, det: dict) -> None:
    """Fills the followed vehicle's box with a translucent tint, on top of
    its already-drawn class-colored box/label -- see leading_vehicle.py for
    the classifier that picks this detection. Alpha-blended (not a solid
    fill) so the vehicle itself stays visible underneath, and drawn as a
    thick outline too so the highlight still reads even on a very thin/
    distant box where the fill area is tiny."""
    h, w = frame.shape[:2]
    x, y = int(det["x"] * w), int(det["y"] * h)
    bw, bh = int(det["w"] * w), int(det["h"] * h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), LEADING_TINT_BGR, -1)
    cv2.addWeighted(overlay, LEADING_TINT_ALPHA, frame, 1 - LEADING_TINT_ALPHA, 0, dst=frame)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), LEADING_TINT_BGR, 3)

    label = "FOLLOWING"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
    label_y = y - 8 if y - 8 - th > 0 else y + bh + th + 8
    cv2.putText(frame, label, (x + 4, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, LEADING_TINT_BGR, 2, cv2.LINE_AA)


def draw_hud(frame, entry: dict, thermal: Optional[tuple]) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thickness = 1.0, 2

    def put(text: str, x: int, y: int, color) -> None:
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def right_aligned(text: str) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        return w - tw - 12

    if thermal is not None:
        _, state, percent = thermal
        state_text = "OK" if state == "nominal" else state.upper()
        color = HUD_DEFAULT_COLOR_BGR
        if state == "fair":
            color = HUD_YELLOW_BGR
        elif state in ("serious", "critical"):
            color = HUD_RED_BGR  # blinking on-device isn't reproduced here
        put(f"thermal: {state_text} {percent}%", 12, TOP_ROW_Y, color)

    put(f"two-pass: {'on' if entry['twoPass'] else 'off'}", 12, h - 50, HUD_DEFAULT_COLOR_BGR)
    model_text = MODEL_DISPLAY_NAMES.get(entry["model"], entry["model"])
    put(model_text, 12, h - 12, HUD_DEFAULT_COLOR_BGR)

    low_state = "on" if entry["lowLightEnabled"] else "off"
    low_text = (
        f"low-light: auto ({low_state})" if entry["autoLowLightEnabled"] else f"low-light: {low_state}"
    )
    put(low_text, right_aligned(low_text), h - 12, HUD_DEFAULT_COLOR_BGR)

    stab_text = f"stabilization: {'on' if entry['stabilizationEnabled'] else 'off'}"
    put(stab_text, right_aligned(stab_text), TOP_ROW_Y, HUD_DEFAULT_COLOR_BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Recording to annotate, named however you like")
    parser.add_argument(
        "--detections", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"detections.jsonl, or a directory of them (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument(
        "--debug-log", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"overlay-debug.log, or a directory of them, for sync + the thermal HUD line (default: {DEFAULT_LOGS_DIR})",
    )
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <video>-annotated.mp4")
    parser.add_argument(
        "--reid", dest="reid", action="store_true", default=True,
        help="Use appearance/ReID matching in the tracker, in addition to motion/IoU (default: on)",
    )
    parser.add_argument("--no-reid", dest="reid", action="store_false", help="Geometry-only tracking")
    parser.add_argument("--reid-model", default=DEFAULT_REID_MODEL)
    parser.add_argument("--reid-device", default="mps")
    parser.add_argument(
        "--gmc", dest="gmc", action="store_true", default=False,
        help="Compensate track predictions for estimated camera motion (see gmc.py). Off by default -- experimental.",
    )
    parser.add_argument("--no-gmc", dest="gmc", action="store_false")
    parser.add_argument(
        "--highlight-leading", dest="highlight_leading", action="store_true", default=True,
        help="Tint the classified forward-leading vehicle (see leading_vehicle.py) (default: on)",
    )
    parser.add_argument("--no-highlight-leading", dest="highlight_leading", action="store_false")
    parser.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    parser.add_argument("--band-half-width", type=float, default=DEFAULT_BAND_HALF_WIDTH)
    parser.add_argument("--leading-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--leading-confirm-frames", type=int, default=DEFAULT_CONFIRM_FRAMES)
    parser.add_argument("--leading-grace-frames", type=int, default=DEFAULT_GRACE_FRAMES)
    args = parser.parse_args()

    output = args.output or args.video.with_name(args.video.stem + "-annotated.mp4")
    if output.resolve() == args.video.resolve():
        sys.exit("Refusing to overwrite the source recording — pass a different --output.")

    start_epoch, thermal_log_path = resolve_start_epoch(args.video, args.debug_log)
    detections = load_detections(args.detections)

    reid_encoder = build_reid_encoder(args.reid_model, device=args.reid_device) if args.reid else None
    tracker = ByteTracker(reid_encoder=reid_encoder, use_gmc=args.gmc)
    leading_lock = LeadingVehicleLock(args.leading_confirm_frames, args.leading_grace_frames)
    leading_velocity = VelocityEstimator()

    thermal = load_thermal(thermal_log_path)
    thermal_keys = build_key_index(thermal, lambda e: e[0])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Couldn't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"Couldn't open {output} for writing")

    # Tracking runs inline with this same sequential decode, not as a separate
    # seek-based pre-pass -- a pre-pass that re-seeks the video once per
    # detection entry (rather than once per raw frame here) turned out to
    # cost 30-40x more per lookup than a plain sequential read (measured: a
    # 5-6x slowdown overall on a real session), because OpenCV/ffmpeg must
    # decode forward from the nearest keyframe on every arbitrary seek. Each
    # entry is instead fed to the tracker using whichever already-decoded
    # frame is current when that entry's own timestamp is first reached --
    # entries land close enough to this video's own frame rate that this is
    # a fine approximation of the frame the entry actually corresponds to.
    next_idx = 0
    current_entry, current_track_ids = None, []
    # Persists across video frames the same way current_entry does -- the
    # lock is only updated when a *new* detection entry is processed below,
    # at the detections.jsonl cadence, not once per rendered video frame.
    current_locked_id = None

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Actual embedded PTS for the frame just read (ms, relative to the
        # start of the recording) — not `i / fps`, which would assume a
        # perfectly constant frame rate and can drift over a long drive if
        # any frames were ever dropped or timing wasn't perfectly steady.
        frame_epoch = start_epoch + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        while next_idx < len(detections) and detections[next_idx]["t"] <= frame_epoch:
            current_entry = detections[next_idx]
            current_track_ids = tracker.update(current_entry["detections"], frame=frame)
            if args.highlight_leading:
                # classify_leading reads trackID off each det -- draw_box below
                # only takes it as a separate zip'd arg, so stamp it on here too.
                for det, track_id in zip(current_entry["detections"], current_track_ids):
                    det["trackID"] = track_id
                leading_velocity.update(current_entry["detections"], current_entry["t"])
                raw_leading = classify_leading(
                    current_entry["detections"], args.center_x, args.band_half_width, args.leading_threshold,
                )
                current_locked_id = leading_lock.update(
                    raw_leading["trackID"] if raw_leading is not None else None
                )
            next_idx += 1

        if current_entry is not None:
            for det, track_id in zip(current_entry["detections"], current_track_ids):
                draw_box(frame, det, track_id)
            if args.highlight_leading and current_locked_id is not None:
                # The locked vehicle may not be this exact entry's raw
                # classify_leading winner (grace-period stickiness) -- look
                # it up by trackID among this entry's own detections instead.
                # If it's not there at all (mid-gap, nothing detected for it
                # this entry), skip drawing rather than fabricate a box.
                locked_det = next(
                    (d for d in current_entry["detections"] if d.get("trackID") == current_locked_id), None
                )
                if locked_det is not None:
                    draw_leading_tint(frame, locked_det)
            thermal_entry = nearest_at_or_before(thermal, thermal_keys, frame_epoch)
            draw_hud(frame, current_entry, thermal_entry)

        writer.write(frame)
        i += 1
        if i % max(1, int(fps) * 30) == 0:
            # Not shown as a fraction of frame_count: OpenCV's frame-count
            # estimate for a live-recorded/fragmented .mov is unreliable (seen
            # under-reporting by 3x on a real file), so this is just an
            # elapsed-progress heartbeat, not a percentage.
            print(f"  {i} frames processed ({i / fps:.0f}s of video)", file=sys.stderr)

    cap.release()
    writer.release()
    print(f"Wrote {i} frames to {output}")
    print(f"Original recording untouched: {args.video}")


if __name__ == "__main__":
    main()
