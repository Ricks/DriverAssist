#!/usr/bin/env python3
"""
Cuts + concatenates a list of {start_t, end_t} time intervals out of a video
into a single short highlight reel, via ffmpeg.

Re-encodes each segment rather than stream-copying -- stream-copy cuts snap
to the nearest keyframe, which can land noticeably off the intended cut
point; these cuts need to land on the actual labeled/algorithm disagreement
boundaries (see find_disagreement_intervals.py), not roughly near them.
`-ss` before `-i` still seeks (fast, not a full decode-from-start) and,
combined with re-encoding, modern ffmpeg decodes the small remaining gap to
land exactly on the requested timestamp -- both fast and accurate.

Usage:
    python3 cut_highlight_reel.py <input.mp4> <intervals.json> <output.mp4>
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("intervals", type=Path, help="JSON list of {start_t, end_t, ...}")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    intervals = json.loads(args.intervals.read_text())
    print(f"Cutting {len(intervals)} intervals from {args.input.name}...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        segment_paths = []
        for i, iv in enumerate(intervals):
            seg_path = tmp_path / f"seg_{i:04d}.mp4"
            duration = iv["end_t"] - iv["start_t"]
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(iv["start_t"]), "-i", str(args.input),
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-an",
                    str(seg_path),
                ],
                check=True,
            )
            segment_paths.append(seg_path)
            print(f"  [{i+1}/{len(intervals)}] {iv['start_t']:.1f}s-{iv['end_t']:.1f}s ({duration:.1f}s) "
                  f"{'/'.join(iv.get('kinds', []))}")

        concat_list = tmp_path / "concat_list.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in segment_paths))

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy",
                str(args.output),
            ],
            check=True,
        )

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
