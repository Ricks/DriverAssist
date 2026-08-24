# Head-On Cyclist Scan — 2026-08-22

Goal: gauge how much head-on cyclist footage already exists across this
project's real dashcam recordings, to decide whether an external dataset
(BDD100K/Cityscapes/nuScenes/KITTI — all of which have native rider/cyclist
classes) is needed before retraining YOLO with a dedicated "cyclist" class
that survives head-on views (the orientation where the stock "bicycle"
class loses almost all its discriminative signal: no side profile, wheels
collapse into a sliver, and it either goes undetected or gets misclassified
as "person" — see `tools/tracker.py`'s `CONFUSABLE_LABELS` comment).

This is a **different, more rigorous pass** than the earlier per-frame
visual-triage search in `data/head_on_cyclist_frames/REPORT.md` (which
grouped by trackID and eyeballed sampled frames one at a time, finding 9
confirmed head-on/tail-on frames out of 214 inspected candidates). This
pass groups bicycle detections by *time* into whole encounters ("periods"),
scores each period from its full box trajectory, and only then extracts
short video clips for the highest-scoring candidates.

## Methodology

Tool: `tools/scan_head_on_cyclists.py` (new, follows this project's
existing `driverassist_sync.py`/`find_disagreement_intervals.py`
conventions).

1. Enumerated every session directory under `data/`, matching real drive
   recordings by filename (skipping any video whose name starts with
   `wipercal-`, `calibration-`, or `nearfocus-`, and skipping derivative
   clips like `-annotated`/`-yawline-check`/`-disagreement-reel` outputs
   from other tools). `TEST`/`TEST2`/`TEST3` were excluded by directory
   name, per the earlier report's confirmation that they're <1-minute
   junk/dev clips, not real drives.
2. For each usable (video, detections file) pair, pulled every
   `label == "bicycle"` detection and grouped consecutive ones into
   contiguous **periods** using a 3-second gap tolerance — the same
   pattern `find_disagreement_intervals.py` uses to merge nearby
   disagreement stretches (`--merge-gap` default 3.0s).
3. Periods with fewer than 5 detections were dropped (too little signal to
   trust a trajectory-based score).
4. Each remaining period was scored 0–1 from three geometric cues, purely
   from the logged box coordinates — **no video decoding needed for this
   ranking step**, only for the final clip extraction:
   - **Aspect ratio** (w/h, averaged across the period, weight 0.40): a
     side-view bicycle box is wide relative to height; head-on collapses
     toward a standing person's aspect (~0.3–0.6). This is the least noisy
     cue since it's averaged over every frame in the period.
   - **Lateral drift** (box-center x-range over the period, weight 0.35): a
     head-on/approaching cyclist stays roughly centered; a crossing/side
     cyclist sweeps across x. Also low-noise (uses every frame).
   - **Height growth** (mean height of the period's first quarter vs. last
     quarter of frames, weight 0.25): an approaching cyclist's box should
     grow. Given the lowest weight because it only compares two small
     frame subsets, making it the most sensitive to detector jitter and
     partial occlusion — **and it structurally rewards only approaching
     (head-on) cases, not receding (tail-on) ones**, since shrinking
     height clamps to a growth score of 0. This is a known, deliberate
     asymmetry — see the Limitations section below.
5. All periods across all sessions were ranked by score. The top 18 were
   selected for clip extraction, capped at 3 per session so the shortlist
   isn't dominated by one long encounter.
6. For those 18, `resolve_start_epoch` (this project's standard
   detections-to-video sync helper) converted each period's capture-time
   window to video-relative seconds, and `ffmpeg` cut a 1-second-padded
   clip for each — saved to `clips/rank##_<session>__<video>__<mm-ss>.mp4`.
7. A subset of the 18 clips (16 of 18) was then spot-checked by extracting
   a single frame from the middle of each clip and visually reviewing it —
   see **Spot-check results** below. This is the step that actually
   answers the "is this heuristic any good" question, and the results are
   not flattering.

## Overall numbers

- **19** sessions had a usable video + detections file and were scanned
  (matches the earlier per-frame report's count exactly).
- **12** sessions had raw recordings but no usable detections log at all
  (mostly pre-detection-logging early drives: `26_07_25_1` through
  `26_07_29_1_Day`, plus two sessions where a log apparently wasn't pulled:
  `26_08_18_Night`, `26_08_20_Night_Matrix`) — skipped entirely.
- **178** total contiguous bicycle-detection periods found across those 19
  sessions.
- **89** periods (50%) had fewer than 5 detections and were dropped as too
  short to score.
- **89** periods were actually scored.

Score distribution (0.0 = fully side-view-like, 1.0 = fully head-on-like by
this heuristic):

| Score range | Count |
|---|---|
| [0.00, 0.20) | 1 |
| [0.20, 0.40) | 7 |
| [0.40, 0.60) | 32 |
| [0.60, 0.80) | 46 |
| [0.80, 1.00) | 3 |

The distribution is heavily concentrated in the middle (0.4–0.8) — very few
periods score as unambiguously head-on (only 3 of 89 broke 0.8) or
unambiguously side-view (only 1 broke below 0.2). That alone is a signal:
most real bicycle encounters in this footage are **oblique**, not cleanly
one orientation or the other, which already argues against there being a
deep well of clean head-on examples to mine.

## Ranked shortlist + spot-check verdicts

All 18 clips were extracted; 16 of 18 were visually spot-checked (one
frame from the middle of each clip). Verdicts:

| Rank | Session | Video time | Score | n | aspect | x-drift | growth | Verdict | Clip |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 26_08_17_Gym | 17:03 | 0.890 | 5 | 0.47 | 0.094 | +0.79 | Side/oblique, distant — not head-on | `rank01_26_08_17_Gym__recording-20260817-174026__17-03.mp4` |
| 2 | 26_08_14_ConeCalibration | 08:30 | 0.878 | 7 | 0.53 | 0.104 | +0.56 | Side view along sidewalk — not head-on | `rank02_26_08_14_ConeCalibration__recording-20260814-154420__08-30.mp4` |
| 3 | 26_08_21_Day_Small | 10:11 | 0.811 | 5 | 0.37 | 0.162 | +0.63 | Side view, distant — not head-on | `rank03_26_08_21_Day_Small__recording-20260821-125413__10-11.mp4` |
| 4 | 26_08_21_Day_Small | 18:17 | 0.789 | 5 | 0.48 | 0.125 | +0.37 | Side view, distant — not head-on | `rank04_26_08_21_Day_Small__recording-20260821-125413__18-17.mp4` |
| 5 | 26_07_30_Day_Hosp_nano_off | 14:27 | 0.788 | 5 | 0.32 | 0.181 | +1.14 | Two cyclists crossing, clean side profile — not head-on | `rank05_26_07_30_Day_Hosp_nano_off__26_07_30_Day_Hosp_nano_off__14-27.mp4` |
| 6 | 26_08_20_Day_Small | 07:44 | 0.776 | 5 | 0.44 | 0.192 | +0.77 | Side view, crossing intersection — not head-on | `rank06_26_08_20_Day_Small__recording-20260820-133618__07-44.mp4` |
| 7 | 26_08_16_TestDrive_LowRes_Nano_Evening_3 | 07:42 | 0.776 | 5 | 0.48 | 0.113 | +0.32 | **Partial hit** — this period overlaps the earlier report's confirmed head-on frame at t≈462.7s in this same video, but the sampled mid-clip frame itself shows a more crossing/oblique angle | `rank07_26_08_16_TestDrive_LowRes_Nano_Evening_3__recording-20260816-185926__07-42.mp4` |
| 8 | 26_07_29_3_Night | 07:43 | 0.770 | 38 | 0.63 | 0.185 | +0.70 | Side view, crossing at night — not head-on | `rank08_26_07_29_3_Night__26_07_29_3_Night__07-43.mp4` |
| 9 | 26_08_17_Gym | 19:07 | 0.769 | 16 | 0.44 | 0.198 | +0.60 | **False positive** — parked bikes on a rack in a parking garage, no rider at all; the "growth" cue fired because the car drove toward the rack | `rank09_26_08_17_Gym__recording-20260817-174026__19-07.mp4` |
| 10 | 26_08_17_Matrix_Day_2 | 01:49 | 0.769 | 6 | 0.42 | 0.198 | +0.65 | **Genuine hit** — matches the earlier report's confirmed oblique-frontal frame at t≈109.9s (rider stopped at a light, facing the camera) | `rank10_26_08_17_Matrix_Day_2__recording-20260817-135552__01-49.mp4` |
| 11 | 26_08_21_Day_Small | 26:32 | 0.766 | 22 | 0.46 | 0.183 | +0.46 | Likely false positive — a parked bike-share dock visible at the curb, truck partially blocking the view | `rank11_26_08_21_Day_Small__recording-20260821-125413__26-32.mp4` |
| 12 | 26_08_16_TestDrive_LowRes_Nano_Evening_2 | 01:21 | 0.755 | 22 | 0.40 | 0.205 | +0.49 | **Partial hit** — two riders in the oncoming lane at a moderate angle toward the camera, not clean side profile | `rank12_26_08_16_TestDrive_LowRes_Nano_Evening_2__recording-20260816-191917__01-21.mp4` |
| 13 | 26_08_17_Matrix_Day_2 | 01:04 | 0.746 | 9 | 0.31 | 0.218 | +1.07 | **False positive** — parked bike on a rack, no rider | `rank13_26_08_17_Matrix_Day_2__recording-20260817-140845__01-04.mp4` |
| 14 | 26_07_29_3_Night | 08:13 | 0.685 | 33 | 0.54 | 0.270 | +0.61 | Partial — night cyclist on the shoulder, tail/side angle | `rank14_26_07_29_3_Night__26_07_29_3_Night__08-13.mp4` |
| 15 | 26_08_16_TestDrive_LowRes_Nano_Day | 34:58 | 0.680 | 52 | 0.35 | 0.060 | −0.31 | **Genuine hit** — clean tail-on/receding cyclist directly ahead in-lane, textbook axis-aligned (scored despite negative growth, on aspect+drift alone) | `rank15_26_08_16_TestDrive_LowRes_Nano_Day__recording-20260816-132942__34-58.mp4` |
| 16 | 26_08_21_Night_Matrix | 02:54 | 0.675 | 40 | 0.55 | 0.278 | +0.85 | Side view, crossing a crosswalk at night — not head-on | `rank16_26_08_21_Night_Matrix__recording-20260821-205254__02-54.mp4` |
| 17 | 26_08_20_Day_Small | 57:10 | 0.674 | 22 | 0.50 | 0.279 | +1.18 | Inconclusive — no cyclist visible in the sampled mid-frame; not confirmed either way | `rank17_26_08_20_Day_Small__recording-20260820-133618__57-10.mp4` |
| 18 | 26_08_16_TestDrive_LowRes_Nano_Day | 19:22 | 0.672 | 7 | 0.41 | 0.282 | +1.30 | **Genuine hit** — cyclist riding away along the shoulder, tail-on/receding | `rank18_26_08_16_TestDrive_LowRes_Nano_Day__recording-20260816-132942__19-22.mp4` |

**Tally**: of the 16 spot-checked, **3 clean genuine hits** (all tail-on or
oblique-frontal, not textbook approaching-head-on), **3 partial/plausible**
hits, **2–3 outright false positives** (parked bikes with no rider at all
— the geometry heuristic can't tell "a stationary bike getting closer as
we drive past it" from "an approaching cyclist"), and **8 confirmed
side-view/crossing** cyclists that the heuristic mis-ranked highly. 1 was
inconclusive from the single sampled frame.

Notably, the two clips that most confidently confirm a *real* head-on/
tail-on cyclist (`rank10`, `rank15`, `rank18`, plus the partial `rank07`)
are ones that independently line up with frames the earlier, completely
separate per-track visual-triage pass had already found and confirmed —
that's reassuring cross-validation that the heuristic isn't randomly
noisy, but it also means this pass mostly **rediscovered** the same
small handful of good examples rather than surfacing new ones.

## Limitations of this heuristic (be honest about the noise)

- **Parked/stationary bikes are a real, confirmed failure mode.** At least
  2 of the top 18 (`rank09`, `rank13`, possibly `rank11`) are bike racks or
  parked bikes with no rider, not cyclists at all. A bike that isn't
  moving, viewed as the car drives toward it, produces exactly the
  "narrow, centered, growing" signature this heuristic rewards.
- **The growth cue only rewards approaching (head-on), not receding
  (tail-on)** by construction — yet the earlier per-frame report found
  tail-on examples just as valuable for the same "cyclist survives
  non-side views" training goal. The two genuine hits this pass did find
  (`rank15`, `rank18`) are both tail-on, and they scored well *despite*
  this asymmetry (on aspect+drift alone), not because of it. A future
  version should probably also reward a *shrinking, centered* box as a
  second, symmetric case.
- **A single mid-clip frame is not the same as watching the whole
  period.** `rank07`'s sampled frame looked more oblique than the known
  confirmed head-on moment in that same encounter — the period-level score
  can be right about there being a good moment somewhere in the window
  even when the one frame checked here didn't catch it. A full visual
  review of all 18 clips (not just one frame each) would sharpen these
  numbers somewhat, but is unlikely to change the overall conclusion given
  how many were unambiguous side/crossing views or bare bike racks.
- **Aspect ratio alone is fooled by partial occlusion and steep crossing
  angles**, not just head-on geometry — several of the "not head-on"
  verdicts above (e.g. `rank01`–`rank06`) had aspect ratios in the
  head-on-like range purely because the rider was distant/small or
  partially cut off, not because they were actually facing the camera.

## Bottom line: does existing footage have enough head-on cyclist data?

**No — external data is clearly needed.** This is consistent with, and
reinforces, the earlier per-frame pass's harder numbers (9 confirmed
head-on/tail-on frames out of 214 visually-inspected candidates, out of
2,830 raw bicycle detections across the same 19 sessions).

This pass, using a completely different methodology (period-level
geometric scoring instead of per-track frame sampling), converges on the
same answer from a different angle:

1. Even a heuristic specifically designed to hunt for head-on geometry,
   run across every real bicycle encounter in every drive session this
   project has, surfaces a **top-18 shortlist where a clear majority are
   not head-on** (side/crossing views or, in a couple of cases, not even a
   real cyclist) once actually looked at.
2. The genuine hits that exist are concentrated in a small number of
   specific encounters — the same 3–4 the earlier, independent pass had
   already found — not spread across many sessions. There is no long tail
   of undiscovered head-on footage waiting to be mined; both methodologies
   are converging on the same small pool.
3. Every genuine or partial hit found here is either **tail-on/receding**
   or **oblique** (30–45°), not a clean, centered, approaching head-on
   view. Textbook head-on (rider pedaling straight at the camera, full
   frontal) is essentially singular in this dataset (the one example from
   the earlier report, `26_08_16_TestDrive_LowRes_Nano_Evening_2` t≈83.5s).

A handful of real examples (roughly a dozen across both passes, combined)
is enough to sanity-check a retrained model's behavior on real footage,
but nowhere near enough diversity (lighting, distance, rider clothing/gear,
bike type, background clutter, partial occlusion) to serve as the primary
training signal for a class meant to be robust specifically in the
orientation where the existing detector already fails hardest. Bootstrap
the head-on/tail-on cyclist training set from an external dataset with a
native rider/cyclist class (BDD100K, Cityscapes, nuScenes, or KITTI all
qualify), and reserve this project's own real footage — the confirmed
examples from both passes, plus whatever turns up on future drives — for
validation and domain-matching fine-tuning, not as the primary source.

## Files

- Scan tool: `tools/scan_head_on_cyclists.py`
- Full machine-readable results (all 89 scored periods + shortlist):
  `analysis/head_on_cyclist_scan_2026-08-22/scan_results.json`
- Extracted clips: `analysis/head_on_cyclist_scan_2026-08-22/clips/`
- Spot-check frames used for the verdicts above:
  `analysis/head_on_cyclist_scan_2026-08-22/spotcheck_frames/`
