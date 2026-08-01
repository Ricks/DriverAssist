#!/usr/bin/env python3
"""
Multi-object tracker (ByteTrack-style, with optional appearance/ReID
matching) that turns per-frame detection boxes into persistent track IDs, so
the same vehicle/pedestrian/cyclist keeps its identity across frames instead
of being re-detected from scratch each time.

Usage (standalone -- prints a per-class track summary for a packaged session):
    python3 tracker.py <session_dir>

As a library, the entry points are:
  - track_entries(entries): geometry-only, no video needed.
  - track_video_entries(entries, video_path, start_epoch, reid_encoder=...):
    same, but seeks the source video to each entry's capture time and feeds
    that frame in -- required for appearance matching (inert otherwise).

Both take the list of detection entries a session's detections.jsonl already
loads into (driverassist_sync.load_detections's format -- each entry has a
"t" and a "detections" list of {label, confidence, x, y, w, h}), and return a
parallel list of per-detection track IDs (None for a low-confidence detection
that matched nothing -- see below).

How this differs from plain SORT: each frame's detections are split at two
confidence tiers. High-confidence detections are matched to existing tracks
first (by IoU, Hungarian assignment) and are the only ones allowed to spawn a
new track. Any track still unmatched after that gets a second chance against
the *low*-confidence detections -- this is what lets a track survive a bad
frame (occlusion, motion blur, glare) instead of losing its ID, without
letting a low-confidence box (more likely a false positive) start a new track
of its own.

IMPORTANT CURRENT LIMITATION: the on-device app only logs detections at or
above CONFIDENCE_THRESHOLD (0.25, see benchmark.py) -- nothing below that
threshold is in detections.jsonl today. So on existing/current recordings,
every detection lands in the "high" tier and the low-confidence recovery
stage never has anything to match against -- this degrades to plain
Kalman-filter SORT until the app is changed to also log a lower confidence
band for the tracker to draw on.

Tracking is done independently per class label (a person track can never
absorb a car detection), but track IDs are globally unique across classes.

Each track's motion is a constant-velocity Kalman filter over
[cx, cy, w, h] (box center, width, height) in normalized [0,1] coordinates --
simpler than classic SORT's [cx, cy, scale, aspect-ratio] parameterization,
and doesn't assume a roughly-constant aspect ratio, which doesn't hold well
for a pedestrian turning or a vehicle changing heading relative to the
camera. Every call to predict() advances one discrete step regardless of the
actual wall-clock gap since the last detection entry (the same simplification
classic SORT makes over video frames) -- fine given how close together this
app's detection entries land in practice, but worth knowing if entries ever
have large, irregular gaps.

Optional appearance/ReID matching: pass a `reid_encoder` (build one with
build_reid_encoder()) and a video frame per update() call, and box-content
similarity (cosine distance between embeddings, exponentially smoothed per
track) is fused with IoU exactly the way BoT-SORT's own get_dists() does --
`combined = min(iou_dist, embedding_dist)`, with embedding distance only
allowed to influence spatially-plausible candidates (it can't create a match
between two boxes that aren't already IoU-adjacent) and only when it's
itself confident (below appearance_thresh). This uses
ultralytics.trackers.utils.reid.ReID, which is a standalone embedding
extractor decoupled from BoT-SORT's own tracker/state classes -- adopting
BoT-SORT's actual tracker would mean rewriting how detections flow in (it
expects its own Results objects); this only borrows its embedding model.

Optional camera motion compensation (use_gmc=True, see gmc.py): the
constant-velocity model above has no idea the *camera* is moving, not just
the tracked object -- for a dashcam, most of a parked car's or pole's
apparent motion is entirely the vehicle's own movement. GMC estimates that
frame-to-frame camera motion (via sparse optical flow) and warps each
track's predicted position/velocity by it before matching, so a turn or tilt
doesn't desync every track's prediction simultaneously. Only rotation/pan/
tilt is corrected this way; forward-motion parallax needs per-object depth
to compensate properly and isn't attempted.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import iou  # noqa: E402
from driverassist_sync import DEFAULT_LOGS_DIR, load_detections, resolve_start_epoch  # noqa: E402
from gmc import GMC  # noqa: E402
from package_session import find_session_video  # noqa: E402

# Association gate -- deliberately looser than benchmark.py's 0.5 evaluation
# threshold, since a track's *predicted* box (extrapolated by the Kalman
# filter, possibly several frames since its last real match) has more slack
# than two independent models' boxes on the same frame.
DEFAULT_IOU_THRESHOLD = 0.2
DEFAULT_MAX_AGE = 5          # frames a track may go unmatched before it's dropped
DEFAULT_MIN_HITS = 1         # matches needed before a track ID is reported (see module docstring)
DEFAULT_HIGH_CONF_THRESHOLD = 0.25  # matches CONFIDENCE_THRESHOLD in benchmark.py / the app
DEFAULT_LOW_CONF_THRESHOLD = 0.1    # inert on today's data -- see module docstring
DEFAULT_APPEARANCE_THRESHOLD = 0.7  # cosine distance (1 - similarity) below this counts as a real match --
# validated via track_benchmark.py sweep (0.25/0.4/0.5/0.6/0.7) across 5 real driving sessions: idf1 improved
# and num_switches dropped monotonically from 0.25 up through 0.7 (the loosest value tested) in every
# large-sample config -- e.g. 26_07_29_2_Night went from idf1=0.279/4569 switches to idf1=0.306/3565 switches.
# The trend hadn't reversed by 0.7, so this may not be the true ceiling, but it's the best validated value.
DEFAULT_REID_MODEL = "yolo26n-reid.onnx"  # lightest yolo26-family ReID checkpoint; auto-downloaded

# How far apart (center-to-center, scaled by the larger box's own diagonal)
# a track/detection pair may be and still have appearance similarity
# consulted at all. Deliberately much looser than DEFAULT_IOU_THRESHOLD and
# expressed as a distance rather than IoU: IoU is exactly 0 for *any*
# non-overlapping pair, so it can't distinguish a track that drifted just
# past matching range from one that's nowhere near the detection -- without
# this, appearance can only ever help pairs that were already about to match
# on geometry alone, defeating the point of adding it. The final acceptance
# bar (iou_threshold, applied to the fused cost) is unchanged -- this only
# widens which pairs are considered, not what counts as a real match.
DEFAULT_APPEARANCE_ELIGIBILITY_SCALE = 3.0


def _box_to_cxcywh(box: dict) -> np.ndarray:
    return np.array([box["x"] + box["w"] / 2, box["y"] + box["h"] / 2, box["w"], box["h"]], dtype=float)


def _cxcywh_to_xywh(state) -> dict:
    cx, cy, w, h = state[:4]
    return {"x": float(cx - w / 2), "y": float(cy - h / 2), "w": float(w), "h": float(h)}


def _dets_to_pixel_xywh(detections: list, frame_shape) -> np.ndarray:
    """ultralytics.trackers.utils.reid.ReID expects pixel-space center-xywh
    boxes; our detections are normalized [0,1] top-left x/y/w/h."""
    fh, fw = frame_shape[:2]
    arr = np.zeros((len(detections), 4), dtype=np.float32)
    for i, d in enumerate(detections):
        arr[i] = [(d["x"] + d["w"] / 2) * fw, (d["y"] + d["h"] / 2) * fh, d["w"] * fw, d["h"] * fh]
    return arr


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-12)


def _box_diagonal(box: dict) -> float:
    return (box["w"] ** 2 + box["h"] ** 2) ** 0.5


def _center_distance(a: dict, b: dict) -> float:
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def build_reid_encoder(model: str = DEFAULT_REID_MODEL, device: str = "mps"):
    """Loads the appearance-embedding model used for optional ReID matching.
    `model` defaults to the lightest yolo26-family ReID checkpoint (an
    ultralytics-hosted asset, auto-downloaded on first use, sized to match
    the model family already used elsewhere in this project); pass a bigger
    tier (yolo26{s,m,l,x}-reid.onnx) for stronger but slower embeddings.
    Imported lazily since ultralytics.trackers.utils.reid is an internal
    (not public-API) module path."""
    from ultralytics.trackers.utils.reid import ReID

    return ReID(model, device=device)


SIZE_VELOCITY_DAMPING = 0.3  # see KalmanBoxTracker's F matrix -- caps runaway size extrapolation


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over an 8-dim state
    [cx, cy, w, h, vcx, vcy, vw, vh]; observes the 4-dim [cx, cy, w, h]."""

    def __init__(self, box: dict):
        self.x = np.zeros(8)
        self.x[:4] = _box_to_cxcywh(box)

        # Position velocity feeds forward at full strength each step, but
        # size (w, h) velocity is damped: during an extended unmatched
        # ("coasting") stretch, a handful of noisy early observations can fit
        # a spurious growth trend that then compounds linearly for as long as
        # the track goes unmatched, inflating the predicted box far past its
        # real size (observed: a predicted width ~2.6x the real one after 14
        # coasted steps, which alone was enough to fail the IoU gate on
        # re-detection). Box size changes far more slowly and noisily than
        # position does frame-to-frame for a person at normal walking speed,
        # so trusting a fitted size-velocity at the same strength as position
        # velocity isn't warranted. Note this damps the *mean* prediction
        # itself (via F), not just uncertainty (Q) -- during a gap with zero
        # measurements the state only ever advances via x = F @ x, so Q (which
        # only grows P, the covariance) can't constrain this on its own.
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0 if i < 2 else SIZE_VELOCITY_DAMPING

        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        # Modest measurement noise; larger process noise on the velocity
        # terms since we have no real prior on how fast a box's size/position
        # changes frame-to-frame.
        self.R = np.eye(4) * 0.01
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 5.0

        # Start with high uncertainty on velocity (unknown at init) and
        # moderate uncertainty on position (we just observed it).
        self.P = np.eye(8)
        self.P[4:, 4:] *= 100.0

    def predict(self) -> dict:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _cxcywh_to_xywh(self.x)

    def apply_gmc(self, warp: np.ndarray, frame_w: int, frame_h: int) -> None:
        """Corrects this track's just-predicted state for the camera's own
        estimated motion (see gmc.py) -- call after predict(), before
        matching. `warp` is a 2x3 affine in pixel space; applied in pixel
        space here too (not directly in our normalized [0,1] state) because
        a non-square frame normalizes x and y by different factors, which
        would distort a rotation if the warp were applied in normalized
        coordinates directly. Position is a point (full affine, including
        translation); velocity is a direction (rotation/scale only, no
        translation) -- size and its own velocity scale uniformly with the
        transform's scale factor. Note: only the mean state is corrected,
        not the covariance P -- a simplification, but the mean is what
        matching is actually scored against."""
        R, t = warp[:, :2], warp[:, 2]
        scale = float(np.sqrt(max(np.linalg.det(R), 1e-12)))

        pos_px = self.x[0] * frame_w, self.x[1] * frame_h
        new_pos_px = R @ np.array(pos_px) + t
        self.x[0], self.x[1] = new_pos_px[0] / frame_w, new_pos_px[1] / frame_h

        vel_px = self.x[4] * frame_w, self.x[5] * frame_h
        new_vel_px = R @ np.array(vel_px)
        self.x[4], self.x[5] = new_vel_px[0] / frame_w, new_vel_px[1] / frame_h

        self.x[2] *= scale
        self.x[3] *= scale
        self.x[6] *= scale
        self.x[7] *= scale

    def update(self, box: dict) -> None:
        z = _box_to_cxcywh(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> dict:
        return _cxcywh_to_xywh(self.x)


class Track:
    __slots__ = ("id", "label", "kf", "hits", "time_since_update", "age", "feat")

    def __init__(self, track_id: int, label: str, box: dict, feat=None):
        self.id = track_id
        self.label = label
        self.kf = KalmanBoxTracker(box)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.feat = None
        if feat is not None:
            self.update_feat(feat)

    def update_feat(self, feat: np.ndarray, alpha: float = 0.9) -> None:
        """Exponential moving average of the track's appearance embedding
        (mirrors BoT-SORT's BOTrack.update_features) -- smooths out a single
        bad crop rather than snapping to it."""
        feat = _normalize(np.asarray(feat, dtype=float))
        self.feat = feat if self.feat is None else _normalize(alpha * self.feat + (1 - alpha) * feat)


def _match(
    track_boxes: list, det_boxes: list, iou_threshold: float,
    track_feats: list = None, det_feats: list = None, appearance_thresh: float = DEFAULT_APPEARANCE_THRESHOLD,
    appearance_eligibility_scale: float = DEFAULT_APPEARANCE_ELIGIBILITY_SCALE,
) -> tuple:
    """Hungarian assignment minimizing 1-IoU distance, gated at iou_threshold.
    If track_feats/det_feats (parallel lists, entries may be None) are given,
    appearance distance is fused in exactly like BoT-SORT's get_dists():
    `combined = min(iou_dist, embedding_dist)`. Appearance is consulted for
    any pair within appearance_eligibility_scale * (larger box's diagonal) --
    a size-scaled *distance* gate, not the iou_threshold itself (see
    DEFAULT_APPEARANCE_ELIGIBILITY_SCALE for why IoU alone can't do this
    gating) -- and only counts if the embedding distance itself clears
    appearance_thresh. The final acceptance bar below is still iou_threshold,
    applied to the fused cost, so a strong appearance match can rescue a pair
    IoU alone would've missed, but a weak one can't invent a match out of
    nothing.
    Returns (matches, unmatched_track_idxs, unmatched_det_idxs), all indices
    local to the track_boxes/det_boxes lists passed in."""
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

    cost = np.array([[1.0 - iou(t, d) for d in det_boxes] for t in track_boxes])

    if track_feats is not None and det_feats is not None and any(f is not None for f in track_feats + det_feats):
        emb_cost = np.ones_like(cost)
        for i, (t, tf) in enumerate(zip(track_boxes, track_feats)):
            if tf is None:
                continue
            for j, (d, df) in enumerate(zip(det_boxes, det_feats)):
                if df is None:
                    continue
                max_diag = max(_box_diagonal(t), _box_diagonal(d))
                if _center_distance(t, d) > appearance_eligibility_scale * max_diag:
                    continue
                emb_cost[i, j] = 1.0 - float(np.dot(tf, df))
        emb_cost[emb_cost > (1.0 - appearance_thresh)] = 1.0
        cost = np.minimum(cost, emb_cost)

    row_idx, col_idx = linear_sum_assignment(cost)
    matches, matched_rows, matched_cols = [], set(), set()
    for r, c in zip(row_idx, col_idx):
        if 1.0 - cost[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_rows]
    unmatched_dets = [j for j in range(len(det_boxes)) if j not in matched_cols]
    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_age: int = DEFAULT_MAX_AGE,
        min_hits: int = DEFAULT_MIN_HITS,
        high_conf_threshold: float = DEFAULT_HIGH_CONF_THRESHOLD,
        low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD,
        reid_encoder=None,
        appearance_thresh: float = DEFAULT_APPEARANCE_THRESHOLD,
        appearance_eligibility_scale: float = DEFAULT_APPEARANCE_ELIGIBILITY_SCALE,
        use_gmc: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.reid_encoder = reid_encoder
        self.appearance_thresh = appearance_thresh
        self.appearance_eligibility_scale = appearance_eligibility_scale
        self.gmc = GMC() if use_gmc else None
        self.tracks: list = []
        self._next_id = 1

    def update(
        self, detections: list, frame: np.ndarray = None, embeddings: list = None,
        gmc_warp: np.ndarray = None, frame_shape: tuple = None,
    ) -> list:
        """detections: this frame's list of {label, confidence, x, y, w, h}.
        `frame` (a BGR image, e.g. from cv2) is only needed for appearance
        matching -- ignored if this tracker has no reid_encoder. Pass
        precomputed `embeddings` (parallel to `detections`, entries may be
        None) instead of `frame` to reuse embeddings computed on a previous
        run rather than re-running the encoder -- see track_benchmark.py's
        caching, which needs this to make sweeping tracker parameters cheap
        without re-running the (slow) reference model and encoder every time.
        Similarly, pass a precomputed `gmc_warp` (a 2x3 pixel-space affine,
        see gmc.py) instead of relying on `frame` + this tracker's own GMC
        estimator, to reuse a warp computed on a previous run -- apply_gmc
        needs the frame's pixel dimensions to convert to/from our normalized
        state, so pass `frame_shape` (h, w) explicitly when reusing a warp
        this way without also having the actual frame (a video's resolution
        is constant for its whole session, so this only needs setting once).
        Returns a list the same length/order as `detections`, each entry the
        assigned track ID (int) or None if it matched nothing and didn't
        spawn a new track (only possible for a low-confidence detection)."""
        for t in self.tracks:
            t.kf.predict()
            t.age += 1

        warp = gmc_warp
        if warp is None and self.gmc is not None and frame is not None:
            warp = self.gmc.apply(frame)
        if warp is not None and self.tracks:
            frame_h, frame_w = frame.shape[:2] if frame is not None else frame_shape
            for t in self.tracks:
                t.kf.apply_gmc(warp, frame_w, frame_h)

        if embeddings is None:
            embeddings = [None] * len(detections)
            if self.reid_encoder is not None and frame is not None and detections:
                pixel_boxes = _dets_to_pixel_xywh(detections, frame.shape)
                raw = self.reid_encoder(frame, pixel_boxes)
                embeddings = [None if e is None else np.asarray(e, dtype=float) for e in raw]

        track_ids_out = [None] * len(detections)
        high_idxs = [i for i, d in enumerate(detections) if d["confidence"] >= self.high_conf_threshold]
        low_idxs = [
            i for i, d in enumerate(detections)
            if self.low_conf_threshold <= d["confidence"] < self.high_conf_threshold
        ]

        labels = {t.label for t in self.tracks} | {detections[i]["label"] for i in high_idxs} | {
            detections[i]["label"] for i in low_idxs
        }

        for label in labels:
            track_idxs = [i for i, t in enumerate(self.tracks) if t.label == label]
            label_high_idxs = [i for i in high_idxs if detections[i]["label"] == label]

            track_boxes = [self.tracks[i].kf.box for i in track_idxs]
            track_feats = [self.tracks[i].feat for i in track_idxs]
            det_boxes = [detections[i] for i in label_high_idxs]
            det_feats = [embeddings[i] for i in label_high_idxs]
            matches, unmatched_t, unmatched_d = _match(
                track_boxes, det_boxes, self.iou_threshold,
                track_feats=track_feats, det_feats=det_feats, appearance_thresh=self.appearance_thresh,
                appearance_eligibility_scale=self.appearance_eligibility_scale,
            )

            for t_local, d_local in matches:
                track = self.tracks[track_idxs[t_local]]
                det_idx = label_high_idxs[d_local]
                track.kf.update(detections[det_idx])
                if embeddings[det_idx] is not None:
                    track.update_feat(embeddings[det_idx])
                track.hits += 1
                track.time_since_update = 0
                track_ids_out[det_idx] = track.id

            # Second stage: remaining unmatched tracks of this label vs.
            # this frame's low-confidence detections of this label.
            remaining_track_idxs = [track_idxs[i] for i in unmatched_t]
            label_low_idxs = [i for i in low_idxs if detections[i]["label"] == label]
            track_boxes2 = [self.tracks[i].kf.box for i in remaining_track_idxs]
            track_feats2 = [self.tracks[i].feat for i in remaining_track_idxs]
            det_boxes2 = [detections[i] for i in label_low_idxs]
            det_feats2 = [embeddings[i] for i in label_low_idxs]
            matches2, unmatched_t2, _ = _match(
                track_boxes2, det_boxes2, self.iou_threshold,
                track_feats=track_feats2, det_feats=det_feats2, appearance_thresh=self.appearance_thresh,
                appearance_eligibility_scale=self.appearance_eligibility_scale,
            )

            still_unmatched_local = set(range(len(remaining_track_idxs)))
            for t_local, d_local in matches2:
                track = self.tracks[remaining_track_idxs[t_local]]
                det_idx = label_low_idxs[d_local]
                track.kf.update(detections[det_idx])
                if embeddings[det_idx] is not None:
                    track.update_feat(embeddings[det_idx])
                track.time_since_update = 0
                track_ids_out[det_idx] = track.id
                still_unmatched_local.discard(t_local)

            for t_local in still_unmatched_local:
                track = self.tracks[remaining_track_idxs[t_local]]
                track.time_since_update += 1

            # Unmatched high-confidence detections spawn new tracks.
            # Low-confidence detections never do (they're the less reliable
            # signal -- see module docstring).
            for d_local in unmatched_d:
                det_idx = label_high_idxs[d_local]
                new_track = Track(self._next_id, label, detections[det_idx], feat=embeddings[det_idx])
                self._next_id += 1
                self.tracks.append(new_track)
                track_ids_out[det_idx] = new_track.id

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        for i, tid in enumerate(track_ids_out):
            if tid is not None:
                track = next((t for t in self.tracks if t.id == tid), None)
                if track is not None and track.hits < self.min_hits:
                    track_ids_out[i] = None  # not yet confirmed -- see DEFAULT_MIN_HITS

        return track_ids_out


def track_entries(entries: list, **tracker_kwargs) -> list:
    """Geometry-only tracking -- see module docstring. entries: chronologically-
    sorted detection entries (as loaded by driverassist_sync.load_detections).
    Returns a parallel list of per-entry track-ID lists, each aligned with
    that entry's "detections" list."""
    tracker = ByteTracker(**tracker_kwargs)
    return [tracker.update(entry["detections"]) for entry in entries]


def track_video_entries(entries: list, video_path, start_epoch: float, reid_encoder=None, **tracker_kwargs) -> list:
    """Like track_entries(), but seeks the source video to each entry's own
    capture time and feeds that frame to the tracker -- needed for appearance/
    ReID matching (inert without reid_encoder, in which case this is just a
    slower track_entries()). Mirrors how benchmark.py seeks per-entry to run
    the reference model on the same frame the on-device model actually saw."""
    tracker = ByteTracker(reid_encoder=reid_encoder, **tracker_kwargs)
    if reid_encoder is None:
        return [tracker.update(e["detections"]) for e in entries]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open {video_path} for appearance matching")

    all_ids = []
    for entry in entries:
        capture_time = entry["t"] - entry["elapsedMs"] / 1000.0
        offset_ms = (capture_time - start_epoch) * 1000.0
        frame = None
        if offset_ms >= 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
            ok, f = cap.read()
            frame = f if ok else None
        all_ids.append(tracker.update(entry["detections"], frame=frame))

    cap.release()
    return all_ids


def summarize(entries: list, all_track_ids: list) -> None:
    """Prints a per-class track-count/duration summary -- not a full MOT
    benchmark (that needs a reference tracker to compare against, which the
    reference model doesn't produce), but a concrete look at how the tracker
    is actually behaving on real data."""
    track_label = {}
    track_frame_count = defaultdict(int)
    track_first_t = {}
    track_last_t = {}

    for entry, ids in zip(entries, all_track_ids):
        for det, tid in zip(entry["detections"], ids):
            if tid is None:
                continue
            track_label[tid] = det["label"]
            track_frame_count[tid] += 1
            track_first_t.setdefault(tid, entry["t"])
            track_last_t[tid] = entry["t"]

    by_class = defaultdict(list)
    for tid, label in track_label.items():
        duration = track_last_t[tid] - track_first_t[tid]
        by_class[label].append((track_frame_count[tid], duration))

    print(f"{'class':<12} {'tracks':>7} {'1-frame':>8} {'median frames':>14} {'median secs':>12} {'longest secs':>13}")
    for label in sorted(by_class):
        counts = [c for c, _ in by_class[label]]
        durations = [d for _, d in by_class[label]]
        one_frame = sum(1 for c in counts if c == 1)
        print(
            f"{label:<12} {len(counts):>7} {one_frame:>8} "
            f"{np.median(counts):>14.1f} {np.median(durations):>12.2f} {max(durations):>13.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Packaged session directory (see package_session.py)")
    parser.add_argument("--detections", type=Path, default=None, help="Defaults to <session_dir>/detections.jsonl")
    parser.add_argument("--video", type=Path, default=None, help="Defaults to the one raw recording in <session_dir>")
    parser.add_argument("--debug-log", type=Path, default=None, help="Defaults to <session_dir>/overlay-debug.log")
    parser.add_argument("--reid", dest="reid", action="store_true", default=True, help="Use appearance/ReID matching (default: on)")
    parser.add_argument("--no-reid", dest="reid", action="store_false", help="Geometry-only tracking, no video needed")
    parser.add_argument("--reid-model", default=DEFAULT_REID_MODEL)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    detections_path = args.detections or (args.session_dir / "detections.jsonl")
    if not detections_path.exists():
        sys.exit(f"{detections_path} doesn't exist.")
    entries = load_detections(detections_path)

    if args.reid:
        video = args.video or find_session_video(args.session_dir)
        debug_log = args.debug_log or (args.session_dir / "overlay-debug.log")
        start_epoch, _ = resolve_start_epoch(video, debug_log if debug_log.exists() else DEFAULT_LOGS_DIR)
        reid_encoder = build_reid_encoder(args.reid_model, device=args.device)
        all_track_ids = track_video_entries(entries, video, start_epoch, reid_encoder=reid_encoder)
    else:
        all_track_ids = track_entries(entries)

    total_tracks = len({tid for ids in all_track_ids for tid in ids if tid is not None})
    print(f"{len(entries)} detection entries -> {total_tracks} distinct tracks\n")
    summarize(entries, all_track_ids)


if __name__ == "__main__":
    main()
