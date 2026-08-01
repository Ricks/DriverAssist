#!/usr/bin/env python3
"""
Global Motion Compensation (GMC) -- estimates the camera's own frame-to-frame
motion via sparse optical flow, so tracker.py can correct a track's predicted
position for how much the *scene* shifted due to the vehicle turning/tilting,
not just how the tracked object itself moved.

Why this exists: KalmanBoxTracker's constant-velocity model fits a track's
velocity purely from how its box moves in raw frame coordinates, with no
notion of the camera moving. For a dashcam that's a real gap -- most tracked
objects (parked cars, poles, signs) aren't moving at all; their entire
apparent motion is the vehicle's own movement. That's a fine approximation
while the vehicle's motion stays roughly steady, but breaks the moment it
changes (braking, turning) -- every track's fitted velocity goes stale
*simultaneously*, since what actually changed was the camera, not the
individual objects.

Method: goodFeaturesToTrack + calcOpticalFlowPyrLK between consecutive
frames (on a downscaled grayscale copy, for speed), then
cv2.estimateAffinePartial2D (RANSAC) fits a similarity transform (rotation +
uniform scale + translation) from the matched point pairs. Sparse
feature-based flow was chosen over whole-image alignment (e.g. ECC) because
it's markedly more robust on the low-texture, noisy night footage this
project mostly deals with -- ECC needs a reasonably strong gradient signal
across the whole frame to converge, which a dark street scene often lacks.
This mirrors BoT-SORT's own GMC module (ultralytics/trackers/utils/gmc.py),
which supports several methods including this one; we didn't reuse BoT-SORT's
implementation directly since (like the rest of tracker.py) this is a
from-scratch tracker, not built on BoT-SORT's own track/state classes.

Only rotation/tilt/pan is meaningfully corrected this way -- forward-motion
parallax (a near object growing faster than a far one for the same distance
traveled) isn't a uniform transform and needs per-object depth to compensate
properly, which isn't attempted here.
"""
import cv2
import numpy as np

DEFAULT_DOWNSCALE = 2
DEFAULT_MAX_CORNERS = 200
DEFAULT_MIN_MATCHES = 8  # below this, an estimated affine is too unreliable to trust


class GMC:
    """Stateful frame-to-frame camera motion estimator. Call apply(frame) once
    per frame, in chronological order -- it remembers the previous frame's
    keypoints internally."""

    def __init__(
        self,
        downscale: int = DEFAULT_DOWNSCALE,
        max_corners: int = DEFAULT_MAX_CORNERS,
        min_matches: int = DEFAULT_MIN_MATCHES,
    ):
        self.downscale = downscale
        self.max_corners = max_corners
        self.min_matches = min_matches
        self.prev_gray = None
        self.prev_points = None

    def _to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // self.downscale), max(1, h // self.downscale)))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def _detect_points(self, gray: np.ndarray):
        return cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=10)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Returns a 2x3 affine warp matrix, in *pixel space at the frame's
        original resolution*, mapping the previous frame's coordinates to
        this frame's -- identity if this is the first frame seen, or if
        estimation fails/has too few matches to trust."""
        gray = self._to_gray_small(frame)
        warp = np.eye(2, 3, dtype=np.float32)

        if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) >= self.min_matches:
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)
            status = status.reshape(-1).astype(bool)
            prev_matched, curr_matched = self.prev_points[status], curr_points[status]

            if len(prev_matched) >= self.min_matches:
                M, _ = cv2.estimateAffinePartial2D(prev_matched, curr_matched, method=cv2.RANSAC)
                if M is not None:
                    warp = M.astype(np.float32)
                    # The rotation/scale block is scale-invariant under our
                    # uniform downscale, but translation was measured in
                    # downscaled pixels -- rescale it back to full resolution.
                    warp[:, 2] *= self.downscale

        self.prev_gray = gray
        self.prev_points = self._detect_points(gray)
        return warp
