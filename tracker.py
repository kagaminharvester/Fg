"""
tracker.py
==========

This module provides object tracking utilities.  For the sake of
simplicity we implement a lightweight template‑matching tracker which
tracks the movement of a template (ROI) across successive frames.

The tracker records the vertical centre of the bounding box for each
frame.  This signal is later mapped to the funscript range (0–100).

For production use you should replace this with a more robust
tracking solution such as Deep SORT or ByteTrack.  Those algorithms
associate detections across frames and handle occlusion and identity
switches better than naive template matching.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, List


class SimpleTracker:
    """A minimal template‑matching tracker.

    The tracker is initialised with the first frame and an ROI.  It
    extracts the ROI as a template.  For each subsequent frame the
    template is matched using ``cv2.matchTemplate`` to find the
    region with the highest correlation.  The returned bounding box
    approximates the new position of the object.

    Parameters
    ----------
    method:
        OpenCV matching method.  Defaults to ``cv2.TM_CCOEFF_NORMED``.
    search_radius:
        Pixel radius around the previous ROI in which to search for the
        template.  Setting this to ``None`` searches the entire frame,
        which is slower but robust to large displacements.
    """

    def __init__(self, method: int = cv2.TM_CCOEFF_NORMED, search_radius: Optional[int] = 100) -> None:
        self.template: Optional[np.ndarray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.method = method
        self.search_radius = search_radius
        self.positions: List[float] = []

    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = roi
        self.template = frame[y1:y2, x1:x2].copy()
        self.roi = roi

    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        if self.template is None or self.roi is None:
            raise RuntimeError("Tracker has not been initialised")
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        tmpl = self.template
        th, tw = tmpl.shape[:2]
        # define search region
        if self.search_radius is not None:
            sx1 = max(0, x1 - self.search_radius)
            sy1 = max(0, y1 - self.search_radius)
            sx2 = min(w - tw, x1 + self.search_radius)
            sy2 = min(h - th, y1 + self.search_radius)
            search_img = frame[sy1: sy2 + th, sx1: sx2 + tw]
        else:
            sx1 = 0
            sy1 = 0
            search_img = frame
        res = cv2.matchTemplate(search_img, tmpl, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_loc = min_loc
        else:
            best_loc = max_loc
        bx = sx1 + best_loc[0]
        by = sy1 + best_loc[1]
        self.roi = (bx, by, bx + tw, by + th)
        # record vertical centre
        cy = by + th / 2
        self.positions.append(float(cy))
        return self.roi
