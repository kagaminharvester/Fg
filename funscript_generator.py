"""
funscript_generator.py
======================

This module converts a series of vertical object positions into a
Funscript.  A Funscript is a simple JSON file containing an
``actions`` array, where each element has an ``at`` field (time in
milliseconds) and a ``pos`` field (position from 0–100).  Many
interactive devices (e.g. The Handy) consume Funscript files to
coordinate motion with video playback.

The generation process maps raw positions (pixel coordinates) into
percentage positions relative to the frame height.  Optional
post‑processing can be applied, including boosting, thresholding,
smoothing and randomness.  See the documentation for details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class Funscript:
    """Represents a complete funscript.

    Attributes
    ----------
    actions:
        List of {at: ms, pos: int} dictionaries.
    meta:
        Optional metadata dictionary.  Most players ignore this field
        but it can be used to store descriptive information.
    """
    actions: List[Dict[str, Any]]
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps({"version": "1.0", "actions": self.actions, **({"meta": self.meta} if self.meta else {})}, indent=2)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def map_positions(
    positions: List[float],
    frame_height: int,
    fps: float,
    min_pos: int = 0,
    max_pos: int = 100,
    boost_up_percent: float = 0.0,
    boost_down_percent: float = 0.0,
    threshold_low: Optional[float] = None,
    threshold_high: Optional[float] = None,
    smoothing_window: int = 1,
    randomness: float = 0.0,
) -> Funscript:
    """Convert raw vertical positions into a funscript.

    Parameters
    ----------
    positions:
        Sequence of vertical centre positions (in pixels) for each frame.
    frame_height:
        Height of the video frame in pixels.
    fps:
        Frames per second of the video.  Used to compute time stamps.
    min_pos / max_pos:
        Output range in funscript coordinates (0–100 by default).
    boost_up_percent / boost_down_percent:
        Percentages to expand the dynamic range of the signal.  Positive
        values push peaks further away from the mean.  Negative values
        compress the range.
    threshold_low / threshold_high:
        Raw positions below ``threshold_low`` map to ``min_pos`` and
        above ``threshold_high`` map to ``max_pos``.  Thresholds
        specified as percent of frame height (0–1).  If ``None`` no
        thresholding is applied.
    smoothing_window:
        Width of the moving average filter to reduce jitter.  Set to 1
        to disable smoothing.
    randomness:
        Amount of noise (0–1) to inject into the signal.  A value of
        0.05 adds up to ±5% random variation.
    """
    if len(positions) == 0:
        return Funscript(actions=[])
    positions_np = np.array(positions, dtype=np.float32)
    # normalise to [0, 1] range (0 at top of frame) and invert (0 => top => high pos)
    normalised = 1.0 - (positions_np / frame_height)
    # apply thresholding
    if threshold_low is not None:
        normalised = np.where(normalised < threshold_low, 0.0, normalised)
    if threshold_high is not None:
        normalised = np.where(normalised > threshold_high, 1.0, normalised)
    # apply boosting
    if boost_up_percent != 0 or boost_down_percent != 0:
        mean = np.mean(normalised)
        above = normalised > mean
        below = normalised <= mean
        if boost_up_percent != 0:
            normalised[above] = mean + (normalised[above] - mean) * (1 + boost_up_percent)
        if boost_down_percent != 0:
            normalised[below] = mean + (normalised[below] - mean) * (1 + boost_down_percent)
        normalised = np.clip(normalised, 0.0, 1.0)
    # smoothing
    normalised = _moving_average(normalised, smoothing_window)
    # randomness
    if randomness > 0:
        noise = (np.random.rand(len(normalised)) - 0.5) * 2 * randomness
        normalised = np.clip(normalised + noise, 0.0, 1.0)
    # map to output range and round
    mapped = min_pos + normalised * (max_pos - min_pos)
    mapped = np.clip(mapped, min_pos, max_pos)
    positions_int = np.round(mapped).astype(int)
    # generate timestamps in ms
    timestamps_ms = (np.arange(len(positions_int)) / fps * 1000.0).astype(int)
    actions = [{"at": int(t), "pos": int(p)} for t, p in zip(timestamps_ms, positions_int)]
    return Funscript(actions=actions)
