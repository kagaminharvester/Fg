"""
video_loader.py
================

This module encapsulates loading and decoding of video files.  The
``VideoLoader`` class provides a simple interface for reading frames
from a video while automatically resizing them for efficient GPU
processing.  For stereo VR formats (side‑by‑side, over‑under,
equirectangular) the loader will split the frames into left and right
images.

The loader uses OpenCV to perform video decoding.  When available
``cv2.cuda.VideoCapture`` will be used to leverage NVIDIA's NVDEC on
RTX‑class GPUs for maximum throughput.  If CUDA support is not
available the loader gracefully falls back to the CPU decoder.

The detection of video type is currently naive and based on simple
filename patterns (``_LR_`` for side‑by‑side, ``_OU_`` for
over‑under, ``_FISHEYE``/``_MKX`` for fisheye and ``_180`` for
equirectangular).  You can extend ``_detect_format`` to read
container metadata if more robust detection is needed.
"""

from __future__ import annotations

import os
import cv2
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional


@dataclass
class VideoInfo:
    """Simple data container describing a video stream."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    format: str  # one of "2D", "SBS", "OU", "EQUI"


class VideoLoader:
    """Load video files and iterate through frames.

    Parameters
    ----------
    path:
        Path to the video file.  Files containing stereo VR footage
        should follow the naming convention described in
        ``_detect_format``.
    target_width:
        Optional width to which frames are resized.  Maintaining
        consistent frame dimensions improves inference throughput.  If
        ``None`` no resizing is performed.
    device:
        Optional CUDA device index.  When set, the loader will try to
        use ``cv2.cuda.VideoCapture`` to decode frames on the GPU.
    """

    def __init__(self, path: str, target_width: Optional[int] = None, device: Optional[int] = None) -> None:
        self.path = path
        self.cap: cv2.VideoCapture | cv2.cuda.VideoCapture
        self.target_width = target_width
        self.device = device
        self.info: Optional[VideoInfo] = None
        self._open()

    def _open(self) -> None:
        if self.device is not None and hasattr(cv2, "cuda"):
            # try to open with CUDA
            try:
                self.cap = cv2.cuda.VideoCapture(self.path)
                if not self.cap.isOpened():
                    raise RuntimeError("Could not open video with CUDA")
            except Exception:
                # fallback to CPU
                self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {self.path}")
        # gather info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fmt = self._detect_format(os.path.basename(self.path))
        self.info = VideoInfo(self.path, width, height, fps, frame_count, fmt)

    def _detect_format(self, filename: str) -> str:
        name = filename.lower()
        if "_lr_" in name or "sbs" in name or "sidebyside" in name:
            return "SBS"  # side‑by‑side
        if "_ou_" in name or "overunder" in name:
            return "OU"  # over‑under
        if any(flag in name for flag in ["fisheye", "mkx", "180"]):
            return "EQUI"  # equirectangular or fisheye 180°
        return "2D"

    def __iter__(self) -> Iterator[Tuple[int, cv2.Mat | Tuple[cv2.Mat, cv2.Mat]]]:
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.target_width is not None:
                # compute new height preserving aspect ratio
                h, w = frame.shape[:2]
                ratio = self.target_width / float(w)
                new_size = (self.target_width, int(h * ratio))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            fmt = self.info.format if self.info else "2D"
            if fmt == "SBS":
                # split side‑by‑side into left/right halves
                h, w = frame.shape[:2]
                left = frame[:, : w // 2]
                right = frame[:, w // 2 :]
                yield idx, (left, right)
            elif fmt == "OU":
                # split over‑under
                h, w = frame.shape[:2]
                top = frame[: h // 2]
                bottom = frame[h // 2 :]
                yield idx, (top, bottom)
            else:
                yield idx, frame
            idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
