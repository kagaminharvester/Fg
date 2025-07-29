"""
detector.py
===========

This module defines the ``ObjectDetector`` class used by the FunGen
pipeline.  In its simplest form the detector takes a frame and
returns one or more bounding boxes representing objects of interest.

For a production system you would integrate a YOLO model here.  The
``ultralytics`` library makes it straightforward to load a .pt file
and perform inference.  If you have converted the model to TensorRT
(.engine) for maximum speed on RTX‑class GPUs you can load it with
the ``tensorrt`` runtime instead.

To avoid a heavy dependency in this template we fall back to a
placeholder implementation that simply returns the full frame bounds.
You can plug in your own detection logic by subclassing
``ObjectDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    # try to import yolov8 via ultralytics if available
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

try:
    import tensorrt as trt  # type: ignore
except Exception:
    trt = None  # type: ignore


@dataclass
class Detection:
    """Represents a single detection output.

    Attributes
    ----------
    box:
        A tuple (x1, y1, x2, y2) in pixels.
    score:
        Confidence score for the detection.
    label:
        String label for the detected class (e.g. 'hand', 'mouth').
    """
    box: Tuple[int, int, int, int]
    score: float
    label: str


class ObjectDetector:
    """Encapsulates the detection model.

    ``model_path`` may refer to a PyTorch (.pt), ONNX (.onnx) or
    TensorRT (.engine) model.  When using .engine the detector will
    attempt to load via TensorRT for optimum performance on NVIDIA
    GPUs.  See `generate_tensorrt.py` for details on converting .pt
    models to .engine.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        suffix = path.split(".")[-1].lower()
        if suffix == "engine" and trt is not None:
            # TensorRT engine
            # For brevity we leave out the full TensorRT loader.  See
            # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
            # for details.  In production you should load the engine
            # and create an execution context here.
            self.model = None
        elif suffix in {"pt", "pth"} and YOLO is not None:
            self.model = YOLO(path)
            self.model.fuse()
        elif suffix == "onnx":
            import onnxruntime  # type: ignore
            self.model = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a single frame.

        In this template implementation the entire frame is returned as
        a single detection with a confidence of 1.0 and label
        ``'roi'``.  Replace this method with your preferred detection
        logic.
        """
        h, w = frame.shape[:2]
        box = (0, 0, w, h)
        return [Detection(box=box, score=1.0, label='roi')]

    def detect_stereo(self, left: np.ndarray, right: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Perform detection on a stereo pair.

        Returns a pair of detection lists for the left and right images.
        """
        return self.detect(left), self.detect(right)
