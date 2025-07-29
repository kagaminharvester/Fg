"""
generate_tensorrt.py
====================

This script demonstrates how to convert a YOLOv8 PyTorch model to a
TensorRT engine for high‑performance inference on NVIDIA GPUs.
TensorRT engines dramatically reduce inference latency by fusing
operations and utilising FP16/INT8 precision when available.

To use this script you must have the ``tensorrt`` and ``torch``
packages installed as well as the ``ultralytics`` package for
loading the YOLO model.  The script first exports the model to
ONNX using Ultralytics and then invokes the ``trtexec`` CLI to
build the engine.  ``trtexec`` is bundled with the TensorRT SDK.

Example:

```
python generate_tensorrt.py --weights models/yolov8n.pt --output models/yolov8n.engine --fp16
```

Note: This is a simple wrapper; consult the TensorRT and Ultralytics
documentation for more advanced options.
"""

import argparse
import os
import subprocess

from ultralytics import YOLO  # type: ignore


def export_to_onnx(weights: str, onnx_path: str) -> None:
    model = YOLO(weights)
    model.export(format="onnx", dynamic=True, simplify=True, opset=12, output=onnx_path)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, workspace: int = 8192) -> None:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
    ]
    if fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLOv8 model to TensorRT engine")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--output", required=True, help="Path to output .engine file")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--workspace", type=int, default=8192, help="Workspace size in MB")
    args = parser.parse_args()
    onnx_path = os.path.splitext(args.output)[0] + ".onnx"
    print(f"Exporting {args.weights} to ONNX...")
    export_to_onnx(args.weights, onnx_path)
    print(f"Building TensorRT engine at {args.output} ...")
    build_engine(onnx_path, args.output, fp16=args.fp16, workspace=args.workspace)
    print("Done.")


if __name__ == "__main__":
    main()
