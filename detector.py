"""
detector.py
===========

Enhanced object detector optimized for RTX 3090 with YOLO support.
Provides GPU-accelerated detection with TensorRT optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None


@dataclass
class Detection:
    """Represents a single detection output."""
    box: Tuple[int, int, int, int]
    score: float
    label: str
    class_id: int = -1


class EnhancedObjectDetector:
    """GPU-accelerated object detector with YOLO and TensorRT support."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", 
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.model_type = "none"
        self.input_size = (640, 640)
        
        # Performance tracking
        self.inference_times = []
        self.last_fps = 0.0
        
        # TensorRT specific
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        
        if model_path:
            self._load_model(model_path)
            
    def _load_model(self, path: str) -> None:
        """Load detection model based on file extension."""
        suffix = path.split(".")[-1].lower()
        
        if suffix == "engine" and TENSORRT_AVAILABLE:
            self._load_tensorrt_engine(path)
        elif suffix in {"pt", "pth"} and ULTRALYTICS_AVAILABLE:
            self._load_yolo_model(path)
        elif suffix == "onnx":
            self._load_onnx_model(path)
        else:
            print(f"Unsupported model format: {suffix}")
            self.model_type = "dummy"
            
    def _load_yolo_model(self, path: str) -> None:
        """Load YOLO model with Ultralytics."""
        try:
            self.model = YOLO(path)
            if self.device == "cuda":
                self.model.to('cuda')
            self.model_type = "yolo"
            print(f"Loaded YOLO model: {path}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model_type = "dummy"
            
    def _load_tensorrt_engine(self, path: str) -> None:
        """Load TensorRT engine for maximum performance."""
        if not TENSORRT_AVAILABLE:
            print("TensorRT not available")
            self.model_type = "dummy"
            return
            
        try:
            # Load TensorRT engine
            with open(path, 'rb') as f:
                engine_data = f.read()
                
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self.trt_context = self.trt_engine.create_execution_context()
            
            # Setup I/O bindings
            self._setup_tensorrt_bindings()
            self.model_type = "tensorrt"
            print(f"Loaded TensorRT engine: {path}")
            
        except Exception as e:
            print(f"Failed to load TensorRT engine: {e}")
            self.model_type = "dummy"
            
    def _setup_tensorrt_bindings(self) -> None:
        """Setup TensorRT input/output bindings."""
        if not self.trt_engine:
            return
            
        # Get input/output information
        for i in range(self.trt_engine.num_bindings):
            name = self.trt_engine.get_binding_name(i)
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
            shape = self.trt_context.get_binding_shape(i)
            
            if self.trt_engine.binding_is_input(i):
                self.input_size = (shape[-1], shape[-2])  # Width, Height
                
            # Allocate memory
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            binding = {
                'name': name,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype
            }
            
            self.trt_bindings.append(binding)
            
            if self.trt_engine.binding_is_input(i):
                self.trt_inputs.append(binding)
            else:
                self.trt_outputs.append(binding)
                
    def _load_onnx_model(self, path: str) -> None:
        """Load ONNX model with ONNX Runtime."""
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model = ort.InferenceSession(path, providers=providers)
            self.model_type = "onnx"
            print(f"Loaded ONNX model: {path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.model_type = "dummy"
            
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on a single frame."""
        start_time = time.time()
        
        if self.model_type == "yolo":
            detections = self._detect_yolo(frame)
        elif self.model_type == "tensorrt":
            detections = self._detect_tensorrt(frame)
        elif self.model_type == "onnx":
            detections = self._detect_onnx(frame)
        else:
            detections = self._detect_dummy(frame)
            
        # Update performance metrics
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
            
        if len(self.inference_times) >= 10:
            avg_time = sum(self.inference_times[-10:]) / 10
            self.last_fps = 1.0 / avg_time if avg_time > 0 else 0
            
        return detections
        
    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """YOLO detection implementation."""
        if not self.model:
            return []
            
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, 
                               iou=self.nms_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, score, cls in zip(boxes, scores, classes):
                        if score >= self.confidence_threshold:
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            label = self.model.names.get(cls, f"class_{cls}")
                            
                            detections.append(Detection(
                                box=(x1, y1, x2, y2),
                                score=float(score),
                                label=label,
                                class_id=cls
                            ))
                            
            return detections
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
            
    def _detect_tensorrt(self, frame: np.ndarray) -> List[Detection]:
        """TensorRT detection implementation."""
        if not self.trt_context:
            return []
            
        try:
            # Preprocess
            input_tensor = self._preprocess_tensorrt(frame)
            
            # Copy input to GPU
            cuda.memcpy_htod(self.trt_inputs[0]['device'], input_tensor)
            
            # Run inference
            self.trt_context.execute_v2([binding['device'] for binding in self.trt_bindings])
            
            # Copy output from GPU
            outputs = []
            for output in self.trt_outputs:
                cuda.memcpy_dtoh(output['host'], output['device'])
                outputs.append(output['host'].reshape(output['shape']))
                
            # Postprocess
            return self._postprocess_tensorrt(outputs, frame.shape)
            
        except Exception as e:
            print(f"TensorRT detection error: {e}")
            return []
            
    def _preprocess_tensorrt(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for TensorRT inference."""
        # Resize to input size
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor.astype(np.float32)
        
    def _postprocess_tensorrt(self, outputs: List[np.ndarray], 
                            original_shape: Tuple[int, int, int]) -> List[Detection]:
        """Postprocess TensorRT outputs."""
        if not outputs:
            return []
            
        # Assuming YOLO-style output format
        predictions = outputs[0]  # Shape: [batch, num_detections, 85]
        
        detections = []
        h_orig, w_orig = original_shape[:2]
        h_input, w_input = self.input_size[1], self.input_size[0]
        
        scale_x = w_orig / w_input
        scale_y = h_orig / h_input
        
        for detection in predictions[0]:  # Remove batch dimension
            confidence = detection[4]
            if confidence < self.confidence_threshold:
                continue
                
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence < self.confidence_threshold:
                continue
                
            # Convert coordinates
            cx, cy, w, h = detection[:4]
            x1 = int((cx - w/2) * scale_x)
            y1 = int((cy - h/2) * scale_y)
            x2 = int((cx + w/2) * scale_x)
            y2 = int((cy + h/2) * scale_y)
            
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(class_confidence),
                label=f"class_{class_id}",
                class_id=class_id
            ))
            
        return detections
        
    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        """ONNX detection implementation."""
        if not self.model:
            return []
            
        try:
            # Preprocess (similar to TensorRT)
            input_tensor = self._preprocess_tensorrt(frame)
            
            # Get input name
            input_name = self.model.get_inputs()[0].name
            
            # Run inference
            outputs = self.model.run(None, {input_name: input_tensor})
            
            # Postprocess
            return self._postprocess_tensorrt(outputs, frame.shape)
            
        except Exception as e:
            print(f"ONNX detection error: {e}")
            return []
            
    def _detect_dummy(self, frame: np.ndarray) -> List[Detection]:
        """Dummy detection for testing."""
        h, w = frame.shape[:2]
        return [Detection(
            box=(0, 0, w, h),
            score=1.0,
            label='frame',
            class_id=0
        )]
        
    def detect_stereo(self, left: np.ndarray, right: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Perform detection on stereo pair."""
        return self.detect(left), self.detect(right)
        
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'model_type': self.model_type,
            'fps': self.last_fps,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'device': self.device
        }
        
    def warmup(self, shape: Tuple[int, int, int] = (640, 640, 3)) -> None:
        """Warm up the model with dummy input."""
        dummy_frame = np.zeros(shape, dtype=np.uint8)
        for _ in range(5):  # Run 5 warmup iterations
            self.detect(dummy_frame)
            
    def release(self) -> None:
        """Release resources."""
        if self.trt_context:
            del self.trt_context
        if self.trt_engine:
            del self.trt_engine
        for binding in self.trt_bindings:
            if 'device' in binding:
                binding['device'].free()


# Legacy compatibility
ObjectDetector = EnhancedObjectDetector
