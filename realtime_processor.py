"""
realtime_processor.py
====================

Real-time video processing pipeline optimized for RTX 3090.
This module provides GPU-accelerated processing with worker threads
to achieve 150+ fps for analysis and generation.
"""

from __future__ import annotations

import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Any, Tuple, List
import numpy as np
import cv2
from PyQt6 import QtCore

from video_loader import VideoLoader, VideoInfo
from enhanced_tracker import GPUTracker, AdaptiveTracker
from detector import EnhancedObjectDetector, Detection
from funscript_generator import map_positions, Funscript


@dataclass
class ProcessingStats:
    """Performance statistics for real-time processing."""
    fps: float = 0.0
    frames_processed: int = 0
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    total_time_ms: float = 0.0
    gpu_memory_mb: float = 0.0


class RealtimeProcessor(QtCore.QObject):
    """Real-time video processor with GPU acceleration.
    
    This processor runs in a separate thread and provides:
    - GPU-accelerated video decoding and processing
    - Real-time object detection and tracking
    - Live position updates for funscript generation
    - Performance monitoring and optimization
    """
    
    # Signals for UI updates
    frameProcessed = QtCore.pyqtSignal(int, np.ndarray, list)  # frame_idx, frame, detections
    positionUpdated = QtCore.pyqtSignal(float, float)  # timestamp, position
    statsUpdated = QtCore.pyqtSignal(ProcessingStats)
    processingComplete = QtCore.pyqtSignal()
    errorOccurred = QtCore.pyqtSignal(str)
    
    def __init__(self, target_fps: float = 150.0):
        super().__init__()
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # Processing components
        self.video_loader: Optional[VideoLoader] = None
        self.detector: Optional[EnhancedObjectDetector] = None
        self.tracker: Optional[AdaptiveTracker] = None
        
        # Processing state
        self.is_processing = False
        self.should_stop = False
        self.current_roi: Optional[Tuple[int, int, int, int]] = None
        self.positions: List[float] = []
        self.timestamps: List[float] = []
        
        # Performance tracking
        self.stats = ProcessingStats()
        self.frame_times: List[float] = []
        self.last_stats_update = 0.0
        
        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # GPU optimization
        self.use_gpu = self._check_gpu_availability()
        self.cuda_stream = None
        if self.use_gpu and hasattr(cv2, 'cuda'):
            try:
                self.cuda_stream = cv2.cuda.Stream()
            except Exception:
                self.use_gpu = False
    
    def _check_gpu_availability(self) -> bool:
        """Check if CUDA is available for OpenCV."""
        try:
            cuda_available = cv2.getBuildInformation().find('CUDA') != -1
            if cuda_available:
                # Test if we can actually use CUDA
                try:
                    test_mat = cv2.cuda.GpuMat()
                    return True
                except:
                    print("CUDA detected but not functional, using CPU")
                    return False
            else:
                print("CUDA not available, using CPU optimizations")
                return False
        except Exception:
            return False
    
    def set_video(self, video_path: str) -> bool:
        """Load a video file for processing."""
        try:
            if self.video_loader:
                self.video_loader.release()
            
            # Use GPU device 0 for RTX 3090
            self.video_loader = VideoLoader(video_path, target_width=1280, device=0)
            
            if not self.video_loader.info:
                raise RuntimeError("Failed to load video info")
            
            return True
        except Exception as e:
            self.errorOccurred.emit(f"Failed to load video: {str(e)}")
            return False
    
    def set_detector(self, model_path: Optional[str] = None) -> None:
        """Set the object detector."""
        try:
            self.detector = EnhancedObjectDetector(model_path, device="cuda" if self.use_gpu else "cpu")
            if self.detector:
                self.detector.warmup()  # Warm up the model
        except Exception as e:
            self.errorOccurred.emit(f"Failed to load detector: {str(e)}")
    
    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set the region of interest for tracking."""
        self.current_roi = (x1, y1, x2, y2)
        self.tracker = None  # Reset tracker
        self.positions.clear()
        self.timestamps.clear()
    
    def start_processing(self, max_frames: Optional[int] = None) -> None:
        """Start real-time processing."""
        if self.is_processing:
            return
        
        if not self.video_loader or not self.current_roi:
            self.errorOccurred.emit("Video and ROI must be set before processing")
            return
        
        self.is_processing = True
        self.should_stop = False
        self.stats = ProcessingStats()
        self.frame_times.clear()
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(
            target=self._process_video,
            args=(max_frames,),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_processing(self) -> None:
        """Stop processing."""
        self.should_stop = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.is_processing = False
    
    def _process_video(self, max_frames: Optional[int] = None) -> None:
        """Main processing loop running in separate thread."""
        try:
            start_time = time.time()
            frame_count = 0
            
            for frame_idx, frame in self.video_loader:
                if self.should_stop:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                frame_start = time.time()
                
                # Handle stereo frames (take left eye)
                if isinstance(frame, tuple):
                    frame = frame[0]
                
                # Process first frame - initialize tracker
                if frame_idx == 0:
                    self._initialize_tracker(frame)
                    cy = self.current_roi[1] + (self.current_roi[3] - self.current_roi[1]) / 2
                    self.positions.append(cy)
                    self.timestamps.append(0.0)
                else:
                    # Perform detection and tracking
                    detections = self._detect_objects(frame)
                    position = self._update_tracking(frame)
                    
                    if position is not None:
                        timestamp = frame_idx / self.video_loader.info.fps
                        self.positions.append(position)
                        self.timestamps.append(timestamp)
                        
                        # Emit position update
                        self.positionUpdated.emit(timestamp, position)
                
                # Emit frame update
                detections = self.detector.detect(frame) if self.detector else []
                self.frameProcessed.emit(frame_idx, frame, detections)
                
                # Update performance stats
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                frame_count += 1
                
                # Update stats periodically
                current_time = time.time()
                if current_time - self.last_stats_update > 0.5:  # Every 500ms
                    self._update_stats(current_time - start_time, frame_count)
                    self.last_stats_update = current_time
                
                # Frame rate limiting for target FPS
                if frame_time < self.target_frame_time:
                    time.sleep(self.target_frame_time - frame_time)
            
            self.processingComplete.emit()
            
        except Exception as e:
            self.errorOccurred.emit(f"Processing error: {str(e)}")
        finally:
            self.is_processing = False
    
    def _initialize_tracker(self, frame: np.ndarray) -> None:
        """Initialize tracker with first frame and ROI."""
        if not self.current_roi:
            return
        
        self.tracker = AdaptiveTracker(use_gpu=self.use_gpu)
        self.tracker.init(frame, self.current_roi)
    
    def _detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on frame."""
        if not self.detector:
            return []
        
        detect_start = time.time()
        detections = self.detector.detect(frame)
        self.stats.detection_time_ms = (time.time() - detect_start) * 1000
        
        return detections
    
    def _update_tracking(self, frame: np.ndarray) -> Optional[float]:
        """Update object tracking and return position."""
        if not self.tracker:
            return None
        
        track_start = time.time()
        roi = self.tracker.update(frame)
        self.stats.tracking_time_ms = (time.time() - track_start) * 1000
        
        # Return vertical center position
        _, y1, _, y2 = roi
        return y1 + (y2 - y1) / 2
    
    def _update_stats(self, total_time: float, frame_count: int) -> None:
        """Update performance statistics."""
        if frame_count > 0:
            self.stats.fps = frame_count / total_time
            self.stats.frames_processed = frame_count
            self.stats.total_time_ms = total_time * 1000
            
            if self.frame_times:
                avg_frame_time = sum(self.frame_times[-30:]) / min(30, len(self.frame_times))
                self.stats.fps = min(self.stats.fps, 1.0 / avg_frame_time if avg_frame_time > 0 else 0)
        
        # GPU memory usage (if available)
        if self.use_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.stats.gpu_memory_mb = info.used / 1024 / 1024
            except Exception:
                # GPU monitoring not available
                self.stats.gpu_memory_mb = 0.0
        
        self.statsUpdated.emit(self.stats)
    
    def get_current_funscript(self, **params) -> Optional[Funscript]:
        """Generate funscript from current positions."""
        if not self.positions or not self.video_loader:
            return None
        
        return map_positions(
            positions=self.positions,
            frame_height=self.video_loader.info.height,
            fps=self.video_loader.info.fps,
            **params
        )
    
    def update_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update ROI during processing."""
        self.current_roi = (x1, y1, x2, y2)
        # Re-initialize tracker if processing
        if self.is_processing and self.processing_thread:
            # Note: In a full implementation, you'd want to synchronize this
            # with the processing thread more carefully
            pass