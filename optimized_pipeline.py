"""
optimized_pipeline.py
====================

Optimized processing pipeline for 150+ FPS performance.
Uses memory pooling, frame skipping, and parallel processing.
"""

from __future__ import annotations

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Tuple
import numpy as np
import cv2

from video_loader import VideoLoader
from enhanced_tracker import GPUTracker
from detector import EnhancedObjectDetector
from funscript_generator import map_positions, Funscript


@dataclass
class FrameData:
    """Container for frame processing data."""
    idx: int
    frame: np.ndarray
    timestamp: float
    roi: Optional[Tuple[int, int, int, int]] = None
    position: Optional[float] = None
    detections: List[Any] = None


class MemoryPool:
    """Memory pool for efficient frame buffer management."""
    
    def __init__(self, shape: Tuple[int, int, int], max_size: int = 20):
        self.shape = shape
        self.pool = deque()
        self.max_size = max_size
        self.lock = threading.Lock()
        
        # Pre-allocate buffers
        for _ in range(max_size):
            self.pool.append(np.zeros(shape, dtype=np.uint8))
    
    def get_buffer(self) -> np.ndarray:
        """Get a buffer from the pool."""
        with self.lock:
            if self.pool:
                return self.pool.popleft()
            else:
                # Pool exhausted, allocate new
                return np.zeros(self.shape, dtype=np.uint8)
    
    def return_buffer(self, buffer: np.ndarray) -> None:
        """Return a buffer to the pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(buffer)


class OptimizedProcessor:
    """Optimized processor for maximum FPS performance."""
    
    def __init__(self, target_fps: float = 150.0, num_workers: int = None):
        self.target_fps = target_fps
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        # Processing components
        self.video_loader: Optional[VideoLoader] = None
        self.detector: Optional[EnhancedObjectDetector] = None
        self.tracker: Optional[GPUTracker] = None
        
        # Memory management
        self.memory_pool: Optional[MemoryPool] = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.frame_queue = deque()
        self.result_queue = deque()
        self.processing_lock = threading.Lock()
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = 0.0
        self.positions: List[float] = []
        self.timestamps: List[float] = []
        
        # Optimization settings
        self.frame_skip = 1  # Process every N frames
        self.detection_interval = 5  # Run detection every N frames
        self.tracking_only_mode = False
        
    def set_video(self, video_path: str) -> bool:
        """Load video and optimize settings."""
        try:
            self.video_loader = VideoLoader(video_path, target_width=800, device=0)
            
            if self.video_loader.info:
                # Initialize memory pool based on frame size
                h, w = 450, 800  # Approximate processed frame size
                self.memory_pool = MemoryPool((h, w, 3), max_size=30)
                
                # Optimize frame skip based on video FPS
                video_fps = self.video_loader.info.fps
                if video_fps > 60:
                    self.frame_skip = max(1, int(video_fps / 60))
                
                print(f"Optimized settings for {video_fps} FPS video:")
                print(f"  Frame skip: {self.frame_skip}")
                print(f"  Detection interval: {self.detection_interval}")
                
                return True
            return False
            
        except Exception as e:
            print(f"Failed to load video: {e}")
            return False
    
    def set_detector(self, model_path: Optional[str] = None) -> None:
        """Set detector with optimization."""
        self.detector = EnhancedObjectDetector(model_path, device="cpu")
        if self.detector:
            self.detector.warmup()
    
    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set tracking ROI."""
        if self.tracker:
            self.tracker.release()
        self.tracker = GPUTracker(use_gpu=False)  # Use CPU for stability
        self.roi = (x1, y1, x2, y2)
        
    def process_frame_parallel(self, frame_data: FrameData) -> FrameData:
        """Process single frame in parallel worker."""
        try:
            # Detection (less frequent)
            if frame_data.idx % self.detection_interval == 0 and self.detector:
                detections = self.detector.detect(frame_data.frame)
                frame_data.detections = detections
            
            # Tracking (every frame)
            if self.tracker and hasattr(self, 'roi'):
                if frame_data.idx == 0:
                    self.tracker.init(frame_data.frame, self.roi)
                    cy = self.roi[1] + (self.roi[3] - self.roi[1]) / 2
                    frame_data.position = cy
                else:
                    roi = self.tracker.update(frame_data.frame)
                    frame_data.roi = roi
                    cy = roi[1] + (roi[3] - roi[1]) / 2
                    frame_data.position = cy
            
            return frame_data
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame_data
    
    def process_video_optimized(self, max_frames: Optional[int] = None, 
                              callback: Optional[Callable] = None) -> Tuple[List[float], List[float]]:
        """Process video with maximum optimization."""
        if not self.video_loader:
            raise ValueError("No video loaded")
        
        self.start_time = time.time()
        self.frame_count = 0
        self.dropped_frames = 0
        self.positions.clear()
        self.timestamps.clear()
        
        # Target frame time for rate limiting
        target_frame_time = 1.0 / self.target_fps
        
        # Processing loop
        futures = []
        last_fps_report = time.time()
        
        try:
            for frame_idx, frame in self.video_loader:
                # Handle stereo frames
                if isinstance(frame, tuple):
                    frame = frame[0]
                
                # Frame skipping for performance
                if frame_idx % self.frame_skip != 0:
                    self.dropped_frames += 1
                    continue
                
                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    break
                
                process_start = time.time()
                
                # Create frame data
                frame_data = FrameData(
                    idx=self.frame_count,
                    frame=frame.copy(),
                    timestamp=frame_idx / self.video_loader.info.fps
                )
                
                # Submit for parallel processing
                if len(futures) < self.num_workers:
                    future = self.executor.submit(self.process_frame_parallel, frame_data)
                    futures.append(future)
                else:
                    # Wait for oldest future and process result
                    completed_future = futures.pop(0)
                    try:
                        result = completed_future.result(timeout=0.1)
                        self._handle_result(result, callback)
                    except Exception as e:
                        print(f"Processing timeout: {e}")
                    
                    # Submit new frame
                    future = self.executor.submit(self.process_frame_parallel, frame_data)
                    futures.append(future)
                
                self.frame_count += 1
                
                # Rate limiting
                process_time = time.time() - process_start
                if process_time < target_frame_time:
                    time.sleep(target_frame_time - process_time)
                
                # FPS reporting
                current_time = time.time()
                if current_time - last_fps_report > 1.0:
                    elapsed = current_time - self.start_time
                    current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processing FPS: {current_fps:.1f} (target: {self.target_fps})")
                    last_fps_report = current_time
            
            # Process remaining futures
            for future in futures:
                try:
                    result = future.result(timeout=1.0)
                    self._handle_result(result, callback)
                except Exception as e:
                    print(f"Final processing error: {e}")
        
        finally:
            # Cleanup
            for future in futures:
                if not future.done():
                    future.cancel()
        
        # Final statistics
        total_time = time.time() - self.start_time
        final_fps = self.frame_count / total_time if total_time > 0 else 0
        
        print(f"\nProcessing complete:")
        print(f"  Processed frames: {self.frame_count}")
        print(f"  Dropped frames: {self.dropped_frames}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {final_fps:.1f}")
        print(f"  Target achieved: {'✓' if final_fps >= self.target_fps else '✗'}")
        
        return self.positions.copy(), self.timestamps.copy()
    
    def _handle_result(self, result: FrameData, callback: Optional[Callable]) -> None:
        """Handle processed frame result."""
        if result.position is not None:
            self.positions.append(result.position)
            self.timestamps.append(result.timestamp)
        
        if callback:
            callback(result)
    
    def generate_funscript(self, **params) -> Optional[Funscript]:
        """Generate funscript from processed positions."""
        if not self.positions or not self.video_loader:
            return None
        
        return map_positions(
            positions=self.positions,
            frame_height=self.video_loader.info.height,
            fps=self.video_loader.info.fps,
            **params
        )
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.tracker:
            self.tracker.release()
        if self.video_loader:
            self.video_loader.release()


class StreamProcessor:
    """Real-time stream processor for live input."""
    
    def __init__(self, target_fps: float = 150.0):
        self.target_fps = target_fps
        self.is_processing = False
        self.processor = OptimizedProcessor(target_fps, num_workers=2)
        
    def start_webcam_processing(self, camera_index: int = 0) -> None:
        """Start processing webcam input."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_processing = True
        frame_idx = 0
        start_time = time.time()
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create frame data and process
                frame_data = FrameData(
                    idx=frame_idx,
                    frame=frame,
                    timestamp=(time.time() - start_time)
                )
                
                # Process frame
                result = self.processor.process_frame_parallel(frame_data)
                
                # Display result (would be sent to UI in real application)
                if result.roi:
                    x1, y1, x2, y2 = result.roi
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                frame_idx += 1
                
                # Rate limiting
                target_time = 1.0 / self.target_fps
                time.sleep(max(0, target_time - 0.001))  # Small processing overhead
        
        finally:
            cap.release()
            self.processor.cleanup()
    
    def stop_processing(self) -> None:
        """Stop processing."""
        self.is_processing = False