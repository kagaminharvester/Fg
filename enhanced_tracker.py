"""
enhanced_tracker.py
==================

GPU-accelerated tracker optimized for RTX 3090 performance.
Provides multiple tracking algorithms with CUDA acceleration.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
import time


class GPUTracker:
    """GPU-accelerated object tracker using CUDA."""
    
    def __init__(self, method: str = "TEMPLATE", use_gpu: bool = True):
        self.method = method
        self.use_gpu = use_gpu and self._check_cuda_support()
        self.tracker = None
        self.template: Optional[np.ndarray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.positions: List[float] = []
        
        # GPU memory objects
        self.gpu_frame = None
        self.gpu_template = None
        self.cuda_stream = None
        
        if self.use_gpu:
            try:
                self.cuda_stream = cv2.cuda.Stream()
            except Exception:
                self.use_gpu = False
                
    def _check_cuda_support(self) -> bool:
        """Check if CUDA is available for OpenCV."""
        try:
            cuda_available = cv2.getBuildInformation().find('CUDA') != -1
            if cuda_available:
                # Test if we can actually use CUDA
                try:
                    test_mat = cv2.cuda.GpuMat()
                    return True
                except:
                    return False
            return False
        except Exception:
            return False
            
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """Initialize tracker with first frame and ROI."""
        self.roi = roi
        x1, y1, x2, y2 = roi
        
        # Extract template
        self.template = frame[y1:y2, x1:x2].copy()
        
        if self.use_gpu:
            try:
                # Upload to GPU
                self.gpu_frame = cv2.cuda.GpuMat()
                self.gpu_template = cv2.cuda.GpuMat()
                self.gpu_frame.upload(frame, self.cuda_stream)
                self.gpu_template.upload(self.template, self.cuda_stream)
            except Exception:
                self.use_gpu = False
                
        # Initialize OpenCV tracker as fallback
        if self.method == "TEMPLATE":
            # Use template matching only
            self.tracker = None
        elif self.method == "CSRT":
            try:
                self.tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                try:
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                except AttributeError:
                    print("CSRT tracker not available, falling back to template matching")
                    self.tracker = None
        elif self.method == "KCF":
            try:
                self.tracker = cv2.TrackerKCF_create()
            except AttributeError:
                try:
                    self.tracker = cv2.legacy.TrackerKCF_create()
                except AttributeError:
                    print("KCF tracker not available, falling back to template matching")
                    self.tracker = None
        elif self.method == "MOSSE":
            try:
                self.tracker = cv2.TrackerMOSSE_create()
            except AttributeError:
                try:
                    self.tracker = cv2.legacy.TrackerMOSSE_create()
                except AttributeError:
                    print("MOSSE tracker not available, falling back to template matching")
                    self.tracker = None
        else:
            # Try CSRT first, then fallback
            try:
                self.tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                try:
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                except AttributeError:
                    print("No OpenCV trackers available, using template matching")
                    self.tracker = None
            
        if self.tracker:
            self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Update tracker with new frame."""
        if self.use_gpu and self.gpu_template is not None:
            return self._update_gpu(frame)
        else:
            return self._update_cpu(frame)
            
    def _update_gpu(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """GPU-accelerated tracking update."""
        try:
            # Upload frame to GPU
            self.gpu_frame.upload(frame, self.cuda_stream)
            
            # Convert to grayscale for better performance
            gpu_gray = cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY, stream=self.cuda_stream)
            gpu_template_gray = cv2.cuda.cvtColor(self.gpu_template, cv2.COLOR_BGR2GRAY, stream=self.cuda_stream)
            
            # Template matching on GPU
            result = cv2.cuda.matchTemplate(gpu_gray, gpu_template_gray, cv2.TM_CCOEFF_NORMED, stream=self.cuda_stream)
            
            # Download result
            result_cpu = result.download(stream=self.cuda_stream)
            
            # Find best match
            _, max_val, _, max_loc = cv2.minMaxLoc(result_cpu)
            
            # Update ROI
            th, tw = self.template.shape[:2]
            x1, y1 = max_loc
            x2, y2 = x1 + tw, y1 + th
            
            self.roi = (x1, y1, x2, y2)
            
            # Record position
            cy = y1 + th / 2
            self.positions.append(float(cy))
            
            return self.roi
            
        except Exception:
            # Fallback to CPU
            return self._update_cpu(frame)
            
    def _update_cpu(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """CPU tracking update."""
        if self.tracker and self.roi:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                self.roi = (x, y, x + w, y + h)
                
                # Record position
                cy = y + h / 2
                self.positions.append(float(cy))
                
                return self.roi
                
        # Fallback to template matching
        return self._template_match_cpu(frame)
        
    def _template_match_cpu(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """CPU template matching fallback."""
        if self.template is None or self.roi is None:
            return self.roi or (0, 0, 0, 0)
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Define search region around last known position
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        th, tw = template_gray.shape[:2]
        
        search_radius = 100
        sx1 = max(0, x1 - search_radius)
        sy1 = max(0, y1 - search_radius)
        sx2 = min(w - tw, x1 + search_radius)
        sy2 = min(h - th, y1 + search_radius)
        
        if sx2 > sx1 and sy2 > sy1:
            search_region = gray[sy1:sy2 + th, sx1:sx2 + tw]
            result = cv2.matchTemplate(search_region, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Update position
            bx = sx1 + max_loc[0]
            by = sy1 + max_loc[1]
            self.roi = (bx, by, bx + tw, by + th)
            
            # Record position
            cy = by + th / 2
            self.positions.append(float(cy))
            
        return self.roi
        
    def get_confidence(self) -> float:
        """Get tracking confidence score."""
        # Simple confidence based on template matching
        if self.template is None or self.roi is None:
            return 0.0
            
        # In a real implementation, you'd compute actual confidence
        return 0.8  # Placeholder
        
    def release(self) -> None:
        """Release GPU resources."""
        if self.gpu_frame is not None:
            self.gpu_frame.release()
        if self.gpu_template is not None:
            self.gpu_template.release()


class MultiObjectTracker:
    """Track multiple objects simultaneously with GPU acceleration."""
    
    def __init__(self, max_objects: int = 5, use_gpu: bool = True):
        self.max_objects = max_objects
        self.use_gpu = use_gpu
        self.trackers: List[GPUTracker] = []
        self.object_ids: List[int] = []
        self.next_id = 0
        
    def add_object(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> int:
        """Add new object to track."""
        if len(self.trackers) >= self.max_objects:
            return -1
            
        tracker = GPUTracker(use_gpu=self.use_gpu)
        tracker.init(frame, roi)
        
        self.trackers.append(tracker)
        self.object_ids.append(self.next_id)
        
        object_id = self.next_id
        self.next_id += 1
        
        return object_id
        
    def update(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Update all trackers and return results."""
        results = []
        
        for i, tracker in enumerate(self.trackers):
            try:
                roi = tracker.update(frame)
                confidence = tracker.get_confidence()
                object_id = self.object_ids[i]
                results.append((object_id, roi, confidence))
            except Exception:
                # Remove failed tracker
                self.trackers.pop(i)
                self.object_ids.pop(i)
                
        return results
        
    def remove_object(self, object_id: int) -> bool:
        """Remove object from tracking."""
        try:
            idx = self.object_ids.index(object_id)
            self.trackers[idx].release()
            self.trackers.pop(idx)
            self.object_ids.pop(idx)
            return True
        except (ValueError, IndexError):
            return False
            
    def clear(self) -> None:
        """Clear all trackers."""
        for tracker in self.trackers:
            tracker.release()
        self.trackers.clear()
        self.object_ids.clear()
        self.next_id = 0


class AdaptiveTracker:
    """Adaptive tracker that switches algorithms based on performance."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.trackers = {
            "CSRT": GPUTracker("CSRT", use_gpu),
            "KCF": GPUTracker("KCF", use_gpu),
            "MOSSE": GPUTracker("MOSSE", use_gpu),
        }
        self.current_tracker = "CSRT"
        self.performance_history = {name: [] for name in self.trackers.keys()}
        self.switch_threshold = 5  # frames
        self.poor_performance_count = 0
        
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """Initialize all trackers."""
        for tracker in self.trackers.values():
            tracker.init(frame, roi)
            
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Update with adaptive tracker switching."""
        start_time = time.time()
        
        # Update current tracker
        current = self.trackers[self.current_tracker]
        roi = current.update(frame)
        
        update_time = time.time() - start_time
        confidence = current.get_confidence()
        
        # Record performance
        self.performance_history[self.current_tracker].append({
            'time': update_time,
            'confidence': confidence
        })
        
        # Check if we should switch trackers
        if confidence < 0.3:  # Low confidence
            self.poor_performance_count += 1
            if self.poor_performance_count >= self.switch_threshold:
                self._switch_tracker(frame)
                self.poor_performance_count = 0
        else:
            self.poor_performance_count = 0
            
        return roi
        
    def _switch_tracker(self, frame: np.ndarray) -> None:
        """Switch to better performing tracker."""
        best_tracker = self.current_tracker
        best_score = 0.0
        
        for name, tracker in self.trackers.items():
            if name == self.current_tracker:
                continue
                
            history = self.performance_history[name]
            if len(history) > 0:
                # Compute average performance
                avg_confidence = np.mean([h['confidence'] for h in history[-10:]])
                avg_time = np.mean([h['time'] for h in history[-10:]])
                
                # Score based on confidence and speed
                score = avg_confidence - avg_time * 10  # Prefer fast and confident
                
                if score > best_score:
                    best_score = score
                    best_tracker = name
                    
        if best_tracker != self.current_tracker:
            print(f"Switching tracker from {self.current_tracker} to {best_tracker}")
            self.current_tracker = best_tracker
            
    def release(self) -> None:
        """Release all tracker resources."""
        for tracker in self.trackers.values():
            tracker.release()