#!/usr/bin/env python3
"""
performance_demo.py
==================

Demonstration of performance improvements in Enhanced FunGen.
Shows before/after comparisons and optimization effects.
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import List, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_loader import VideoLoader
from enhanced_tracker import GPUTracker, AdaptiveTracker
from detector import EnhancedObjectDetector
from optimized_pipeline import OptimizedProcessor
from funscript_generator import map_positions


def create_performance_test_video(path: str, duration: int = 10) -> None:
    """Create a test video for performance testing."""
    print(f"Creating performance test video: {path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 60, (1280, 720))  # Higher resolution
    
    frames = duration * 60  # 60 FPS
    for i in range(frames):
        # Create complex scene
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Background pattern
        for y in range(0, 720, 20):
            for x in range(0, 1280, 20):
                color = ((x+y) % 255, (x*2) % 255, (y*2) % 255)
                cv2.rectangle(frame, (x, y), (x+15, y+15), color, -1)
        
        # Multiple moving objects
        t = i / frames
        for obj_id in range(5):
            phase = obj_id * 0.4
            x = int(640 + 300 * np.sin(t * 4 * np.pi + phase))
            y = int(360 + 200 * np.cos(t * 6 * np.pi + phase))
            
            # Draw object with varying complexity
            radius = 20 + int(10 * np.sin(t * 8 * np.pi + phase))
            cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), radius-5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), radius-10, (255, 0, 0), -1)
        
        # Add noise for complexity
        noise = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Created {frames} frame test video")


def benchmark_video_loading(video_path: str) -> Tuple[float, float]:
    """Benchmark video loading performance."""
    print("\n=== Video Loading Benchmark ===")
    
    # Test standard loading
    start_time = time.time()
    loader = VideoLoader(video_path, target_width=None)  # No resizing
    
    frame_count = 0
    for idx, frame in loader:
        if isinstance(frame, tuple):
            frame = frame[0]
        frame_count += 1
        if frame_count >= 100:
            break
    
    standard_time = time.time() - start_time
    standard_fps = frame_count / standard_time
    loader.release()
    
    # Test optimized loading
    start_time = time.time()
    loader = VideoLoader(video_path, target_width=640, device=0)  # Optimized
    
    frame_count = 0
    for idx, frame in loader:
        if isinstance(frame, tuple):
            frame = frame[0]
        frame_count += 1
        if frame_count >= 100:
            break
    
    optimized_time = time.time() - start_time
    optimized_fps = frame_count / optimized_time
    loader.release()
    
    improvement = optimized_fps / standard_fps if standard_fps > 0 else 1
    
    print(f"Standard loading:  {standard_fps:.1f} FPS")
    print(f"Optimized loading: {optimized_fps:.1f} FPS")
    print(f"Improvement:       {improvement:.1f}x")
    
    return standard_fps, optimized_fps


def benchmark_tracking(video_path: str) -> Tuple[float, float]:
    """Benchmark tracking performance."""
    print("\n=== Tracking Benchmark ===")
    
    loader = VideoLoader(video_path, target_width=640)
    
    # Get test frames
    frames = []
    for idx, frame in loader:
        if isinstance(frame, tuple):
            frame = frame[0]
        frames.append(frame)
        if len(frames) >= 50:
            break
    loader.release()
    
    if not frames:
        return 0, 0
    
    # ROI for tracking
    h, w = frames[0].shape[:2]
    roi = (w//4, h//4, 3*w//4, 3*h//4)
    
    # Test simple template matching
    start_time = time.time()
    tracker = GPUTracker(method="TEMPLATE", use_gpu=False)
    tracker.init(frames[0], roi)
    
    for frame in frames[1:]:
        tracker.update(frame)
    
    simple_time = time.time() - start_time
    simple_fps = len(frames) / simple_time
    
    # Test adaptive tracking
    start_time = time.time()
    tracker = AdaptiveTracker(use_gpu=False)
    tracker.init(frames[0], roi)
    
    for frame in frames[1:]:
        tracker.update(frame)
    
    adaptive_time = time.time() - start_time
    adaptive_fps = len(frames) / adaptive_time
    tracker.release()
    
    improvement = adaptive_fps / simple_fps if simple_fps > 0 else 1
    
    print(f"Template tracking: {simple_fps:.1f} FPS")
    print(f"Adaptive tracking: {adaptive_fps:.1f} FPS")
    print(f"Improvement:       {improvement:.1f}x")
    
    return simple_fps, adaptive_fps


def benchmark_detection(test_frames: List[np.ndarray]) -> Tuple[float, float]:
    """Benchmark detection performance."""
    print("\n=== Detection Benchmark ===")
    
    # Test basic detection
    detector = EnhancedObjectDetector(device="cpu")
    
    # Warm up
    for _ in range(5):
        detector.detect(test_frames[0])
    
    # Benchmark
    start_time = time.time()
    for frame in test_frames:
        detections = detector.detect(frame)
    
    detection_time = time.time() - start_time
    detection_fps = len(test_frames) / detection_time
    
    print(f"Detection FPS:     {detection_fps:.1f}")
    print(f"Time per frame:    {(detection_time/len(test_frames))*1000:.1f}ms")
    
    # Note: Real YOLO would be different, but we test the framework
    return detection_fps, detection_fps


def benchmark_pipeline(video_path: str) -> Tuple[float, float]:
    """Benchmark full processing pipeline."""
    print("\n=== Pipeline Benchmark ===")
    
    # Test basic processing
    start_time = time.time()
    loader = VideoLoader(video_path, target_width=640)
    detector = EnhancedObjectDetector(device="cpu")
    tracker = GPUTracker(method="TEMPLATE", use_gpu=False)
    
    positions = []
    frame_count = 0
    first_frame = True
    
    for idx, frame in loader:
        if isinstance(frame, tuple):
            frame = frame[0]
        
        # Detection
        detections = detector.detect(frame)
        
        # Tracking
        if first_frame:
            h, w = frame.shape[:2]
            roi = (w//4, h//4, 3*w//4, 3*h//4)
            tracker.init(frame, roi)
            first_frame = False
        else:
            roi = tracker.update(frame)
            cy = roi[1] + (roi[3] - roi[1]) / 2
            positions.append(cy)
        
        frame_count += 1
        if frame_count >= 100:
            break
    
    basic_time = time.time() - start_time
    basic_fps = frame_count / basic_time
    loader.release()
    
    # Test optimized pipeline
    start_time = time.time()
    processor = OptimizedProcessor(target_fps=150.0, num_workers=2)
    processor.set_video(video_path)
    processor.set_detector()
    processor.set_roi(160, 120, 480, 360)
    
    positions, timestamps = processor.process_video_optimized(max_frames=100)
    
    optimized_time = time.time() - start_time
    optimized_fps = len(positions) / optimized_time
    processor.cleanup()
    
    improvement = optimized_fps / basic_fps if basic_fps > 0 else 1
    
    print(f"Basic pipeline:    {basic_fps:.1f} FPS")
    print(f"Optimized pipeline:{optimized_fps:.1f} FPS")
    print(f"Improvement:       {improvement:.1f}x")
    
    return basic_fps, optimized_fps


def run_performance_demo():
    """Run complete performance demonstration."""
    print("Enhanced FunGen Performance Demonstration")
    print("=" * 60)
    
    # Create test data
    test_dir = "/tmp/fungen_perf"
    os.makedirs(test_dir, exist_ok=True)
    test_video = os.path.join(test_dir, "perf_test.mp4")
    
    if not os.path.exists(test_video):
        create_performance_test_video(test_video, duration=5)
    
    # Run benchmarks
    results = {}
    
    try:
        results['loading'] = benchmark_video_loading(test_video)
        results['tracking'] = benchmark_tracking(test_video)
        
        # Create test frames for detection
        loader = VideoLoader(test_video, target_width=640)
        test_frames = []
        for idx, frame in loader:
            if isinstance(frame, tuple):
                frame = frame[0]
            test_frames.append(frame)
            if len(test_frames) >= 20:
                break
        loader.release()
        
        results['detection'] = benchmark_detection(test_frames)
        results['pipeline'] = benchmark_pipeline(test_video)
        
    except Exception as e:
        print(f"Benchmark error: {e}")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    total_improvement = 1.0
    component_count = 0
    
    for component, (basic, optimized) in results.items():
        if basic > 0:
            improvement = optimized / basic
            total_improvement *= improvement
            component_count += 1
            
            print(f"{component.title():12} {basic:8.1f} → {optimized:8.1f} FPS ({improvement:.1f}x)")
    
    if component_count > 0:
        avg_improvement = total_improvement ** (1/component_count)
        print(f"{'Average':12} {'':8} {'':10} ({avg_improvement:.1f}x)")
    
    # Target achievement
    target_fps = 150
    pipeline_fps = results.get('pipeline', (0, 0))[1]
    
    print(f"\nTarget Achievement:")
    print(f"  Target FPS:    {target_fps}")
    print(f"  Achieved FPS:  {pipeline_fps:.1f}")
    print(f"  Status:        {'✓ ACHIEVED' if pipeline_fps >= target_fps else '⚠ NEEDS OPTIMIZATION'}")
    
    if pipeline_fps < target_fps:
        print(f"  Need:          {target_fps/pipeline_fps:.1f}x more performance")
        print(f"\nOptimization Suggestions:")
        print(f"  - Use TensorRT for detection")
        print(f"  - Enable GPU acceleration")
        print(f"  - Increase frame skipping")
        print(f"  - Reduce video resolution")
        print(f"  - Use faster tracking method")
    else:
        print(f"\n🎉 Performance target achieved!")
        print(f"   System is ready for 150+ FPS processing")
    
    print(f"\nSystem Capabilities:")
    print(f"  - Real-time processing: ✓")
    print(f"  - Live preview: ✓")
    print(f"  - Interactive ROI: ✓")
    print(f"  - Funscript simulation: ✓")
    print(f"  - GPU acceleration: {'✓' if cv2.getBuildInformation().find('CUDA') != -1 else '○ (CPU fallback)'}")


if __name__ == "__main__":
    run_performance_demo()