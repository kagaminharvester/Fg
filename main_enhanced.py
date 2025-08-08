"""
main_enhanced.py
===============

Enhanced entry point for FunGen VR with modern features and command-line options.
High-performance, GPU-accelerated application optimized for RTX 3090.

Usage:
    python main_enhanced.py              # Launch modern GUI
    python main_enhanced.py --classic    # Launch classic GUI
    python main_enhanced.py --benchmark  # Run performance benchmarks
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_environment():
    """Setup environment optimizations."""
    import os
    
    # Enable optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Disable Qt warnings in headless environments
    os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '')


def run_modern_gui():
    """Launch the modern PyQt6 GUI."""
    try:
        from modern_gui import main as modern_main
        logging.info("Launching modern GUI with GPU acceleration")
        modern_main()
    except ImportError as e:
        logging.error(f"Modern GUI not available: {e}")
        logging.info("Falling back to classic GUI")
        run_classic_gui()
    except Exception as e:
        logging.error(f"Failed to launch modern GUI: {e}")
        logging.info("Falling back to classic GUI")
        run_classic_gui()


def run_classic_gui():
    """Launch the classic PyQt6 GUI."""
    from main import main as classic_main
    logging.info("Launching classic GUI")
    classic_main()


def run_benchmarks():
    """Run performance benchmarks."""
    import time
    import torch
    from performance_optimizer import PerformanceOptimizer, benchmark_inference, test_memory_bandwidth
    from detector import ObjectDetector
    from tracker import AdvancedTracker, TrackingMethod
    
    print("=== FunGen VR Performance Benchmarks ===\n")
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    print("Hardware Information:")
    hw = optimizer.hardware_info
    print(f"  GPU: {hw.gpu_name}")
    print(f"  GPU Memory: {hw.gpu_memory_total}MB")
    print(f"  CUDA Available: {hw.cuda_available}")
    print(f"  TensorRT Available: {hw.tensorrt_available}")
    print(f"  CPU Cores: {hw.cpu_cores}")
    print(f"  RAM: {hw.ram_total}MB")
    
    print("\nOptimization Recommendations:")
    for key, value in optimizer.recommendations.items():
        print(f"  {key}: {value}")
    
    # Test memory bandwidth
    if hw.cuda_available:
        print(f"\nGPU Memory Bandwidth: {test_memory_bandwidth():.1f} GB/s")
    
    # Test detector performance
    print("\n=== Object Detection Benchmark ===")
    detector = ObjectDetector(device="cuda" if hw.cuda_available else "cpu")
    
    # Create dummy frame
    import numpy as np
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        detector.detect(frame)
    
    # Benchmark detection
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        detections = detector.detect(frame)
    end_time = time.time()
    
    detection_fps = iterations / (end_time - start_time)
    print(f"Detection FPS: {detection_fps:.1f}")
    print(f"Detection Time: {(end_time - start_time) / iterations * 1000:.2f}ms")
    
    # Test tracker performance
    print("\n=== Tracking Benchmark ===")
    tracker = AdvancedTracker(
        method=TrackingMethod.TEMPLATE_MATCHING,
        use_gpu=hw.cuda_available
    )
    
    # Initialize tracker
    roi = (100, 100, 200, 200)
    tracker.init(frame, roi)
    
    # Benchmark tracking
    start_time = time.time()
    for _ in range(iterations):
        result = tracker.update(frame)
    end_time = time.time()
    
    tracking_fps = iterations / (end_time - start_time)
    print(f"Tracking FPS: {tracking_fps:.1f}")
    print(f"Tracking Time: {(end_time - start_time) / iterations * 1000:.2f}ms")
    
    # Combined pipeline benchmark
    print("\n=== Combined Pipeline Benchmark ===")
    start_time = time.time()
    for _ in range(50):  # Fewer iterations for combined test
        detections = detector.detect(frame)
        if detections:
            # Use first detection as ROI
            box = detections[0].box
            if len(box) == 4:
                result = tracker.update(frame)
    end_time = time.time()
    
    pipeline_fps = 50 / (end_time - start_time)
    print(f"Combined Pipeline FPS: {pipeline_fps:.1f}")
    
    print(f"\n=== Performance Summary ===")
    print(f"Target: 150+ FPS analysis capability")
    print(f"Achieved Detection FPS: {detection_fps:.1f}")
    print(f"Achieved Tracking FPS: {tracking_fps:.1f}")
    print(f"Achieved Pipeline FPS: {pipeline_fps:.1f}")
    
    if pipeline_fps >= 150:
        print("✓ PERFORMANCE TARGET ACHIEVED!")
    elif pipeline_fps >= 100:
        print("⚠ Good performance, close to target")
    else:
        print("✗ Performance below target - consider optimization")
    
    optimizer.cleanup()


def main():
    """Main entry point with command line argument support."""
    parser = argparse.ArgumentParser(
        description="FunGen VR - High-Performance Funscript Generator"
    )
    parser.add_argument(
        "--classic", 
        action="store_true", 
        help="Launch classic GUI interface"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_environment()
    
    if args.benchmark:
        run_benchmarks()
    elif args.classic:
        run_classic_gui()
    else:
        run_modern_gui()


if __name__ == "__main__":
    main()