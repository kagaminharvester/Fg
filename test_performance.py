#!/usr/bin/env python3
"""
test_performance.py
===================

Performance validation script for the enhanced FunGen application.
Tests GPU acceleration, tracking performance, and validates 150+ FPS capability.
"""

import time
import numpy as np
import cv2
import torch
from pathlib import Path

from enhanced_tracker import GPUAcceleratedTracker, TrackingMethod
from enhanced_detector import EnhancedObjectDetector, DetectionBackend
from performance_optimizer import setup_rtx3090_environment, validate_150fps_capability
from live_streaming import DeviceSimulator


def generate_test_frames(width=1280, height=720, num_frames=300):
    """Generate synthetic test frames for performance testing."""
    frames = []
    
    for i in range(num_frames):
        # Create frame with moving object
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some noise
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame += noise
        
        # Add moving rectangle (simulated object)
        x = int((width * 0.3) + (width * 0.4) * np.sin(i * 0.1))
        y = int((height * 0.3) + (height * 0.4) * np.cos(i * 0.1))
        
        cv2.rectangle(frame, (x-50, y-50), (x+50, y+50), (255, 255, 255), -1)
        cv2.rectangle(frame, (x-40, y-40), (x+40, y+40), (0, 128, 255), -1)
        
        frames.append(frame)
        
    return frames


def test_tracking_performance():
    """Test tracking performance with different methods."""
    print("\n🎯 Testing Tracking Performance")
    print("=" * 50)
    
    frames = generate_test_frames(num_frames=150)  # Test with 150 frames for 150 FPS target
    roi = (590, 310, 690, 410)  # Center region
    
    methods = [
        TrackingMethod.TEMPLATE_MATCHING,
        TrackingMethod.OPTICAL_FLOW,
        TrackingMethod.KALMAN_FILTER
    ]
    
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.value}...")
        
        # GPU version
        tracker_gpu = GPUAcceleratedTracker(method=method, use_gpu=True)
        if tracker_gpu.init(frames[0], roi):
            start_time = time.time()
            
            for frame in frames[1:]:
                result = tracker_gpu.update(frame)
                
            gpu_time = time.time() - start_time
            gpu_fps = len(frames) / gpu_time
            
            print(f"  GPU: {gpu_fps:.1f} FPS ({gpu_time*1000:.1f}ms total)")
            results[f"{method.value}_gpu"] = gpu_fps
        else:
            print(f"  GPU: Failed to initialize")
            results[f"{method.value}_gpu"] = 0
        
        # CPU version
        tracker_cpu = GPUAcceleratedTracker(method=method, use_gpu=False)
        if tracker_cpu.init(frames[0], roi):
            start_time = time.time()
            
            for frame in frames[1:]:
                result = tracker_cpu.update(frame)
                
            cpu_time = time.time() - start_time
            cpu_fps = len(frames) / cpu_time
            
            print(f"  CPU: {cpu_fps:.1f} FPS ({cpu_time*1000:.1f}ms total)")
            results[f"{method.value}_cpu"] = cpu_fps
        else:
            print(f"  CPU: Failed to initialize")
            results[f"{method.value}_cpu"] = 0
    
    return results


def test_detection_performance():
    """Test detection performance with different backends."""
    print("\n🔍 Testing Detection Performance")
    print("=" * 50)
    
    frames = generate_test_frames(num_frames=100)
    
    backends = [DetectionBackend.PYTORCH]  # Others require model files
    
    results = {}
    
    for backend in backends:
        print(f"\nTesting {backend.value}...")
        
        detector = EnhancedObjectDetector(
            backend=backend,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if torch.cuda.is_available():
            detector.optimize_for_rtx3090()
        
        start_time = time.time()
        
        for frame in frames:
            detections = detector.detect(frame)
            
        total_time = time.time() - start_time
        fps = len(frames) / total_time
        
        print(f"  {backend.value}: {fps:.1f} FPS ({total_time*1000:.1f}ms total)")
        results[f"{backend.value}"] = fps
        
        # Get average metrics
        avg_metrics = detector.get_average_metrics()
        print(f"  Avg inference time: {avg_metrics.inference_time:.1f}ms")
        print(f"  Avg total time: {avg_metrics.total_time:.1f}ms")
        
    return results


def test_streaming_performance():
    """Test live streaming performance."""
    print("\n📡 Testing Streaming Performance")
    print("=" * 50)
    
    simulator = DeviceSimulator()
    
    # Test connection
    import asyncio
    
    async def test_async():
        await simulator.connect("test")
        
        # Test command throughput
        num_commands = 1000
        start_time = time.time()
        
        for i in range(num_commands):
            from live_streaming import DeviceCommand
            command = DeviceCommand(
                timestamp=int(time.time() * 1000),
                position=int(50 + 25 * np.sin(i * 0.1))
            )
            await simulator.send_command(command)
            
        total_time = time.time() - start_time
        commands_per_sec = num_commands / total_time
        
        print(f"Commands/sec: {commands_per_sec:.1f}")
        print(f"Latency per command: {total_time/num_commands*1000:.2f}ms")
        
        await simulator.disconnect()
        
        return commands_per_sec
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(test_async())
        return result
    finally:
        loop.close()


def test_memory_usage():
    """Test memory usage and optimization."""
    print("\n💾 Testing Memory Usage")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Clear cache first
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Setup optimizer
        optimizer = setup_rtx3090_environment()
        
        after_optimizer = torch.cuda.memory_allocated() / 1024**3
        print(f"After optimization: {after_optimizer:.2f} GB")
        
        # Create some test data
        frames = generate_test_frames(num_frames=50)
        tracker = GPUAcceleratedTracker(use_gpu=True)
        
        tracker.init(frames[0], (100, 100, 200, 200))
        
        for frame in frames[1:]:
            tracker.update(frame)
            
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        final_memory = torch.cuda.memory_allocated() / 1024**3
        
        print(f"Peak GPU memory: {peak_memory:.2f} GB")
        print(f"Final GPU memory: {final_memory:.2f} GB")
        
        # Memory efficiency
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        efficiency = (peak_memory / total_gpu_memory) * 100
        
        print(f"Memory efficiency: {efficiency:.1f}% of {total_gpu_memory:.0f}GB")
        
        return {
            'peak_memory_gb': peak_memory,
            'efficiency_percent': efficiency,
            'total_memory_gb': total_gpu_memory
        }
    else:
        print("CUDA not available - skipping GPU memory test")
        return {}


def generate_performance_report(results):
    """Generate comprehensive performance report."""
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 60)
    
    # System info
    print("\n🖥️ System Information:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"  GPU: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  GPU: Not Available")
    
    print(f"  PyTorch Version: {torch.__version__}")
    
    # Performance summary
    print("\n⚡ Performance Summary:")
    
    tracking_results = results.get('tracking', {})
    best_tracking_fps = max(tracking_results.values()) if tracking_results else 0
    print(f"  Best Tracking FPS: {best_tracking_fps:.1f}")
    
    detection_results = results.get('detection', {})
    best_detection_fps = max(detection_results.values()) if detection_results else 0
    print(f"  Best Detection FPS: {best_detection_fps:.1f}")
    
    streaming_fps = results.get('streaming', 0)
    print(f"  Streaming Commands/sec: {streaming_fps:.1f}")
    
    # 150 FPS Assessment
    print("\n🎯 150+ FPS Assessment:")
    can_achieve_150fps = best_tracking_fps >= 150
    
    if can_achieve_150fps:
        print("  ✅ System CAN achieve 150+ FPS for analysis")
        print("  🚀 Ready for real-time processing!")
    else:
        print("  ⚠️  System may not consistently achieve 150+ FPS")
        print(f"  📈 Current best: {best_tracking_fps:.1f} FPS")
        
        if best_tracking_fps >= 100:
            print("  💡 Still excellent for real-time applications")
        elif best_tracking_fps >= 60:
            print("  💡 Good for standard real-time applications")
        else:
            print("  💡 Consider optimizations or hardware upgrade")
    
    # Memory assessment
    memory_results = results.get('memory', {})
    if memory_results:
        print(f"\n💾 Memory Usage:")
        print(f"  Peak Usage: {memory_results.get('peak_memory_gb', 0):.2f} GB")
        print(f"  Efficiency: {memory_results.get('efficiency_percent', 0):.1f}%")
        
        if memory_results.get('efficiency_percent', 0) < 50:
            print("  ✅ Efficient memory usage")
        else:
            print("  ⚠️  High memory usage - consider optimization")
    
    print("\n" + "=" * 60)


def main():
    """Run complete performance test suite."""
    print("🚀 Enhanced FunGen Performance Test Suite")
    print("=" * 60)
    
    # Validate system capability
    validate_150fps_capability()
    
    # Run tests
    results = {}
    
    try:
        results['tracking'] = test_tracking_performance()
    except Exception as e:
        print(f"❌ Tracking test failed: {e}")
        results['tracking'] = {}
    
    try:
        results['detection'] = test_detection_performance()
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        results['detection'] = {}
    
    try:
        results['streaming'] = test_streaming_performance()
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        results['streaming'] = 0
    
    try:
        results['memory'] = test_memory_usage()
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        results['memory'] = {}
    
    # Generate report
    generate_performance_report(results)
    
    print("\n✅ Performance testing complete!")


if __name__ == "__main__":
    main()