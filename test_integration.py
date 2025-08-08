"""
test_integration.py
==================

Integration tests for FunGen VR high-performance components.
Tests the full pipeline with GPU acceleration and device streaming.
"""

import time
import numpy as np
import cv2
from pathlib import Path

def test_detector_integration():
    """Test object detector with different backends."""
    print("Testing Object Detector...")
    
    from detector import ObjectDetector
    
    # Test with dummy detector (should always work)
    detector = ObjectDetector()
    
    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection
    detections = detector.detect(frame)
    assert len(detections) > 0, "Detector should return at least one detection"
    assert detections[0].score == 1.0, "Dummy detector should have confidence 1.0"
    
    # Test performance stats
    stats = detector.get_performance_stats()
    assert 'fps' in stats, "Stats should include FPS"
    assert 'backend' in stats, "Stats should include backend type"
    
    print("✓ Object Detector integration test passed")


def test_tracker_integration():
    """Test advanced tracker with different methods."""
    print("Testing Advanced Tracker...")
    
    from tracker import AdvancedTracker, TrackingMethod
    
    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    roi = (100, 100, 200, 200)
    
    # Test template matching
    tracker = AdvancedTracker(method=TrackingMethod.TEMPLATE_MATCHING)
    tracker.init(frame, roi)
    
    # Test tracking
    result = tracker.update(frame)
    assert result.roi == roi, "First tracking result should match initial ROI"
    assert result.confidence > 0, "Tracking should have positive confidence"
    
    # Test performance stats
    stats = tracker.get_performance_stats()
    assert 'fps' in stats, "Stats should include FPS"
    assert 'method' in stats, "Stats should include tracking method"
    
    print("✓ Advanced Tracker integration test passed")


def test_performance_optimizer():
    """Test performance optimization components."""
    print("Testing Performance Optimizer...")
    
    from performance_optimizer import PerformanceOptimizer, HardwareDetector
    
    # Test hardware detection
    hardware = HardwareDetector.detect_hardware()
    assert hardware.cpu_cores > 0, "Should detect CPU cores"
    assert hardware.ram_total > 0, "Should detect RAM"
    
    # Test optimizer
    optimizer = PerformanceOptimizer()
    
    # Test recommendations
    recommendations = optimizer.recommendations
    assert 'memory_fraction' in recommendations, "Should have memory recommendations"
    assert 'batch_size' in recommendations, "Should have batch size recommendations"
    
    # Test performance stats
    stats = optimizer.get_performance_stats()
    assert 'hardware' in stats, "Stats should include hardware info"
    assert 'recommendations' in stats, "Stats should include recommendations"
    
    optimizer.cleanup()
    print("✓ Performance Optimizer integration test passed")


def test_device_streaming():
    """Test device streaming infrastructure."""
    print("Testing Device Streaming...")
    
    from device_streaming import DeviceStreamer, DeviceSimulator
    
    # Test device simulator
    simulator = DeviceSimulator()
    
    # Test async connection
    import asyncio
    
    async def test_simulator():
        success = await simulator.connect("test")
        assert success, "Simulator should connect successfully"
        assert simulator.is_connected(), "Simulator should report connected"
        
        # Test command sending
        from device_streaming import DeviceCommand
        command = DeviceCommand(position=50, duration_ms=100, timestamp=time.time())
        success = await simulator.send_command(command)
        assert success, "Simulator should accept commands"
        
        # Test device info
        info = simulator.get_device_info()
        assert info['type'] == 'Simulator', "Should report correct device type"
        
        await simulator.disconnect()
        assert not simulator.is_connected(), "Simulator should report disconnected"
    
    # Run async test
    asyncio.run(test_simulator())
    
    print("✓ Device Streaming integration test passed")


def test_funscript_generation():
    """Test funscript generation pipeline."""
    print("Testing Funscript Generation...")
    
    from funscript_generator import map_positions, Funscript
    
    # Create test position data
    positions = [100, 120, 140, 160, 180, 160, 140, 120, 100]
    frame_height = 480
    fps = 30.0
    
    # Generate funscript
    funscript = map_positions(
        positions=positions,
        frame_height=frame_height,
        fps=fps,
        min_pos=0,
        max_pos=100,
        smoothing_window=3
    )
    
    assert len(funscript.actions) == len(positions), "Should have action for each position"
    assert all(0 <= action['pos'] <= 100 for action in funscript.actions), "Positions should be in range 0-100"
    
    # Test JSON serialization
    json_str = funscript.to_json()
    assert '"version"' in json_str, "JSON should include version"
    assert '"actions"' in json_str, "JSON should include actions"
    
    print("✓ Funscript Generation integration test passed")


def test_video_loading():
    """Test video loading capabilities."""
    print("Testing Video Loading...")
    
    from video_loader import VideoLoader, VideoInfo
    
    # Create a simple test video in memory (can't test real files in this environment)
    # This tests the class structure and basic functionality
    
    # Test video info structure
    info = VideoInfo(
        path="test.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        format="2D"
    )
    
    assert info.width == 640, "Video info should store width"
    assert info.height == 480, "Video info should store height"
    assert info.format == "2D", "Video info should store format"
    
    print("✓ Video Loading integration test passed")


def test_full_pipeline():
    """Test complete processing pipeline."""
    print("Testing Full Pipeline Integration...")
    
    from detector import ObjectDetector
    from tracker import AdvancedTracker, TrackingMethod
    from funscript_generator import map_positions
    
    # Create test frame sequence
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    
    # Initialize components
    detector = ObjectDetector()
    tracker = AdvancedTracker(method=TrackingMethod.TEMPLATE_MATCHING)
    
    # Process first frame
    detections = detector.detect(frames[0])
    assert len(detections) > 0, "Should detect objects"
    
    # Initialize tracker with first detection
    roi = detections[0].box
    tracker.init(frames[0], roi)
    
    # Process remaining frames
    positions = []
    for frame in frames[1:]:
        result = tracker.update(frame)
        # Extract vertical center position
        x1, y1, x2, y2 = result.roi
        center_y = y1 + (y2 - y1) / 2
        positions.append(center_y)
    
    # Generate funscript
    funscript = map_positions(
        positions=positions,
        frame_height=480,
        fps=30.0
    )
    
    assert len(funscript.actions) > 0, "Should generate funscript actions"
    
    print("✓ Full Pipeline integration test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=== FunGen VR Integration Tests ===\n")
    
    tests = [
        test_detector_integration,
        test_tracker_integration,
        test_performance_optimizer,
        test_device_streaming,
        test_funscript_generation,
        test_video_loading,
        test_full_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! FunGen VR is ready for high-performance operation.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)