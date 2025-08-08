#!/usr/bin/env python3
"""
test_enhanced_fungen.py
=======================

Test script for the enhanced FunGen application.
Creates a simple test interface to verify functionality.
"""

import sys
import os
import numpy as np
import cv2
from PyQt6 import QtWidgets, QtCore, QtGui

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_enhanced import EnhancedMainWindow


def create_test_video(path: str, duration: int = 10, fps: int = 30):
    """Create a simple test video for demonstration."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    
    frames = duration * fps
    for i in range(frames):
        # Create frame with moving rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(480):
            frame[y, :] = [y//2, (y//3) % 255, 128]
        
        # Moving object
        t = i / frames
        x = int(320 + 200 * np.sin(t * 4 * np.pi))
        y = int(240 + 100 * np.cos(t * 6 * np.pi))
        
        cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {path}")


def main():
    # Create QApplication
    app = QtWidgets.QApplication(sys.argv)
    
    # Set up test environment
    test_dir = "/tmp/fungen_test"
    os.makedirs(test_dir, exist_ok=True)
    
    test_video_path = os.path.join(test_dir, "test_video.mp4")
    
    # Create test video if it doesn't exist
    if not os.path.exists(test_video_path):
        print("Creating test video...")
        create_test_video(test_video_path, duration=5, fps=30)
    
    # Create and show main window
    window = EnhancedMainWindow()
    
    # Add some helpful information
    info_text = f"""
Enhanced FunGen Test Application

Test video created at: {test_video_path}

Instructions:
1. Click 'Open Video' and select the test video
2. Draw an ROI around the moving green circle
3. Click 'Start Real-time' to begin processing
4. Watch the live plot update in real-time
5. Adjust parameters to see live changes

Performance Notes:
- Target: 150 FPS analysis
- GPU: {'Available' if window.processor.use_gpu else 'Not Available (CPU mode)'}
- Real-time processing with live preview
"""
    
    # Show info dialog
    QtWidgets.QMessageBox.information(window, "Enhanced FunGen Test", info_text)
    
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())