"""
main_enhanced.py
===============

Enhanced entry point for the improved FunGen application optimized for RTX 3090.
This script provides real-time processing, live preview, and 150+ fps capabilities.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from video_loader import VideoLoader
from tracker import SimpleTracker
from funscript_generator import map_positions, Funscript
from roi_selector import ROISelector
from realtime_processor import RealtimeProcessor, ProcessingStats


class PerformanceWidget(QtWidgets.QWidget):
    """Widget displaying real-time performance statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # FPS display
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        layout.addWidget(self.fps_label)
        
        # Progress bar for target FPS
        self.fps_progress = QtWidgets.QProgressBar()
        self.fps_progress.setRange(0, 150)
        self.fps_progress.setFormat("Target: 150 FPS (%p%)")
        layout.addWidget(self.fps_progress)
        
        # Processing stats
        self.stats_label = QtWidgets.QLabel("Processing: Ready")
        layout.addWidget(self.stats_label)
        
        # GPU usage (if available)
        self.gpu_label = QtWidgets.QLabel("GPU: N/A")
        layout.addWidget(self.gpu_label)
        
        # Frames processed
        self.frames_label = QtWidgets.QLabel("Frames: 0")
        layout.addWidget(self.frames_label)
        
    def update_stats(self, stats: ProcessingStats):
        """Update performance display with new statistics."""
        self.fps_label.setText(f"FPS: {stats.fps:.1f}")
        self.fps_progress.setValue(int(stats.fps))
        
        # Color coding for FPS
        if stats.fps >= 120:
            color = "green"
        elif stats.fps >= 60:
            color = "orange"
        else:
            color = "red"
        self.fps_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        
        self.stats_label.setText(
            f"Detection: {stats.detection_time_ms:.1f}ms | "
            f"Tracking: {stats.tracking_time_ms:.1f}ms"
        )
        
        if stats.gpu_memory_mb > 0:
            self.gpu_label.setText(f"GPU Memory: {stats.gpu_memory_mb:.1f} MB")
        
        self.frames_label.setText(f"Frames: {stats.frames_processed}")


class LivePlotWidget(QtWidgets.QWidget):
    """Enhanced plot widget with live updates and simulation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.positions = []
        self.timestamps = []
        self.simulation_pos = 0
        self.simulation_timer = QtCore.QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Matplotlib canvas
        self.figure, (self.ax_main, self.ax_sim) = plt.subplots(2, 1, figsize=(8, 4), height_ratios=[3, 1])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Simulation controls
        controls_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("▶ Play Simulation")
        self.play_btn.clicked.connect(self.toggle_simulation)
        self.sim_speed_spin = QtWidgets.QDoubleSpinBox()
        self.sim_speed_spin.setRange(0.1, 5.0)
        self.sim_speed_spin.setValue(1.0)
        self.sim_speed_spin.setSuffix("x")
        controls_layout.addWidget(QtWidgets.QLabel("Simulation:"))
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(QtWidgets.QLabel("Speed:"))
        controls_layout.addWidget(self.sim_speed_spin)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize the plot axes."""
        # Main funscript plot
        self.ax_main.set_xlabel("Time (s)")
        self.ax_main.set_ylabel("Position (0-100)")
        self.ax_main.set_ylim(0, 100)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title("Live Funscript Generation")
        
        # Simulation plot (position over time with current indicator)
        self.ax_sim.set_xlabel("Time (s)")
        self.ax_sim.set_ylabel("Pos")
        self.ax_sim.set_ylim(0, 100)
        self.ax_sim.set_title("Simulation")
        
        self.canvas.draw()
        
    def add_position(self, timestamp: float, position: float):
        """Add a new position point to the live plot."""
        self.timestamps.append(timestamp)
        self.positions.append(position)
        
        # Keep only last 1000 points for performance
        if len(self.positions) > 1000:
            self.timestamps = self.timestamps[-1000:]
            self.positions = self.positions[-1000:]
        
        self.update_main_plot()
        
    def update_main_plot(self):
        """Update the main funscript plot."""
        if not self.positions:
            return
            
        self.ax_main.clear()
        self.ax_main.plot(self.timestamps, self.positions, 'b-', linewidth=2, alpha=0.8)
        self.ax_main.set_xlabel("Time (s)")
        self.ax_main.set_ylabel("Position (0-100)")
        self.ax_main.set_ylim(0, 100)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title(f"Live Funscript Generation ({len(self.positions)} points)")
        
        # Highlight recent points
        if len(self.positions) > 10:
            recent_x = self.timestamps[-10:]
            recent_y = self.positions[-10:]
            self.ax_main.plot(recent_x, recent_y, 'r-', linewidth=3, alpha=0.8)
        
        self.canvas.draw_idle()
        
    def update_simulation_plot(self, funscript: Funscript):
        """Update the simulation plot with full funscript."""
        if not funscript.actions:
            return
            
        times = [act["at"] / 1000.0 for act in funscript.actions]
        positions = [act["pos"] for act in funscript.actions]
        
        self.ax_sim.clear()
        self.ax_sim.plot(times, positions, 'g-', linewidth=2)
        self.ax_sim.set_xlabel("Time (s)")
        self.ax_sim.set_ylabel("Pos")
        self.ax_sim.set_ylim(0, 100)
        self.ax_sim.set_title("Simulation Preview")
        
        self.canvas.draw_idle()
        
    def toggle_simulation(self):
        """Start/stop simulation playback."""
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.play_btn.setText("▶ Play Simulation")
        else:
            if self.positions:
                self.simulation_pos = 0
                self.simulation_timer.start(50)  # 20 FPS simulation
                self.play_btn.setText("⏸ Pause Simulation")
                
    def update_simulation(self):
        """Update simulation position indicator."""
        if not self.timestamps:
            return
            
        speed = self.sim_speed_spin.value()
        self.simulation_pos += 0.05 * speed  # 50ms * speed
        
        if self.simulation_pos >= self.timestamps[-1]:
            self.simulation_pos = 0
            
        # Find closest position
        closest_idx = 0
        for i, t in enumerate(self.timestamps):
            if t <= self.simulation_pos:
                closest_idx = i
            else:
                break
                
        # Update simulation plot with position indicator
        self.ax_sim.clear()
        if len(self.timestamps) > 1:
            self.ax_sim.plot(self.timestamps, self.positions, 'g-', linewidth=2)
            if closest_idx < len(self.positions):
                current_pos = self.positions[closest_idx]
                self.ax_sim.axvline(x=self.simulation_pos, color='red', linewidth=2, alpha=0.8)
                self.ax_sim.plot(self.simulation_pos, current_pos, 'ro', markersize=8)
                
        self.ax_sim.set_xlabel("Time (s)")
        self.ax_sim.set_ylabel("Pos")
        self.ax_sim.set_ylim(0, 100)
        self.ax_sim.set_title(f"Simulation @ {self.simulation_pos:.1f}s")
        
        self.canvas.draw_idle()


class EnhancedMainWindow(QtWidgets.QMainWindow):
    """Enhanced main window with real-time processing capabilities."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FunGen VR - RTX 3090 Optimized")
        self.resize(1600, 1000)
        
        # Processing components
        self.processor = RealtimeProcessor(target_fps=150.0)
        self.current_frame: Optional[np.ndarray] = None
        self.processing_active = False
        
        # UI state
        self.video_path: Optional[str] = None
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        
        # Main layout: Left (video + controls) | Right (plots + performance)
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # Left panel
        left_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=2)
        
        # Video display with enhanced ROI selector
        self.video_widget = ROISelector()
        self.video_widget.setMinimumSize(800, 450)
        left_panel.addWidget(self.video_widget, stretch=3)
        
        # File controls
        file_group = QtWidgets.QGroupBox("File Controls")
        file_layout = QtWidgets.QHBoxLayout(file_group)
        
        self.open_btn = QtWidgets.QPushButton("📁 Open Video")
        self.open_btn.clicked.connect(self.open_video)
        
        self.start_btn = QtWidgets.QPushButton("🚀 Start Real-time")
        self.start_btn.clicked.connect(self.start_realtime_processing)
        self.start_btn.setEnabled(False)
        
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        self.save_btn = QtWidgets.QPushButton("💾 Save Funscript")
        self.save_btn.clicked.connect(self.save_funscript)
        
        file_layout.addWidget(self.open_btn)
        file_layout.addWidget(self.start_btn)
        file_layout.addWidget(self.stop_btn)
        file_layout.addWidget(self.save_btn)
        
        left_panel.addWidget(file_group)
        
        # Parameter controls
        self.setup_parameter_controls(left_panel)
        
        # Right panel
        right_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=3)
        
        # Performance monitoring
        perf_group = QtWidgets.QGroupBox("Performance Monitor")
        perf_layout = QtWidgets.QVBoxLayout(perf_group)
        self.performance_widget = PerformanceWidget()
        perf_layout.addWidget(self.performance_widget)
        right_panel.addWidget(perf_group)
        
        # Live plot
        plot_group = QtWidgets.QGroupBox("Live Funscript Generation & Simulation")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.plot_widget = LivePlotWidget()
        plot_layout.addWidget(self.plot_widget)
        right_panel.addWidget(plot_group, stretch=2)
        
        # Processing options
        options_group = QtWidgets.QGroupBox("Processing Options")
        options_layout = QtWidgets.QFormLayout(options_group)
        
        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(0, 10000)
        self.max_frames_spin.setValue(0)  # 0 = unlimited
        self.max_frames_spin.setSpecialValueText("Unlimited")
        
        self.use_gpu_check = QtWidgets.QCheckBox()
        self.use_gpu_check.setChecked(True)
        
        self.target_fps_spin = QtWidgets.QSpinBox()
        self.target_fps_spin.setRange(30, 300)
        self.target_fps_spin.setValue(150)
        
        options_layout.addRow("Max Frames:", self.max_frames_spin)
        options_layout.addRow("Use GPU:", self.use_gpu_check)
        options_layout.addRow("Target FPS:", self.target_fps_spin)
        
        right_panel.addWidget(options_group)
        
    def setup_parameter_controls(self, parent_layout):
        """Setup parameter control panel."""
        param_group = QtWidgets.QGroupBox("Funscript Parameters")
        param_layout = QtWidgets.QFormLayout(param_group)
        
        # Position range
        self.min_pos_spin = QtWidgets.QSpinBox()
        self.min_pos_spin.setRange(0, 100)
        self.min_pos_spin.setValue(0)
        
        self.max_pos_spin = QtWidgets.QSpinBox()
        self.max_pos_spin.setRange(0, 100)
        self.max_pos_spin.setValue(100)
        
        param_layout.addRow("Min Position:", self.min_pos_spin)
        param_layout.addRow("Max Position:", self.max_pos_spin)
        
        # Boosting
        self.boost_up_spin = QtWidgets.QDoubleSpinBox()
        self.boost_up_spin.setRange(0.0, 1.0)
        self.boost_up_spin.setSingleStep(0.05)
        self.boost_up_spin.setValue(0.0)
        
        self.boost_down_spin = QtWidgets.QDoubleSpinBox()
        self.boost_down_spin.setRange(-1.0, 1.0)
        self.boost_down_spin.setSingleStep(0.05)
        self.boost_down_spin.setValue(0.0)
        
        param_layout.addRow("Boost Up %:", self.boost_up_spin)
        param_layout.addRow("Boost Down %:", self.boost_down_spin)
        
        # Thresholding
        self.thresh_low_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_low_spin.setRange(0.0, 1.0)
        self.thresh_low_spin.setSingleStep(0.05)
        self.thresh_low_spin.setValue(0.0)
        
        self.thresh_high_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_high_spin.setRange(0.0, 1.0)
        self.thresh_high_spin.setSingleStep(0.05)
        self.thresh_high_spin.setValue(1.0)
        
        param_layout.addRow("Threshold Low:", self.thresh_low_spin)
        param_layout.addRow("Threshold High:", self.thresh_high_spin)
        
        # Smoothing and randomness
        self.smooth_spin = QtWidgets.QSpinBox()
        self.smooth_spin.setRange(1, 51)
        self.smooth_spin.setValue(5)
        
        self.random_spin = QtWidgets.QDoubleSpinBox()
        self.random_spin.setRange(0.0, 0.5)
        self.random_spin.setSingleStep(0.01)
        self.random_spin.setValue(0.0)
        
        param_layout.addRow("Smoothing Window:", self.smooth_spin)
        param_layout.addRow("Randomness:", self.random_spin)
        
        parent_layout.addWidget(param_group)
        
        # Connect parameter changes for live updates
        for widget in [self.min_pos_spin, self.max_pos_spin, self.boost_up_spin,
                      self.boost_down_spin, self.thresh_low_spin, self.thresh_high_spin,
                      self.smooth_spin, self.random_spin]:
            widget.valueChanged.connect(self.update_live_funscript)
            
    def connect_signals(self):
        """Connect processor signals to UI updates."""
        self.video_widget.roiSelected.connect(self.on_roi_selected)
        self.processor.frameProcessed.connect(self.on_frame_processed)
        self.processor.positionUpdated.connect(self.on_position_updated)
        self.processor.statsUpdated.connect(self.on_stats_updated)
        self.processor.processingComplete.connect(self.on_processing_complete)
        self.processor.errorOccurred.connect(self.on_error)
        
    def frame_to_qimage(self, frame: np.ndarray) -> QtGui.QImage:
        """Convert BGR frame to QImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        
    def open_video(self):
        """Open video file dialog."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", os.getcwd(), 
            "Videos (*.mp4 *.mkv *.mov *.avi *.webm)"
        )
        
        if not path:
            return
            
        self.video_path = path
        
        # Load video in processor
        if self.processor.set_video(path):
            # Show first frame
            loader = VideoLoader(path, target_width=800, device=0)
            try:
                for idx, frame in loader:
                    if isinstance(frame, tuple):
                        frame = frame[0]
                    self.current_frame = frame
                    qimg = self.frame_to_qimage(frame)
                    self.video_widget.setImage(qimg)
                    break
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load video: {e}")
                return
            finally:
                loader.release()
                
            self.start_btn.setEnabled(True)
            self.setWindowTitle(f"FunGen VR - {os.path.basename(path)}")
        
    def on_roi_selected(self, x1: int, y1: int, x2: int, y2: int):
        """Handle ROI selection."""
        self.processor.set_roi(x1, y1, x2, y2)
        
    def start_realtime_processing(self):
        """Start real-time video processing."""
        if not self.video_path or not self.processor.current_roi:
            QtWidgets.QMessageBox.information(
                self, "Info", "Please load a video and select an ROI first."
            )
            return
            
        # Update processor settings
        self.processor.target_fps = self.target_fps_spin.value()
        
        # Clear previous data
        self.plot_widget.positions.clear()
        self.plot_widget.timestamps.clear()
        self.plot_widget.setup_plots()
        
        # Start processing
        max_frames = self.max_frames_spin.value() if self.max_frames_spin.value() > 0 else None
        self.processor.start_processing(max_frames)
        
        # Update UI state
        self.processing_active = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        
    def stop_processing(self):
        """Stop processing."""
        self.processor.stop_processing()
        self.processing_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        
    def on_frame_processed(self, frame_idx: int, frame: np.ndarray, detections: list):
        """Handle processed frame update."""
        # Update video display
        display_frame = frame.copy()
        
        # Draw ROI if active
        if self.processor.current_roi:
            x1, y1, x2, y2 = self.processor.current_roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"{detection.label} ({detection.score:.2f})",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        qimg = self.frame_to_qimage(display_frame)
        self.video_widget.setImage(qimg)
        
    def on_position_updated(self, timestamp: float, position: float):
        """Handle position update from tracker."""
        # Add to live plot
        self.plot_widget.add_position(timestamp, position)
        
    def on_stats_updated(self, stats: ProcessingStats):
        """Handle performance statistics update."""
        self.performance_widget.update_stats(stats)
        
    def on_processing_complete(self):
        """Handle processing completion."""
        self.stop_processing()
        QtWidgets.QMessageBox.information(self, "Complete", "Processing finished!")
        
        # Update simulation plot with final funscript
        funscript = self.processor.get_current_funscript(**self.get_funscript_params())
        if funscript:
            self.plot_widget.update_simulation_plot(funscript)
        
    def on_error(self, error_msg: str):
        """Handle processing error."""
        QtWidgets.QMessageBox.critical(self, "Error", error_msg)
        self.stop_processing()
        
    def get_funscript_params(self) -> dict:
        """Get current funscript generation parameters."""
        return {
            'min_pos': self.min_pos_spin.value(),
            'max_pos': self.max_pos_spin.value(),
            'boost_up_percent': self.boost_up_spin.value(),
            'boost_down_percent': self.boost_down_spin.value(),
            'threshold_low': self.thresh_low_spin.value(),
            'threshold_high': self.thresh_high_spin.value(),
            'smoothing_window': self.smooth_spin.value(),
            'randomness': self.random_spin.value(),
        }
        
    def update_live_funscript(self):
        """Update funscript with current parameters during processing."""
        if not self.processing_active:
            return
            
        funscript = self.processor.get_current_funscript(**self.get_funscript_params())
        if funscript:
            self.plot_widget.update_simulation_plot(funscript)
            
    def save_funscript(self):
        """Save current funscript."""
        funscript = self.processor.get_current_funscript(**self.get_funscript_params())
        if not funscript or not funscript.actions:
            QtWidgets.QMessageBox.information(self, "Info", "No funscript data to save.")
            return
            
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Funscript", os.getcwd(), "Funscript (*.funscript *.json)"
        )
        
        if path:
            funscript.save(path)
            QtWidgets.QMessageBox.information(self, "Saved", f"Funscript saved to {path}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Dark theme
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(0, 0, 0))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(255, 255, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(dark_palette)
    
    window = EnhancedMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()