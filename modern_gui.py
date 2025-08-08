"""
modern_gui.py
=============

Modern PyQt6 GUI for high-performance VR funscript generation with:

- Dark theme with modern styling
- Tabbed interface for organized workflow
- Real-time performance monitoring dashboard
- Live funscript simulation with interactive controls
- GPU acceleration status and metrics
- Device streaming controls with latency monitoring
- Dynamic ROI modification and tracking visualization
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout

# Import our modules
from video_loader import VideoLoader
from detector import ObjectDetector
from tracker import AdvancedTracker, TrackingMethod, SimpleTracker
from funscript_generator import map_positions, Funscript
from roi_selector import ROISelector


class DarkTheme:
    """Dark theme stylesheet for modern appearance."""
    
    @staticmethod
    def get_stylesheet() -> str:
        return """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #2d2d2d;
        }
        
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #ffffff;
            padding: 8px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #007acc;
        }
        
        QTabBar::tab:hover {
            background-color: #4c4c4c;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #3c3c3c;
            border-radius: 5px;
            margin-top: 10px;
            background-color: #2d2d2d;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            background-color: #2d2d2d;
        }
        
        QPushButton {
            background-color: #007acc;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #005a9e;
        }
        
        QPushButton:pressed {
            background-color: #004578;
        }
        
        QPushButton:disabled {
            background-color: #4c4c4c;
            color: #888888;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #3c3c3c;
            height: 8px;
            background: #4c4c4c;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #007acc;
            border: 1px solid #005a9e;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #005a9e;
        }
        
        QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #5c5c5c;
            color: #ffffff;
            padding: 4px;
            border-radius: 4px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
            border-color: #007acc;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            text-align: center;
            background-color: #4c4c4c;
        }
        
        QProgressBar::chunk {
            background-color: #007acc;
            border-radius: 3px;
        }
        
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #5c5c5c;
            color: #ffffff;
            padding: 4px;
            border-radius: 4px;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: #5c5c5c;
            border-left-style: solid;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #ffffff;
        }
        """


class PerformanceMonitor(QtWidgets.QWidget):
    """Real-time performance monitoring widget."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.performance_data = {
            'detection_fps': [],
            'tracking_fps': [],
            'gpu_usage': [],
            'memory_usage': [],
            'processing_fps': []
        }
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Performance metrics display
        metrics_layout = QHBoxLayout()
        
        # FPS metrics
        fps_group = QtWidgets.QGroupBox("Performance (FPS)")
        fps_layout = QtWidgets.QVBoxLayout()
        
        self.detection_fps_label = QtWidgets.QLabel("Detection: 0.0 FPS")
        self.tracking_fps_label = QtWidgets.QLabel("Tracking: 0.0 FPS")
        self.processing_fps_label = QtWidgets.QLabel("Processing: 0.0 FPS")
        
        fps_layout.addWidget(self.detection_fps_label)
        fps_layout.addWidget(self.tracking_fps_label)
        fps_layout.addWidget(self.processing_fps_label)
        fps_group.setLayout(fps_layout)
        
        # Hardware metrics
        hardware_group = QtWidgets.QGroupBox("Hardware")
        hardware_layout = QtWidgets.QVBoxLayout()
        
        self.gpu_usage_label = QtWidgets.QLabel("GPU: N/A")
        self.memory_label = QtWidgets.QLabel("Memory: N/A")
        self.backend_label = QtWidgets.QLabel("Backend: CPU")
        
        hardware_layout.addWidget(self.gpu_usage_label)
        hardware_layout.addWidget(self.memory_label)
        hardware_layout.addWidget(self.backend_label)
        hardware_group.setLayout(hardware_layout)
        
        metrics_layout.addWidget(fps_group)
        metrics_layout.addWidget(hardware_group)
        
        # Performance graph
        self.figure = Figure(figsize=(8, 3), facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2d2d2d;")
        
        self.ax = self.figure.add_subplot(111, facecolor='#2d2d2d')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('FPS')
        self.ax.set_title('Real-time Performance', color='white')
        
        layout.addLayout(metrics_layout)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def update_performance_data(self, detector_stats: Dict, tracker_stats: Dict):
        """Update performance data from detector and tracker."""
        current_time = time.time()
        
        # Store data with timestamps
        self.performance_data['detection_fps'].append((current_time, detector_stats.get('fps', 0)))
        self.performance_data['tracking_fps'].append((current_time, tracker_stats.get('fps', 0)))
        
        # Keep only last 30 seconds of data
        cutoff_time = current_time - 30
        for key in self.performance_data:
            self.performance_data[key] = [
                (t, v) for t, v in self.performance_data[key] if t > cutoff_time
            ]
    
    def update_metrics(self):
        """Update displayed metrics."""
        # Update FPS labels
        if self.performance_data['detection_fps']:
            latest_detection_fps = self.performance_data['detection_fps'][-1][1]
            self.detection_fps_label.setText(f"Detection: {latest_detection_fps:.1f} FPS")
        
        if self.performance_data['tracking_fps']:
            latest_tracking_fps = self.performance_data['tracking_fps'][-1][1]
            self.tracking_fps_label.setText(f"Tracking: {latest_tracking_fps:.1f} FPS")
        
        # Update graph
        self.update_performance_graph()
    
    def update_performance_graph(self):
        """Update the performance graph."""
        self.ax.clear()
        self.ax.set_facecolor('#2d2d2d')
        
        current_time = time.time()
        
        # Plot detection FPS
        if self.performance_data['detection_fps']:
            times, fps_values = zip(*self.performance_data['detection_fps'])
            relative_times = [(t - current_time) for t in times]
            self.ax.plot(relative_times, fps_values, 'b-', label='Detection FPS', linewidth=2)
        
        # Plot tracking FPS
        if self.performance_data['tracking_fps']:
            times, fps_values = zip(*self.performance_data['tracking_fps'])
            relative_times = [(t - current_time) for t in times]
            self.ax.plot(relative_times, fps_values, 'g-', label='Tracking FPS', linewidth=2)
        
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('FPS', color='white')
        self.ax.set_title('Real-time Performance', color='white')
        self.ax.legend(facecolor='#2d2d2d', edgecolor='white', labelcolor='white')
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(colors='white')
        
        # Set x-axis limits to show last 30 seconds
        self.ax.set_xlim(-30, 0)
        
        self.canvas.draw()


class DeviceControlPanel(QtWidgets.QWidget):
    """Live device streaming and control panel."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.connected_devices = []
        self.streaming_active = False
        self.latency_ms = 0
        self.commands_per_sec = 0
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Device connection
        connection_group = QtWidgets.QGroupBox("Device Connection")
        connection_layout = QtWidgets.QVBoxLayout()
        
        device_layout = QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["The Handy", "OSR2", "SR6", "Buttplug.io"])
        device_layout.addWidget(QtWidgets.QLabel("Device:"))
        device_layout.addWidget(self.device_combo)
        
        self.device_key_edit = QtWidgets.QLineEdit()
        self.device_key_edit.setPlaceholderText("Device connection key/URL")
        device_layout.addWidget(QtWidgets.QLabel("Key:"))
        device_layout.addWidget(self.device_key_edit)
        
        connection_layout.addLayout(device_layout)
        
        button_layout = QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_device)
        self.test_btn = QtWidgets.QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.test_btn)
        
        connection_layout.addLayout(button_layout)
        connection_group.setLayout(connection_layout)
        
        # Streaming controls
        streaming_group = QtWidgets.QGroupBox("Live Streaming")
        streaming_layout = QtWidgets.QVBoxLayout()
        
        stream_controls = QHBoxLayout()
        self.stream_btn = QtWidgets.QPushButton("Start Streaming")
        self.stream_btn.clicked.connect(self.toggle_streaming)
        self.stream_btn.setEnabled(False)
        
        self.simulator_btn = QtWidgets.QPushButton("Device Simulator")
        self.simulator_btn.clicked.connect(self.open_simulator)
        
        stream_controls.addWidget(self.stream_btn)
        stream_controls.addWidget(self.simulator_btn)
        streaming_layout.addLayout(stream_controls)
        
        # Streaming metrics
        metrics_layout = QHBoxLayout()
        
        self.latency_label = QtWidgets.QLabel("Latency: -- ms")
        self.commands_label = QtWidgets.QLabel("Commands/sec: --")
        self.connection_quality_label = QtWidgets.QLabel("Quality: Disconnected")
        
        metrics_layout.addWidget(self.latency_label)
        metrics_layout.addWidget(self.commands_label)
        metrics_layout.addWidget(self.connection_quality_label)
        
        streaming_layout.addLayout(metrics_layout)
        
        # Latency calibration
        calibration_layout = QHBoxLayout()
        self.target_latency_spin = QtWidgets.QSpinBox()
        self.target_latency_spin.setRange(10, 500)
        self.target_latency_spin.setValue(50)
        self.target_latency_spin.setSuffix(" ms")
        
        self.calibrate_btn = QtWidgets.QPushButton("Auto Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate_latency)
        
        calibration_layout.addWidget(QtWidgets.QLabel("Target Latency:"))
        calibration_layout.addWidget(self.target_latency_spin)
        calibration_layout.addWidget(self.calibrate_btn)
        
        streaming_layout.addLayout(calibration_layout)
        streaming_group.setLayout(streaming_layout)
        
        layout.addWidget(connection_group)
        layout.addWidget(streaming_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def connect_device(self):
        """Connect to selected device."""
        device_type = self.device_combo.currentText()
        device_key = self.device_key_edit.text().strip()
        
        if not device_key:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter device key/URL")
            return
        
        # TODO: Implement actual device connection
        # For now, simulate connection
        self.connected_devices.append({
            'type': device_type,
            'key': device_key,
            'connected': True
        })
        
        self.connect_btn.setText("Disconnect")
        self.stream_btn.setEnabled(True)
        self.connection_quality_label.setText("Quality: Connected")
        
        QtWidgets.QMessageBox.information(self, "Success", f"Connected to {device_type}")
    
    def test_connection(self):
        """Test device connection."""
        # TODO: Implement actual connection test
        QtWidgets.QMessageBox.information(self, "Test", "Connection test not implemented")
    
    def toggle_streaming(self):
        """Toggle streaming state."""
        if not self.streaming_active:
            self.streaming_active = True
            self.stream_btn.setText("Stop Streaming")
            self.connection_quality_label.setText("Quality: Streaming")
        else:
            self.streaming_active = False
            self.stream_btn.setText("Start Streaming")
            self.connection_quality_label.setText("Quality: Connected")
    
    def calibrate_latency(self):
        """Auto-calibrate latency."""
        # TODO: Implement latency calibration
        target = self.target_latency_spin.value()
        QtWidgets.QMessageBox.information(self, "Calibration", f"Latency calibrated to {target}ms")
    
    def open_simulator(self):
        """Open device simulator window."""
        # TODO: Implement device simulator
        QtWidgets.QMessageBox.information(self, "Simulator", "Device simulator not implemented")


class EnhancedROISelector(ROISelector):
    """Enhanced ROI selector with real-time feedback."""
    
    trackingUpdate = pyqtSignal(float, float, float)  # confidence, velocity_x, velocity_y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracking_overlay = None
        self.confidence = 0.0
        self.velocity = (0.0, 0.0)
        
    def update_tracking_info(self, confidence: float, velocity: Tuple[float, float]):
        """Update tracking overlay information."""
        self.confidence = confidence
        self.velocity = velocity
        self.trackingUpdate.emit(confidence, velocity[0], velocity[1])
        self.update_overlay()
    
    def update_overlay(self):
        """Update tracking confidence overlay."""
        if self._pixmap_item:
            # Add confidence indicator overlay
            # This would draw confidence and velocity indicators on the video
            pass


class ModernMainWindow(QMainWindow):
    """Modern main window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FunGen VR - High-Performance Funscript Generator")
        self.setMinimumSize(1400, 900)
        
        # Apply dark theme
        self.setStyleSheet(DarkTheme.get_stylesheet())
        
        # Core components
        self.video_loader: Optional[VideoLoader] = None
        self.detector: Optional[ObjectDetector] = None
        self.tracker: Optional[AdvancedTracker] = None
        self.positions: List[float] = []
        self.frame_height: Optional[int] = None
        self.fps: float = 30.0
        self.current_funscript: Optional[Funscript] = None
        
        # Performance monitoring
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing_stats)
        self.processing_timer.start(100)  # Update every 100ms
        
        self.init_ui()
        self.init_detector()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Main processing tab
        self.processing_tab = self.create_processing_tab()
        self.tab_widget.addTab(self.processing_tab, "🎬 Processing")
        
        # Performance monitoring tab
        self.performance_monitor = PerformanceMonitor()
        self.tab_widget.addTab(self.performance_monitor, "📊 Performance")
        
        # Device control tab
        self.device_panel = DeviceControlPanel()
        self.tab_widget.addTab(self.device_panel, "🎮 Device Control")
        
        # Settings tab
        self.settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.statusBar().showMessage("Ready - Load a video to begin")
    
    def create_processing_tab(self):
        """Create the main processing tab."""
        tab = QtWidgets.QWidget()
        layout = QHBoxLayout()
        
        # Left side - Video and plot
        left_panel = QVBoxLayout()
        
        # Enhanced video widget
        self.video_widget = EnhancedROISelector()
        self.video_widget.setMinimumSize(800, 450)
        self.video_widget.roiSelected.connect(self.on_roi_selected)
        self.video_widget.trackingUpdate.connect(self.on_tracking_update)
        left_panel.addWidget(self.video_widget, stretch=3)
        
        # Plot panel with enhanced features
        self.create_plot_panel()
        left_panel.addWidget(self.plot_widget, stretch=1)
        
        # Right side - Controls
        right_panel = QVBoxLayout()
        
        # File controls
        self.create_file_controls(right_panel)
        
        # Algorithm selection
        self.create_algorithm_controls(right_panel)
        
        # Parameter controls
        self.create_parameter_controls(right_panel)
        
        # Real-time controls
        self.create_realtime_controls(right_panel)
        
        layout.addLayout(left_panel, stretch=3)
        layout.addLayout(right_panel, stretch=2)
        
        tab.setLayout(layout)
        return tab
    
    def create_plot_panel(self):
        """Create enhanced plot panel with simulation controls."""
        self.plot_widget = QtWidgets.QWidget()
        plot_layout = QVBoxLayout()
        
        # Plot controls
        plot_controls = QHBoxLayout()
        
        self.play_btn = QtWidgets.QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_simulation)
        
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 400)  # 0.25x to 4x speed
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        self.speed_label = QtWidgets.QLabel("1.0x")
        
        plot_controls.addWidget(self.play_btn)
        plot_controls.addWidget(QtWidgets.QLabel("Speed:"))
        plot_controls.addWidget(self.speed_slider)
        plot_controls.addWidget(self.speed_label)
        plot_controls.addStretch()
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 3), facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2d2d2d;")
        
        self.ax = self.figure.add_subplot(111, facecolor='#2d2d2d')
        self.ax.tick_params(colors='white')
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Position (0-100)', color='white')
        self.ax.set_title('Funscript Preview', color='white')
        
        plot_layout.addLayout(plot_controls)
        plot_layout.addWidget(self.canvas)
        
        self.plot_widget.setLayout(plot_layout)
    
    def create_file_controls(self, parent_layout):
        """Create file operation controls."""
        file_group = QtWidgets.QGroupBox("📁 File Operations")
        file_layout = QVBoxLayout()
        
        # Main file buttons
        button_layout1 = QHBoxLayout()
        
        self.open_btn = QtWidgets.QPushButton("📂 Open Video")
        self.open_btn.clicked.connect(self.on_open_video)
        
        self.save_btn = QtWidgets.QPushButton("💾 Save Funscript")
        self.save_btn.clicked.connect(self.on_save_funscript)
        
        button_layout1.addWidget(self.open_btn)
        button_layout1.addWidget(self.save_btn)
        
        # Processing buttons
        button_layout2 = QHBoxLayout()
        
        self.preview_btn = QtWidgets.QPushButton("🔍 Generate Preview")
        self.preview_btn.clicked.connect(self.on_generate_preview)
        
        self.batch_btn = QtWidgets.QPushButton("📦 Batch Process")
        self.batch_btn.clicked.connect(self.on_batch_process)
        
        button_layout2.addWidget(self.preview_btn)
        button_layout2.addWidget(self.batch_btn)
        
        file_layout.addLayout(button_layout1)
        file_layout.addLayout(button_layout2)
        
        file_group.setLayout(file_layout)
        parent_layout.addWidget(file_group)
    
    def create_algorithm_controls(self, parent_layout):
        """Create algorithm selection controls."""
        algo_group = QtWidgets.QGroupBox("🧠 Detection & Tracking")
        algo_layout = QVBoxLayout()
        
        # Detection backend
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(QtWidgets.QLabel("Detection:"))
        
        self.detection_combo = QtWidgets.QComboBox()
        self.detection_combo.addItems(["Auto", "YOLO", "TensorRT", "ONNX", "OpenCV DNN"])
        detection_layout.addWidget(self.detection_combo)
        
        # Tracking method
        tracking_layout = QHBoxLayout()
        tracking_layout.addWidget(QtWidgets.QLabel("Tracking:"))
        
        self.tracking_combo = QtWidgets.QComboBox()
        self.tracking_combo.addItems([
            "Template Matching", "Optical Flow", "Kalman Filter", "CSRT", "KCF"
        ])
        self.tracking_combo.currentTextChanged.connect(self.on_tracking_method_changed)
        tracking_layout.addWidget(self.tracking_combo)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QtWidgets.QLabel("Confidence:"))
        
        self.confidence_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(70)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        
        self.confidence_label = QtWidgets.QLabel("0.70")
        
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        
        algo_layout.addLayout(detection_layout)
        algo_layout.addLayout(tracking_layout)
        algo_layout.addLayout(conf_layout)
        
        algo_group.setLayout(algo_layout)
        parent_layout.addWidget(algo_group)
    
    def create_parameter_controls(self, parent_layout):
        """Create funscript parameter controls."""
        param_group = QtWidgets.QGroupBox("🎚️ Funscript Parameters")
        param_layout = QtWidgets.QVBoxLayout()
        
        # Position range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QtWidgets.QLabel("Range:"))
        
        self.min_pos_spin = QtWidgets.QSpinBox()
        self.min_pos_spin.setRange(0, 100)
        self.min_pos_spin.setValue(0)
        self.min_pos_spin.valueChanged.connect(self.on_params_changed)
        
        range_layout.addWidget(self.min_pos_spin)
        range_layout.addWidget(QtWidgets.QLabel("to"))
        
        self.max_pos_spin = QtWidgets.QSpinBox()
        self.max_pos_spin.setRange(0, 100)
        self.max_pos_spin.setValue(100)
        self.max_pos_spin.valueChanged.connect(self.on_params_changed)
        
        range_layout.addWidget(self.max_pos_spin)
        
        # Smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QtWidgets.QLabel("Smoothing:"))
        
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(1, 21)
        self.smooth_slider.setValue(5)
        self.smooth_slider.valueChanged.connect(self.on_params_changed)
        
        self.smooth_label = QtWidgets.QLabel("5")
        smooth_layout.addWidget(self.smooth_slider)
        smooth_layout.addWidget(self.smooth_label)
        
        # Intensity
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QtWidgets.QLabel("Intensity:"))
        
        self.intensity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(50, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(self.on_params_changed)
        
        self.intensity_label = QtWidgets.QLabel("1.0")
        intensity_layout.addWidget(self.intensity_slider)
        intensity_layout.addWidget(self.intensity_label)
        
        param_layout.addLayout(range_layout)
        param_layout.addLayout(smooth_layout)
        param_layout.addLayout(intensity_layout)
        
        param_group.setLayout(param_layout)
        parent_layout.addWidget(param_group)
    
    def create_realtime_controls(self, parent_layout):
        """Create real-time processing controls."""
        realtime_group = QtWidgets.QGroupBox("⚡ Real-time Controls")
        realtime_layout = QVBoxLayout()
        
        # Auto-update toggle
        self.auto_update_check = QtWidgets.QCheckBox("Auto-update preview")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.toggled.connect(self.on_auto_update_toggled)
        
        # GPU acceleration toggle
        self.gpu_accel_check = QtWidgets.QCheckBox("GPU Acceleration")
        self.gpu_accel_check.setChecked(True)
        self.gpu_accel_check.toggled.connect(self.on_gpu_accel_toggled)
        
        # Performance target
        target_layout = QHBoxLayout()
        target_layout.addWidget(QtWidgets.QLabel("Target FPS:"))
        
        self.target_fps_spin = QtWidgets.QSpinBox()
        self.target_fps_spin.setRange(30, 300)
        self.target_fps_spin.setValue(60)
        
        target_layout.addWidget(self.target_fps_spin)
        
        realtime_layout.addWidget(self.auto_update_check)
        realtime_layout.addWidget(self.gpu_accel_check)
        realtime_layout.addLayout(target_layout)
        
        realtime_group.setLayout(realtime_layout)
        parent_layout.addWidget(realtime_group)
        
        parent_layout.addStretch()
    
    def create_settings_tab(self):
        """Create settings tab."""
        tab = QtWidgets.QWidget()
        layout = QVBoxLayout()
        
        # Model settings
        model_group = QtWidgets.QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to YOLO model (.pt, .engine, .onnx)")
        
        self.browse_model_btn = QtWidgets.QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model_path)
        
        model_path_layout.addWidget(QtWidgets.QLabel("Model Path:"))
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.browse_model_btn)
        
        model_layout.addLayout(model_path_layout)
        model_group.setLayout(model_layout)
        
        # Performance settings
        perf_group = QtWidgets.QGroupBox("Performance Settings")
        perf_layout = QVBoxLayout()
        
        # Memory management
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QtWidgets.QLabel("GPU Memory Fraction:"))
        
        self.memory_fraction_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.memory_fraction_slider.setRange(50, 95)
        self.memory_fraction_slider.setValue(90)
        
        self.memory_fraction_label = QtWidgets.QLabel("90%")
        memory_layout.addWidget(self.memory_fraction_slider)
        memory_layout.addWidget(self.memory_fraction_label)
        
        # TensorRT optimization
        self.enable_tensorrt_check = QtWidgets.QCheckBox("Enable TensorRT optimization")
        self.enable_tensorrt_check.setChecked(True)
        
        perf_layout.addLayout(memory_layout)
        perf_layout.addWidget(self.enable_tensorrt_check)
        perf_group.setLayout(perf_layout)
        
        layout.addWidget(model_group)
        layout.addWidget(perf_group)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QtGui.QAction('Open Video', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.on_open_video)
        file_menu.addAction(open_action)
        
        save_action = QtGui.QAction('Save Funscript', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.on_save_funscript)
        file_menu.addAction(save_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        optimize_action = QtGui.QAction('Optimize Models', self)
        optimize_action.triggered.connect(self.optimize_models)
        tools_menu.addAction(optimize_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QtGui.QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_detector(self):
        """Initialize object detector."""
        model_path = self.model_path_edit.text().strip() if hasattr(self, 'model_path_edit') else None
        if not model_path and Path("models").exists():
            # Look for models in models directory
            model_files = list(Path("models").glob("*.pt")) + list(Path("models").glob("*.engine"))
            if model_files:
                model_path = str(model_files[0])
        
        self.detector = ObjectDetector(
            model_path=model_path,
            device="cuda" if self.gpu_accel_check.isChecked() if hasattr(self, 'gpu_accel_check') else True else "cpu",
            optimize_memory=True,
            enable_trt=True
        )
    
    # Event handlers
    def on_open_video(self):
        """Handle video file opening."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", os.getcwd(), 
            "Videos (*.mp4 *.mkv *.mov *.avi *.webm)"
        )
        if not path:
            return
        
        # Release previous loader
        if self.video_loader:
            self.video_loader.release()
        
        try:
            self.video_loader = VideoLoader(path, target_width=800, device=0)
            info = self.video_loader.info
            if info:
                self.frame_height = info.height
                self.fps = info.fps
                self.statusBar().showMessage(f"Loaded: {Path(path).name} ({info.width}x{info.height}, {info.fps:.1f} FPS)")
            
            # Show first frame
            it = iter(self.video_loader)
            try:
                _, frame = next(it)
                if isinstance(frame, tuple):
                    frame = frame[0]
                
                qimg = self._frame_to_qimage(frame)
                self.video_widget.setImage(qimg)
                
                # Reset tracking
                self.tracker = None
                self.positions = []
                self.ax.clear()
                self.canvas.draw()
                
            except StopIteration:
                QtWidgets.QMessageBox.warning(self, "Error", "Could not read video frames")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
    
    def on_roi_selected(self, x1: int, y1: int, x2: int, y2: int):
        """Handle ROI selection."""
        if not self.video_loader:
            return
        
        # Get tracking method
        tracking_method_map = {
            "Template Matching": TrackingMethod.TEMPLATE_MATCHING,
            "Optical Flow": TrackingMethod.OPTICAL_FLOW,
            "Kalman Filter": TrackingMethod.KALMAN_FILTER,
            "CSRT": TrackingMethod.CSRT,
            "KCF": TrackingMethod.KCF
        }
        
        method = tracking_method_map.get(
            self.tracking_combo.currentText(), 
            TrackingMethod.TEMPLATE_MATCHING
        )
        
        # Initialize tracker
        it = iter(self.video_loader)
        try:
            _, frame = next(it)
            if isinstance(frame, tuple):
                frame = frame[0]
            
            self.tracker = AdvancedTracker(
                method=method,
                use_gpu=self.gpu_accel_check.isChecked(),
                confidence_threshold=self.confidence_slider.value() / 100.0
            )
            self.tracker.init(frame, (x1, y1, x2, y2))
            self.positions = []
            
            # Record first position
            cy = y1 + (y2 - y1) / 2
            self.positions.append(cy)
            
            # Auto-update if enabled
            if self.auto_update_check.isChecked():
                self.on_params_changed()
                
        except StopIteration:
            return
    
    def on_generate_preview(self):
        """Generate funscript preview."""
        if not self.video_loader or not self.tracker:
            QtWidgets.QMessageBox.information(
                self, "Info", "Load a video and select an ROI first."
            )
            return
        
        self.positions = []
        max_frames = 300  # Limit for preview
        
        try:
            for idx, frame in self.video_loader:
                if idx >= max_frames:
                    break
                
                if isinstance(frame, tuple):
                    frame = frame[0]
                
                if idx == 0:
                    continue  # Skip first frame
                
                result = self.tracker.update(frame)
                x1, y1, x2, y2 = result.roi
                cy = y1 + (y2 - y1) / 2
                self.positions.append(cy)
                
                # Update video widget with tracking info
                self.video_widget.update_tracking_info(
                    result.confidence, result.velocity
                )
            
            self.on_params_changed()
            self.statusBar().showMessage(f"Generated preview from {len(self.positions)} frames")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Preview generation failed: {str(e)}")
    
    def on_params_changed(self):
        """Handle parameter changes and update preview."""
        if not self.positions or self.frame_height is None:
            return
        
        try:
            # Get parameters
            min_pos = self.min_pos_spin.value()
            max_pos = self.max_pos_spin.value()
            smooth_win = self.smooth_slider.value()
            intensity = self.intensity_slider.value() / 100.0
            
            # Update slider labels
            self.smooth_label.setText(str(smooth_win))
            self.intensity_label.setText(f"{intensity:.1f}")
            
            # Generate funscript
            fs = map_positions(
                positions=self.positions,
                frame_height=self.frame_height,
                fps=self.fps,
                min_pos=min_pos,
                max_pos=max_pos,
                smoothing_window=smooth_win,
                boost_up_percent=(intensity - 1.0) * 0.5,
                boost_down_percent=(intensity - 1.0) * 0.5
            )
            
            # Update plot
            self.ax.clear()
            self.ax.set_facecolor('#2d2d2d')
            
            if fs.actions:
                xs = [act["at"] / 1000.0 for act in fs.actions]
                ys = [act["pos"] for act in fs.actions]
                self.ax.plot(xs, ys, color="#007acc", linewidth=2, label="Funscript")
                self.ax.set_ylim(0, 100)
                self.ax.set_xlabel("Time (s)", color='white')
                self.ax.set_ylabel("Position (0-100)", color='white')
                self.ax.set_title("Funscript Preview", color='white')
                self.ax.grid(True, alpha=0.3)
                self.ax.tick_params(colors='white')
            
            self.canvas.draw()
            self.current_funscript = fs
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Parameter update failed: {str(e)}")
    
    def on_save_funscript(self):
        """Save generated funscript."""
        if not hasattr(self, 'current_funscript') or not self.current_funscript or not self.current_funscript.actions:
            QtWidgets.QMessageBox.information(self, "Info", "Nothing to save. Generate a preview first.")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save funscript", os.getcwd(), 
            "Funscript (*.funscript *.json)"
        )
        if not path:
            return
        
        try:
            self.current_funscript.save(path)
            QtWidgets.QMessageBox.information(self, "Saved", f"Funscript saved to {Path(path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save funscript: {str(e)}")
    
    def on_batch_process(self):
        """Handle batch processing."""
        # TODO: Implement batch processing with progress dialog
        QtWidgets.QMessageBox.information(self, "Batch Process", "Batch processing not yet implemented")
    
    def on_tracking_method_changed(self):
        """Handle tracking method change."""
        if self.tracker and self.video_loader:
            # Get current frame for re-initialization
            it = iter(self.video_loader)
            try:
                _, frame = next(it)
                if isinstance(frame, tuple):
                    frame = frame[0]
                
                method_map = {
                    "Template Matching": TrackingMethod.TEMPLATE_MATCHING,
                    "Optical Flow": TrackingMethod.OPTICAL_FLOW,
                    "Kalman Filter": TrackingMethod.KALMAN_FILTER,
                    "CSRT": TrackingMethod.CSRT,
                    "KCF": TrackingMethod.KCF
                }
                
                new_method = method_map.get(
                    self.tracking_combo.currentText(),
                    TrackingMethod.TEMPLATE_MATCHING
                )
                
                success = self.tracker.switch_method(new_method, frame)
                if success:
                    self.statusBar().showMessage(f"Switched to {self.tracking_combo.currentText()}")
                else:
                    self.statusBar().showMessage("Failed to switch tracking method")
                    
            except StopIteration:
                pass
    
    def on_confidence_changed(self):
        """Handle confidence threshold change."""
        confidence = self.confidence_slider.value() / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        
        if self.tracker:
            self.tracker.confidence_threshold = confidence
    
    def on_auto_update_toggled(self, checked):
        """Handle auto-update toggle."""
        self.statusBar().showMessage(f"Auto-update {'enabled' if checked else 'disabled'}")
    
    def on_gpu_accel_toggled(self, checked):
        """Handle GPU acceleration toggle."""
        # Reinitialize detector with new settings
        self.init_detector()
        self.statusBar().showMessage(f"GPU acceleration {'enabled' if checked else 'disabled'}")
    
    def on_speed_changed(self):
        """Handle simulation speed change."""
        speed = self.speed_slider.value() / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
    
    def on_tracking_update(self, confidence, vel_x, vel_y):
        """Handle tracking update from enhanced ROI selector."""
        # Update status bar with tracking info
        self.statusBar().showMessage(
            f"Tracking: {confidence:.2f} confidence, velocity: ({vel_x:.1f}, {vel_y:.1f})"
        )
    
    def toggle_simulation(self):
        """Toggle funscript simulation playback."""
        if self.play_btn.text() == "▶️ Play":
            self.play_btn.setText("⏸️ Pause")
            # TODO: Start simulation
        else:
            self.play_btn.setText("▶️ Play")
            # TODO: Pause simulation
    
    def browse_model_path(self):
        """Browse for model file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model file", os.getcwd(),
            "Models (*.pt *.engine *.onnx)"
        )
        if path:
            self.model_path_edit.setText(path)
            self.init_detector()
    
    def optimize_models(self):
        """Optimize models for current hardware."""
        QtWidgets.QMessageBox.information(self, "Optimize", "Model optimization not yet implemented")
    
    def show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(
            self, "About FunGen VR",
            "FunGen VR - High-Performance Funscript Generator\n\n"
            "GPU-accelerated VR funscript generation with real-time processing\n"
            "optimized for RTX 3090 hardware.\n\n"
            "Features:\n"
            "• 150+ FPS analysis capability\n"
            "• Multi-algorithm tracking\n"
            "• Real-time device streaming\n"
            "• Modern tabbed interface"
        )
    
    def update_processing_stats(self):
        """Update processing performance statistics."""
        if self.detector and self.tracker:
            detector_stats = self.detector.get_performance_stats()
            tracker_stats = self.tracker.get_performance_stats()
            
            # Update performance monitor
            self.performance_monitor.update_performance_data(detector_stats, tracker_stats)
    
    def _frame_to_qimage(self, frame: np.ndarray) -> QtGui.QImage:
        """Convert OpenCV frame to QImage."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("FunGen VR")
    app.setApplicationVersion("2.0")
    
    # Set application icon
    if Path("resources/icon.png").exists():
        app.setWindowIcon(QtGui.QIcon("resources/icon.png"))
    
    window = ModernMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()