"""
main.py
=======

Entry point for the improved FunGen application.  This script
launches a modern PyQt6 GUI that allows users to load videos, select
an ROI, tweak funscript parameters and preview the generated
motion curve.  It is designed to be user friendly and to take
advantage of high‑performance GPUs like the RTX 3090 by offloading
video decoding and detection to CUDA whenever possible.

To run the application install the dependencies listed in
``requirements.txt`` and execute:

```
python main.py
```

The GUI comprises several panels:

* **Video panel** – displays the current frame and allows the user to
  draw a rectangular region of interest.  The ROI is tracked across
  frames to generate a motion signal.
* **Parameter panel** – a collection of sliders and spin boxes that
  control the mapping from positions to funscript values (range,
  boosting, thresholding, smoothing, randomness).
* **Plot panel** – shows the resulting funscript curve over time.
* **Control buttons** – open videos, generate a preview, save
  funscript, batch‑process folders, and stream a live preview to a
  connected device.
"""

from __future__ import annotations

import os
import sys
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Improved FunGen VR Funscript Generator")
        self.resize(1200, 800)

        # internal state
        self.video_loader: Optional[VideoLoader] = None
        self.tracker: Optional[SimpleTracker] = None
        self.positions: List[float] = []
        self.frame_height: Optional[int] = None
        self.fps: float = 30.0

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        # video and plot area
        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, stretch=3)

        # video panel
        self.video_widget = ROISelector()
        self.video_widget.setMinimumSize(640, 360)
        self.video_widget.roiSelected.connect(self.on_roi_selected)
        left_panel.addWidget(self.video_widget, stretch=3)

        # plot panel
        self.figure, self.ax = plt.subplots(figsize=(6, 2.5))
        self.canvas = FigureCanvas(self.figure)
        left_panel.addWidget(self.canvas, stretch=1)

        # right panel (controls)
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)

        # file controls
        file_box = QtWidgets.QGroupBox("File")
        file_layout = QtWidgets.QHBoxLayout()
        file_box.setLayout(file_layout)
        self.open_btn = QtWidgets.QPushButton("Open video")
        self.open_btn.clicked.connect(self.on_open_video)
        self.preview_btn = QtWidgets.QPushButton("Generate preview")
        self.preview_btn.clicked.connect(self.on_generate_preview)
        self.save_btn = QtWidgets.QPushButton("Save funscript")
        self.save_btn.clicked.connect(self.on_save_funscript)
        self.batch_btn = QtWidgets.QPushButton("Batch process")
        self.batch_btn.clicked.connect(self.on_batch_process)
        file_layout.addWidget(self.open_btn)
        file_layout.addWidget(self.preview_btn)
        file_layout.addWidget(self.save_btn)
        file_layout.addWidget(self.batch_btn)
        right_panel.addWidget(file_box)

        # parameter controls
        param_box = QtWidgets.QGroupBox("Parameters")
        form = QtWidgets.QFormLayout()
        param_box.setLayout(form)
        # min/max positions
        self.min_pos_spin = QtWidgets.QSpinBox()
        self.min_pos_spin.setRange(0, 100)
        self.min_pos_spin.setValue(0)
        self.max_pos_spin = QtWidgets.QSpinBox()
        self.max_pos_spin.setRange(0, 100)
        self.max_pos_spin.setValue(100)
        form.addRow("Min position", self.min_pos_spin)
        form.addRow("Max position", self.max_pos_spin)
        # boosting
        self.boost_up_spin = QtWidgets.QDoubleSpinBox()
        self.boost_up_spin.setRange(0.0, 1.0)
        self.boost_up_spin.setSingleStep(0.05)
        self.boost_up_spin.setValue(0.0)
        self.boost_down_spin = QtWidgets.QDoubleSpinBox()
        self.boost_down_spin.setRange(-1.0, 1.0)
        self.boost_down_spin.setSingleStep(0.05)
        self.boost_down_spin.setValue(0.0)
        form.addRow("Boost up %", self.boost_up_spin)
        form.addRow("Boost down %", self.boost_down_spin)
        # thresholding
        self.thresh_low_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_low_spin.setRange(0.0, 1.0)
        self.thresh_low_spin.setSingleStep(0.05)
        self.thresh_low_spin.setValue(0.0)
        self.thresh_high_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_high_spin.setRange(0.0, 1.0)
        self.thresh_high_spin.setSingleStep(0.05)
        self.thresh_high_spin.setValue(1.0)
        form.addRow("Thresh low", self.thresh_low_spin)
        form.addRow("Thresh high", self.thresh_high_spin)
        # smoothing
        self.smooth_spin = QtWidgets.QSpinBox()
        self.smooth_spin.setRange(1, 51)
        self.smooth_spin.setValue(5)
        form.addRow("Smoothing window", self.smooth_spin)
        # randomness
        self.random_spin = QtWidgets.QDoubleSpinBox()
        self.random_spin.setRange(0.0, 0.5)
        self.random_spin.setSingleStep(0.01)
        self.random_spin.setValue(0.0)
        form.addRow("Randomness", self.random_spin)
        right_panel.addWidget(param_box)

        # live preview controls
        live_box = QtWidgets.QGroupBox("Live preview")
        live_layout = QtWidgets.QHBoxLayout()
        live_box.setLayout(live_layout)
        self.hand_key_edit = QtWidgets.QLineEdit()
        self.hand_key_edit.setPlaceholderText("Device key (The Handy)")
        self.live_btn = QtWidgets.QPushButton("Stream preview")
        self.live_btn.clicked.connect(self.on_live_preview)
        live_layout.addWidget(self.hand_key_edit)
        live_layout.addWidget(self.live_btn)
        right_panel.addWidget(live_box)

        # connect parameter changes to preview update
        for w in [self.min_pos_spin, self.max_pos_spin, self.boost_up_spin,
                  self.boost_down_spin, self.thresh_low_spin, self.thresh_high_spin,
                  self.smooth_spin, self.random_spin]:
            w.valueChanged.connect(self.on_params_changed)

    # utility to convert BGR frame to QImage
    def _frame_to_qimage(self, frame: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

    def on_open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", os.getcwd(), "Videos (*.mp4 *.mkv *.mov *.avi)")
        if not path:
            return
        # release any previous loader
        if self.video_loader:
            self.video_loader.release()
        self.video_loader = VideoLoader(path, target_width=640, device=0)
        info = self.video_loader.info
        if info:
            self.frame_height = info.height
            self.fps = info.fps
        # show first frame
        it = iter(self.video_loader)
        try:
            _, frame = next(it)
        except StopIteration:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not read video")
            return
        # for stereo frames take left eye by default
        if isinstance(frame, tuple):
            frame = frame[0]
        qimg = self._frame_to_qimage(frame)
        self.video_widget.setImage(qimg)
        self.tracker = None
        self.positions = []
        self.ax.clear()
        self.canvas.draw()

    def on_roi_selected(self, x1: int, y1: int, x2: int, y2: int) -> None:
        # initialise tracker on ROI
        if not self.video_loader:
            return
        it = iter(self.video_loader)
        try:
            _, frame = next(it)
        except StopIteration:
            return
        if isinstance(frame, tuple):
            frame = frame[0]
        self.tracker = SimpleTracker()
        self.tracker.init(frame, (x1, y1, x2, y2))
        self.positions = []
        # record first position
        cy = y1 + (y2 - y1) / 2
        self.positions.append(cy)
        # when ROI selected we immediately regenerate preview for first frame
        self.on_params_changed()

    def on_generate_preview(self) -> None:
        if not self.video_loader or not self.tracker:
            QtWidgets.QMessageBox.information(self, "Info", "Load a video and select an ROI first.")
            return
        self.positions = []
        # iterate through frames (limit to e.g. 300 frames for quick preview)
        max_frames = 300  # adjust as desired
        for idx, frame in self.video_loader:
            if idx >= max_frames:
                break
            if isinstance(frame, tuple):
                frame = frame[0]
            if idx == 0:
                # skip first, already handled by ROI selection
                continue
            roi = self.tracker.update(frame)
            _, y1, _, y2 = roi
            cy = y1 + (y2 - y1) / 2
            self.positions.append(cy)
        self.on_params_changed()

    def on_params_changed(self) -> None:
        if not self.positions or self.frame_height is None:
            return
        min_pos = self.min_pos_spin.value()
        max_pos = self.max_pos_spin.value()
        boost_up = self.boost_up_spin.value()
        boost_down = self.boost_down_spin.value()
        thresh_low = self.thresh_low_spin.value()
        thresh_high = self.thresh_high_spin.value()
        smooth_win = self.smooth_spin.value()
        randomness = self.random_spin.value()
        fs = map_positions(
            positions=self.positions,
            frame_height=self.frame_height,
            fps=self.fps,
            min_pos=min_pos,
            max_pos=max_pos,
            boost_up_percent=boost_up,
            boost_down_percent=boost_down,
            threshold_low=thresh_low,
            threshold_high=thresh_high,
            smoothing_window=smooth_win,
            randomness=randomness,
        )
        # update plot
        self.ax.clear()
        if fs.actions:
            xs = [act["at"] / 1000.0 for act in fs.actions]
            ys = [act["pos"] for act in fs.actions]
            self.ax.plot(xs, ys, color="blue")
            self.ax.set_ylim(0, 100)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Position (0-100)")
        self.canvas.draw()
        # store current funscript
        self.current_funscript = fs

    def on_save_funscript(self) -> None:
        if not hasattr(self, "current_funscript") or not self.current_funscript.actions:
            QtWidgets.QMessageBox.information(self, "Info", "Nothing to save. Generate a preview first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save funscript", os.getcwd(), "Funscript (*.funscript *.json)")
        if not path:
            return
        self.current_funscript.save(path)
        QtWidgets.QMessageBox.information(self, "Saved", f"Funscript saved to {path}")

    def on_batch_process(self) -> None:
        # run generation on all videos in a folder using current settings
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder of videos", os.getcwd())
        if not folder:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder for funscripts", os.getcwd())
        if not out_dir:
            return
        # gather video files
        video_ext = {".mp4", ".mkv", ".mov", ".avi"}
        vids = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in video_ext]
        if not vids:
            QtWidgets.QMessageBox.warning(self, "Warning", "No video files found in the selected folder.")
            return
        for vid in vids:
            loader = VideoLoader(vid, target_width=640, device=0)
            info = loader.info
            tracker = None
            positions: List[float] = []
            # automatically select full frame ROI for batch mode
            if not info:
                continue
            it = iter(loader)
            try:
                _, frame0 = next(it)
            except StopIteration:
                continue
            if isinstance(frame0, tuple):
                frame0 = frame0[0]
            h, w = frame0.shape[:2]
            tracker = SimpleTracker()
            tracker.init(frame0, (0, 0, w, h))
            positions.append(h / 2)
            for idx, frame in loader:
                if idx >= 300:  # limit frames
                    break
                if isinstance(frame, tuple):
                    frame = frame[0]
                if idx == 0:
                    continue
                roi = tracker.update(frame)
                _, y1, _, y2 = roi
                cy = y1 + (y2 - y1) / 2
                positions.append(cy)
            fs = map_positions(
                positions=positions,
                frame_height=info.height,
                fps=info.fps,
                min_pos=self.min_pos_spin.value(),
                max_pos=self.max_pos_spin.value(),
                boost_up_percent=self.boost_up_spin.value(),
                boost_down_percent=self.boost_down_spin.value(),
                threshold_low=self.thresh_low_spin.value(),
                threshold_high=self.thresh_high_spin.value(),
                smoothing_window=self.smooth_spin.value(),
                randomness=self.random_spin.value(),
            )
            base_name = os.path.splitext(os.path.basename(vid))[0]
            out_path = os.path.join(out_dir, f"{base_name}.funscript")
            fs.save(out_path)
        QtWidgets.QMessageBox.information(self, "Done", f"Processed {len(vids)} videos.")

    def on_live_preview(self) -> None:
        # stub: in a real implementation this would connect to The Handy using
        # the provided API key and stream a short funscript.
        if not hasattr(self, "current_funscript") or not self.current_funscript.actions:
            QtWidgets.QMessageBox.information(self, "Info", "Generate a preview first.")
            return
        device_key = self.hand_key_edit.text().strip()
        if not device_key:
            QtWidgets.QMessageBox.information(self, "Info", "Enter your device key to stream the preview.")
            return
        QtWidgets.QMessageBox.information(self, "Streaming", "Live preview streaming is not implemented in this template.")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
