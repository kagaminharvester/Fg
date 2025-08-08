# Improved FunGen VR Funscript Generator

This repository contains an enhanced version of the original
[FunGen](https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator)
project.  The goal of this fork is to provide a streamlined, fast
and user‑friendly tool for generating Funscript files from VR and
2D videos.  The emphasis is on running locally on high‑performance
hardware (e.g. an RTX 3090) without relying on cloud services.

## Features

- **Modular architecture** – the codebase has been reorganised into
  clear modules (`video_loader.py`, `detector.py`, `tracker.py`,
  `funscript_generator.py`, `roi_selector.py`) to make it easier to
  extend and maintain.
- **Improved GUI** – written in PyQt6, the new graphical interface
  exposes all tuning parameters (range mapping, boosting,
  thresholding, smoothing, randomness) and updates the preview curve
  on the fly.  Users can draw a region of interest on the first
  frame to focus tracking on a specific part of the scene.
- **Live preview (stub)** – a placeholder interface is included to
  stream a short preview to *The Handy* or other interactive devices.
- **Batch processing** – select a folder of videos and process them
  all in one go with your current settings.  The resulting
  `.funscript` files are saved into your chosen output folder.
- **TensorRT support** – a helper script (`generate_tensorrt.py`) and
  batch file (`GenerateTensorRT.bat`) are provided to convert YOLO
  weights into optimised TensorRT engines for maximum inference
  performance on NVIDIA GPUs.
- **One‑click installer** – an Inno Setup script (`improved_fungen.iss`)
  packages the application into a Windows installer when combined
  with the PyInstaller build.

## Installation

1. **Create a Python environment** – on Windows the easiest way is
   via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

   ```bash
   conda create -n fungen python=3.11
   conda activate fungen
   ```

2. **Install dependencies** – install the packages listed in
   `requirements.txt`.  On an RTX 3090 you should also install
   TensorRT from the NVIDIA developer site.

   ```bash
   pip install -r requirements.txt
   pip install nvidia-pyindex tensorrt  # if available
   ```

3. **Download a detection model** – place your YOLO weights (e.g.
   `yolov8n.pt`) in the `models/` directory.  You can convert it to a
   TensorRT engine using:

   ```bash
   python generate_tensorrt.py --weights models/yolov8n.pt --output models/yolov8n.engine --fp16
   ```

4. **Run the application** – from the project root run

   ```bash
   python main.py
   ```

5. **Build the Windows installer** – after verifying that the
   application runs correctly, you can package it using PyInstaller
   and Inno Setup:

   ```bash
   pyinstaller improved_fungen.spec
   iscc installer\improved_fungen.iss
   ```

   The resulting `FunGenVRSetup.exe` can be distributed to end users
   who do not have Python installed.

## Usage

1. **Open a video** – click *Open video* and choose a file.  For VR
   content the loader will automatically detect side‑by‑side or
   over‑under formats and split frames accordingly.
2. **Select an ROI** – draw a rectangle on the first frame to focus
   tracking on a specific region (e.g. hands or genitals).  The
   rectangle can be redrawn until you are satisfied.
3. **Generate a preview** – click *Generate preview* to process the
   first few seconds of the video (default 300 frames).  The motion
   curve will appear on the plot.
4. **Adjust parameters** – tune the range mapping, boosting,
   thresholding, smoothing and randomness until the preview curve
   matches your expectations.  The plot updates automatically.
5. **Save the script** – once satisfied click *Save funscript* and
   choose a filename.  The `.funscript` can be loaded into a
   compatible player or device.
6. **Batch mode** – use *Batch process* to generate scripts for an
   entire folder of videos using the current settings.

## Limitations and future work

This template provides a working baseline but omits several
optimisations present in the original FunGen project:

- The detection step currently uses a dummy implementation that
  returns the full frame as a single ROI.  You should integrate your
  YOLO detection model by extending `ObjectDetector` in
  `detector.py`.
- The tracking uses a simple template matcher (`SimpleTracker`).
  More sophisticated trackers like Deep SORT or ByteTrack can
  improve robustness, especially in cluttered scenes.
- Live preview streaming is stubbed out; integration with the
  [Buttplug.io](https://buttplug.io/) library or The Handy SDK is
  required to control a device directly from the GUI.
- Multi‑axis funscript generation is not implemented.  This
  repository focuses solely on up‑down motion.

We welcome contributions and bug reports.  Fork the project and
submit a pull request with your improvements!
