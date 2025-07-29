# Technical Instructions for Upgrading FunGen

These instructions describe how to refactor and extend the
[`FunGen`](https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator)
project into a fast, accurate and user‑friendly local funscript
generator.  The goal is to support both 2D and 3D VR videos on
Windows, take full advantage of a powerful GPU (e.g. RTX 3090) and
offer a modern GUI with live preview and customisation controls.
Multi‑axis output is explicitly out of scope.

## 1. Refactoring and Project Setup

* **Create a new branch** off the original FunGen repository to avoid
  disrupting the upstream project.  Use a descriptive name such as
  `feature/gui-revamp`.
* **Modularise the code**.  Break the monolithic `FSGenerator.py` into
  a number of focused modules:

  - `video_loader.py` – encapsulates video decoding using OpenCV,
    supports 2D, side‑by‑side (SBS), over‑under (OU) and
    equirectangular formats.  Resizes frames to a fixed width for
    consistent GPU throughput.  Detects the format from the filename.
  - `detector.py` – hides the object detection implementation behind
    an `ObjectDetector` class.  It accepts a model path (.pt, .onnx
    or .engine) and exposes `detect(frame)` / `detect_stereo(left,
    right)` methods returning bounding boxes.  In production you
    should integrate YOLOv8 via Ultralytics or TensorRT.
  - `tracker.py` – implements object tracking.  A simple template
    matcher (`SimpleTracker`) is included for prototyping; swap it
    for Deep SORT or ByteTrack in production.
  - `roi_selector.py` – reusable PyQt6 widget that lets the user
    draw a rectangular region of interest (ROI) on an image.  Emits
    a signal with the ROI coordinates when selection completes.
  - `funscript_generator.py` – maps raw vertical positions to a
    funscript.  Implements range mapping, boosting, thresholding,
    smoothing, randomness and produces valid JSON.
  - `main.py` – entry point for the GUI.  Composes the widgets,
    wires up signal/slot connections and orchestrates the pipeline.

* **Dependencies**: list all required Python packages in
  `requirements.txt`.  Use a separate `gui.requirements.txt` if you
  plan to support headless processing.  Key packages include
  `PyQt6`, `opencv-python`, `numpy`, `matplotlib`, `torch`,
  `ultralytics`, `tensorrt`, `onnxruntime`, `scipy` and
  `buttplug-py`.
* **TensorRT conversion**: add a script `generate_tensorrt.py` and a
  convenience batch file `GenerateTensorRT.bat` to convert YOLO
  weights into TensorRT engines.  Document the process using
  `trtexec` or `torch2trt`.

## 2. New GUI

* **Framework** – implement the GUI in PyQt6.  Avoid Qt Designer and
  construct widgets manually for clarity and flexibility.
* **Layout** – divide the main window into a left panel for video
  display and plot, and a right panel for controls.  The left panel
  contains the `ROISelector` for drawing the ROI and a Matplotlib
  canvas for visualising the generated curve.  The right panel hosts
  file controls (Open, Generate, Save, Batch), parameter controls
  (range, boosting, thresholding, smoothing, randomness) and a live
  preview section with a text field for the device key.
* **ROI selection** – after loading a video, show the first frame.
  When the user draws a rectangle on this frame, initialise the
  tracker with that ROI.  Record the vertical centre of the ROI for
  every subsequent frame.
* **Parameter controls** – expose spin boxes for minimum and maximum
  positions (0–100), boosting percentages (positive to expand the
  range, negative to compress), low/high thresholds (0–1 fractions
  of frame height), smoothing window (number of frames) and
  randomness.  Connect each control to a handler that regenerates
  the preview when changed.
* **Plot panel** – embed a Matplotlib figure using
  `FigureCanvasQTAgg`.  When a preview is generated, plot the
  position values (0–100) against time (seconds).  Label the axes
  and clamp the y‑axis to [0,100].
* **Batch processing** – provide a dialog to choose an input folder
  and an output folder.  For each video use the full frame as the
  ROI and generate a short script (e.g. first 300 frames).  Save the
  resulting `.funscript` with the same base name into the output
  folder.
* **Live preview** – add fields for the device key and a button to
  stream a short preview to *The Handy* or other interactive devices.
  Stub this out initially; a full implementation would use the
  [Buttplug.io](https://buttplug.io/) library or the Handy's
  Bluetooth API.

## 3. Optimisation for RTX 3090

* **Detection model** – download or train a custom YOLOv8 model for
  VR/adult scenes.  Place the `.pt` file into a `models/` directory.
  Convert it to a `.engine` file with `generate_tensorrt.py` to
  utilise TensorRT and FP16 on an RTX 3090.  Expose a setting in
  the GUI to choose between `.pt`, `.onnx` and `.engine` models.
* **Video decoding** – leverage `cv2.cuda.VideoCapture` when CUDA is
  available to decode frames on the GPU.  For CPU fallback use
  `cv2.VideoCapture`.  Resize frames to a fixed width to improve
  cache locality and reduce memory consumption.
* **Parallelism** – for batch processing allow the user to specify
  the number of worker processes.  Each worker should load its own
  copy of the model and process a different video to utilise the GPU
  more effectively.  On a 24 GB 3090 three or four workers are
  possible.
* **Mixed precision** – wrap inference in `torch.cuda.amp.autocast()`
  when using PyTorch and ensure FP16 layers are enabled when using
  TensorRT.

## 4. One‑Click Windows Installer

* **PyInstaller** – create a spec file (`improved_fungen.spec`) that
  bundles `main.py` with all helper modules, dynamic libraries and
  resources.  Use the `--add-data` flag to include the `models/`
  folder, icons and configuration files.  Disable the console for a
  clean user experience and assign an application icon.
* **Inno Setup** – write a script (`installer/improved_fungen.iss`)
  that copies the PyInstaller output to `Program Files\FunGenVR`,
  creates start menu and desktop shortcuts and registers the
  `.funscript` file extension to open with the installed program.
  Offer optional tasks such as creating a desktop icon.
* **Updates** – optionally implement an update checker that queries
  the GitHub releases API on startup and notifies the user when a
  new version is available.  Allow the user to disable update
  checks in the settings.

## 5. Testing & Documentation

* **Unit tests** – write tests for each module.  For example, verify
  that `map_positions` produces the expected range mapping and
  smoothing behaviour.  Use a small synthetic video or random data
  for tests to avoid large binary files in the repository.
* **Continuous Integration** – configure a GitHub Actions workflow to
  run tests on every push.  Test both CPU and CUDA builds if
  possible.
* **User documentation** – update the README with installation and
  usage instructions.  Include screenshots or animated GIFs showing
  ROI selection, parameter adjustment and the preview curve.  Note
  that the software is for personal, non‑commercial use only.

## 6. Exclusions

* **Multi‑axis support** – generating multi‑axis funscripts (e.g.
  twisting/roll motions) is outside the scope of this project.  Only
  a single axis (up‑down) is generated.

Following these steps will yield a powerful local funscript
generator with a modern, intuitive GUI, GPU‑accelerated inference and
batch processing capabilities.  Feel free to iterate on this design
to further optimise performance and user experience.
