@echo off
REM Generate TensorRT engine from a YOLO .pt weights file
REM Usage: drag a .pt file onto this script or run from cmd:
REM    GenerateTensorRT.bat models\yolov8n.pt

IF "%1"=="" (
    echo Please specify the path to your YOLO .pt model.
    echo Example: GenerateTensorRT.bat models\yolov8n.pt
    exit /b 1
)
SET WEIGHTS=%~1
SET OUT=%~dpn1.engine

echo Converting %WEIGHTS% to TensorRT engine %OUT%
python generate_tensorrt.py --weights "%WEIGHTS%" --output "%OUT%" --fp16
if %errorlevel% neq 0 (
    echo Error during conversion.
    exit /b 1
)
echo Engine generated at %OUT%
