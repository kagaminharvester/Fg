; Inno Setup script for building a one‑click installer for the improved FunGen
; This script installs the packaged executable (created via PyInstaller)
; into the Program Files directory and registers the .funscript file type.

[Setup]
AppName=FunGenVR
AppVersion=1.0.0
DefaultDirName={pf}\FunGenVR
DefaultGroupName=FunGenVR
DisableDirPage=yes
DisableProgramGroupPage=yes
OutputBaseFilename=FunGenVRSetup
Compression=lzma
SolidCompression=yes
SetupIconFile=resources\icon.ico

[Files]
; copy all files from the PyInstaller dist folder
Source: "dist\FunGenVR\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\FunGenVR"; Filename: "{app}\FunGenVR.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\FunGenVR"; Filename: "{app}\FunGenVR.exe"; Tasks: desktopicon

[Registry]
; register .funscript file extension
Root: HKCR; Subkey: ".funscript"; ValueType: string; ValueName: ""; ValueData: "FunGenVR.funscript"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "FunGenVR.funscript"; ValueType: string; ValueName: ""; ValueData: "FunGenVR Funscript"; Flags: uninsdeletekey
Root: HKCR; Subkey: "FunGenVR.funscript\DefaultIcon"; ValueType: string; ValueData: "{app}\FunGenVR.exe,1"; Flags: uninsdeletekey
Root: HKCR; Subkey: "FunGenVR.funscript\Shell\Open\Command"; ValueType: string; ValueData: '"{app}\FunGenVR.exe" "%1"'; Flags: uninsdeletekey

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; Flags: unchecked
