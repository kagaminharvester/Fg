# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# PyInstaller spec file for building the improved FunGen executable.
# Run with `pyinstaller improved_fungen.spec` from within the project
# directory.  The resulting executable will be placed in the
# `dist/FunGenVR` folder.

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('models/*', 'models'),
        ('resources/*', 'resources'),
        ('config/*', 'config'),
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'matplotlib',
        'numpy',
        'cv2',
        'ultralytics',
        'tensorrt',
        'onnxruntime',
        'buttplug',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='FunGenVR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='resources/icon.ico' if os.path.exists('resources/icon.ico') else None,
)