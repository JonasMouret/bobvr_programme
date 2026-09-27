# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the Windows build.

Two executables share one folder: BobVr.exe opens the window, bobvr-cli.exe
keeps a console for `info`, `probe`, `render` and `label`. A one-folder build
rather than one-file, on purpose -- a one-file .exe unpacks 200 MB of Qt into
a temporary directory on every launch, and leaves nowhere obvious to drop
ffmpeg.exe.

Build from the project root:

    pyinstaller installer/bobvr.spec --noconfirm --clean

Anything found in installer/vendor/ (ffmpeg.exe, ffprobe.exe, exiftool.exe and
their DLLs) is copied into the result, so the folder that comes out is
self-contained. Leave it empty and the app will look for those tools on PATH
instead.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

PROJECT = Path(SPECPATH).resolve().parent
VENDOR = PROJECT / "installer" / "vendor"
ICON = PROJECT / "installer" / "bobvr.ico"

# The renderer reads this off disk at run time to specialise it per capture
# resolution, so it has to travel as data, not as code.
datas = [
    (str(PROJECT / "bobvr" / "render" / "kernels" / "gopromax_equirect.cl"),
     "bobvr/render/kernels"),
]

# UI images read at run time: the window icon, and the splash logo if the
# builder dropped a bobvr/ui/splash.png in. The .ico below is the icon of the
# .exe file itself, which is a different thing.
for image in sorted((PROJECT / "bobvr" / "ui").glob("*.png")):
    datas.append((str(image), "bobvr/ui"))

# Helper executables, if the builder dropped any in. Declared as datas rather
# than binaries: PyInstaller rewrites the load paths of things it considers
# binaries, and these are complete programs that must be left exactly as they
# came.
if VENDOR.is_dir():
    for item in sorted(VENDOR.rglob("*")):
        if item.is_file() and item.suffix.lower() not in {".md", ".txt"}:
            datas.append((str(item), str(Path("vendor") / item.relative_to(VENDOR).parent)))

hiddenimports = collect_submodules("bobvr")

# Qt ships far more than a five-window desktop app uses, and every module
# dragged in is tens of megabytes the operator has to copy to the track.
excludes = [
    "tkinter", "unittest", "pydoc_data", "pytest", "setuptools", "pip",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.Qt3DCore",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtBluetooth", "PySide6.QtNfc",
    "PySide6.QtPositioning", "PySide6.QtSerialPort", "PySide6.QtTest",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtSql", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
]

common = dict(
    pathex=[str(PROJECT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

gui_analysis = Analysis([str(PROJECT / "installer" / "entry_gui.py")], **common)
cli_analysis = Analysis([str(PROJECT / "installer" / "entry_cli.py")], **common)

gui_pyz = PYZ(gui_analysis.pure, gui_analysis.zipped_data)
cli_pyz = PYZ(cli_analysis.pure, cli_analysis.zipped_data)

icon = str(ICON) if ICON.is_file() else None

gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="BobVr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console: this one is double-clicked. Errors reach the operator
    # through the window's own log panel and message boxes.
    console=False,
    icon=icon,
)

cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="bobvr-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=icon,
)

COLLECT(
    gui_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    cli_exe,
    cli_analysis.binaries,
    cli_analysis.datas,
    strip=False,
    upx=False,
    name="BobVr",
)
