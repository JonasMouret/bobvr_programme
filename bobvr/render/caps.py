"""Discover what the host can actually accelerate, once, at startup.

Rendering a .360 file is the expensive part of the whole pipeline, and the
gap between hardware and software paths is not marginal -- on the reference
machine it is roughly eleven-fold. So rather than assume a fixed toolchain,
probe ffmpeg for what it really offers and let the renderer pick accordingly.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

#: OpenCL platforms whose name marks them as a CPU fallback. Running the
#: projection kernel on these is slower than ffmpeg's own CPU filter, so they
#: are never selected.
_CPU_OPENCL_HINTS = ("portable computing language", "pocl", "oclgrind")

_OPENCL_LINE = re.compile(r"^\s*\[.*?\]\s*(\d+\.\d+):\s*(.+?)\s*$", re.MULTILINE)


def subprocess_kwargs() -> dict:
    """Keyword arguments that keep helper processes from flashing a console.

    Only meaningful on Windows; a no-op elsewhere.
    """
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": startupinfo, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


class FFmpegMissingError(RuntimeError):
    """Neither ffmpeg nor ffprobe could be located."""


@dataclass(frozen=True)
class Capabilities:
    """What this machine can do, as reported by its own ffmpeg."""

    ffmpeg: str
    ffprobe: str
    version: str
    opencl_device: str | None = None
    opencl_name: str | None = None
    encoders: frozenset[str] = field(default_factory=frozenset)
    hwaccels: frozenset[str] = field(default_factory=frozenset)
    filters: frozenset[str] = field(default_factory=frozenset)

    # -- capability questions the renderer actually asks ------------------

    @property
    def has_nvenc(self) -> bool:
        return "h264_nvenc" in self.encoders

    @property
    def has_nvdec(self) -> bool:
        return "cuda" in self.hwaccels

    @property
    def has_opencl_kernel(self) -> bool:
        """Can we run the single-pass projection kernel on a real GPU?"""
        return self.opencl_device is not None and "program_opencl" in self.filters

    @property
    def has_v360(self) -> bool:
        """Software fallback for the projection."""
        return "v360" in self.filters

    @property
    def video_encoder(self) -> str:
        if self.has_nvenc:
            return "h264_nvenc"
        if "h264_vaapi" in self.encoders:
            return "h264_vaapi"
        return "libx264"

    def summary(self) -> str:
        bits = [f"ffmpeg {self.version}"]
        bits.append(f"decode: {'NVDEC' if self.has_nvdec else 'CPU'}")
        if self.has_opencl_kernel:
            bits.append(f"project: OpenCL ({self.opencl_name})")
        elif self.has_v360:
            bits.append("project: CPU (v360)")
        else:
            bits.append("project: unavailable")
        bits.append(f"encode: {self.video_encoder}")
        return " | ".join(bits)

    def explain_gaps(self) -> list[str]:
        """Human-readable warnings about missing acceleration."""
        gaps = []
        if not self.has_nvdec:
            gaps.append(
                "Décodage matériel indisponible : le décodage HEVC se fera sur "
                "le CPU (environ deux fois plus lent)."
            )
        if not self.has_opencl_kernel:
            gaps.append(
                "OpenCL indisponible : la projection équirectangulaire se fera "
                "sur le CPU, ce qui est nettement plus lent."
            )
        if not self.has_nvenc:
            gaps.append(
                "NVENC indisponible : l'encodage H.264 se fera sur le CPU."
            )
        return gaps


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("probe command failed: %s (%s)", cmd, exc)
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def _names(output: str, pattern: str) -> frozenset[str]:
    return frozenset(re.findall(pattern, output, re.MULTILINE))


def _probe_opencl(ffmpeg: str) -> tuple[str | None, str | None]:
    """Return the best usable ``platform.device`` spec, or ``(None, None)``.

    ffmpeg has no "list OpenCL devices" mode, but it enumerates them to the
    log while selecting one, whether or not that selection succeeds. So ask it
    to auto-select, read the listing it prints, then confirm the device we want
    really initialises.
    """
    listing = _run(
        [
            ffmpeg, "-hide_banner", "-v", "verbose",
            "-init_hw_device", "opencl",
            "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.04",
            "-f", "null", "-",
        ]
    )
    candidates = [
        (spec, name)
        for spec, name in _OPENCL_LINE.findall(listing)
        if not any(hint in name.lower() for hint in _CPU_OPENCL_HINTS)
    ]
    if not candidates:
        log.debug("no non-CPU OpenCL device found in:\n%s", listing)
        return None, None

    for spec, name in candidates:
        check = _run(
            [
                ffmpeg, "-hide_banner", "-v", "error",
                "-init_hw_device", f"opencl=ocl:{spec}",
                "-filter_hw_device", "ocl",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04",
                "-vf", "format=yuv420p,hwupload,hwdownload,format=yuv420p",
                "-f", "null", "-",
            ]
        )
        if "Device creation failed" in check or "No such device" in check:
            log.debug("OpenCL device %s (%s) refused initialisation", spec, name)
            continue
        return spec, name
    return None, None


@lru_cache(maxsize=1)
def detect(ffmpeg: str | None = None, ffprobe: str | None = None) -> Capabilities:
    """Probe the host toolchain. Cached: this costs a few seconds."""
    ffmpeg_path = ffmpeg or _which("ffmpeg")
    ffprobe_path = ffprobe or _which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise FFmpegMissingError(
            "ffmpeg et ffprobe sont introuvables. Installez-les "
            "(sous Ubuntu : sudo apt install ffmpeg) ou renseignez leur "
            "chemin dans les réglages."
        )

    banner = _run([ffmpeg_path, "-hide_banner", "-version"])
    version_match = re.search(r"ffmpeg version (\S+)", banner)
    version = version_match.group(1) if version_match else "inconnue"

    encoders = _names(_run([ffmpeg_path, "-hide_banner", "-encoders"]), r"^\s*V\S*\s+(\w+)")
    filters = _names(_run([ffmpeg_path, "-hide_banner", "-filters"]), r"^\s*\S+\s+(\w+)\s")
    hwaccel_out = _run([ffmpeg_path, "-hide_banner", "-hwaccels"])
    hwaccels = frozenset(
        line.strip()
        for line in hwaccel_out.splitlines()[1:]
        if line.strip() and " " not in line.strip()
    )

    device, name = _probe_opencl(ffmpeg_path)

    caps = Capabilities(
        ffmpeg=ffmpeg_path,
        ffprobe=ffprobe_path,
        version=version,
        opencl_device=device,
        opencl_name=name,
        encoders=encoders,
        hwaccels=hwaccels,
        filters=filters,
    )
    log.info("capabilities: %s", caps.summary())
    return caps


def app_dir() -> Path:
    """Folder the application was started from.

    Frozen builds must use ``sys.executable``: ``sys.argv[0]`` can be a
    relative path or a launcher shim, and the helper tools shipped alongside
    the .exe would then be looked for in the wrong place.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(sys.argv[0]).resolve().parent


#: Where a helper binary may sit next to the application, in the order tried.
#: ``vendor`` is where the Windows build drops ffmpeg and exiftool;
#: ``ffmpeg/bin`` matches the layout of the official Windows archives, so an
#: operator can unzip one whole and be done.
_TOOL_SUBDIRS = (".", "vendor", "ffmpeg/bin", "exiftool")


def find_tool(name: str) -> str | None:
    """Locate a helper executable: on PATH first, then beside the app.

    A packaged Windows build has no PATH to speak of -- the operator unzips a
    folder and double-clicks -- so shipping the tools inside it has to work
    without any environment setup.
    """
    found = shutil.which(name)
    if found:
        return found
    filename = name + (".exe" if sys.platform == "win32" else "")
    root = app_dir()
    for relative in _TOOL_SUBDIRS:
        candidate = (root / relative / filename).resolve()
        if candidate.is_file():
            return str(candidate)
    return None


def _which(name: str) -> str | None:
    return find_tool(name)
