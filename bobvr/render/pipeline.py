"""Turning a .360 file into a phone-playable equirectangular H.264 MP4.

Two routes exist. The fast one uploads both HEVC tracks to the GPU and runs
the whole projection as a single OpenCL kernel; the fallback rebuilds the
cubemap with ffmpeg's own crop/stack filters and projects it with ``v360`` on
the CPU. On the reference machine the first is about eleven times quicker, so
the fallback exists to keep the app usable on a host without a GPU, not as an
equal alternative.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from ..media import MaxVideoInfo
from .caps import Capabilities, subprocess_kwargs
from .geometry import (
    MAX_FOV,
    MIN_FOV,
    NEUTRAL_FOV,
    MaxLayout,
    Orientation,
    ROLLING_AXES,
    ROLLING_SPAN,
    TooManyFramesError,
    materialise_kernel,
    view_scale,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[["RenderProgress"], None]


def _escape_filtergraph_path(path) -> str:
    """Make a file path safe as a value inside an ffmpeg ``-filter_complex``.

    Two things bite on Windows. A backslash is the filtergraph escape
    character, so the path uses forward slashes instead (ffmpeg accepts them on
    Windows). And the drive colon in ``C:/…`` is the option separator: it has
    to survive *two* levels of unescaping -- the graph level and the
    filter-arguments level -- so it is written ``\\:`` (two backslashes). With a
    single backslash the graph level consumes it and the argument parser then
    splits the value at the bare colon, so ``program_opencl`` only ever sees
    ``C`` and reports "Unable to open program source file". Forward-slashed
    POSIX paths carry no colon and pass through untouched.
    """
    return str(path).replace("\\", "/").replace(":", r"\\:")


class RenderError(RuntimeError):
    """The render failed or was refused."""


class RenderCancelled(RenderError):
    """The render was stopped on request."""


@dataclass(frozen=True)
class RenderSettings:
    """Output shape and quality.

    ``quality`` is a constant-quantiser target: lower is better and larger.
    23 lands around 40 Mbit/s at 4K, which modern phones stream comfortably.
    """

    width: int = 3840
    height: int = 1920
    orientation: Orientation = Orientation()
    cubic: bool = True
    #: How wide a view the file opens on, in degrees. Baked into the
    #: projection: see ``NEUTRAL_FOV`` for why it cannot be metadata.
    initial_fov: float = NEUTRAL_FOV
    #: Time constant of the stabilisation smoother, in seconds; 0 turns it off.
    #: Bigger is calmer, and in 360 that costs nothing but a larger rotation --
    #: the sphere is turned, not cropped, so there is no margin to run out of.
    #:
    #: Off by default until the result has been judged on real runs. Note what
    #: cannot settle that: any measure built on correlating consecutive frames.
    #: There is signal enough -- 0.6 degrees a frame is about 7 pixels at 3840
    #: wide -- but the tunnel's walls repeat, so the correlation locks onto the
    #: wrong peak often enough to throw impossible values (a whole turn inside
    #: a second where the gyroscope is quiet). Checking this means looking at
    #: the picture, or tracking the sled, which cannot be mistaken for itself.
    stabilise_seconds: float = 0.0
    #: Sensor scan order, for undoing the rolling shutter's within-frame skew;
    #: ``None`` corrects each frame as a whole and leaves the skew. See
    #: ``geometry.ROLLING_AXES``. Only does anything alongside
    #: ``stabilise_seconds``.
    rolling_axis: str | None = None
    #: How long the scan takes to cross the sphere, in frame periods.
    rolling_span: float = ROLLING_SPAN
    quality: int = 23
    max_bitrate_kbps: int = 60_000
    audio_bitrate_kbps: int = 192
    gop_seconds: float = 2.0
    force_cpu: bool = False
    #: Optional trim, used for preview renders.
    start: float | None = None
    duration: float | None = None

    def validate(self) -> None:
        if self.width % 2 or self.height % 2:
            raise RenderError("Les dimensions de sortie doivent être paires.")
        if self.width != 2 * self.height:
            raise RenderError(
                f"Une vidéo équirectangulaire doit être au format 2:1 ; "
                f"{self.width}x{self.height} ne l'est pas. La navigation 360 "
                "serait déformée dans les lecteurs."
            )
        if not 1 <= self.quality <= 51:
            raise RenderError("La qualité doit être comprise entre 1 et 51.")
        if not MIN_FOV <= self.initial_fov <= MAX_FOV:
            raise RenderError(
                f"Le champ de vision doit être compris entre {MIN_FOV:g}° et "
                f"{MAX_FOV:g}°."
            )
        if self.rolling_axis is not None and self.rolling_axis not in ROLLING_AXES:
            raise RenderError(
                f"Ordre de lecture du capteur inconnu : {self.rolling_axis!r}."
            )
        if self.stabilise_seconds < 0:
            raise RenderError(
                "La constante de lissage de la stabilisation ne peut pas être "
                "négative ; 0 la désactive."
            )


@dataclass(frozen=True)
class RenderProgress:
    frame: int
    fps: float
    seconds_done: float
    speed: float
    total_seconds: float

    @property
    def fraction(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return min(1.0, self.seconds_done / self.total_seconds)

    @property
    def eta_seconds(self) -> float | None:
        remaining = self.total_seconds - self.seconds_done
        if remaining <= 0 or self.speed <= 0:
            return None
        return remaining / self.speed


_PROGRESS_LINE = re.compile(r"^(\w+)=(.*)$")


class Renderer:
    """Builds and runs render commands for a given host."""

    def __init__(self, caps: Capabilities, cache_dir: Path):
        self.caps = caps
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------ planning

    def uses_gpu(self, settings: RenderSettings) -> bool:
        return self.caps.has_opencl_kernel and not settings.force_cpu

    def describe_route(self, settings: RenderSettings) -> str:
        if self.uses_gpu(settings):
            return f"GPU · kernel OpenCL ({self.caps.opencl_name})"
        if self.caps.has_v360:
            return "CPU · filtre v360 (lent)"
        return "indisponible"

    def build_command(
        self,
        info: MaxVideoInfo,
        output: Path,
        settings: RenderSettings,
        container: str | None = None,
    ) -> list[str]:
        cmd = self._decode_and_project(info, settings)
        cmd += ["-map", "[v]"]
        if info.audio_index is not None:
            cmd += ["-map", f"0:{info.audio_index}"]

        cmd += self._encoder_args(info, settings)
        if container:
            # Renders land on a ".part" name first, which gives ffmpeg no
            # extension to infer a muxer from.
            cmd += ["-f", container]
        cmd += ["-progress", "pipe:1", "-nostats", str(output)]
        return cmd

    def _decode_and_project(
        self, info: MaxVideoInfo, settings: RenderSettings
    ) -> list[str]:
        """Everything up to and including the ``[v]`` projected stream.

        Shared by full renders and single-frame previews so the two can never
        drift apart -- a preview that does not match the render is worse than
        no preview.
        """
        settings.validate()
        layout = MaxLayout(src_w=info.track_width, src_h=info.track_height)
        layout.validate()

        cmd = [self.caps.ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error"]

        use_gpu = self.uses_gpu(settings)
        if use_gpu:
            cmd += ["-init_hw_device", f"opencl=ocl:{self.caps.opencl_device}",
                    "-filter_hw_device", "ocl"]
        # Hardware decode helps both routes: the two HEVC tracks alone can
        # saturate a modest CPU.
        if self.caps.has_nvdec:
            cmd += ["-hwaccel", "cuda"]

        if settings.start is not None:
            cmd += ["-ss", f"{settings.start:.3f}"]
        cmd += ["-i", str(info.path)]
        if settings.duration is not None:
            cmd += ["-t", f"{settings.duration:.3f}"]

        a, b = info.video[0].index, info.video[1].index
        if use_gpu:
            table, offset = self._stabilisation(info, settings)
            graph = self._gpu_graph(a, b, layout, settings, table, offset)
        elif self.caps.has_v360:
            if settings.stabilise_seconds > 0:
                # v360 takes one fixed orientation for the whole render; there
                # is nowhere to put a rotation that changes with the frame.
                raise RenderError(
                    "La stabilisation demande le kernel OpenCL : le repli "
                    "logiciel ne sait appliquer qu'une orientation fixe."
                )
            graph = self._cpu_graph(a, b, layout, settings)
        else:
            raise RenderError(
                "Cette installation de ffmpeg ne propose ni OpenCL ni le filtre "
                "v360 ; la projection équirectangulaire est impossible."
            )
        cmd += ["-filter_complex", graph]
        return cmd

    # ------------------------------------------------------------- graphs

    def _stabilisation(
        self, info: MaxVideoInfo, settings: RenderSettings
    ) -> tuple[list[tuple[float, float, float, float]] | None, int]:
        """The per-frame rotations for this clip, and where to start reading.

        The table is indexed by the clip's own frame numbering, while the
        kernel counts from zero at whatever reaches it -- so a render that
        seeks first has to say how far in it started.
        """
        if settings.stabilise_seconds <= 0:
            return None, 0
        from .. import orientation, telemetry

        try:
            tel = telemetry.extract(info.path, info, self.caps)
            frame = orientation.solve_gyro_frame(tel)
            # One entry per *video* frame, which is what the kernel counts.
            #
            # CORI runs at the video frame rate, and sample k belongs to frame
            # k: the table is therefore sampled from the first quaternion's
            # own instant, one frame period apart. That alignment was measured
            # on the picture two independent ways -- the launch, where a still
            # image and a still gyroscope start moving together (-0.188 s),
            # and a fit of image rotation against gyroscope rotation over the
            # hardest curve (-0.1815 s). Both land on CORI[0] = -0.1851 s.
            #
            # Do *not* subtract ``clock_offset_us`` here. VPTS - STMP is 1.1 s
            # on this camera and putting it in shifts the table by 38 frames,
            # which is what the measurements above rule out.
            fps = info.fps or 30.0
            count = round((info.duration or 0.0) * fps) + 1
            stamps = tel.sensor_times("CORI")
            first = stamps[0] if stamps else 0.0
            times = [first + k / fps for k in range(count)]
            table = orientation.stabilise(
                tel, times, settings.stabilise_seconds, frame)
        except (telemetry.TelemetryError, orientation.OrientationError) as exc:
            raise RenderError(f"Stabilisation impossible : {exc}") from exc
        if not frame.trustworthy:
            raise RenderError(
                f"Les axes du gyroscope n'ont pas pu être établis de façon "
                f"fiable sur {info.path.name} (résidu {frame.residual:.0%} sur "
                f"{frame.windows} fenêtres) ; la stabilisation serait fausse."
            )
        offset = round((settings.start or 0.0) * (info.fps or 30.0))
        return table, offset

    def _gpu_graph(
        self, a: int, b: int, layout: MaxLayout, settings: RenderSettings,
        stabilisation: list[tuple[float, float, float, float]] | None = None,
        frame_offset: int = 0,
    ) -> str:
        try:
            kernel = materialise_kernel(
                self.cache_dir, layout, settings.orientation, settings.cubic,
                settings.initial_fov, stabilisation, frame_offset,
                settings.rolling_axis, settings.rolling_span,
            )
        except TooManyFramesError as exc:
            raise RenderError(str(exc)) from exc
        source = _escape_filtergraph_path(kernel)
        return (
            f"[0:{a}]format=yuv420p,hwupload[t0];"
            f"[0:{b}]format=yuv420p,hwupload[t1];"
            f"[t0][t1]program_opencl="
            f"source={source}:kernel=gopromax_equirect:inputs=2:"
            f"size={settings.width}x{settings.height},"
            f"hwdownload,format=yuv420p[v]"
        )

    def _cpu_graph(
        self, a: int, b: int, layout: MaxLayout, settings: RenderSettings
    ) -> str:
        """Rebuild the EAC frame with plain filters, then project it.

        The seams are collapsed by keeping one lens and discarding the other's
        copy. GoPro's own stitch cross-fades them instead, but doing that with
        ffmpeg's ``geq`` costs more than the entire rest of the render, so this
        route trades a faint seam for staying usable at all.
        """
        h = layout.src_h
        half = layout.seam_src_w // 2
        parts = []
        for tag, stream in (("A", a), ("B", b)):
            # (source x, source width, width to stretch it to)
            segs = [
                (0, layout.seam0_eac, None),
                (layout.seam0_src, half, layout.seam_eac_w),
                (layout.seam0_src + layout.seam_src_w,
                 layout.seam1_eac - (layout.seam0_eac + layout.seam_eac_w), None),
                (layout.seam1_src, half, layout.seam_eac_w),
                (layout.seam1_src + layout.seam_src_w,
                 layout.eac_w - (layout.seam1_eac + layout.seam_eac_w), None),
            ]
            labels = []
            for i, (x, w, scale_to) in enumerate(segs):
                label = f"{tag}{i}"
                chain = f"[0:{stream}]crop={w}:{h}:{x}:0"
                if scale_to:
                    chain += f",scale={scale_to}:{h}"
                parts.append(f"{chain}[{label}]")
                labels.append(label)
            merged = labels[0]
            for i, label in enumerate(labels[1:], start=1):
                out = f"{tag}m{i}"
                parts.append(f"[{merged}][{label}]hstack[{out}]")
                merged = out
            parts.append(f"[{merged}]null[{tag}row]")
        parts.append("[Arow][Brow]vstack[eac]")
        if view_scale(settings.initial_fov) != 1.0:
            # v360 projects the sphere; it cannot redistribute it. Every way of
            # faking the field of view with plain filters -- shrinking the
            # sphere onto a smaller canvas, or scaling the angles -- curves the
            # straight lines, which is precisely what the setting exists to
            # avoid. Better to say so than to hand back a bent picture.
            raise RenderError(
                f"Le champ de vision ({settings.initial_fov:g}°) ne peut être "
                f"appliqué que par le kernel OpenCL. Sans GPU, gardez la valeur "
                f"neutre de {NEUTRAL_FOV:g}° : le repli logiciel ne sait pas "
                "redistribuer la sphère sans courber les lignes droites."
            )
        parts.append(
            f"[eac]v360=eac:e:w={settings.width}:h={settings.height}"
            f":interp={'cubic' if settings.cubic else 'linear'}"
            f":yaw={settings.orientation.yaw}"
            f":pitch={settings.orientation.pitch}"
            f":roll={settings.orientation.roll}[v]"
        )
        return ";".join(parts)

    # ------------------------------------------------------------ encoding

    def _encoder_args(self, info: MaxVideoInfo, settings: RenderSettings) -> list[str]:
        fps = info.fps or 30.0
        gop = max(1, int(round(fps * settings.gop_seconds)))
        encoder = self.caps.video_encoder
        args = ["-c:v", encoder]

        if encoder == "h264_nvenc":
            args += [
                "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", str(settings.quality),
                "-b:v", "0",
                "-maxrate", f"{settings.max_bitrate_kbps}k",
                "-bufsize", f"{settings.max_bitrate_kbps * 2}k",
                # Phones decode High profile at these sizes; keep B-frames
                # modest so seeking stays responsive.
                "-profile:v", "high", "-bf", "2",
            ]
        else:
            args += [
                "-preset", "medium", "-crf", str(settings.quality),
                "-maxrate", f"{settings.max_bitrate_kbps}k",
                "-bufsize", f"{settings.max_bitrate_kbps * 2}k",
                "-profile:v", "high",
            ]

        args += [
            "-g", str(gop), "-keyint_min", str(gop),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if info.audio_index is not None:
            args += ["-c:a", "aac", "-b:a", f"{settings.audio_bitrate_kbps}k", "-ac", "2"]
        return args

    # ------------------------------------------------------------- running

    def render(
        self,
        info: MaxVideoInfo,
        output: Path,
        settings: RenderSettings,
        on_progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        """Render to ``output``, returning its path.

        Writes to a sibling ``.part`` file first, so an interrupted render
        never leaves something that looks like a finished video.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".part")
        partial.unlink(missing_ok=True)

        cmd = self.build_command(info, partial, settings, container="mp4")
        total = settings.duration or info.duration
        log.info("render %s -> %s (%s)", info.path.name, output.name,
                 self.describe_route(settings))
        log.debug("command: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **subprocess_kwargs(),
        )
        stderr_tail: list[str] = []
        stderr_thread = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_tail), daemon=True
        )
        stderr_thread.start()

        fields: dict[str, str] = {}
        try:
            for line in proc.stdout:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    raise RenderCancelled("Rendu annulé.")
                match = _PROGRESS_LINE.match(line.strip())
                if not match:
                    continue
                key, value = match.group(1), match.group(2)
                fields[key] = value
                if key == "progress" and on_progress is not None:
                    on_progress(_progress_from(fields, total))
        finally:
            proc.stdout.close()
            returncode = proc.wait()
            stderr_thread.join(timeout=2.0)

        if cancel is not None and cancel.is_set():
            partial.unlink(missing_ok=True)
            raise RenderCancelled("Rendu annulé.")
        if returncode != 0:
            partial.unlink(missing_ok=True)
            detail = "\n".join(stderr_tail[-12:]).strip() or f"code de sortie {returncode}"
            raise RenderError(f"Échec du rendu de {info.path.name} :\n{detail}")
        if not partial.exists() or partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise RenderError(f"Le rendu de {info.path.name} n'a produit aucune donnée.")

        partial.replace(output)
        return output

    def extract_frame(
        self, info: MaxVideoInfo, output: Path, settings: RenderSettings, at: float
    ) -> Path:
        """Render a single projected frame, for previews."""
        output = Path(output)
        one = replace(settings, start=at, duration=None)
        cmd = self._decode_and_project(info, one)
        cmd += ["-map", "[v]", "-frames:v", "1", "-update", "1", str(output)]
        output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            cmd, capture_output=True, text=True, **subprocess_kwargs()
        )
        if result.returncode != 0 or not output.exists():
            raise RenderError(
                f"Impossible d'extraire l'aperçu : {result.stderr.strip()[-400:]}"
            )
        return output


def _drain(stream, sink: list[str]) -> None:
    for line in stream:
        sink.append(line.rstrip())
        if len(sink) > 200:
            del sink[:100]
    stream.close()


def _progress_from(fields: dict[str, str], total: float) -> RenderProgress:
    def num(key: str, default: float = 0.0) -> float:
        raw = fields.get(key, "")
        try:
            return float(raw)
        except ValueError:
            return default

    speed_raw = fields.get("speed", "").strip().rstrip("x")
    try:
        speed = float(speed_raw)
    except ValueError:
        speed = 0.0

    return RenderProgress(
        frame=int(num("frame")),
        fps=num("fps"),
        seconds_done=num("out_time_us") / 1_000_000.0,
        speed=speed,
        total_seconds=total,
    )


# -------------------------------------------------------- 360 metadata

#: The Spherical Video V1 box identity and payload. Players (VLC, YouTube,
#: GoPro Player) look for this exact uuid inside the video track and read the
#: RDF/XML to switch into 360. Written here in pure Python rather than shelled
#: out to exiftool: the tag is four fixed lines, and bundling exiftool (a whole
#: Perl runtime) on Windows just to write them was never worth the weight.
_SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
_SPHERICAL_XML = (
    "<rdf:SphericalVideo\n"
    " xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'\n"
    " xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>\n"
    "  <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>\n"
    "  <GSpherical:Spherical>True</GSpherical:Spherical>\n"
    "  <GSpherical:Stitched>True</GSpherical:Stitched>\n"
    "  <GSpherical:StitchingSoftware>{software}</GSpherical:StitchingSoftware>\n"
    "</rdf:SphericalVideo>"
)

#: MP4 boxes that hold other boxes; the rest are leaves whose bytes we copy.
_MP4_CONTAINERS = frozenset(
    {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}
)


def _mp4_boxes(buf, start: int, end: int):
    """Yield ``(offset, type, header_len, size)`` for each box in a range."""
    p = start
    while p + 8 <= end:
        size = int.from_bytes(buf[p:p + 4], "big")
        typ = bytes(buf[p + 4:p + 8])
        header = 8
        if size == 1:                       # 64-bit size follows the type
            size = int.from_bytes(buf[p + 8:p + 16], "big")
            header = 16
        elif size == 0:                     # runs to the end of its parent
            size = end - p
        if size < header or p + size > end:
            break
        yield p, typ, header, size
        p += size


def _mp4_find(buf, start: int, end: int, path: tuple[bytes, ...]):
    """Find the first box matching a nested ``path``; ``None`` if absent."""
    want = path[0]
    for offset, typ, header, size in _mp4_boxes(buf, start, end):
        if typ == want:
            if len(path) == 1:
                return offset, header, size
            if typ in _MP4_CONTAINERS:
                found = _mp4_find(buf, offset + header, offset + size, path[1:])
                if found:
                    return found
    return None


def _mp4_traks(buf, moov_start: int, header: int, size: int):
    for offset, typ, h, sz in _mp4_boxes(buf, moov_start + header, moov_start + size):
        if typ == b"trak":
            yield offset, h, sz


def _mp4_handler(buf, trak_start: int, header: int, size: int) -> bytes | None:
    """The four-letter handler of a track: ``b"vide"``, ``b"soun"``, …"""
    found = _mp4_find(buf, trak_start + header, trak_start + size, (b"mdia", b"hdlr"))
    if not found:
        return None
    hs, hh, _ = found
    # hdlr payload: version+flags (4), pre_defined (4), handler_type (4).
    return bytes(buf[hs + hh + 8:hs + hh + 12])


def inject_spherical_metadata(path: Path, software: str = "BobVr") -> bool:
    """Tag ``path`` as a stitched equirectangular 360 video, in place.

    Writes the Spherical Video V1 ``uuid`` box into the video track. Without
    it, players show the frame flat and finger-drag navigation never appears.
    Returns False (with a warning) rather than raising, so an MP4 we cannot
    parse costs the metadata but not the render.

    When ``moov`` sits before ``mdat`` -- which it does here, because the render
    muxes with ``+faststart`` -- growing ``moov`` pushes the media data down, so
    every chunk offset in ``stco``/``co64`` is shifted to match. Get that wrong
    and the file still lists as spherical but decodes to garbage.

    Note what is deliberately *not* written: a field of view. No
    spherical-video specification carries one, so it is settled in the
    projection instead; see ``NEUTRAL_FOV``.
    """
    try:
        buf = bytearray(path.read_bytes())
    except OSError as exc:
        log.warning("360 : lecture de %s impossible (%s)", path.name, exc)
        return False

    moov = mdat = None
    for offset, typ, header, size in _mp4_boxes(buf, 0, len(buf)):
        if typ == b"moov":
            moov = (offset, header, size)
        elif typ == b"mdat":
            mdat = offset
    if moov is None:
        log.warning("360 : pas de boîte moov dans %s", path.name)
        return False
    moov_start, moov_header, moov_size = moov

    video = None
    for ts, th, tz in _mp4_traks(buf, moov_start, moov_header, moov_size):
        if _mp4_handler(buf, ts, th, tz) == b"vide":
            video = (ts, th, tz)
            break
    if video is None:
        log.warning("360 : pas de piste vidéo dans %s", path.name)
        return False
    vtrak_start, _, vtrak_size = video

    # Already tagged (e.g. a re-run): leave it be rather than stack a second box.
    if _SPHERICAL_UUID in buf[vtrak_start:vtrak_start + vtrak_size]:
        return True

    xml = _SPHERICAL_XML.format(software=software).encode("utf-8")
    box = (8 + 16 + len(xml)).to_bytes(4, "big") + b"uuid" + _SPHERICAL_UUID + xml
    added = len(box)
    # Faststart puts moov first, so the media below it slides down by `added`.
    shift = added if (mdat is not None and mdat > moov_start) else 0

    moov_bytes = bytearray(buf[moov_start:moov_start + moov_size])
    if shift:
        for ts, th, tz in _mp4_traks(buf, moov_start, moov_header, moov_size):
            for name, entry in ((b"stco", 4), (b"co64", 8)):
                found = _mp4_find(
                    buf, ts + th, ts + tz, (b"mdia", b"minf", b"stbl", name)
                )
                if not found:
                    continue
                cs, ch, _ = found
                rel = cs - moov_start
                count = int.from_bytes(moov_bytes[rel + ch + 4:rel + ch + 8], "big")
                pos = rel + ch + 8
                for _ in range(count):
                    value = int.from_bytes(moov_bytes[pos:pos + entry], "big") + shift
                    moov_bytes[pos:pos + entry] = value.to_bytes(entry, "big")
                    pos += entry

    # Insert the box as the last child of the video track, then grow the two
    # container sizes that now enclose it.
    rel_trak = vtrak_start - moov_start
    insert_at = rel_trak + vtrak_size
    moov_bytes[insert_at:insert_at] = box
    new_trak = int.from_bytes(moov_bytes[rel_trak:rel_trak + 4], "big") + added
    moov_bytes[rel_trak:rel_trak + 4] = new_trak.to_bytes(4, "big")
    moov_bytes[0:4] = (moov_size + added).to_bytes(4, "big")

    try:
        path.write_bytes(
            bytes(buf[:moov_start]) + bytes(moov_bytes) + bytes(buf[moov_start + moov_size:])
        )
    except OSError as exc:
        log.warning("360 : écriture de %s impossible (%s)", path.name, exc)
        return False
    return True
