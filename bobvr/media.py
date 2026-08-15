"""Inspecting .360 files with ffprobe.

A GoPro MAX .360 is an MP4 carrying two HEVC video tracks (the two lenses),
one AAC track, and a ``gpmd`` data track holding telemetry. Everything the
renderer needs to build its filter graph is decided from here, so this module
is deliberately strict: a file that does not look like a .360 is rejected up
front rather than producing a confusing ffmpeg error later.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .render.caps import Capabilities, subprocess_kwargs

log = logging.getLogger(__name__)


class NotAMaxVideoError(ValueError):
    """The file is not a two-lens GoPro MAX capture."""


@dataclass(frozen=True)
class Track:
    index: int
    codec: str
    width: int
    height: int


@dataclass(frozen=True)
class MaxVideoInfo:
    """The parts of a .360 file the pipeline cares about."""

    path: Path
    duration: float
    size: int
    fps: float
    video: tuple[Track, Track]
    audio_index: int | None
    telemetry_index: int | None
    created: datetime | None

    @property
    def track_width(self) -> int:
        return self.video[0].width

    @property
    def track_height(self) -> int:
        return self.video[0].height

    @property
    def has_telemetry(self) -> bool:
        return self.telemetry_index is not None


def _parse_fps(value: str | None) -> float:
    if not value or "/" not in value:
        return 0.0
    num, _, den = value.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return 0.0
    return n / d if d else 0.0


def _parse_created(tags: dict) -> datetime | None:
    raw = tags.get("creation_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def probe(path: Path, caps: Capabilities, timeout: float = 60.0) -> MaxVideoInfo:
    """Read a .360 file's structure.

    Raises:
        NotAMaxVideoError: the file is unreadable, or is not a MAX capture.
    """
    cmd = [
        caps.ffprobe, "-v", "error",
        "-show_streams", "-show_format",
        "-print_format", "json", str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=True, **subprocess_kwargs(),
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip()
        if "moov atom not found" in detail:
            # GoPro writes the index last, so a missing moov means the tail of
            # the file never made it. Usually an interrupted copy or a card
            # pulled while the camera was still recording.
            raise NotAMaxVideoError(
                f"{path.name} : fichier incomplet — l'index (moov) est absent. "
                "La copie sur la carte a été interrompue, ou la carte a été "
                "retirée pendant l'enregistrement. Le fichier est inutilisable "
                "et doit être recopié depuis la source."
            ) from exc
        raise NotAMaxVideoError(
            f"{path.name} : ffprobe n'a pas pu lire le fichier ({detail})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise NotAMaxVideoError(f"{path.name} : analyse interrompue (délai dépassé)") from exc

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise NotAMaxVideoError(f"{path.name} : sortie ffprobe illisible") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    videos = [
        Track(
            index=int(s["index"]),
            codec=s.get("codec_name", "?"),
            width=int(s.get("width") or 0),
            height=int(s.get("height") or 0),
        )
        for s in streams
        if s.get("codec_type") == "video"
    ]
    if len(videos) != 2:
        raise NotAMaxVideoError(
            f"{path.name} : {len(videos)} piste(s) vidéo trouvée(s), 2 attendues. "
            "Ce fichier n'est probablement pas une capture GoPro MAX en 360."
        )
    if videos[0].width != videos[1].width or videos[0].height != videos[1].height:
        raise NotAMaxVideoError(
            f"{path.name} : les deux pistes vidéo ont des tailles différentes "
            f"({videos[0].width}x{videos[0].height} et {videos[1].width}x{videos[1].height})."
        )

    audio = next((int(s["index"]) for s in streams if s.get("codec_type") == "audio"), None)
    telemetry = next(
        (
            int(s["index"])
            for s in streams
            if s.get("codec_tag_string") == "gpmd"
        ),
        None,
    )

    fps = _parse_fps(next((s.get("avg_frame_rate") for s in streams if s.get("codec_type") == "video"), None))

    return MaxVideoInfo(
        path=path,
        duration=float(fmt.get("duration") or 0.0),
        size=int(fmt.get("size") or path.stat().st_size),
        fps=fps,
        video=(videos[0], videos[1]),
        audio_index=audio,
        telemetry_index=telemetry,
        created=_parse_created(fmt.get("tags", {}) or {}),
    )
