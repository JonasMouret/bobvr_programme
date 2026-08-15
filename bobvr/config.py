"""Application settings, persisted as JSON next to the user's other app data.

Everything an operator might reasonably want to change lives here, so the UI
never has to reach into the pipeline's internals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from platformdirs import user_config_dir, user_videos_dir
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

APP_NAME = "BobVr"

#: The sleds on the track, and how many of each. Card labels are a prefix
#: followed by a number, e.g. "B1" is the first bobsleigh.
DEFAULT_FLEET: dict[str, int] = {"B": 6, "S": 6, "R": 5}

#: What each prefix means, for display.
FLEET_LABELS: dict[str, str] = {
    "B": "Bobsleigh",
    "S": "Speedluge",
    "R": "Bob Racing",
}


def config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


class RenderConfig(BaseModel):
    """Output shape and quality for the equirectangular render."""

    width: int = 3840
    height: int = 1920
    quality: int = Field(default=23, ge=1, le=51)
    max_bitrate_kbps: int = Field(default=60_000, gt=0)
    audio_bitrate_kbps: int = Field(default=192, gt=0)
    cubic: bool = True
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    force_cpu: bool = False

    #: How wide a view the finished MP4 opens on, in degrees. Baked into the
    #: projection, because no metadata field carries a field of view and no
    #: player would read one. 80 leaves the sphere untouched; above that the
    #: view opens wider, below it tightens.
    initial_fov: float = Field(default=80.0, ge=40.0, le=150.0)

    @field_validator("width")
    @classmethod
    def _even_width(cls, v: int) -> int:
        if v % 2:
            raise ValueError("la largeur doit être paire")
        return v

    @field_validator("height")
    @classmethod
    def _even_height(cls, v: int) -> int:
        if v % 2:
            raise ValueError("la hauteur doit être paire")
        return v

    def model_post_init(self, _context) -> None:
        if self.width != 2 * self.height:
            raise ValueError(
                f"une vidéo équirectangulaire doit être au format 2:1 ; "
                f"{self.width}x{self.height} ne l'est pas"
            )


class Settings(BaseModel):
    """Everything the app remembers between runs."""

    library_root: Path = Field(
        default_factory=lambda: Path(user_videos_dir()) / APP_NAME
    )
    #: Sub-directories under ``library_root``. Both are grouped by capture date.
    archive_dirname: str = "originaux"
    render_dirname: str = "equirect"

    fleet: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_FLEET))

    #: How deep to look for .360 files on a card. Cards are searched
    #: recursively because the camera's folder sits at the root on some cards
    #: and under DCIM on others, and rolls over to 101GOPRO, 102GOPRO... as it
    #: fills. Four levels covers every layout seen without crawling a card
    #: that has been used as general storage.
    card_search_depth: int = Field(default=4, ge=1, le=8)

    auto_ingest: bool = True
    auto_delete_source: bool = True
    #: Once a card's clips are copied and verified, remove the camera's own
    #: folder (``DCIM/100GOPRO``) whole and put an empty one back, rather than
    #: unlinking the .360 files and leaving the sidecars behind.
    clear_camera_folder: bool = True
    auto_eject: bool = True
    verify_checksum: bool = True
    auto_render: bool = True

    render: RenderConfig = Field(default_factory=RenderConfig)

    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None

    @property
    def archive_root(self) -> Path:
        return self.library_root / self.archive_dirname

    @property
    def render_root(self) -> Path:
        return self.library_root / self.render_dirname

    def known_card_ids(self) -> list[str]:
        return [
            f"{prefix}{n}"
            for prefix, count in sorted(self.fleet.items())
            for n in range(1, count + 1)
        ]

    def describe_card(self, card_id: str) -> str:
        prefix = card_id[:1].upper()
        name = FLEET_LABELS.get(prefix, "Engin")
        return f"{name} {card_id[1:]}"

    # ------------------------------------------------------------ storage

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        target = path or config_path()
        if not target.exists():
            log.info("no settings file at %s, using defaults", target)
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("settings at %s are unreadable (%s); using defaults", target, exc)
            return cls()
        try:
            return cls.model_validate(data)
        except ValueError as exc:
            log.warning("settings at %s are invalid (%s); using defaults", target, exc)
            return cls()

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=False)
        # Write through a temporary file: a crash mid-write must not leave the
        # app unable to start next time.
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return target
