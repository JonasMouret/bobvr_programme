"""Shared fixtures.

Tests that need a real .360 file use ``BOBVR_TEST_360`` if it points at one,
and skip otherwise, so the suite stays runnable on a machine without sample
footage.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bobvr.cards.base import Card, Volume, VolumeBackend
from bobvr.config import Settings


@pytest.fixture(scope="session")
def sample_360() -> Path:
    raw = os.environ.get("BOBVR_TEST_360")
    if not raw:
        pytest.skip("BOBVR_TEST_360 n'est pas défini")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"{path} est introuvable")
    return path


@pytest.fixture(scope="session")
def caps():
    from bobvr.render.caps import FFmpegMissingError, detect

    try:
        return detect()
    except FFmpegMissingError:
        pytest.skip("ffmpeg n'est pas installé")


class FakeBackend(VolumeBackend):
    """A volume backend driven by the test, with ejection recorded."""

    def __init__(self, volumes: list[Volume] | None = None):
        self.volumes = list(volumes or [])
        self.ejected: list[Volume] = []
        self.eject_result = (True, "ok")

    def list_volumes(self) -> list[Volume]:
        return list(self.volumes)

    def eject(self, volume: Volume) -> tuple[bool, str]:
        self.ejected.append(volume)
        return self.eject_result


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(library_root=tmp_path / "library")


@pytest.fixture
def make_card():
    """Build a Card standing in for a mounted, correctly labelled SD card."""

    def build(card_id: str, mountpoint: Path) -> Card:
        return Card(
            card_id=card_id,
            volume=Volume(
                label=card_id,
                mountpoint=mountpoint,
                device=f"/dev/fake-{card_id}",
            ),
        )

    return build
