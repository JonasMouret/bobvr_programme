"""Card detection, with the right backend for the running platform."""

from __future__ import annotations

import sys

from .base import Card, CardWatcher, Volume, VolumeBackend, parse_card_id

__all__ = [
    "Card",
    "CardWatcher",
    "Volume",
    "VolumeBackend",
    "parse_card_id",
    "make_backend",
]


def make_backend() -> VolumeBackend:
    """Return a volume backend for this OS.

    Raises:
        RuntimeError: the platform is unsupported, or its tooling is missing.
    """
    if sys.platform == "win32":
        from .windows import WindowsVolumeBackend

        return WindowsVolumeBackend()
    if sys.platform.startswith("linux"):
        from .linux import LinuxVolumeBackend

        return LinuxVolumeBackend()
    raise RuntimeError(
        f"La détection des cartes n'est pas prise en charge sur {sys.platform}."
    )
