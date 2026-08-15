"""Watching for SD cards, independently of how the OS reports them.

Cards are identified by their volume label: a card labelled ``B1`` is the
first bobsleigh. That choice keeps the card self-describing -- it survives
being moved between readers and machines, and an operator can read it off the
card itself.

Detection works by polling the list of mounted volumes rather than listening
for device events. The card still has to be mounted by the desktop before we
can read it, so an event would only tell us to start waiting; polling costs a
trivial amount and behaves identically on Linux and Windows.
"""

from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

#: A usable card label: one fleet letter followed by a number.
CARD_LABEL = re.compile(r"^([A-Z])(\d{1,2})$")


@dataclass(frozen=True)
class Volume:
    """A mounted removable volume."""

    label: str
    mountpoint: Path
    device: str | None = None
    uuid: str | None = None
    fstype: str | None = None

    @property
    def key(self) -> str:
        """Stable identity for diffing between polls."""
        return self.device or str(self.mountpoint)


@dataclass(frozen=True)
class Card:
    """A volume recognised as belonging to the fleet."""

    card_id: str
    volume: Volume

    @property
    def mountpoint(self) -> Path:
        return self.volume.mountpoint

    @property
    def prefix(self) -> str:
        return self.card_id[:1]


def parse_card_id(label: str, fleet: dict[str, int]) -> str | None:
    """Return the canonical card id for ``label``, or None if it is not ours.

    Labels are matched case-insensitively and against the configured fleet, so
    a stray USB stick labelled ``B99`` is ignored rather than treated as a
    bobsleigh that does not exist.
    """
    if not label:
        return None
    match = CARD_LABEL.match(label.strip().upper())
    if not match:
        return None
    prefix, number = match.group(1), int(match.group(2))
    limit = fleet.get(prefix)
    if limit is None or not 1 <= number <= limit:
        return None
    return f"{prefix}{number}"


class VolumeBackend(ABC):
    """Platform-specific half of card detection."""

    @abstractmethod
    def list_volumes(self) -> list[Volume]:
        """Every mounted removable volume, with its label."""

    @abstractmethod
    def eject(self, volume: Volume) -> tuple[bool, str]:
        """Unmount and power down ``volume``. Returns (ok, message)."""


class CardWatcher:
    """Polls a backend and reports cards appearing and disappearing.

    Callbacks run on the watcher's own thread; the UI is expected to hop back
    to its own thread.
    """

    def __init__(
        self,
        backend: VolumeBackend,
        fleet: dict[str, int],
        interval: float = 1.5,
    ):
        self.backend = backend
        self.fleet = dict(fleet)
        self.interval = interval
        self.on_card_added: Callable[[Card], None] | None = None
        self.on_card_removed: Callable[[Card], None] | None = None
        self.on_unknown_volume: Callable[[Volume], None] | None = None
        self._seen: dict[str, Card] = {}
        self._reported_unknown: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="card-watcher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                # A backend hiccup (a volume vanishing mid-enumeration) must
                # not kill the watcher, or cards stop being noticed silently.
                log.exception("card poll failed")
            self._stop.wait(self.interval)

    # ------------------------------------------------------------ polling

    def current_cards(self) -> list[Card]:
        with self._lock:
            return list(self._seen.values())

    def poll(self) -> None:
        """One detection pass. Safe to call directly, e.g. from a Refresh button."""
        volumes = self.backend.list_volumes()
        found: dict[str, Card] = {}
        for volume in volumes:
            card_id = parse_card_id(volume.label, self.fleet)
            if card_id is None:
                if volume.key not in self._reported_unknown:
                    self._reported_unknown.add(volume.key)
                    log.debug("ignoring volume %r (label %r)", volume.key, volume.label)
                    if self.on_unknown_volume:
                        self.on_unknown_volume(volume)
                continue
            found[volume.key] = Card(card_id=card_id, volume=volume)

        with self._lock:
            added = [c for k, c in found.items() if k not in self._seen]
            removed = [c for k, c in self._seen.items() if k not in found]
            self._seen = found
        live_keys = {v.key for v in volumes}
        self._reported_unknown &= live_keys

        for card in removed:
            log.info("card removed: %s", card.card_id)
            if self.on_card_removed:
                self.on_card_removed(card)
        for card in added:
            log.info("card detected: %s at %s", card.card_id, card.mountpoint)
            if self.on_card_added:
                self.on_card_added(card)

    def forget(self, card: Card) -> None:
        """Drop a card from the seen set so re-inserting it triggers again."""
        with self._lock:
            self._seen.pop(card.volume.key, None)
