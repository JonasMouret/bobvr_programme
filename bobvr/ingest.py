"""Copying clips off a card, then clearing and ejecting it.

This is the only part of the app that destroys data, so it is deliberately
cautious: nothing is deleted from a card until the copy has been read back and
its hash matches. A card that fails verification is left untouched and
reported, because a re-copy costs a minute and a lost run cannot be recovered.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterator

from .cards.base import Card, VolumeBackend
from .config import Settings
from .db import Database
from .media import MaxVideoInfo, NotAMaxVideoError, probe
from .render.caps import Capabilities

log = logging.getLogger(__name__)

#: GoPro writes .360 for dual-lens captures; .mp4 files on the same card are
#: single-lens and not our business.
MEDIA_SUFFIXES = (".360",)

#: The camera's own media folder, as it appears on the card -- normally
#: ``DCIM/100GOPRO``, then ``101GOPRO`` and so on as each one fills. This is
#: the only directory the app ever removes, so the pattern is kept exact.
CAMERA_DIR = re.compile(r"^\d{3}GOPRO$", re.IGNORECASE)

#: Operating-system bookkeeping directories: never captures, often
#: unreadable. Deliberately limited to names no one would give a real folder --
#: the depth cap already bounds the search, so guessing at camera-specific
#: names would only risk skipping somewhere a clip really sits.
SKIP_DIRS = frozenset(
    {
        "system volume information",
        "$recycle.bin",
        "recycler",
        "lost+found",
    }
)

_CHUNK = 8 * 1024 * 1024

PHASE_COPY = "copie"
PHASE_VERIFY = "vérification"
PHASE_CLEAN = "nettoyage"


class IngestError(RuntimeError):
    """Ingestion could not proceed."""


@dataclass
class IngestProgress:
    card_id: str
    phase: str
    file_index: int
    file_count: int
    source_name: str
    bytes_done: int
    bytes_total: int

    @property
    def fraction(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        return min(1.0, self.bytes_done / self.bytes_total)


@dataclass
class IngestReport:
    card_id: str
    copied: list[int] = field(default_factory=list)      # clip ids
    skipped: list[str] = field(default_factory=list)     # source filenames
    failures: list[tuple[str, str]] = field(default_factory=list)
    deleted: int = 0
    #: Camera folders removed and put back empty, e.g. ``["100GOPRO"]``.
    cleared_dirs: list[str] = field(default_factory=list)
    ejected: bool = False
    eject_message: str = ""
    #: No .360 file was found at all -- worth saying out loud rather than
    #: reporting "0 copied" as if that were a normal outcome.
    empty: bool = False

    @property
    def ok(self) -> bool:
        """Nothing went wrong with the files we were able to copy."""
        return not self.failures

    @property
    def complete(self) -> bool:
        """Every file on the card was copied, and at least one was.

        Stricter than :attr:`ok` on purpose. A skipped file is one we could not
        read -- often a capture truncated by a card pulled mid-recording. The
        card still holds it, so the card is not finished with, and ejecting it
        would hide a problem the operator needs to see.
        """
        return bool(self.copied) and not self.failures and not self.skipped

    def summary(self) -> str:
        if self.empty:
            return "aucun fichier .360 trouvé"
        bits = [f"{len(self.copied)} clip(s) copié(s)"]
        if self.deleted:
            bits.append(f"{self.deleted} fichier(s) supprimé(s) de la carte")
        if self.cleared_dirs:
            bits.append(f"{', '.join(self.cleared_dirs)} recréé(s) à vide")
        if self.skipped:
            bits.append(f"{len(self.skipped)} ignoré(s)")
        if self.failures:
            bits.append(f"{len(self.failures)} en échec")
        return ", ".join(bits)


ProgressCallback = Callable[[IngestProgress], None]


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        caps: Capabilities,
        backend: VolumeBackend,
    ):
        self.settings = settings
        self.db = db
        self.caps = caps
        self.backend = backend

    # ------------------------------------------------------------ finding

    def find_media(self, card: Card) -> list[Path]:
        """Every .360 on the card, in capture order.

        Searched recursively rather than at a fixed path: cards in the field
        put ``100GOPRO`` at the root or under ``DCIM`` depending on how they
        were formatted, and the camera rolls over to ``101GOPRO``, ``102GOPRO``
        and so on as folders fill. A depth-limited walk covers all of it
        without depending on a layout we happened to see once.

        Results are ordered by file name, which follows the camera's own
        numbering and therefore the order the runs were filmed.
        """
        found: dict[Path, None] = {}
        for directory in self._walk(card.mountpoint, self.settings.card_search_depth):
            try:
                entries = sorted(directory.iterdir())
            except OSError as exc:
                log.warning("cannot list %s: %s", directory, exc)
                continue
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.suffix.lower() not in MEDIA_SUFFIXES:
                    continue
                try:
                    if not entry.is_file():
                        continue
                    found.setdefault(entry.resolve(), None)
                except OSError:
                    continue
        return sorted(found, key=lambda p: (p.name.lower(), str(p)))

    def _walk(self, root: Path, max_depth: int) -> Iterator[Path]:
        """Yield ``root`` and its sub-directories, breadth first, depth-capped.

        The cap keeps a card someone used as general storage from turning
        ingestion into a full filesystem crawl.
        """
        level = [root]
        for _ in range(max(1, max_depth)):
            if not level:
                return
            yield from level
            children: list[Path] = []
            for directory in level:
                try:
                    entries = list(directory.iterdir())
                except OSError as exc:
                    log.debug("cannot descend into %s: %s", directory, exc)
                    continue
                for entry in entries:
                    if entry.name.startswith("."):
                        continue
                    if entry.name.lower() in SKIP_DIRS:
                        continue
                    try:
                        if entry.is_dir():
                            children.append(entry)
                    except OSError:
                        continue
            level = children

    # ---------------------------------------------------------- ingesting

    def ingest_card(
        self,
        card: Card,
        on_progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
        capture_day: date | None = None,
    ) -> IngestReport:
        """Copy everything off ``card``, then clear and eject it.

        Deleting and ejecting are governed by settings and only ever happen
        when every file on the card copied cleanly.
        """
        report = IngestReport(card_id=card.card_id)
        media = self.find_media(card)
        if not media:
            # Deliberately left mounted. An empty card is either already
            # processed or the wrong card, and ejecting it immediately gives
            # the operator nothing to look at -- which is exactly how a
            # mis-detected layout stays invisible.
            log.info("card %s holds no .360 files", card.card_id)
            report.empty = True
            report.eject_message = (
                f"Aucun fichier .360 trouvé sur la carte {card.card_id} "
                f"({card.mountpoint}). La carte est laissée en place : "
                "vérifiez qu'il s'agit de la bonne carte, ou retirez-la "
                "manuellement si elle a déjà été traitée."
            )
            return report

        day = capture_day or date.today()
        total_bytes = sum(_size_of(p) for p in media)
        self._check_free_space(total_bytes)

        copied_sources: list[Path] = []
        for index, source in enumerate(media, start=1):
            if cancel is not None and cancel.is_set():
                raise IngestError("Ingestion annulée.")
            try:
                clip_id, archived = self._ingest_one(
                    card, source, day, index, len(media), on_progress, cancel
                )
            except NotAMaxVideoError as exc:
                log.warning("skipping %s: %s", source.name, exc)
                report.skipped.append(source.name)
                continue
            except (OSError, IngestError) as exc:
                log.exception("failed to ingest %s", source.name)
                report.failures.append((source.name, str(exc)))
                continue
            report.copied.append(clip_id)
            copied_sources.append(source)

        # Only clear a card that gave up everything it had without complaint.
        if self.settings.auto_delete_source and report.complete:
            report.deleted, report.cleared_dirs = self._delete_sources(
                copied_sources, card, on_progress
            )
        elif not report.complete:
            log.warning(
                "card %s not cleared: %d failed, %d unreadable",
                card.card_id, len(report.failures), len(report.skipped),
            )

        if self.settings.auto_eject and report.complete:
            report.ejected, report.eject_message = self.backend.eject(card.volume)
        elif not report.complete:
            report.eject_message = (
                f"Carte {card.card_id} laissée en place : "
                + (
                    f"{len(report.failures)} fichier(s) en échec de copie"
                    if report.failures
                    else f"{len(report.skipped)} fichier(s) illisible(s)"
                )
                + ". Rien n'a été effacé de la carte."
            )
        return report

    def _ingest_one(
        self,
        card: Card,
        source: Path,
        day: date,
        index: int,
        count: int,
        on_progress: ProgressCallback | None,
        cancel: threading.Event | None,
    ) -> tuple[int, Path]:
        # Reject non-MAX files before reserving a name, so a stray file cannot
        # burn a sequence number.
        info = probe(source, self.caps)

        clip = self.db.allocate_clip(card.card_id, day, source_name=source.name)
        destination = (
            self.settings.archive_root / day.isoformat() / f"{clip.name}{source.suffix.lower()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")

        try:
            size = _size_of(source)
            digest = self._copy(
                source, partial,
                lambda done: _emit(on_progress, card, PHASE_COPY, index, count,
                                   source.name, done, size),
                cancel,
            )
            if self.settings.verify_checksum:
                check = self._hash(
                    partial,
                    lambda done: _emit(on_progress, card, PHASE_VERIFY, index, count,
                                       source.name, done, size),
                    cancel,
                )
                if check != digest:
                    raise IngestError(
                        f"La copie de {source.name} ne correspond pas à l'original "
                        "(empreinte différente). Le fichier de la carte est conservé."
                    )
            partial.replace(destination)
        except BaseException:
            # Drop the reservation entirely: a clip that never landed should
            # not consume a sequence number the operator would look for.
            partial.unlink(missing_ok=True)
            self.db.delete(clip.id)
            raise

        self.db.mark_ingested(
            clip.id,
            archive_path=destination,
            size=size,
            checksum=digest,
            duration=info.duration,
        )
        log.info("ingested %s -> %s", source.name, destination)
        return clip.id, destination

    def _copy(
        self,
        source: Path,
        destination: Path,
        report: Callable[[int], None],
        cancel: threading.Event | None,
    ) -> str:
        """Copy, hashing as we go so the source is read only once."""
        hasher = hashlib.blake2b(digest_size=32)
        done = 0
        with open(source, "rb") as src, open(destination, "wb") as dst:
            while True:
                if cancel is not None and cancel.is_set():
                    raise IngestError("Ingestion annulée.")
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                report(done)
            dst.flush()
            os.fsync(dst.fileno())
        return hasher.hexdigest()

    def _hash(
        self,
        path: Path,
        report: Callable[[int], None],
        cancel: threading.Event | None,
    ) -> str:
        hasher = hashlib.blake2b(digest_size=32)
        done = 0
        with open(path, "rb") as handle:
            while True:
                if cancel is not None and cancel.is_set():
                    raise IngestError("Ingestion annulée.")
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                hasher.update(chunk)
                done += len(chunk)
                report(done)
        return hasher.hexdigest()

    def _delete_sources(
        self,
        sources: list[Path],
        card: Card,
        on_progress: ProgressCallback | None,
    ) -> tuple[int, list[str]]:
        deleted = 0
        for index, source in enumerate(sources, start=1):
            _emit(on_progress, card, PHASE_CLEAN, index, len(sources),
                  source.name, index, len(sources))
            try:
                source.unlink()
                deleted += 1
            except OSError as exc:
                log.warning("could not delete %s from card: %s", source, exc)
        cleared: list[str] = []
        if self.settings.clear_camera_folder:
            cleared = self._reset_camera_folders(sources, card)
        return deleted, cleared

    def _reset_camera_folders(self, sources: list[Path], card: Card) -> list[str]:
        """Remove each camera folder we took clips from, then put it back empty.

        Unlinking the .360 files alone leaves the folder full of the camera's
        own leftovers -- ``.LRV`` proxies, ``.THM`` thumbnails, the odd
        single-lens ``.MP4`` -- which accumulate over a season and make a card
        that has been emptied look as though it has not. Deleting
        ``DCIM/100GOPRO`` outright and recreating it gives the camera back
        exactly the folder it expects, on a card that is genuinely empty.

        Only directories named like ``100GOPRO`` are touched, only inside the
        card, and only once nothing that looks like a capture is left inside.
        Anything else on the card is none of our business.
        """
        cleared: list[str] = []
        mount = card.mountpoint.resolve()
        folders = sorted(
            {
                source.parent
                for source in sources
                if CAMERA_DIR.match(source.parent.name)
            }
        )
        for folder in folders:
            if mount not in folder.parents:
                log.warning("not clearing %s: outside the card %s", folder, mount)
                continue
            if folder.is_symlink() or not folder.is_dir():
                continue
            leftovers = [
                path for path in folder.rglob("*")
                if path.suffix.lower() in MEDIA_SUFFIXES and path.is_file()
            ]
            if leftovers:
                # Belt and braces: we only get here when every capture copied
                # cleanly, so anything left is something we never looked at.
                log.warning(
                    "not clearing %s: %d capture(s) still inside",
                    folder, len(leftovers),
                )
                continue
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                log.warning("could not remove %s: %s", folder, exc)
                continue
            cleared.append(folder.name)
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                log.warning(
                    "%s was removed but could not be recreated: %s", folder, exc
                )
            else:
                log.info("camera folder %s emptied and recreated", folder)
        return cleared

    def _check_free_space(self, needed: int) -> None:
        root = self.settings.archive_root
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        # Leave headroom: the render lands on the same volume by default.
        required = int(needed * 1.5)
        if free < required:
            raise IngestError(
                f"Espace disque insuffisant dans {root} : "
                f"{free / 1e9:.1f} Go libres, {required / 1e9:.1f} Go nécessaires "
                "(copie + rendu). Libérez de l'espace puis relancez."
            )


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _emit(
    callback: ProgressCallback | None,
    card: Card,
    phase: str,
    index: int,
    count: int,
    name: str,
    done: int,
    total: int,
) -> None:
    if callback is None:
        return
    callback(
        IngestProgress(
            card_id=card.card_id,
            phase=phase,
            file_index=index,
            file_count=count,
            source_name=name,
            bytes_done=done,
            bytes_total=total,
        )
    )
