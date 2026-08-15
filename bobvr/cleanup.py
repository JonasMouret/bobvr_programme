"""Removing clips from the record, and optionally from the disk.

Kept apart from the workflow itself because this is the second place in the app
that destroys data, and the first -- clearing a card -- earned its caution the
hard way. The rules here are the same: never touch a file outside the library,
never touch a clip a worker is still holding, and report exactly what happened
rather than assuming it worked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .config import Settings
from .db import STATUS_COPYING, STATUS_RENDERING, Clip, Database

#: States that mean a worker owns the clip right now. Deleting one of these
#: rows would leave the worker writing a file nothing remembers -- the copy
#: would finish into a library the app no longer has a record of.
BUSY_STATUSES = (STATUS_COPYING, STATUS_RENDERING)

log = logging.getLogger(__name__)


@dataclass
class PurgeReport:
    """What a purge actually did."""

    forgotten: int = 0
    renders_deleted: int = 0
    archives_deleted: int = 0
    bytes_freed: int = 0
    #: Clips left alone because a worker was using them.
    busy: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if not self.forgotten and not self.busy:
            return "aucune descente à retirer"
        bits = [f"{self.forgotten} descente(s) retirée(s) de la liste"]
        if self.renders_deleted:
            bits.append(f"{self.renders_deleted} rendu(s) supprimé(s)")
        if self.archives_deleted:
            bits.append(f"{self.archives_deleted} original(aux) supprimé(s)")
        if self.bytes_freed:
            bits.append(f"{self.bytes_freed / 1e9:.1f} Go libérés")
        if self.busy:
            bits.append(f"{len(self.busy)} en cours, conservée(s)")
        if self.failures:
            bits.append(f"{len(self.failures)} en échec")
        return ", ".join(bits)


def disk_usage(clips: Sequence[Clip]) -> tuple[int, int]:
    """Bytes held by these clips' renders and originals, as they are now."""
    renders = sum(_size(c.render_path) for c in clips)
    archives = sum(_size(c.archive_path) for c in clips)
    return renders, archives


def purge(
    settings: Settings,
    db: Database,
    clips: Iterable[Clip],
    *,
    delete_renders: bool = False,
    delete_archives: bool = False,
    keep: Iterable[int] = (),
) -> PurgeReport:
    """Forget ``clips``, optionally deleting the files they point at.

    Clips a worker owns are skipped whole, whether the caller names them in
    ``keep`` or their status says so: deleting the record of a copy or a render
    in flight would leave the file orphaned and the operator none the wiser.
    """
    report = PurgeReport()
    protected = {int(i) for i in keep}
    doomed: list[int] = []

    for clip in clips:
        if clip.id in protected or clip.status in BUSY_STATUSES:
            report.busy.append(clip.name)
            continue
        if delete_renders and _remove(
            clip.render_path, settings.render_root, clip.name, report
        ):
            report.renders_deleted += 1
        if delete_archives and _remove(
            clip.archive_path, settings.archive_root, clip.name, report
        ):
            report.archives_deleted += 1
        doomed.append(clip.id)

    if doomed:
        report.forgotten = db.delete_many(doomed)
    _prune_empty_days(settings)
    log.info("purge: %s", report.summary())
    return report


def _remove(path: Path | None, root: Path, name: str, report: PurgeReport) -> bool:
    """Delete one file, but only from inside the library. True if it went."""
    if path is None:
        return False
    try:
        resolved = path.resolve()
        inside = resolved.is_relative_to(root.resolve())
    except OSError as exc:
        report.failures.append(f"{name} : {exc}")
        return False
    if not inside:
        # Someone moved the library, or the row predates the current settings.
        # Forget the clip, keep the file: we cannot vouch for what it is.
        report.failures.append(
            f"{name} : {path} est hors de la bibliothèque, fichier conservé"
        )
        return False
    size = _size(resolved)
    try:
        resolved.unlink()
    except FileNotFoundError:
        return False                    # already gone; nothing to report
    except OSError as exc:
        report.failures.append(f"{name} : {exc}")
        return False
    report.bytes_freed += size
    return True


def _size(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _prune_empty_days(settings: Settings) -> None:
    """Drop day folders nothing lives in any more.

    Cosmetic, but a library full of empty dated folders is a library nobody
    trusts to tell them what is there.
    """
    for root in (settings.render_root, settings.archive_root):
        if not root.is_dir():
            continue
        for day in sorted(root.iterdir()):
            try:
                if day.is_dir() and not any(day.iterdir()):
                    day.rmdir()
            except OSError as exc:
                log.debug("could not remove %s: %s", day, exc)
