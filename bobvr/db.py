"""Durable record of every clip the app has seen.

The database is what makes the workflow safe to interrupt: a card can be
pulled, the app can crash, the machine can lose power, and on restart we still
know which clips were copied, which were verified, and which still owe a
render. It is also what guarantees sequence numbers are never reused.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Lifecycle of a clip. A clip only moves forward, except that FAILED can be
#: retried back to QUEUED.
STATUS_COPYING = "copying"
STATUS_INGESTED = "ingested"
STATUS_QUEUED = "queued"
STATUS_RENDERING = "rendering"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id       TEXT    NOT NULL,
    capture_date  TEXT    NOT NULL,
    sequence      INTEGER NOT NULL,
    name          TEXT    NOT NULL,
    source_name   TEXT,
    archive_path  TEXT,
    render_path   TEXT,
    size          INTEGER,
    duration      REAL,
    checksum      TEXT,
    status        TEXT    NOT NULL,
    error         TEXT,
    created_at    TEXT    NOT NULL,
    ingested_at   TEXT,
    rendered_at   TEXT,
    UNIQUE (card_id, capture_date, sequence)
);

CREATE INDEX IF NOT EXISTS clips_status  ON clips (status);
CREATE INDEX IF NOT EXISTS clips_created ON clips (created_at DESC);
"""


@dataclass
class Clip:
    """One capture, from card to finished render."""

    id: int
    card_id: str
    capture_date: str
    sequence: int
    name: str
    source_name: str | None
    archive_path: Path | None
    render_path: Path | None
    size: int | None
    duration: float | None
    checksum: str | None
    status: str
    error: str | None
    created_at: str
    ingested_at: str | None
    rendered_at: str | None

    @property
    def is_finished(self) -> bool:
        return self.status == STATUS_DONE

    @property
    def needs_render(self) -> bool:
        return self.status in (STATUS_INGESTED, STATUS_QUEUED, STATUS_FAILED)


def _row_to_clip(row: sqlite3.Row) -> Clip:
    return Clip(
        id=row["id"],
        card_id=row["card_id"],
        capture_date=row["capture_date"],
        sequence=row["sequence"],
        name=row["name"],
        source_name=row["source_name"],
        archive_path=Path(row["archive_path"]) if row["archive_path"] else None,
        render_path=Path(row["render_path"]) if row["render_path"] else None,
        size=row["size"],
        duration=row["duration"],
        checksum=row["checksum"],
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        rendered_at=row["rendered_at"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-safe SQLite store.

    Worker threads and the UI both touch this, so every statement runs under a
    single lock on one connection rather than juggling per-thread connections.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -------------------------------------------------------- allocation

    def allocate_clip(
        self, card_id: str, capture_date: date, source_name: str | None = None
    ) -> Clip:
        """Reserve the next sequence number for ``card_id`` on ``capture_date``.

        Done inside a transaction so two cards being read at once can never be
        handed the same name.
        """
        day = capture_date.isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM clips "
                    "WHERE card_id = ? AND capture_date = ?",
                    (card_id, day),
                ).fetchone()
                sequence = int(row["next"])
                name = f"{card_id}_{sequence}"
                cursor = self._conn.execute(
                    "INSERT INTO clips (card_id, capture_date, sequence, name, "
                    "source_name, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (card_id, day, sequence, name, source_name, STATUS_COPYING, _now()),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            return self.clip(int(cursor.lastrowid))

    # ------------------------------------------------------------ updates

    def _update(self, clip_id: int, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = [
            str(v) if isinstance(v, Path) else v for v in fields.values()
        ]
        with self._lock:
            self._conn.execute(
                f"UPDATE clips SET {assignments} WHERE id = ?", (*values, clip_id)
            )

    def mark_ingested(
        self,
        clip_id: int,
        archive_path: Path,
        size: int,
        checksum: str | None,
        duration: float | None,
    ) -> None:
        self._update(
            clip_id,
            archive_path=archive_path,
            size=size,
            checksum=checksum,
            duration=duration,
            status=STATUS_INGESTED,
            error=None,
            ingested_at=_now(),
        )

    def set_status(self, clip_id: int, status: str, error: str | None = None) -> None:
        self._update(clip_id, status=status, error=error)

    def mark_rendered(self, clip_id: int, render_path: Path) -> None:
        self._update(
            clip_id,
            render_path=render_path,
            status=STATUS_DONE,
            error=None,
            rendered_at=_now(),
        )

    def delete(self, clip_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))

    def delete_many(self, clip_ids: Iterable[int]) -> int:
        """Forget several clips at once, returning how many rows went.

        Sequence numbers are drawn from what the table still holds, so a card
        whose clips are all deleted starts again at 1 -- which is exactly what
        an operator means by resetting the day.
        """
        ids = [int(i) for i in clip_ids]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM clips WHERE id IN ({marks})", ids
            )
        return cursor.rowcount

    # ------------------------------------------------------------ queries

    def clip(self, clip_id: int) -> Clip:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM clips WHERE id = ?", (clip_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"clip {clip_id} introuvable")
        return _row_to_clip(row)

    def clips(self, limit: int = 500, status: str | None = None) -> list[Clip]:
        query = "SELECT * FROM clips"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_clip(r) for r in rows]

    def clips_by_status(self, statuses: Sequence[str]) -> list[Clip]:
        """Every clip in any of ``statuses``, newest first."""
        if not statuses:
            return []
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM clips WHERE status IN ({marks}) ORDER BY id DESC",
                list(statuses),
            ).fetchall()
        return [_row_to_clip(r) for r in rows]

    def pending_renders(self) -> list[Clip]:
        """Clips copied but not yet rendered, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM clips WHERE status IN (?, ?) "
                "AND archive_path IS NOT NULL ORDER BY id ASC",
                (STATUS_INGESTED, STATUS_QUEUED),
            ).fetchall()
        return [_row_to_clip(r) for r in rows]

    def recover_interrupted(self) -> int:
        """Reset states that only make sense while the app is running.

        Called at startup: anything left mid-copy died with the last process
        and its partial file is worthless, while anything mid-render can simply
        be rendered again.
        """
        with self._lock:
            stale_copies = self._conn.execute(
                "DELETE FROM clips WHERE status = ?", (STATUS_COPYING,)
            ).rowcount
            stale_renders = self._conn.execute(
                "UPDATE clips SET status = ? WHERE status = ?",
                (STATUS_QUEUED, STATUS_RENDERING),
            ).rowcount
        if stale_copies or stale_renders:
            log.info(
                "recovered from unclean shutdown: dropped %d partial copies, "
                "requeued %d renders", stale_copies, stale_renders
            )
        return stale_copies + stale_renders

    def counts_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM clips GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}
