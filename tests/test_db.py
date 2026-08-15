"""Sequence allocation and crash recovery.

Numbering is what the operator reads off the file name, so a duplicate or a
skipped number is a real problem, not cosmetic.
"""

from __future__ import annotations

import threading
from datetime import date

import pytest

from bobvr.db import (
    STATUS_COPYING,
    STATUS_DONE,
    STATUS_INGESTED,
    STATUS_QUEUED,
    STATUS_RENDERING,
    Database,
)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "clips.sqlite3")
    yield database
    database.close()


def test_sequence_starts_at_one_and_increments(db):
    day = date(2026, 2, 1)
    names = [db.allocate_clip("B1", day).name for _ in range(3)]
    assert names == ["B1_1", "B1_2", "B1_3"]


def test_sequences_are_independent_per_card_and_per_day(db):
    monday, tuesday = date(2026, 2, 1), date(2026, 2, 2)
    assert db.allocate_clip("B1", monday).name == "B1_1"
    assert db.allocate_clip("B2", monday).name == "B2_1"
    assert db.allocate_clip("S1", monday).name == "S1_1"
    assert db.allocate_clip("B1", monday).name == "B1_2"
    # A new day restarts the count.
    assert db.allocate_clip("B1", tuesday).name == "B1_1"


def test_concurrent_allocation_never_repeats_a_number(db):
    """Two cards read at once must not collide on a name."""
    day = date(2026, 2, 1)
    names: list[str] = []
    lock = threading.Lock()

    def grab():
        clip = db.allocate_clip("B1", day)
        with lock:
            names.append(clip.name)

    threads = [threading.Thread(target=grab) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(names) == 20
    assert len(set(names)) == 20, "un numéro de séquence a été attribué deux fois"


def test_deleting_a_reservation_frees_its_number(db):
    """A copy that fails should not leave a hole in the numbering."""
    day = date(2026, 2, 1)
    first = db.allocate_clip("B1", day)
    second = db.allocate_clip("B1", day)
    assert second.name == "B1_2"
    db.delete(second.id)
    assert db.allocate_clip("B1", day).name == "B1_2"
    assert first.name == "B1_1"


def test_pending_renders_only_lists_clips_with_a_file(db, tmp_path):
    day = date(2026, 2, 1)
    reserved = db.allocate_clip("B1", day)          # still copying, no file yet
    ready = db.allocate_clip("B2", day)
    db.mark_ingested(ready.id, tmp_path / "B2_1.360", 100, "abc", 12.0)

    pending = db.pending_renders()
    assert [c.id for c in pending] == [ready.id]
    assert reserved.id not in {c.id for c in pending}


def test_finished_clips_are_not_pending(db, tmp_path):
    day = date(2026, 2, 1)
    clip = db.allocate_clip("B1", day)
    db.mark_ingested(clip.id, tmp_path / "B1_1.360", 100, "abc", 12.0)
    db.mark_rendered(clip.id, tmp_path / "B1_1.mp4")
    assert db.pending_renders() == []
    assert db.clip(clip.id).status == STATUS_DONE
    assert db.clip(clip.id).render_path == tmp_path / "B1_1.mp4"


def test_recovery_drops_partial_copies_and_requeues_renders(db, tmp_path):
    day = date(2026, 2, 1)
    partial = db.allocate_clip("B1", day)           # left in COPYING
    rendering = db.allocate_clip("B2", day)
    db.mark_ingested(rendering.id, tmp_path / "B2_1.360", 100, "abc", 12.0)
    db.set_status(rendering.id, STATUS_RENDERING)

    assert db.clip(partial.id).status == STATUS_COPYING
    db.recover_interrupted()

    with pytest.raises(KeyError):
        db.clip(partial.id)
    assert db.clip(rendering.id).status == STATUS_QUEUED


def test_status_counts(db, tmp_path):
    day = date(2026, 2, 1)
    for card in ("B1", "B2", "B3"):
        clip = db.allocate_clip(card, day)
        db.mark_ingested(clip.id, tmp_path / f"{card}.360", 10, "x", 1.0)
    counts = db.counts_by_status()
    assert counts == {STATUS_INGESTED: 3}
