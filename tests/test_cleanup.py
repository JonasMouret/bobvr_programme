"""Purging clips, and the files they point at.

This is the second place in the app that destroys data, so the tests care most
about what must survive: files outside the library, clips a worker is holding,
and originals nobody asked to delete.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bobvr.cleanup import disk_usage, purge
from bobvr.db import STATUS_DONE, Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "clips.sqlite3")
    yield database
    database.close()


def make_clip(db: Database, settings, name_seed: int, *, rendered: bool = True):
    """A clip with real files under the library, as ingestion would leave it."""
    clip = db.allocate_clip("B1", date(2026, 2, 3), source_name=f"GS{name_seed}.360")
    archive = settings.archive_dir(clip.card_id, "2026-02-03") / f"{clip.name}.360"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"o" * 2048)
    db.mark_ingested(clip.id, archive_path=archive, size=2048, checksum="x", duration=1.0)
    if rendered:
        render = settings.render_dir(clip.card_id, "2026-02-03") / f"{clip.name}.mp4"
        render.parent.mkdir(parents=True, exist_ok=True)
        render.write_bytes(b"r" * 1024)
        db.mark_rendered(clip.id, render)
    return db.clip(clip.id)


def test_forgetting_a_clip_keeps_its_files_by_default(settings, db):
    clip = make_clip(db, settings, 1)

    report = purge(settings, db, [clip])

    assert report.forgotten == 1
    assert report.ok
    assert db.clips() == []
    assert clip.render_path.exists(), "le rendu a été supprimé sans qu'on le demande"
    assert clip.archive_path.exists()


def test_deleting_renders_leaves_the_originals(settings, db):
    clip = make_clip(db, settings, 1)

    report = purge(settings, db, [clip], delete_renders=True)

    assert report.renders_deleted == 1
    assert report.archives_deleted == 0
    assert report.bytes_freed == 1024
    assert not clip.render_path.exists()
    assert clip.archive_path.exists(), "l'original doit survivre : il permet de re-rendre"


def test_deleting_everything_frees_both(settings, db):
    clip = make_clip(db, settings, 1)

    report = purge(settings, db, [clip], delete_renders=True, delete_archives=True)

    assert (report.renders_deleted, report.archives_deleted) == (1, 1)
    assert report.bytes_freed == 1024 + 2048
    assert not clip.render_path.exists() and not clip.archive_path.exists()


def test_a_clip_being_rendered_is_left_completely_alone(settings, db):
    busy = make_clip(db, settings, 1)
    other = make_clip(db, settings, 2)

    report = purge(
        settings, db, [busy, other],
        delete_renders=True, delete_archives=True, keep=[busy.id],
    )

    assert report.busy == [busy.name]
    assert report.forgotten == 1
    assert busy.render_path.exists() and busy.archive_path.exists()
    assert [c.id for c in db.clips()] == [busy.id]


def test_a_copy_in_flight_survives_a_reset(settings, db):
    """« Réinitialiser » pressed while a card is being read must not eat it.

    The row is what the ingest worker updates when the copy lands; deleting it
    would leave a file in the library that nothing remembers.
    """
    done = make_clip(db, settings, 1)
    copying = db.allocate_clip("B1", date(2026, 2, 3))   # status: copying

    report = purge(settings, db, db.clips())

    assert report.busy == [copying.name]
    assert report.forgotten == 1
    assert [c.id for c in db.clips()] == [copying.id]
    assert done.id not in {c.id for c in db.clips()}


def test_a_file_outside_the_library_is_never_deleted(settings, db, tmp_path):
    """A moved library, or a row from another install: forget it, keep it."""
    clip = make_clip(db, settings, 1)
    stray = tmp_path / "ailleurs" / "precieux.mp4"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"important")
    db._update(clip.id, render_path=stray)

    report = purge(settings, db, [db.clip(clip.id)], delete_renders=True)

    assert stray.exists(), "un fichier hors bibliothèque a été supprimé"
    assert not report.ok and "hors de la bibliothèque" in report.failures[0]
    assert report.forgotten == 1, "la descente doit quand même quitter la liste"


def test_a_missing_file_is_not_a_failure(settings, db):
    clip = make_clip(db, settings, 1)
    clip.render_path.unlink()

    report = purge(settings, db, [clip], delete_renders=True)

    assert report.ok
    assert report.renders_deleted == 0


def test_emptied_day_folders_are_tidied_away(settings, db):
    clip = make_clip(db, settings, 1)
    day = clip.render_path.parent

    purge(settings, db, [clip], delete_renders=True)

    assert not day.exists(), "un dossier de jour vide reste dans la bibliothèque"
    assert clip.archive_path.parent.exists(), "le jour des originaux n'est pas vide"


def test_clearing_the_day_restarts_the_numbering(settings, db):
    """What an operator means by « réinitialiser »."""
    first = make_clip(db, settings, 1)
    second = make_clip(db, settings, 2)
    assert (first.name, second.name) == ("B1_1", "B1_2")

    purge(settings, db, [first, second])

    again = db.allocate_clip("B1", date(2026, 2, 3))
    assert again.name == "B1_1"


def test_only_finished_clips_are_offered_to_the_bin(settings, db):
    done = make_clip(db, settings, 1)
    pending = make_clip(db, settings, 2, rendered=False)

    finished = db.clips_by_status([STATUS_DONE])

    assert [c.id for c in finished] == [done.id]
    assert pending.id not in {c.id for c in finished}


def test_disk_usage_reports_both_kinds(settings, db):
    clips = [make_clip(db, settings, i) for i in (1, 2)]
    renders, archives = disk_usage(clips)
    assert (renders, archives) == (2 * 1024, 2 * 2048)
