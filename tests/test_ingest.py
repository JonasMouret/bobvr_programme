"""Ingestion, with an eye on the destructive parts.

The card is wiped after copying, so these tests care less about the happy path
than about every way the app might delete something it should not have.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from bobvr.db import STATUS_INGESTED, Database
from bobvr.ingest import IngestError, Ingestor




@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "clips.sqlite3")
    yield database
    database.close()


@pytest.fixture
def card_dir(tmp_path):
    path = tmp_path / "card" / "DCIM" / "100GOPRO"
    path.mkdir(parents=True)
    return path


def put(card_dir: Path, sample: Path, name: str) -> Path:
    target = card_dir / name
    shutil.copy(sample, target)
    return target


def test_copies_renames_verifies_and_clears_the_card(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    source = put(card_dir, sample_360, "GS010021.360")
    ingestor = Ingestor(settings, db, caps, fake_backend)
    card = make_card("B1", tmp_path / "card")

    report = ingestor.ingest_card(card, capture_day=date(2026, 2, 3))

    assert report.ok, report.failures
    assert len(report.copied) == 1
    archived = settings.archive_root / "2026-02-03" / "B1_1.360"
    assert archived.is_file()
    assert archived.stat().st_size == sample_360.stat().st_size
    # Source cleared and card ejected only after a clean copy.
    assert not source.exists()
    assert report.deleted == 1
    assert fake_backend.ejected

    clip = db.clip(report.copied[0])
    assert clip.name == "B1_1"
    assert clip.status == STATUS_INGESTED
    assert clip.checksum
    assert clip.duration and clip.duration > 0
    assert not list(archived.parent.glob("*.part"))


def test_second_clip_of_the_day_continues_the_numbering(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    put(card_dir, sample_360, "GS010021.360")
    put(card_dir, sample_360, "GS020021.360")
    ingestor = Ingestor(settings, db, caps, fake_backend)

    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert report.ok, report.failures
    names = sorted(c.name for c in (db.clip(i) for i in report.copied))
    assert names == ["B1_1", "B1_2"]
    day = settings.archive_root / "2026-02-03"
    assert {p.name for p in day.glob("*.360")} == {"B1_1.360", "B1_2.360"}


def test_files_outside_the_camera_folder_are_left_alone(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    """Only the camera's own folder is cleared; the rest of the card is not."""
    put(card_dir, sample_360, "GS010021.360")
    root = tmp_path / "card"
    notes = root / "notes.txt"
    notes.write_text("bonjour")
    other = root / "photos"
    other.mkdir()
    (other / "IMG_0001.JPG").write_bytes(b"not ours")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert report.ok
    assert len(report.copied) == 1
    assert notes.exists(), "un fichier hors du dossier caméra a été supprimé"
    assert (other / "IMG_0001.JPG").exists()


def test_camera_folder_is_emptied_and_recreated(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    """The camera's sidecars go with the folder, and the folder comes back.

    The camera writes a .LRV proxy and a .THM thumbnail next to every capture;
    unlinking the .360 alone would leave a card that fills up over a season and
    never looks empty.
    """
    put(card_dir, sample_360, "GS010021.360")
    (card_dir / "GS010021.LRV").write_bytes(b"proxy")
    (card_dir / "GS010021.THM").write_bytes(b"thumb")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert report.ok, report.failures
    assert report.cleared_dirs == ["100GOPRO"]
    assert card_dir.is_dir(), "le dossier caméra n'a pas été recréé"
    assert list(card_dir.iterdir()) == []
    assert card_dir.parent.name == "DCIM"


def test_camera_folder_is_kept_when_the_option_is_off(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    settings.clear_camera_folder = False
    put(card_dir, sample_360, "GS010021.360")
    proxy = card_dir / "GS010021.LRV"
    proxy.write_bytes(b"proxy")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert report.ok
    assert report.cleared_dirs == []
    assert proxy.exists()
    assert not (card_dir / "GS010021.360").exists()


def test_a_clip_at_the_card_root_clears_no_folder(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path
):
    """The mountpoint itself is never a camera folder, whatever else is there."""
    root = tmp_path / "card"
    root.mkdir()
    put(root, sample_360, "GS010021.360")
    keep = root / "notes.txt"
    keep.write_text("bonjour")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(make_card("B1", root), capture_day=date(2026, 2, 3))

    assert report.ok
    assert report.cleared_dirs == []
    assert root.is_dir()
    assert keep.exists()


def test_a_file_that_is_not_a_max_capture_is_skipped_not_deleted(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    good = put(card_dir, sample_360, "GS010021.360")
    broken = card_dir / "GS020021.360"
    broken.write_bytes(b"\x00" * 4096)         # right extension, wrong content

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert report.skipped == ["GS020021.360"]
    assert len(report.copied) == 1
    assert broken.exists(), "un fichier illisible ne doit jamais être supprimé"
    # The unreadable file burned no sequence number.
    assert db.clip(report.copied[0]).name == "B1_1"

    # The card still holds something we could not read, so it is not finished
    # with: nothing is deleted and it stays mounted for the operator to see.
    assert not report.complete
    assert good.exists(), "la carte a été vidée alors qu'un fichier est illisible"
    assert report.deleted == 0
    assert not fake_backend.ejected


def test_a_truncated_capture_is_explained_not_just_rejected(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    """A .360 whose moov never got written is the classic interrupted copy."""
    from bobvr.media import NotAMaxVideoError, probe

    truncated = card_dir / "MOV.360"
    # GoPro writes the index last, so lopping off the tail removes the moov.
    data = sample_360.read_bytes()
    truncated.write_bytes(data[: len(data) // 2])

    with pytest.raises(NotAMaxVideoError, match="incomplet"):
        probe(truncated, caps)

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )
    assert report.skipped == ["MOV.360"]
    assert not report.complete
    assert truncated.exists()
    assert not fake_backend.ejected


def test_checksum_mismatch_keeps_the_card_intact(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path, monkeypatch
):
    source = put(card_dir, sample_360, "GS010021.360")
    ingestor = Ingestor(settings, db, caps, fake_backend)
    # Simulate a bad read-back of the copy.
    monkeypatch.setattr(Ingestor, "_hash", lambda self, *a, **k: "0" * 64)

    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )

    assert not report.ok
    assert report.copied == []
    assert source.exists(), "la carte a été vidée malgré une vérification échouée"
    assert report.deleted == 0
    assert not fake_backend.ejected, "une carte en échec ne doit pas être éjectée"
    assert not list((settings.archive_root / "2026-02-03").glob("*")) or not (
        settings.archive_root / "2026-02-03" / "B1_1.360"
    ).exists()


def test_verification_can_be_switched_off(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path, monkeypatch
):
    settings.verify_checksum = False
    put(card_dir, sample_360, "GS010021.360")
    called = []
    monkeypatch.setattr(
        Ingestor, "_hash", lambda self, *a, **k: called.append(1) or "x"
    )
    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )
    assert report.ok
    assert called == [], "la vérification a tourné alors qu'elle est désactivée"


def test_card_is_not_cleared_when_deletion_is_disabled(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    settings.auto_delete_source = False
    source = put(card_dir, sample_360, "GS010021.360")
    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )
    assert report.ok
    assert source.exists()
    assert report.deleted == 0


def test_files_at_the_card_root_are_found(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path
):
    """Some workflows leave the clip at the root rather than in DCIM."""
    root = tmp_path / "card"
    root.mkdir()
    put(root, sample_360, "B21.360")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(make_card("S3", root), capture_day=date(2026, 2, 3))

    assert report.ok
    assert db.clip(report.copied[0]).name == "S3_1"


@pytest.mark.parametrize(
    "layout",
    [
        "100GOPRO",              # camera folder at the card root
        "DCIM/100GOPRO",         # camera folder under DCIM
        "DCIM/101GOPRO",         # rollover folder once the first one filled
        "MISC/DCIM/100GOPRO",    # an extra level of nesting
    ],
)
def test_clips_are_found_whatever_the_card_layout(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path, layout
):
    """Real cards put the camera folder in several different places."""
    root = tmp_path / "card"
    directory = root / layout
    directory.mkdir(parents=True)
    put(directory, sample_360, "MOV.360")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    assert [p.name for p in ingestor.find_media(make_card("B1", root))] == ["MOV.360"]

    report = ingestor.ingest_card(make_card("B1", root), capture_day=date(2026, 2, 3))
    assert report.ok, report.failures
    assert db.clip(report.copied[0]).name == "B1_1"


def test_clips_spread_across_folders_are_numbered_by_file_name(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path
):
    """The camera's numbering, not the folder order, decides the sequence."""
    root = tmp_path / "card"
    (root / "DCIM" / "100GOPRO").mkdir(parents=True)
    (root / "DCIM" / "101GOPRO").mkdir(parents=True)
    put(root / "DCIM" / "101GOPRO", sample_360, "GS030021.360")
    put(root / "DCIM" / "100GOPRO", sample_360, "GS010021.360")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    found = [p.name for p in ingestor.find_media(make_card("B1", root))]
    assert found == ["GS010021.360", "GS030021.360"]


def test_search_depth_is_respected(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path
):
    """A card used as general storage must not become a full crawl."""
    root = tmp_path / "card"
    deep = root / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    put(deep, sample_360, "MOV.360")

    settings.card_search_depth = 2
    ingestor = Ingestor(settings, db, caps, fake_backend)
    assert ingestor.find_media(make_card("B1", root)) == []

    settings.card_search_depth = 8
    assert len(ingestor.find_media(make_card("B1", root))) == 1


def test_system_directories_are_skipped(
    settings, db, caps, fake_backend, make_card, sample_360, tmp_path
):
    root = tmp_path / "card"
    junk = root / "System Volume Information"
    junk.mkdir(parents=True)
    put(junk, sample_360, "MOV.360")
    hidden = root / ".Trash-1000"
    hidden.mkdir()
    put(hidden, sample_360, "MOV2.360")

    ingestor = Ingestor(settings, db, caps, fake_backend)
    assert ingestor.find_media(make_card("B1", root)) == []


def test_a_file_is_never_ingested_twice_from_both_search_roots(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path
):
    """DCIM sits under the root, so the two searches must not double-count."""
    put(card_dir, sample_360, "GS010021.360")
    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(
        make_card("B1", tmp_path / "card"), capture_day=date(2026, 2, 3)
    )
    assert len(report.copied) == 1


def test_empty_card_is_left_mounted_and_reported(
    settings, db, caps, fake_backend, make_card, card_dir, tmp_path
):
    """Finding nothing usually means the wrong card, or a layout we missed.

    Ejecting straight away hides that; the operator gets a card spat out with
    no explanation.
    """
    ingestor = Ingestor(settings, db, caps, fake_backend)
    report = ingestor.ingest_card(make_card("R2", tmp_path / "card"))

    assert report.empty
    assert report.copied == []
    assert not fake_backend.ejected, "une carte vide ne doit pas être éjectée"
    assert "Aucun fichier .360" in report.eject_message
    assert report.summary() == "aucun fichier .360 trouvé"


def test_insufficient_disk_space_is_refused_before_copying(
    settings, db, caps, fake_backend, make_card, card_dir, sample_360, tmp_path, monkeypatch
):
    put(card_dir, sample_360, "GS010021.360")
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: usage._replace(free=1024)
    )
    ingestor = Ingestor(settings, db, caps, fake_backend)

    with pytest.raises(IngestError, match="Espace disque insuffisant"):
        ingestor.ingest_card(make_card("B1", tmp_path / "card"))

    assert (card_dir / "GS010021.360").exists()


# --------------------------------------------------------------------------
# Clearing the camera folder. Tested straight against the helper: this is the
# most destructive thing the app does, and it must stay covered on a machine
# with no sample footage.


@pytest.fixture
def clearing(settings, db, caps, fake_backend):
    return Ingestor(settings, db, caps, fake_backend)


def test_camera_folder_is_replaced_by_an_empty_one(clearing, make_card, tmp_path):
    root = tmp_path / "card"
    folder = root / "DCIM" / "100GOPRO"
    folder.mkdir(parents=True)
    (folder / "GS010021.LRV").write_bytes(b"proxy")
    (folder / "sub").mkdir()

    cleared = clearing._reset_camera_folders(
        [folder / "GS010021.360"], make_card("B1", root)
    )

    assert cleared == ["100GOPRO"]
    assert folder.is_dir() and list(folder.iterdir()) == []


def test_rollover_folders_are_cleared_too(clearing, make_card, tmp_path):
    root = tmp_path / "card"
    first = root / "DCIM" / "100GOPRO"
    second = root / "DCIM" / "101GOPRO"
    for folder in (first, second):
        folder.mkdir(parents=True)

    cleared = clearing._reset_camera_folders(
        [first / "a.360", second / "b.360"], make_card("B1", root)
    )

    assert cleared == ["100GOPRO", "101GOPRO"]
    assert first.is_dir() and second.is_dir()


def test_only_camera_folders_are_removed(clearing, make_card, tmp_path):
    """A clip found somewhere else must not cost that folder its contents."""
    root = tmp_path / "card"
    folder = root / "DCIM" / "Vacances"
    folder.mkdir(parents=True)
    keep = folder / "IMG_0001.JPG"
    keep.write_bytes(b"souvenir")

    assert clearing._reset_camera_folders(
        [folder / "MOV.360"], make_card("B1", root)
    ) == []
    assert keep.exists()


def test_a_folder_still_holding_a_capture_is_kept(clearing, make_card, tmp_path):
    """Never remove a folder with a .360 we have not accounted for."""
    root = tmp_path / "card"
    folder = root / "DCIM" / "100GOPRO"
    folder.mkdir(parents=True)
    survivor = folder / "GS020021.360"
    survivor.write_bytes(b"still here")

    assert clearing._reset_camera_folders(
        [folder / "GS010021.360"], make_card("B1", root)
    ) == []
    assert survivor.exists()


def test_a_folder_outside_the_card_is_never_touched(clearing, make_card, tmp_path):
    """The card's mountpoint bounds what may be deleted, whatever the name."""
    root = tmp_path / "card"
    root.mkdir()
    elsewhere = tmp_path / "ailleurs" / "100GOPRO"
    elsewhere.mkdir(parents=True)
    keep = elsewhere / "GS010021.LRV"
    keep.write_bytes(b"proxy")

    assert clearing._reset_camera_folders(
        [elsewhere / "GS010021.360"], make_card("B1", root)
    ) == []
    assert keep.exists()
