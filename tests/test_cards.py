"""Recognising fleet cards, and noticing them come and go."""

from __future__ import annotations

from pathlib import Path

import pytest

from bobvr.cards.base import CardWatcher, Volume, parse_card_id

FLEET = {"B": 6, "S": 6, "R": 5}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("B1", "B1"),
        ("B6", "B6"),
        ("b3", "B3"),          # case is not the operator's problem
        (" S2 ", "S2"),        # nor is stray whitespace
        ("R5", "R5"),
    ],
)
def test_fleet_labels_are_accepted(label, expected):
    assert parse_card_id(label, FLEET) == expected


@pytest.mark.parametrize(
    "label",
    [
        "B7",        # only six bobsleighs
        "R6",        # only five racing sleds
        "B0",        # numbering starts at one
        "X1",        # not a fleet prefix
        "BACKUP",    # someone else's disk
        "B",
        "1B",
        "B 1",
        "",
    ],
)
def test_foreign_labels_are_ignored(label):
    assert parse_card_id(label, FLEET) is None


def test_unlabelled_volume_is_ignored():
    """An unlabelled stick must never be mistaken for a card -- we delete from cards."""
    assert parse_card_id("", FLEET) is None


class ScriptedBackend:
    def __init__(self):
        self.volumes: list[Volume] = []

    def list_volumes(self):
        return list(self.volumes)

    def eject(self, volume):
        return True, "ok"


def test_watcher_reports_arrival_and_removal_once():
    backend = ScriptedBackend()
    watcher = CardWatcher(backend, FLEET)
    added, removed = [], []
    watcher.on_card_added = added.append
    watcher.on_card_removed = removed.append

    card = Volume(label="B1", mountpoint=Path("/media/x"), device="/dev/sdz1")
    backend.volumes = [card]
    watcher.poll()
    watcher.poll()                      # steady state: no repeat
    assert [c.card_id for c in added] == ["B1"]
    assert removed == []

    backend.volumes = []
    watcher.poll()
    assert [c.card_id for c in removed] == ["B1"]


def test_watcher_ignores_non_fleet_volumes():
    backend = ScriptedBackend()
    watcher = CardWatcher(backend, FLEET)
    added = []
    unknown = []
    watcher.on_card_added = added.append
    watcher.on_unknown_volume = unknown.append

    backend.volumes = [
        Volume(label="", mountpoint=Path("/media/stick"), device="/dev/sdy1"),
        Volume(label="SAUVEGARDE", mountpoint=Path("/media/bak"), device="/dev/sdx1"),
    ]
    watcher.poll()
    assert added == []
    assert len(unknown) == 2
    # Reported once, not on every poll.
    watcher.poll()
    assert len(unknown) == 2


def test_reinserting_the_same_card_triggers_again():
    backend = ScriptedBackend()
    watcher = CardWatcher(backend, FLEET)
    added = []
    watcher.on_card_added = added.append

    card = Volume(label="S4", mountpoint=Path("/media/s4"), device="/dev/sdw1")
    backend.volumes = [card]
    watcher.poll()
    backend.volumes = []
    watcher.poll()
    backend.volumes = [card]
    watcher.poll()
    assert [c.card_id for c in added] == ["S4", "S4"]


def test_forget_allows_reprocessing_without_physical_removal():
    backend = ScriptedBackend()
    watcher = CardWatcher(backend, FLEET)
    added = []
    watcher.on_card_added = added.append

    card = Volume(label="B2", mountpoint=Path("/media/b2"), device="/dev/sdv1")
    backend.volumes = [card]
    watcher.poll()
    assert len(added) == 1

    watcher.forget(watcher.current_cards()[0])
    watcher.poll()
    assert len(added) == 2
