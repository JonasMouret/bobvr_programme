"""Carrying orchestrator events onto the Qt thread.

Worker threads must not touch widgets. Qt signals emitted from another thread
are delivered as queued connections on the receiver's thread, so wrapping each
listener in a signal is all the marshalling this app needs.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..cards import Card
from ..db import Clip
from ..ingest import IngestProgress, IngestReport
from ..jobs import Orchestrator
from ..render.pipeline import RenderProgress


class Bridge(QObject):
    card_detected = Signal(object)
    card_removed = Signal(object)
    ingest_started = Signal(object)
    ingest_progress = Signal(object)
    ingest_finished = Signal(object)
    render_started = Signal(object)
    render_progress = Signal(object, object)
    render_finished = Signal(object)
    render_failed = Signal(object, str)
    queue_changed = Signal()
    notice = Signal(str, str)

    def attach(self, orchestrator: Orchestrator) -> None:
        listeners = orchestrator.listeners
        listeners.card_detected = self.card_detected.emit
        listeners.card_removed = self.card_removed.emit
        listeners.ingest_started = self.ingest_started.emit
        listeners.ingest_progress = self.ingest_progress.emit
        listeners.ingest_finished = self.ingest_finished.emit
        listeners.render_started = self.render_started.emit
        listeners.render_progress = self.render_progress.emit
        listeners.render_finished = self.render_finished.emit
        listeners.render_failed = self.render_failed.emit
        listeners.queue_changed = self.queue_changed.emit
        listeners.notice = self.notice.emit
