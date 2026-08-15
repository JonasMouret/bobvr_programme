"""The operator's window.

Built around what actually happens on a race day: a card goes in, and the
operator needs to know at a glance whether it is safe to pull out again. So
card state is the most prominent thing on screen, the clip list is the record,
and the render queue runs quietly underneath.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFontDatabase,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..cleanup import purge
from ..config import Settings
from ..db import (
    STATUS_COPYING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_INGESTED,
    STATUS_QUEUED,
    STATUS_RENDERING,
    Clip,
    Database,
)
from ..jobs import Orchestrator
from .bridge import Bridge
from .cleanup_dialog import CleanupDialog
from .settings_dialog import SettingsDialog

log = logging.getLogger(__name__)

STATUS_TEXT = {
    STATUS_COPYING: "copie en cours",
    STATUS_INGESTED: "en attente",
    STATUS_QUEUED: "en file",
    STATUS_RENDERING: "rendu en cours",
    STATUS_DONE: "terminé",
    STATUS_FAILED: "échec",
}

#: Mid-tone colours, legible as text on a light or a dark desktop alike. They
#: mark the status cell only: filling a whole row turns the list into a wall of
#: colour where nothing stands out, which is the opposite of the point.
STATUS_COLOUR = {
    STATUS_COPYING: QColor("#8E8E8E"),
    STATUS_INGESTED: QColor("#8E8E8E"),
    STATUS_QUEUED: QColor("#E08A1E"),
    STATUS_RENDERING: QColor("#3B8FE0"),
    STATUS_DONE: QColor("#3FA45B"),
    STATUS_FAILED: QColor("#D64B4B"),
}


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, db: Database, orchestrator: Orchestrator):
        super().__init__()
        self.settings = settings
        self.db = db
        self.orchestrator = orchestrator

        self.setWindowTitle("BobVr — traitement des vidéos 360")
        self.resize(1180, 780)

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()

        self.bridge = Bridge()
        self.bridge.attach(orchestrator)
        self._connect_signals()

        self.refresh_clips()
        self.refresh_cards()

        # The watcher already reports arrivals; this only keeps the card panel
        # honest if a volume disappears without an event.
        self._card_timer = QTimer(self)
        self._card_timer.timeout.connect(self.refresh_cards)
        self._card_timer.start(3000)

    # -------------------------------------------------------------- build

    def _build_toolbar(self) -> None:
        from PySide6.QtCore import QSize

        bar = self.addToolBar("Actions")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(18, 18))

        def action(text: str, icon, slot, tip: str = "") -> QAction:
            item = QAction(self.style().standardIcon(icon), text, self)
            item.setToolTip(tip or text)
            item.triggered.connect(slot)
            bar.addAction(item)
            return item

        self.action_rescan = action(
            "Rechercher les cartes", QStyle.SP_BrowserReload,
            self.orchestrator.rescan,
            "Relire la liste des volumes montés",
        )
        self.action_ingest = action(
            "Importer", QStyle.SP_ArrowDown, self._ingest_selected_card,
            "Importer la carte sélectionnée",
        )
        bar.addSeparator()
        self.action_retry = action(
            "Relancer le rendu", QStyle.SP_MediaPlay, self._retry_selected,
            "Remettre la descente sélectionnée en file de rendu",
        )
        self.action_cancel = action(
            "Annuler le rendu", QStyle.SP_MediaStop,
            self.orchestrator.cancel_current_render,
            "Arrêter le rendu en cours",
        )
        bar.addSeparator()
        self.action_clear_done = action(
            "Vider les terminées", QStyle.SP_TrashIcon, self._clear_finished,
            "Retirer de la liste les descentes déjà converties",
        )
        self.action_reset = action(
            "Réinitialiser", QStyle.SP_LineEditClearButton, self._reset_all,
            "Vider toute la liste et repartir de la descente n° 1",
        )
        bar.addSeparator()
        self.action_open = action(
            "Bibliothèque", QStyle.SP_DirOpenIcon, self._open_library,
            "Ouvrir le dossier des vidéos",
        )
        self.action_settings = action(
            "Réglages", QStyle.SP_FileDialogDetailedView, self._open_settings,
        )

    def _build_body(self) -> None:
        splitter = QSplitter(Qt.Vertical, self)

        # -- cards ---------------------------------------------------------
        cards_box = QGroupBox("Cartes détectées")
        cards_layout = QVBoxLayout(cards_box)
        self.cards_table = QTableWidget(0, 4)
        self.cards_table.setHorizontalHeaderLabels(
            ["Carte", "Engin", "Emplacement", "État"]
        )
        _tune_table(self.cards_table, stretch_column=2)
        cards_layout.addWidget(self.cards_table)

        self.card_hint = QLabel()
        self.card_hint.setWordWrap(True)
        cards_layout.addWidget(self.card_hint)
        cards_layout.addStretch(1)
        splitter.addWidget(cards_box)

        # -- clips ---------------------------------------------------------
        clips_box = QGroupBox("Descentes")
        clips_layout = QVBoxLayout(clips_box)

        self.clips_summary = QLabel()
        self.clips_summary.setTextFormat(Qt.RichText)
        clips_layout.addWidget(self.clips_summary)

        self.clips_table = QTableWidget(0, 6)
        self.clips_table.setHorizontalHeaderLabels(
            ["Descente", "Jour", "Durée", "Taille", "État", "Fichier"]
        )
        _tune_table(self.clips_table, stretch_column=5)
        self.clips_table.doubleClicked.connect(self._open_selected_render)
        self.clips_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clips_table.customContextMenuRequested.connect(self._clip_menu)
        clips_layout.addWidget(self.clips_table)

        self.clips_hint = QLabel(
            "Double-cliquez une descente pour l'ouvrir · clic droit pour les "
            "autres actions."
        )
        self.clips_hint.setEnabled(False)
        clips_layout.addWidget(self.clips_hint)
        splitter.addWidget(clips_box)

        # -- activity ------------------------------------------------------
        activity = QGroupBox("Activité")
        activity_layout = QVBoxLayout(activity)

        self.ingest_label = QLabel("Aucune importation en cours")
        self.ingest_bar = _bar()
        activity_layout.addWidget(self.ingest_label)
        activity_layout.addWidget(self.ingest_bar)

        self.render_label = QLabel("Aucun rendu en cours")
        self.render_bar = _bar()
        activity_layout.addWidget(self.render_label)
        activity_layout.addWidget(self.render_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        # Ask the platform for its fixed-width face rather than naming one:
        # "monospace" is a Linux alias and resolves to nothing on Windows.
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log_view.setMinimumHeight(90)
        activity_layout.addWidget(self.log_view)
        splitter.addWidget(activity)

        splitter.setStretchFactor(1, 1)
        splitter.setSizes([110, 430, 280])
        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        caps = self.orchestrator.caps
        self.status_label = QLabel(caps.summary())
        self.statusBar().addPermanentWidget(self.status_label)
        for gap in caps.explain_gaps():
            self.note("warning", gap)

    def _connect_signals(self) -> None:
        b = self.bridge
        b.card_detected.connect(self._on_card_detected)
        b.card_removed.connect(lambda _c: self.refresh_cards())
        b.ingest_started.connect(self._on_ingest_started)
        b.ingest_progress.connect(self._on_ingest_progress)
        b.ingest_finished.connect(self._on_ingest_finished)
        b.render_started.connect(self._on_render_started)
        b.render_progress.connect(self._on_render_progress)
        b.render_finished.connect(self._on_render_finished)
        b.render_failed.connect(self._on_render_failed)
        b.queue_changed.connect(self.refresh_clips)
        b.notice.connect(self.note)

    # ------------------------------------------------------------ updating

    def note(self, level: str, message: str) -> None:
        prefix = {"error": "ERREUR", "warning": "ATTENTION", "info": "info"}.get(
            level, level
        )
        self.log_view.appendPlainText(f"[{prefix}] {message}")

    def refresh_cards(self) -> None:
        cards = self.orchestrator.watcher.current_cards()
        busy = self.orchestrator.current_card
        self.cards_table.setRowCount(len(cards))
        for row, card in enumerate(sorted(cards, key=lambda c: c.card_id)):
            state = (
                "importation en cours"
                if busy is not None and busy.card_id == card.card_id
                else "prête"
            )
            for column, value in enumerate(
                [
                    card.card_id,
                    self.settings.describe_card(card.card_id),
                    str(card.mountpoint),
                    state,
                ]
            ):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, card.card_id)
                self.cards_table.setItem(row, column, item)

        # An empty table is a big grey rectangle saying nothing; the hint says
        # it better, so the table only appears once there is something in it.
        self.cards_table.setVisible(bool(cards))
        if cards:
            self.cards_table.setFixedHeight(
                self.cards_table.horizontalHeader().height()
                + len(cards) * self.cards_table.verticalHeader().defaultSectionSize()
                + 8
            )
            self.card_hint.setText("")
        else:
            self.card_hint.setText(
                "Aucune carte détectée. Insérez une carte dont le nom de volume "
                "correspond à un engin (B1 à B6, S1 à S6, R1 à R5)."
            )

    def refresh_clips(self) -> None:
        clips = self.db.clips(limit=300)
        table = self.clips_table
        table.setUpdatesEnabled(False)
        table.setRowCount(len(clips))
        for row, clip in enumerate(clips):
            self._fill_clip_row(row, clip)
        table.setUpdatesEnabled(True)

        self._refresh_summary(clips)
        depth = self.orchestrator.queue_depth()
        self.action_cancel.setEnabled(self.orchestrator.current_clip is not None)
        self.action_clear_done.setEnabled(
            any(c.status == STATUS_DONE for c in clips)
        )
        self.action_reset.setEnabled(bool(clips))
        self.statusBar().showMessage(
            f"{depth} rendu(s) en attente" if depth else "File de rendu vide"
        )

    def _fill_clip_row(self, row: int, clip: Clip) -> None:
        table = self.clips_table

        def cell(column: int, text: str, *, align=None, tip: str = "") -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setData(Qt.UserRole, clip.id)
            if align is not None:
                item.setTextAlignment(align)
            if tip:
                item.setToolTip(tip)
            table.setItem(row, column, item)
            return item

        name = cell(0, clip.name)
        bold = name.font()
        bold.setBold(True)
        name.setFont(bold)

        cell(1, clip.capture_date)
        cell(2, _format_duration(clip.duration), align=Qt.AlignRight | Qt.AlignVCenter)
        cell(3, _format_size(clip.size), align=Qt.AlignRight | Qt.AlignVCenter)

        # The status carries the only colour in the row, so the eye lands on
        # the one thing that changes.
        status = cell(
            4, f"● {STATUS_TEXT.get(clip.status, clip.status)}",
            align=Qt.AlignLeft | Qt.AlignVCenter,
        )
        colour = STATUS_COLOUR.get(clip.status)
        if colour is not None:
            status.setForeground(colour)

        if clip.error:
            detail = cell(5, clip.error.replace("\n", " ")[:200], tip=clip.error)
            detail.setForeground(STATUS_COLOUR[STATUS_FAILED])
        elif clip.render_path is not None:
            # The full path is noise in a column: every row shares all of it
            # but the last component. It stays one hover away.
            detail = cell(5, clip.render_path.name, tip=str(clip.render_path))
        else:
            cell(5, "—")

    def _refresh_summary(self, clips: list[Clip]) -> None:
        if not clips:
            self.clips_summary.setText(
                "Aucune descente enregistrée. Insérez une carte pour commencer."
            )
            return
        counts: dict[str, int] = {}
        for clip in clips:
            counts[clip.status] = counts.get(clip.status, 0) + 1
        parts = [f"<b>{len(clips)}</b> descente(s)"]
        for status in (
            STATUS_DONE, STATUS_RENDERING, STATUS_QUEUED, STATUS_INGESTED,
            STATUS_FAILED,
        ):
            if counts.get(status):
                colour = STATUS_COLOUR[status].name()
                parts.append(
                    f"<span style='color:{colour}'>{counts[status]} "
                    f"{STATUS_TEXT[status]}</span>"
                )
        self.clips_summary.setText(" &nbsp;·&nbsp; ".join(parts))

    # -------------------------------------------------------------- events

    def _on_card_detected(self, card) -> None:
        self.refresh_cards()
        self.note(
            "info",
            f"Carte {card.card_id} ({self.settings.describe_card(card.card_id)}) détectée.",
        )
        if not self.settings.auto_ingest:
            self.note(
                "info",
                "L'importation automatique est désactivée : utilisez "
                "« Importer la carte sélectionnée ».",
            )

    def _on_ingest_started(self, card) -> None:
        self.refresh_cards()
        self.ingest_label.setText(f"Importation de la carte {card.card_id}…")
        self.ingest_bar.setValue(0)

    def _on_ingest_progress(self, progress) -> None:
        self.ingest_label.setText(
            f"Carte {progress.card_id} — {progress.phase} "
            f"({progress.file_index}/{progress.file_count}) {progress.source_name}"
        )
        self.ingest_bar.setValue(int(progress.fraction * 100))

    def _on_ingest_finished(self, report) -> None:
        self.ingest_bar.setValue(100 if not report.empty else 0)
        self.ingest_label.setText(
            f"Carte {report.card_id} : {report.summary()}"
        )
        if report.empty:
            self.note("warning", report.eject_message)
            self.refresh_cards()
            return
        level = "info" if report.ok else "error"
        self.note(level, f"Carte {report.card_id} — {report.summary()}.")
        if report.ejected:
            self.note("info", f"Carte {report.card_id} : {report.eject_message}")
        elif report.eject_message:
            self.note("warning", f"Carte {report.card_id} : {report.eject_message}")
        self.refresh_clips()
        self.refresh_cards()

    def _on_render_started(self, clip) -> None:
        self.render_label.setText(f"Rendu de {clip.name}…")
        self.render_bar.setValue(0)
        self.action_cancel.setEnabled(True)
        self.refresh_clips()

    def _on_render_progress(self, clip, progress) -> None:
        eta = progress.eta_seconds
        eta_text = f" — reste {_format_duration(eta)}" if eta else ""
        self.render_label.setText(
            f"Rendu de {clip.name} — {progress.fraction * 100:.0f} % "
            f"({progress.speed:.2f}× temps réel){eta_text}"
        )
        self.render_bar.setValue(int(progress.fraction * 100))

    def _on_render_finished(self, clip) -> None:
        self.render_bar.setValue(100)
        self.render_label.setText(f"{clip.name} : rendu terminé")
        self.note("info", f"{clip.name} — rendu terminé : {clip.render_path}")
        self.action_cancel.setEnabled(False)
        self.refresh_clips()

    def _on_render_failed(self, clip, message) -> None:
        self.render_label.setText(f"{clip.name} : échec du rendu")
        self.note("error", f"{clip.name} — {message}")
        self.action_cancel.setEnabled(False)
        self.refresh_clips()

    # ------------------------------------------------------------- actions

    def _selected_clip(self) -> Clip | None:
        rows = self.clips_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.clips_table.item(rows[0].row(), 0)
        if item is None:
            return None
        try:
            return self.db.clip(int(item.data(Qt.UserRole)))
        except KeyError:
            return None

    def _retry_selected(self) -> None:
        clip = self._selected_clip()
        if clip is None:
            QMessageBox.information(
                self, "Relancer le rendu",
                "Sélectionnez d'abord une descente dans la liste.",
            )
            return
        self.orchestrator.retry(clip.id)
        self.note("info", f"{clip.name} — remis en file de rendu.")

    def _ingest_selected_card(self) -> None:
        rows = self.cards_table.selectionModel().selectedRows()
        cards = {c.card_id: c for c in self.orchestrator.watcher.current_cards()}
        if not rows:
            QMessageBox.information(
                self, "Importer une carte",
                "Sélectionnez d'abord une carte dans la liste.",
            )
            return
        item = self.cards_table.item(rows[0].row(), 0)
        card = cards.get(item.text() if item else "")
        if card is None:
            self.refresh_cards()
            return
        self.orchestrator.ingest_now(card)

    # ------------------------------------------------------------- purging

    def _clear_finished(self) -> None:
        clips = self.db.clips_by_status([STATUS_DONE])
        if not clips:
            QMessageBox.information(
                self, "Vider les terminées",
                "Aucune descente terminée pour l'instant.",
            )
            return
        self._purge(
            clips,
            title="Vider les descentes terminées",
            lead="Retirer de la liste les descentes déjà converties.",
        )

    def _reset_all(self) -> None:
        clips = self.db.clips(limit=100_000)
        if not clips:
            return
        self._purge(
            clips,
            title="Réinitialiser les descentes",
            lead=(
                "Vider entièrement la liste, y compris les descentes en "
                "attente ou en échec. La numérotation repartira de 1 "
                "(B1_1, B1_2, …)."
            ),
        )

    def _delete_one(self, clip: Clip) -> None:
        self._purge(
            [clip],
            title=f"Supprimer {clip.name}",
            lead=f"Retirer {clip.name} de la liste.",
        )

    def _purge(self, clips: list[Clip], title: str, lead: str) -> None:
        dialog = CleanupDialog(clips, title, lead, self)
        if not dialog.exec():
            return
        current = self.orchestrator.current_clip
        report = purge(
            self.settings, self.db, clips,
            delete_renders=dialog.delete_renders,
            delete_archives=dialog.delete_archives,
            # A render in flight keeps its row: the worker still needs it, and
            # deleting the record would orphan the file it is writing.
            keep=[current.id] if current is not None else [],
        )
        self.note("info" if report.ok else "warning", report.summary() + ".")
        for failure in report.failures:
            self.note("error", failure)
        if report.busy:
            self.note(
                "info",
                f"{', '.join(report.busy)} : rendu en cours, conservée(s). "
                "Relancez l'opération une fois le rendu terminé.",
            )
        self.refresh_clips()

    def _clip_menu(self, position) -> None:
        item = self.clips_table.itemAt(position)
        if item is None:
            return
        self.clips_table.selectRow(item.row())
        clip = self._selected_clip()
        if clip is None:
            return

        menu = QMenu(self)
        playable = clip.render_path is not None and clip.render_path.exists()
        open_action = menu.addAction("Ouvrir la vidéo")
        open_action.setEnabled(playable)
        folder_action = menu.addAction("Ouvrir le dossier")
        folder_action.setEnabled(playable)
        menu.addSeparator()
        retry_action = menu.addAction("Relancer le rendu")
        retry_action.setEnabled(clip.archive_path is not None)
        menu.addSeparator()
        delete_action = menu.addAction("Supprimer cette descente…")

        chosen = menu.exec(self.clips_table.viewport().mapToGlobal(position))
        if chosen is None:
            return
        if chosen is open_action:
            self._open_selected_render()
        elif chosen is folder_action:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(clip.render_path.parent))
            )
        elif chosen is retry_action:
            self._retry_selected()
        elif chosen is delete_action:
            self._delete_one(clip)

    def _open_selected_render(self) -> None:
        clip = self._selected_clip()
        if clip is None or clip.render_path is None:
            return
        if not clip.render_path.exists():
            QMessageBox.warning(
                self, "Fichier introuvable",
                f"{clip.render_path} n'existe plus.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(clip.render_path)))

    def _open_library(self) -> None:
        root = self.settings.library_root
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.orchestrator.caps, self)
        if dialog.exec():
            self.settings = dialog.result_settings
            self.settings.save()
            self.orchestrator.settings = self.settings
            self.orchestrator.ingestor.settings = self.settings
            self.orchestrator.watcher.fleet = dict(self.settings.fleet)
            self.note(
                "info",
                "Réglages enregistrés. Ils s'appliquent aux prochaines "
                "importations et aux prochains rendus.",
            )
            self.refresh_cards()

    def closeEvent(self, event) -> None:
        if self.orchestrator.current_clip or self.orchestrator.current_card:
            answer = QMessageBox.question(
                self, "Quitter BobVr",
                "Une importation ou un rendu est en cours. Quitter maintenant "
                "l'interrompra. Voulez-vous vraiment quitter ?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self._card_timer.stop()
        self.orchestrator.stop()
        event.accept()


def _tune_table(table: QTableWidget, stretch_column: int) -> None:
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)                 # the grid fights the row banding
    table.setWordWrap(False)
    table.verticalHeader().setDefaultSectionSize(26)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setSectionResizeMode(stretch_column, QHeaderView.Stretch)
    header.setHighlightSections(False)
    # The stretched column is wide and its content starts on the left; a
    # centred title above it would sit nowhere near what it labels.
    item = table.horizontalHeaderItem(stretch_column)
    if item is not None:
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    font = header.font()
    font.setBold(True)
    header.setFont(font)


def _bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setTextVisible(True)
    bar.setFixedHeight(16)
    return bar


def _format_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "—"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_size(size: int | None) -> str:
    if not size:
        return "—"
    value = float(size)
    for unit in ("o", "Ko", "Mo", "Go"):
        if value < 1024 or unit == "Go":
            return f"{value:.0f} {unit}" if unit == "o" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} Go"
