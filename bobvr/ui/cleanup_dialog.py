"""Confirming a purge, with the consequences spelled out.

Nothing here decides anything: it counts what is about to go, asks which of
the files should go with it, and hands the answer back. Everything is off by
default -- forgetting a clip is undoable in a minute, deleting the original of
a run that has already been given back to a team is not.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from ..cleanup import disk_usage
from ..db import Clip


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} Go"


class CleanupDialog(QDialog):
    """Ask what to do with a set of clips."""

    def __init__(self, clips: list[Clip], title: str, lead: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self._clips = clips

        renders, archives = disk_usage(clips)

        headline = QLabel(f"<b>{lead}</b>")
        headline.setWordWrap(True)

        count = QLabel(
            f"{len(clips)} descente(s) concernée(s) : "
            f"{', '.join(c.name for c in clips[:8])}"
            + (" …" if len(clips) > 8 else "")
        )
        count.setWordWrap(True)

        self.renders_check = QCheckBox(
            f"Supprimer aussi les vidéos converties (.mp4) — {_gb(renders)}"
        )
        self.archives_check = QCheckBox(
            f"Supprimer aussi les fichiers d'origine (.360) — {_gb(archives)}"
        )
        self.archives_check.setEnabled(bool(archives))

        warning = QLabel()
        warning.setWordWrap(True)
        warning.setTextFormat(Qt.RichText)

        def refresh() -> None:
            if self.archives_check.isChecked():
                warning.setText(
                    "⚠ Sans le fichier d'origine, la descente ne pourra plus "
                    "être re-rendue : ni changement de champ de vision, ni "
                    "changement de résolution, ni télémétrie. "
                    "<b>C'est définitif.</b>"
                )
            elif self.renders_check.isChecked():
                warning.setText(
                    "Les originaux sont conservés : les descentes pourront "
                    "être re-rendues à tout moment."
                )
            else:
                warning.setText(
                    "Aucun fichier ne sera supprimé — les descentes "
                    "disparaissent seulement de la liste."
                )
            ok = self.buttons.button(QDialogButtonBox.Ok)
            ok.setText(
                "Supprimer" if self.archives_check.isChecked() else "Vider la liste"
            )

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.button(QDialogButtonBox.Cancel).setText("Annuler")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.renders_check.toggled.connect(lambda _on: refresh())
        self.archives_check.toggled.connect(lambda _on: refresh())
        refresh()

        layout = QVBoxLayout(self)
        layout.addWidget(headline)
        layout.addWidget(count)
        layout.addSpacing(8)
        layout.addWidget(self.renders_check)
        layout.addWidget(self.archives_check)
        layout.addSpacing(4)
        layout.addWidget(warning)
        layout.addStretch(1)
        layout.addWidget(self.buttons)

    @property
    def delete_renders(self) -> bool:
        return self.renders_check.isChecked()

    @property
    def delete_archives(self) -> bool:
        return self.archives_check.isChecked()
