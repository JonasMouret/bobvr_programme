"""La mise à jour, côté interface.

La logique vit dans :mod:`bobvr.updater` (sans Qt, testable hors ligne). Ce
module ne fait que l'habiller : deux fils d'exécution pour ne jamais bloquer la
fenêtre — l'un vérifie sur GitHub, l'autre télécharge — et une boîte de dialogue
qui montre la version disponible, ses notes, une barre de progression, puis
lance l'installation.

Le réseau et le disque tournent donc hors du fil de l'UI ; ils communiquent par
signaux Qt, livrés sur le fil de la fenêtre, comme le reste de l'application
(voir :mod:`bobvr.ui.bridge`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from .. import __version__
from ..updater import (
    RELEASES_PAGE_URL,
    Release,
    apply_update,
    can_self_update,
    check_for_update,
    download_asset,
    stage_update,
    staging_root,
)

log = logging.getLogger(__name__)

_MB = 1024 * 1024


class CheckWorker(QThread):
    """Interroge GitHub sur un fil séparé et rend une :class:`Release` ou rien."""

    result = Signal(object)  # Release | None

    def run(self) -> None:  # pragma: no cover - réseau
        try:
            self.result.emit(check_for_update())
        except Exception:  # check_for_update avale déjà tout ; ceinture et bretelles
            log.debug("échec inattendu de la vérification", exc_info=True)
            self.result.emit(None)


class DownloadWorker(QThread):
    """Télécharge le zip et le décompresse ; rend le dossier applicatif prêt."""

    progress = Signal(int, int)  # octets reçus, total (0 si inconnu)
    ready = Signal(object)  # Path du dossier à installer
    failed = Signal(str)

    def __init__(self, release: Release, parent=None) -> None:
        super().__init__(parent)
        self._release = release

    def run(self) -> None:  # pragma: no cover - réseau et disque
        try:
            root = staging_root()
            zip_path = download_asset(
                self._release, root, on_progress=self.progress.emit
            )
            app = stage_update(zip_path, root / "staged")
            self.ready.emit(app)
        except Exception as exc:
            log.warning("mise à jour : téléchargement impossible : %s", exc)
            self.failed.emit(str(exc))


class UpdateDialog(QDialog):
    """Propose la version disponible, la télécharge et l'installe."""

    def __init__(self, release: Release, parent=None) -> None:
        super().__init__(parent)
        self._release = release
        self._worker: DownloadWorker | None = None

        self.setWindowTitle("Mise à jour de BobVr")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)

        head = QLabel(
            f"<b>BobVr {release.version}</b> est disponible.<br>"
            f"Vous utilisez actuellement la version {__version__}."
        )
        head.setWordWrap(True)
        layout.addWidget(head)

        if release.notes:
            notes = QTextBrowser()
            notes.setPlainText(release.notes)
            notes.setMinimumHeight(180)
            layout.addWidget(notes)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setVisible(False)
        layout.addWidget(self.status)

        row = QHBoxLayout()
        row.addStretch(1)
        self.later_btn = QPushButton("Plus tard")
        self.later_btn.clicked.connect(self.reject)
        self.install_btn = QPushButton("Télécharger et installer")
        self.install_btn.setDefault(True)
        self.install_btn.clicked.connect(self._start)
        row.addWidget(self.later_btn)
        row.addWidget(self.install_btn)
        layout.addLayout(row)

        # Cas où l'on ne peut pas installer soi-même : pas de zip attaché, ou
        # bien on tourne depuis les sources / hors Windows. On ne cache pas le
        # bouton, on le transforme en « ouvrir la page des versions ».
        if not release.can_install:
            self._offer_manual(
                "Cette version n'a pas de fichier installable ; récupérez-la "
                "depuis la page des versions."
            )
        elif not can_self_update():
            self._offer_manual(
                "L'installation automatique n'est disponible que dans la "
                "version Windows. Ouvrez la page des versions pour mettre à jour."
            )

    def _offer_manual(self, message: str) -> None:
        self.install_btn.setText("Ouvrir la page des versions")
        self.install_btn.clicked.disconnect()
        self.install_btn.clicked.connect(self._open_page)
        self.status.setText(message)
        self.status.setVisible(True)

    def _open_page(self) -> None:
        QDesktopServices.openUrl(QUrl(RELEASES_PAGE_URL))
        self.reject()

    # ------------------------------------------------------------ install

    def _start(self) -> None:
        self.install_btn.setEnabled(False)
        self.later_btn.setEnabled(False)
        self.progress.setVisible(True)
        # Indéterminée tant qu'on ne connaît pas la taille annoncée.
        self.progress.setRange(0, 0)
        self.status.setVisible(True)
        self.status.setText("Téléchargement…")

        self._worker = DownloadWorker(self._release, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, received: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(received)
            self.status.setText(
                f"Téléchargement… {received // _MB} Mo / {total // _MB} Mo"
            )
        else:
            self.status.setText(f"Téléchargement… {received // _MB} Mo")

    def _on_ready(self, app_dir: Path) -> None:
        self.status.setText("Installation… BobVr va redémarrer.")
        try:
            apply_update(app_dir)
        except Exception as exc:  # noqa: BLE001 - on remet l'UI en état
            self._on_failed(str(exc))
            return
        # Le script de bascule attend précisément la fermeture de ce processus
        # pour remplacer les fichiers verrouillés, puis relance BobVr.
        self.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.install_btn.setEnabled(True)
        self.later_btn.setEnabled(True)
        QMessageBox.warning(
            self,
            "Mise à jour impossible",
            f"Le téléchargement ou l'installation a échoué :\n{message}\n\n"
            "Réessayez plus tard, ou récupérez la version depuis la page des "
            "versions.",
        )
