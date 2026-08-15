"""Application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from ..config import APP_NAME, Settings
from ..db import Database
from ..jobs import Orchestrator
from ..render.caps import FFmpegMissingError, detect

log = logging.getLogger(__name__)


def _apply_icon(app: QApplication) -> None:
    """Give the window and the taskbar entry the app's own icon.

    A packaged .exe carries an icon of its own, but that one belongs to the
    file: without this, the running window shows Qt's default and looks like
    somebody's homework in the taskbar.
    """
    from PySide6.QtGui import QIcon

    icon = Path(__file__).resolve().parent / "icon.png"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    if sys.platform == "win32":
        # Windows groups taskbar buttons by "application user model id", and
        # a plain python.exe host would otherwise borrow Python's icon.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f"mouret.{APP_NAME}"
            )
        except Exception:                       # pragma: no cover - Windows only
            log.debug("could not set the taskbar identity", exc_info=True)


def database_path(settings: Settings) -> Path:
    from platformdirs import user_data_dir

    return Path(user_data_dir(APP_NAME, appauthor=False)) / "clips.sqlite3"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    _apply_icon(app)

    settings = Settings.load()

    try:
        caps = detect(settings.ffmpeg_path, settings.ffprobe_path)
    except FFmpegMissingError as exc:
        QMessageBox.critical(None, "ffmpeg introuvable", str(exc))
        return 2

    db = Database(database_path(settings))
    try:
        orchestrator = Orchestrator(settings, db, caps=caps)
    except RuntimeError as exc:
        QMessageBox.critical(None, "Démarrage impossible", str(exc))
        db.close()
        return 2

    # Imported late so a headless run of the rest of the package does not need
    # the widget modules.
    from .main_window import MainWindow

    window = MainWindow(settings, db, orchestrator)
    orchestrator.start()
    window.show()

    try:
        return app.exec()
    finally:
        orchestrator.stop()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
