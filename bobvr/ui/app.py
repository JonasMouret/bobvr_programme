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


def _splash_logo() -> Path | None:
    """The image for the startup screen: a dedicated splash, else the icon.

    Drop a ``splash.png`` next to this file to show the full logo at launch;
    without one, the window icon stands in.
    """
    here = Path(__file__).resolve().parent
    for name in ("splash.png", "icon.png"):
        candidate = here / name
        if candidate.is_file():
            return candidate
    return None


def _show_splash() -> "QSplashScreen | None":
    """A logo screen while the slow startup (hardware probe) runs.

    Returns None if no image is available or Qt cannot load it, so a missing
    asset costs the splash but never the launch.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen

    logo = _splash_logo()
    if logo is None:
        return None
    source = QPixmap(str(logo))
    if source.isNull():
        return None

    width, height = 360, 320
    canvas = QPixmap(width, height)
    canvas.fill(QColor("#ffffff"))          # white, so the logo's dark parts read
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    scaled = source.scaled(248, 248, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    painter.drawPixmap(
        (width - scaled.width()) // 2, (height - 44 - scaled.height()) // 2, scaled
    )
    painter.setPen(QColor("#d9dee3"))       # a hairline so the edge shows on white
    painter.drawRect(0, 0, width - 1, height - 1)
    painter.end()

    splash = QSplashScreen(canvas)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    splash.show()
    return splash


def _splash_note(splash, text: str) -> None:
    if splash is None:
        return
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    splash.showMessage(text, Qt.AlignBottom | Qt.AlignHCenter, QColor("#33475b"))
    QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    _apply_icon(app)

    splash = _show_splash()

    settings = Settings.load()

    _splash_note(splash, "Détection du matériel…")
    try:
        caps = detect(settings.ffmpeg_path, settings.ffprobe_path)
    except FFmpegMissingError as exc:
        if splash is not None:
            splash.close()
        QMessageBox.critical(None, "ffmpeg introuvable", str(exc))
        return 2

    _splash_note(splash, "Ouverture de la bibliothèque…")
    db = Database(database_path(settings))
    try:
        orchestrator = Orchestrator(settings, db, caps=caps)
    except RuntimeError as exc:
        if splash is not None:
            splash.close()
        QMessageBox.critical(None, "Démarrage impossible", str(exc))
        db.close()
        return 2

    # Imported late so a headless run of the rest of the package does not need
    # the widget modules.
    from .main_window import MainWindow

    window = MainWindow(settings, db, orchestrator)
    orchestrator.start()
    window.show()
    if splash is not None:
        splash.finish(window)

    try:
        return app.exec()
    finally:
        orchestrator.stop()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
