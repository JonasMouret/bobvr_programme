"""The running application: watch for cards, ingest them, render the results.

Kept free of any UI toolkit so the workflow can be driven from a window, from
a script, or from a test. Callers subscribe to events; every event is
delivered on a worker thread, so a UI must marshal them onto its own.

Ingestion and rendering run on separate workers. Copying is bound by the card
reader and rendering by the GPU, so overlapping them means the next card can
be read while the previous one is still rendering -- which is exactly the
rhythm of a race day.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from .cards import Card, CardWatcher, VolumeBackend, make_backend
from .config import Settings
from .db import (
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RENDERING,
    Clip,
    Database,
)
from .ingest import IngestProgress, IngestReport, Ingestor
from .media import NotAMaxVideoError, probe
from .render.caps import Capabilities, detect
from .render.geometry import Orientation
from .render.pipeline import (
    RenderCancelled,
    RenderError,
    RenderProgress,
    RenderSettings,
    Renderer,
    inject_spherical_metadata,
)

log = logging.getLogger(__name__)


@dataclass
class Listeners:
    """Callbacks the UI (or a test) can fill in. All are optional."""

    card_detected: Callable[[Card], None] | None = None
    card_removed: Callable[[Card], None] | None = None
    ingest_started: Callable[[Card], None] | None = None
    ingest_progress: Callable[[IngestProgress], None] | None = None
    ingest_finished: Callable[[IngestReport], None] | None = None
    render_started: Callable[[Clip], None] | None = None
    render_progress: Callable[[Clip, RenderProgress], None] | None = None
    render_finished: Callable[[Clip], None] | None = None
    render_failed: Callable[[Clip, str], None] | None = None
    queue_changed: Callable[[], None] | None = None
    notice: Callable[[str, str], None] | None = None   # (level, message)

    def fire(self, name: str, *args) -> None:
        callback = getattr(self, name, None)
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            # A broken listener must never take down a worker mid-copy.
            log.exception("listener %s raised", name)


class Orchestrator:
    """Owns the watcher and the two worker threads."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        caps: Capabilities | None = None,
        backend: VolumeBackend | None = None,
    ):
        self.settings = settings
        self.db = db
        self.caps = caps or detect(settings.ffmpeg_path, settings.ffprobe_path)
        self.backend = backend or make_backend()
        self.listeners = Listeners()

        self.renderer = Renderer(self.caps, _kernel_cache_dir(settings))
        self.ingestor = Ingestor(settings, db, self.caps, self.backend)
        self.watcher = CardWatcher(self.backend, settings.fleet)
        self.watcher.on_card_added = self._card_added
        self.watcher.on_card_removed = self._card_removed

        self._cards: queue.Queue[Card | None] = queue.Queue()
        self._renders: queue.Queue[int | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._cancel_render = threading.Event()
        self._current_clip: Clip | None = None
        self._current_card: Card | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------- lifecycle

    def start(self) -> None:
        recovered = self.db.recover_interrupted()
        if recovered:
            self.listeners.fire(
                "notice", "info",
                f"Reprise après un arrêt inattendu : {recovered} élément(s) réinitialisé(s).",
            )
        for clip in self.db.pending_renders():
            self._renders.put(clip.id)

        self._stopping.clear()
        for name, target in (
            ("ingest-worker", self._ingest_loop),
            ("render-worker", self._render_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self.watcher.start()
        log.info("orchestrator started (%s)", self.caps.summary())

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        self._cancel_render.set()
        self.watcher.stop()
        self._cards.put(None)
        self._renders.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    # ------------------------------------------------------------- status

    @property
    def current_clip(self) -> Clip | None:
        with self._lock:
            return self._current_clip

    @property
    def current_card(self) -> Card | None:
        with self._lock:
            return self._current_card

    def queue_depth(self) -> int:
        return self._renders.qsize()

    # ------------------------------------------------------------ actions

    def ingest_now(self, card: Card) -> None:
        """Queue a card for ingestion, whatever the auto-ingest setting says."""
        self._cards.put(card)

    def render_now(self, clip_id: int) -> None:
        self.db.set_status(clip_id, STATUS_QUEUED)
        self._renders.put(clip_id)
        self.listeners.fire("queue_changed")

    def retry(self, clip_id: int) -> None:
        clip = self.db.clip(clip_id)
        if clip.archive_path is None or not clip.archive_path.exists():
            self.listeners.fire(
                "notice", "error",
                f"{clip.name} : le fichier d'origine est introuvable, "
                "impossible de relancer le rendu.",
            )
            return
        self.render_now(clip_id)

    def cancel_current_render(self) -> None:
        self._cancel_render.set()

    def rescan(self) -> None:
        self.watcher.poll()

    # ------------------------------------------------------------- events

    def _card_added(self, card: Card) -> None:
        self.listeners.fire("card_detected", card)
        if self.settings.auto_ingest:
            self._cards.put(card)

    def _card_removed(self, card: Card) -> None:
        self.listeners.fire("card_removed", card)

    # ------------------------------------------------------------ workers

    def _ingest_loop(self) -> None:
        while not self._stopping.is_set():
            card = self._cards.get()
            if card is None:
                break
            with self._lock:
                self._current_card = card
            try:
                self.listeners.fire("ingest_started", card)
                report = self.ingestor.ingest_card(
                    card,
                    on_progress=lambda p: self.listeners.fire("ingest_progress", p),
                    cancel=self._stopping,
                    capture_day=date.today(),
                )
            except Exception as exc:
                log.exception("ingest of %s failed", card.card_id)
                self.listeners.fire(
                    "notice", "error", f"Carte {card.card_id} : {exc}"
                )
                continue
            finally:
                with self._lock:
                    self._current_card = None

            self.listeners.fire("ingest_finished", report)
            for clip_id in report.copied:
                if self.settings.auto_render:
                    self.render_now(clip_id)
            if report.failures:
                detail = "; ".join(f"{n} ({e})" for n, e in report.failures)
                self.listeners.fire(
                    "notice", "error",
                    f"Carte {card.card_id} conservée intacte, des fichiers ont "
                    f"échoué : {detail}",
                )
            if report.skipped:
                self.listeners.fire(
                    "notice", "warning",
                    f"Carte {card.card_id} : "
                    f"{', '.join(report.skipped)} n'a pas pu être lu et a été "
                    "ignoré. La carte n'a été ni vidée ni éjectée.",
                )

    def _render_loop(self) -> None:
        while not self._stopping.is_set():
            clip_id = self._renders.get()
            if clip_id is None:
                break
            self.listeners.fire("queue_changed")
            try:
                self._render_one(clip_id)
            except Exception:
                log.exception("render worker error on clip %s", clip_id)

    def _render_one(self, clip_id: int) -> None:
        try:
            clip = self.db.clip(clip_id)
        except KeyError:
            return
        if clip.archive_path is None or not clip.archive_path.exists():
            self.db.set_status(clip_id, STATUS_FAILED, "fichier d'origine introuvable")
            self.listeners.fire(
                "render_failed", clip, "Le fichier d'origine est introuvable."
            )
            return

        try:
            info = probe(clip.archive_path, self.caps)
        except NotAMaxVideoError as exc:
            self.db.set_status(clip_id, STATUS_FAILED, str(exc))
            self.listeners.fire("render_failed", clip, str(exc))
            return

        destination = (
            self.settings.render_dir(clip.card_id, clip.capture_date)
            / f"{clip.name}.mp4"
        )
        settings = self._render_settings()

        self._cancel_render.clear()
        self.db.set_status(clip_id, STATUS_RENDERING)
        with self._lock:
            self._current_clip = clip
        self.listeners.fire("render_started", clip)

        try:
            output = self.renderer.render(
                info, destination, settings,
                on_progress=lambda p: self.listeners.fire("render_progress", clip, p),
                cancel=self._cancel_render,
            )
        except RenderCancelled:
            self.db.set_status(clip_id, STATUS_QUEUED)
            self.listeners.fire("notice", "info", f"{clip.name} : rendu annulé.")
            return
        except RenderError as exc:
            self.db.set_status(clip_id, STATUS_FAILED, str(exc))
            self.listeners.fire("render_failed", clip, str(exc))
            return
        finally:
            with self._lock:
                self._current_clip = None
            self.listeners.fire("queue_changed")

        if not inject_spherical_metadata(output):
            self.listeners.fire(
                "notice", "warning",
                f"{clip.name} : rendu terminé, mais les métadonnées 360 n'ont pas "
                "pu être écrites. La vidéo s'ouvrira à plat.",
            )
        self.db.mark_rendered(clip_id, output)
        self.listeners.fire("render_finished", self.db.clip(clip_id))

    def _render_settings(self) -> RenderSettings:
        config = self.settings.render
        return RenderSettings(
            width=config.width,
            height=config.height,
            orientation=Orientation(config.yaw, config.pitch, config.roll),
            cubic=config.cubic,
            initial_fov=config.initial_fov,
            stabilise_seconds=(config.stabilise_seconds
                               if config.stabilise else 0.0),
            quality=config.quality,
            max_bitrate_kbps=config.max_bitrate_kbps,
            audio_bitrate_kbps=config.audio_bitrate_kbps,
            force_cpu=config.force_cpu,
        )


def _kernel_cache_dir(settings: Settings) -> Path:
    from platformdirs import user_cache_dir

    from .config import APP_NAME

    return Path(user_cache_dir(APP_NAME, appauthor=False)) / "kernels"
