"""Settings, grouped the way an operator thinks about them.

Nothing here is applied until OK is pressed, and the dialog hands back a fresh
Settings object rather than mutating the live one, so cancelling is genuinely
free.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import FLEET_LABELS, RenderConfig, Settings
from ..render.caps import Capabilities
from ..render.geometry import MAX_FOV, MIN_FOV, NEUTRAL_FOV

#: Equirectangular output must be 2:1, so presets come in pairs.
RESOLUTIONS = [
    ("2880 × 1440 (2.7K) — fichiers légers", 2880, 1440),
    ("3840 × 1920 (4K) — recommandé", 3840, 1920),
    ("4032 × 2016 — résolution native du stitch", 4032, 2016),
    ("5760 × 2880 (5.7K) — archivage", 5760, 2880),
]

#: Shown when the size fields hold something that is not one of the presets.
CUSTOM_RESOLUTION = "Personnalisée"

#: Widest frame the NVENC H.264 encoder accepts. Beyond it the render fails on
#: the GPU and has to fall back to libx264, which is worth saying before the
#: operator waits for a run to fail.
NVENC_H264_MAX_WIDTH = 4096

QUALITIES = [
    ("Haute (fichiers plus gros)", 20),
    ("Standard (recommandé)", 23),
    ("Légère (fichiers plus petits)", 27),
]


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, caps: Capabilities, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Réglages de BobVr")
        self.resize(700, 700)
        self._settings = settings
        self._caps = caps
        self.result_settings = settings

        tabs = QTabWidget(self)
        tabs.addTab(self._build_storage_tab(), "Stockage")
        tabs.addTab(self._build_cards_tab(), "Cartes")
        tabs.addTab(self._build_render_tab(), "Rendu")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # --------------------------------------------------------------- tabs

    def _build_storage_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.library_edit = QLineEdit(str(self._settings.library_root))
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self._pick_library)
        row = QHBoxLayout()
        row.addWidget(self.library_edit)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Dossier de la bibliothèque", holder)

        self.archive_edit = QLineEdit(self._settings.archive_dirname)
        self.render_edit = QLineEdit(self._settings.render_dirname)
        form.addRow("Sous-dossier des originaux", self.archive_edit)
        form.addRow("Sous-dossier des rendus", self.render_edit)

        note = QLabel(
            "Les fichiers sont rangés par jour : "
            "<code>bibliothèque/originaux/2026-02-03/B1_1.360</code> et "
            "<code>bibliothèque/equirect/2026-02-03/B1_1.mp4</code>.<br>"
            "Les fichiers .360 d'origine sont conservés : ils sont nécessaires "
            "pour refaire un rendu ou exploiter la télémétrie."
        )
        note.setWordWrap(True)
        form.addRow(note)

        self.ffmpeg_edit = QLineEdit(self._settings.ffmpeg_path or "")
        self.ffmpeg_edit.setPlaceholderText(self._caps.ffmpeg)
        self.ffprobe_edit = QLineEdit(self._settings.ffprobe_path or "")
        self.ffprobe_edit.setPlaceholderText(self._caps.ffprobe)
        form.addRow("Chemin de ffmpeg (facultatif)", self.ffmpeg_edit)
        form.addRow("Chemin de ffprobe (facultatif)", self.ffprobe_edit)
        return page

    def _build_cards_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        fleet_box = QGroupBox("Parc d'engins")
        fleet_form = QFormLayout(fleet_box)
        self.fleet_spins: dict[str, QSpinBox] = {}
        for prefix, label in FLEET_LABELS.items():
            spin = QSpinBox()
            spin.setRange(0, 99)
            spin.setValue(self._settings.fleet.get(prefix, 0))
            self.fleet_spins[prefix] = spin
            fleet_form.addRow(f"{label} (préfixe {prefix})", spin)
        hint = QLabel(
            "Une carte est reconnue par le nom de son volume : préfixe puis "
            "numéro, par exemple <b>B1</b> ou <b>S4</b>. Les volumes dont le nom "
            "ne correspond pas sont ignorés."
        )
        hint.setWordWrap(True)
        fleet_form.addRow(hint)
        layout.addWidget(fleet_box)

        search_box = QGroupBox("Recherche des fichiers")
        search_form = QFormLayout(search_box)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 8)
        self.depth_spin.setValue(self._settings.card_search_depth)
        search_form.addRow("Profondeur de recherche", self.depth_spin)
        search_hint = QLabel(
            "Les fichiers .360 sont cherchés dans toute la carte, quelle que "
            "soit l'arborescence : <code>100GOPRO</code> à la racine, "
            "<code>DCIM/100GOPRO</code>, ou les dossiers suivants "
            "(<code>101GOPRO</code>…) créés par la caméra quand le premier est "
            "plein."
        )
        search_hint.setWordWrap(True)
        search_form.addRow(search_hint)
        layout.addWidget(search_box)

        auto_box = QGroupBox("Automatisation")
        auto_layout = QVBoxLayout(auto_box)
        self.auto_ingest_check = _check(
            "Importer automatiquement dès qu'une carte est détectée",
            self._settings.auto_ingest,
        )
        self.verify_check = _check(
            "Vérifier la copie par empreinte avant d'effacer la carte "
            "(fortement recommandé)",
            self._settings.verify_checksum,
        )
        self.delete_check = _check(
            "Effacer les fichiers de la carte après une copie vérifiée",
            self._settings.auto_delete_source,
        )
        self.clear_folder_check = _check(
            "Repartir d'un dossier caméra vide : supprimer DCIM/100GOPRO "
            "puis le recréer",
            self._settings.clear_camera_folder,
        )
        self.eject_check = _check(
            "Éjecter la carte à la fin", self._settings.auto_eject
        )
        self.auto_render_check = _check(
            "Lancer le rendu automatiquement après l'importation",
            self._settings.auto_render,
        )
        for widget in (
            self.auto_ingest_check, self.verify_check, self.delete_check,
            self.clear_folder_check, self.eject_check, self.auto_render_check,
        ):
            auto_layout.addWidget(widget)

        folder_note = QLabel(
            "Le dossier de la caméra est supprimé en entier, ce qui emporte "
            "aussi ses fichiers annexes (.LRV, .THM). Le reste de la carte "
            "n'est jamais touché."
        )
        folder_note.setWordWrap(True)
        folder_note.setIndent(20)
        auto_layout.addWidget(folder_note)
        self.clear_folder_check.toggled.connect(folder_note.setVisible)
        folder_note.setVisible(self._settings.clear_camera_folder)
        # Nothing is deleted from a card at all unless deletion is on.
        self.clear_folder_check.setEnabled(self._settings.auto_delete_source)
        self.delete_check.toggled.connect(self.clear_folder_check.setEnabled)

        warning = QLabel(
            "Sans vérification, un fichier corrompu pendant la copie serait "
            "effacé de la carte sans que rien ne le signale."
        )
        warning.setWordWrap(True)
        auto_layout.addWidget(warning)
        self.verify_check.toggled.connect(
            lambda on: warning.setVisible(not on)
        )
        warning.setVisible(not self._settings.verify_checksum)
        layout.addWidget(auto_box)
        layout.addStretch(1)
        return page

    def _build_render_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        config = self._settings.render

        self.resolution_combo = QComboBox()
        for label, width, height in RESOLUTIONS:
            self.resolution_combo.addItem(label, (width, height))
        self.resolution_combo.addItem(CUSTOM_RESOLUTION, None)
        index = next(
            (i for i, (_, w, h) in enumerate(RESOLUTIONS)
             if (w, h) == (config.width, config.height)),
            self.resolution_combo.count() - 1,
        )
        self.resolution_combo.setCurrentIndex(index)
        self.resolution_combo.activated.connect(self._resolution_chosen)
        form.addRow("Résolution de sortie", self.resolution_combo)

        # The two fields the operator can drive directly. They stay locked to
        # each other: an equirectangular frame that is not 2:1 is not a sphere,
        # and every player would read it as a distorted one.
        self.width_spin = _pixels(config.width, step=4)
        self.height_spin = _pixels(config.height, step=2)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Largeur"))
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("Hauteur"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch(1)
        size_holder = QWidget()
        size_holder.setLayout(size_row)
        form.addRow("Taille (pixels)", size_holder)
        self.width_spin.valueChanged.connect(self._width_edited)
        self.height_spin.valueChanged.connect(self._height_edited)

        self.size_note = QLabel()
        self.size_note.setWordWrap(True)
        form.addRow(self.size_note)
        self._refresh_size_note()

        self.quality_combo = QComboBox()
        for label, value in QUALITIES:
            self.quality_combo.addItem(label, value)
        quality_index = next(
            (i for i, (_, v) in enumerate(QUALITIES) if v == config.quality), 1
        )
        self.quality_combo.setCurrentIndex(quality_index)
        form.addRow("Qualité", self.quality_combo)

        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(5_000, 200_000)
        self.bitrate_spin.setSingleStep(5_000)
        self.bitrate_spin.setSuffix(" kbit/s")
        self.bitrate_spin.setValue(config.max_bitrate_kbps)
        form.addRow("Débit maximal", self.bitrate_spin)

        self.cubic_check = _check(
            "Échantillonnage bicubique (meilleure netteté, coût négligeable "
            "sur GPU)",
            config.cubic,
        )
        form.addRow(self.cubic_check)

        orientation_box = QGroupBox("Orientation de la caméra")
        orientation_form = QFormLayout(orientation_box)
        self.yaw_spin = _angle(config.yaw)
        self.pitch_spin = _angle(config.pitch)
        self.roll_spin = _angle(config.roll)
        orientation_form.addRow("Lacet (yaw)", self.yaw_spin)
        orientation_form.addRow("Tangage (pitch)", self.pitch_spin)
        orientation_form.addRow("Roulis (roll)", self.roll_spin)
        orientation_hint = QLabel(
            "Corrige l'inclinaison de la caméra sur le bob. Le point de vue "
            "initial de la vidéo suit ce réglage."
        )
        orientation_hint.setWordWrap(True)
        orientation_form.addRow(orientation_hint)
        form.addRow(orientation_box)

        view_box = QGroupBox("Vue à l'ouverture")
        view_form = QFormLayout(view_box)
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(MIN_FOV, MAX_FOV)
        self.fov_spin.setSingleStep(5.0)
        self.fov_spin.setDecimals(0)
        self.fov_spin.setSuffix(" °")
        self.fov_spin.setValue(config.initial_fov)
        view_form.addRow("Champ de vision (FOV)", self.fov_spin)
        view_hint = QLabel(
            f"Largeur de la vue à l'ouverture, comme un objectif plus large : "
            f"les lignes droites le restent. <b>{NEUTRAL_FOV:g}°</b> est la "
            "valeur neutre, celle des lecteurs ; <b>130°</b> ouvre nettement "
            "et fait apparaître la nacelle du bob.<br>"
            "Appliqué <b>au rendu</b> — les clips déjà convertis sont à "
            "refaire. La navigation au doigt reste entière."
        )
        view_hint.setWordWrap(True)
        view_form.addRow(view_hint)

        if not self._caps.has_opencl_kernel:
            self.fov_spin.setEnabled(False)
            self.fov_spin.setValue(NEUTRAL_FOV)
            no_gpu = QLabel(
                "Réglage indisponible sans kernel OpenCL : le repli logiciel "
                "ne sait pas redistribuer la sphère sans courber les lignes."
            )
            no_gpu.setWordWrap(True)
            view_form.addRow(no_gpu)
        form.addRow(view_box)

        self.force_cpu_check = _check(
            "Forcer le rendu sur le processeur (diagnostic uniquement)",
            config.force_cpu,
        )
        form.addRow(self.force_cpu_check)

        route = QLabel(
            f"Accélération détectée : {self._caps.summary()}"
        )
        route.setWordWrap(True)
        form.addRow(route)
        return page

    # -------------------------------------------------------- output size

    def _resolution_chosen(self, index: int) -> None:
        """A preset was picked: fill the fields from it."""
        preset = self.resolution_combo.itemData(index)
        if preset is None:                      # "Personnalisée": leave as is
            return
        width, height = preset
        for spin, value in ((self.width_spin, width), (self.height_spin, height)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._refresh_size_note()

    def _width_edited(self, width: int) -> None:
        # A multiple of four, so the half that becomes the height is still
        # even -- H.264 cannot encode an odd dimension either way.
        width = _snap(self.width_spin, width, 4)
        self._set_paired(self.height_spin, width // 2)

    def _height_edited(self, height: int) -> None:
        height = _snap(self.height_spin, height, 2)
        self._set_paired(self.width_spin, height * 2)

    def _set_paired(self, other: QSpinBox, value: int) -> None:
        """Move the other field to match, without it answering back."""
        other.blockSignals(True)
        other.setValue(max(other.minimum(), min(other.maximum(), value)))
        other.blockSignals(False)
        self._match_preset()
        self._refresh_size_note()

    def _match_preset(self) -> None:
        """Point the combo at whichever preset the fields now describe."""
        size = (self.width_spin.value(), self.height_spin.value())
        index = next(
            (i for i, (_, w, h) in enumerate(RESOLUTIONS) if (w, h) == size),
            self.resolution_combo.count() - 1,
        )
        self.resolution_combo.setCurrentIndex(index)

    def _refresh_size_note(self) -> None:
        width, height = self.width_spin.value(), self.height_spin.value()
        note = (
            f"Sortie {width} × {height}. Les deux champs restent liés au "
            "format 2:1 : une image équirectangulaire d'un autre rapport n'est "
            "plus une sphère, et les lecteurs la montreraient déformée."
        )
        if (
            width > NVENC_H264_MAX_WIDTH
            and self._caps.video_encoder == "h264_nvenc"
        ):
            note += (
                f"<br><b>Attention :</b> au-delà de {NVENC_H264_MAX_WIDTH} px "
                "de large, l'encodeur NVENC en H.264 refuse l'image et le "
                "rendu échouera. Restez à 4096 × 2048 pour garder le GPU."
            )
        self.size_note.setText(note)

    # ------------------------------------------------------------ actions

    def _pick_library(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choisir le dossier de la bibliothèque", self.library_edit.text()
        )
        if chosen:
            self.library_edit.setText(chosen)

    def _accept(self) -> None:
        width, height = self.width_spin.value(), self.height_spin.value()
        fleet = {
            prefix: spin.value()
            for prefix, spin in self.fleet_spins.items()
            if spin.value() > 0
        }
        if not fleet:
            QMessageBox.warning(
                self, "Parc vide",
                "Indiquez au moins un engin, sinon aucune carte ne sera reconnue.",
            )
            return

        try:
            render = RenderConfig(
                width=width,
                height=height,
                quality=self.quality_combo.currentData(),
                max_bitrate_kbps=self.bitrate_spin.value(),
                audio_bitrate_kbps=self._settings.render.audio_bitrate_kbps,
                cubic=self.cubic_check.isChecked(),
                yaw=self.yaw_spin.value(),
                pitch=self.pitch_spin.value(),
                roll=self.roll_spin.value(),
                force_cpu=self.force_cpu_check.isChecked(),
                initial_fov=self.fov_spin.value(),
            )
            updated = Settings(
                library_root=Path(self.library_edit.text()).expanduser(),
                archive_dirname=self.archive_edit.text().strip() or "originaux",
                render_dirname=self.render_edit.text().strip() or "equirect",
                fleet=fleet,
                card_search_depth=self.depth_spin.value(),
                auto_ingest=self.auto_ingest_check.isChecked(),
                auto_delete_source=self.delete_check.isChecked(),
                clear_camera_folder=self.clear_folder_check.isChecked(),
                auto_eject=self.eject_check.isChecked(),
                verify_checksum=self.verify_check.isChecked(),
                auto_render=self.auto_render_check.isChecked(),
                render=render,
                ffmpeg_path=self.ffmpeg_edit.text().strip() or None,
                ffprobe_path=self.ffprobe_edit.text().strip() or None,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Réglages invalides", str(exc))
            return

        if self.delete_check.isChecked() and not self.verify_check.isChecked():
            answer = QMessageBox.question(
                self, "Effacer sans vérifier ?",
                "Vous avez choisi d'effacer les cartes sans vérifier la copie. "
                "Une copie corrompue serait alors perdue définitivement.\n\n"
                "Confirmer ce réglage ?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.result_settings = updated
        self.accept()


def _check(text: str, checked: bool) -> QCheckBox:
    box = QCheckBox(text)
    box.setChecked(checked)
    return box


def _pixels(value: int, step: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(640, 15360)
    spin.setSingleStep(step)
    spin.setSuffix(" px")
    spin.setValue(value)
    spin.setAccelerated(True)
    return spin


def _snap(spin: QSpinBox, value: int, multiple: int) -> int:
    """Round a typed value to something encodable, in place and quietly."""
    snapped = max(spin.minimum(), multiple * round(value / multiple))
    if snapped != value:
        spin.blockSignals(True)
        spin.setValue(snapped)
        spin.blockSignals(False)
    return snapped


def _angle(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-180.0, 180.0)
    spin.setSingleStep(1.0)
    spin.setDecimals(1)
    spin.setSuffix(" °")
    spin.setValue(value)
    return spin
