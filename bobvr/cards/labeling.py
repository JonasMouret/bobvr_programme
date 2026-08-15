"""Naming a card so the app recognises it.

Cards are identified by their volume label, which means each card has to be
labelled once. Writing a filesystem label needs privileges the app does not
(and should not) hold, so this module validates the request, works out the
exact command for the filesystem in question, and either runs it -- when the
app happens to have the rights -- or hands the operator a command to paste.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..render.caps import subprocess_kwargs
from .base import parse_card_id

log = logging.getLogger(__name__)

#: Which tool writes a label, per filesystem. GoPro formats large cards exFAT
#: and small ones FAT32.
LABEL_TOOLS: dict[str, str] = {
    "vfat": "fatlabel",
    "fat": "fatlabel",
    "fat32": "fatlabel",
    "msdos": "fatlabel",
    "exfat": "exfatlabel",
    "ntfs": "ntfslabel",
}

#: exFAT and FAT32 labels are limited to 11 characters; ours are far shorter.
MAX_LABEL = 11


class LabelError(RuntimeError):
    """The card could not be labelled."""


@dataclass(frozen=True)
class LabelPlan:
    """What it would take to apply a label."""

    device: str
    card_id: str
    fstype: str
    command: list[str]
    mounted_at: Path | None

    @property
    def needs_unmount(self) -> bool:
        return self.mounted_at is not None

    def as_shell(self) -> str:
        quoted = " ".join(
            part if part.replace("/", "").replace(".", "").isalnum() else f"'{part}'"
            for part in self.command
        )
        return quoted

    def instructions(self) -> str:
        lines = []
        if sys.platform == "win32":
            lines.append(
                "Ouvrez une invite de commandes en tant qu'administrateur, "
                "puis :"
            )
            lines.append(f"    label {self.device} {self.card_id}")
        elif self.needs_unmount:
            lines.append(
                f"1. Démontez la carte :  udisksctl unmount -b {self.device}"
            )
            lines.append(f"2. Écrivez le nom  :  sudo {self.as_shell()}")
        else:
            lines.append(f"Écrivez le nom :  sudo {self.as_shell()}")
        lines.append(
            "Puis retirez et réinsérez la carte pour que BobVr la reconnaisse."
        )
        return "\n".join(lines)


def plan_label(
    device: str,
    card_id: str,
    fstype: str,
    fleet: dict[str, int],
    mounted_at: Path | None = None,
) -> LabelPlan:
    """Validate a labelling request and work out how to carry it out.

    Raises:
        LabelError: the id is not in the fleet, or the filesystem is unknown.
    """
    canonical = parse_card_id(card_id, fleet)
    if canonical is None:
        known = ", ".join(
            f"{prefix}1–{prefix}{count}" for prefix, count in sorted(fleet.items())
        )
        raise LabelError(
            f"« {card_id} » n'est pas un identifiant d'engin valide. "
            f"Attendu : {known}."
        )
    if len(canonical) > MAX_LABEL:
        raise LabelError(
            f"« {canonical} » dépasse {MAX_LABEL} caractères, la limite du "
            "système de fichiers."
        )

    if sys.platform == "win32":
        # Windows names volumes by drive letter, with a command that ships
        # with the system and relabels in place -- no unmount, no extra tool,
        # whatever the filesystem is.
        letter = device.strip().rstrip("\\").rstrip(":")[:1].upper()
        if not letter.isalpha():
            raise LabelError(
                f"« {device} » n'est pas une lettre de lecteur (attendu : E:)."
            )
        return LabelPlan(
            device=f"{letter}:",
            card_id=canonical,
            fstype=fstype,
            command=["cmd", "/c", "label", f"{letter}:", canonical],
            mounted_at=None,
        )

    tool_name = LABEL_TOOLS.get(fstype.lower())
    if tool_name is None:
        raise LabelError(
            f"Système de fichiers « {fstype} » non pris en charge pour "
            "l'étiquetage. Reformatez la carte en exFAT depuis la caméra."
        )
    tool = shutil.which(tool_name) or shutil.which(tool_name, path="/usr/sbin:/sbin")
    if tool is None:
        package = {
            "fatlabel": "dosfstools",
            "exfatlabel": "exfatprogs",
            "ntfslabel": "ntfs-3g",
        }[tool_name]
        raise LabelError(
            f"{tool_name} est introuvable. Installez-le : sudo apt install {package}"
        )

    return LabelPlan(
        device=device,
        card_id=canonical,
        fstype=fstype,
        command=[tool, device, canonical],
        mounted_at=mounted_at,
    )


def apply_label(plan: LabelPlan) -> tuple[bool, str]:
    """Run the plan if we have the rights; otherwise explain what to run.

    Returns (applied, message).
    """
    if plan.needs_unmount:
        return False, (
            f"La carte est montée sur {plan.mounted_at}. Démontez-la d'abord.\n\n"
            + plan.instructions()
        )
    if sys.platform != "win32" and os.geteuid() != 0:
        return False, (
            "L'écriture du nom de volume demande des droits administrateur.\n\n"
            + plan.instructions()
        )

    try:
        proc = subprocess.run(
            plan.command, capture_output=True, text=True, timeout=30,
            **subprocess_kwargs(),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return False, f"Échec de l'étiquetage : {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"code {proc.returncode}"
        return False, f"Échec de l'étiquetage : {detail}"
    return True, (
        f"Carte nommée « {plan.card_id} ». Retirez et réinsérez-la pour que "
        "BobVr la détecte."
    )
