"""Linux volume backend, built on lsblk and udisks.

``lsblk --json`` gives labels, mountpoints and removability in one call, and
``udisksctl`` ejects without needing root -- the same path the desktop uses
when you click Eject, so cards end up in the state operators expect.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from ..render.caps import subprocess_kwargs
from .base import Volume, VolumeBackend

log = logging.getLogger(__name__)

_LSBLK_FIELDS = "NAME,PATH,LABEL,MOUNTPOINT,MOUNTPOINTS,RM,HOTPLUG,TRAN,TYPE,FSTYPE,UUID"


def _flatten(nodes: list[dict], inherited_transport: str | None = None):
    for node in nodes:
        transport = node.get("tran") or inherited_transport
        yield node, transport
        yield from _flatten(node.get("children", []) or [], transport)


class LinuxVolumeBackend(VolumeBackend):
    def __init__(self) -> None:
        self._lsblk = shutil.which("lsblk")
        self._udisksctl = shutil.which("udisksctl")
        self._eject = shutil.which("eject")
        if not self._lsblk:
            raise RuntimeError(
                "lsblk est introuvable ; impossible de détecter les cartes SD."
            )

    def list_volumes(self) -> list[Volume]:
        try:
            proc = subprocess.run(
                [self._lsblk, "--json", "--bytes", "-o", _LSBLK_FIELDS],
                capture_output=True, text=True, timeout=15, check=True,
                **subprocess_kwargs(),
            )
            data = json.loads(proc.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            log.warning("lsblk failed: %s", exc)
            return []

        volumes: list[Volume] = []
        for node, transport in _flatten(data.get("blockdevices", []) or []):
            mountpoint = _first_mountpoint(node)
            if not mountpoint:
                continue
            if node.get("type") not in ("part", "disk"):
                continue
            # Removable, hot-pluggable, or plainly on a USB/MMC bus. Internal
            # disks must never be treated as cards -- we delete from cards.
            if not (
                node.get("rm")
                or node.get("hotplug")
                or (transport or "") in ("usb", "mmc")
            ):
                continue
            volumes.append(
                Volume(
                    label=(node.get("label") or "").strip(),
                    mountpoint=Path(mountpoint),
                    device=node.get("path") or node.get("name"),
                    uuid=node.get("uuid"),
                    fstype=node.get("fstype"),
                )
            )
        return volumes

    def eject(self, volume: Volume) -> tuple[bool, str]:
        """Unmount, then power off the whole reader when possible."""
        device = volume.device
        if not device:
            return False, "Périphérique inconnu ; éjection impossible."

        if self._udisksctl:
            ok, message = self._run(
                [self._udisksctl, "unmount", "-b", device], "démontage"
            )
            if not ok:
                return False, message
            # Powering off is a courtesy: the card is already safe to remove
            # once unmounted, so a failure here is not an error.
            parent = _parent_device(device)
            if parent:
                powered, detail = self._run(
                    [self._udisksctl, "power-off", "-b", parent], "extinction"
                )
                if not powered:
                    log.debug("power-off refused for %s: %s", parent, detail)
            return True, "Carte éjectée ; vous pouvez la retirer."

        if self._eject:
            ok, message = self._run([self._eject, device], "éjection")
            return ok, "Carte éjectée ; vous pouvez la retirer." if ok else message

        return False, "Ni udisksctl ni eject ne sont disponibles."

    def _run(self, cmd: list[str], what: str) -> tuple[bool, str]:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, **subprocess_kwargs()
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return False, f"Échec du {what} : {exc}"
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"code {proc.returncode}"
            return False, f"Échec du {what} : {detail}"
        return True, ""


def _first_mountpoint(node: dict) -> str | None:
    single = node.get("mountpoint")
    if single:
        return single
    for point in node.get("mountpoints") or []:
        if point:
            return point
    return None


def _parent_device(device: str) -> str | None:
    """``/dev/sdd1`` -> ``/dev/sdd``; ``/dev/mmcblk0p1`` -> ``/dev/mmcblk0``."""
    name = device.rstrip("0123456789")
    if name.endswith("p") and name[:-1].rstrip("0123456789") != name[:-1]:
        name = name[:-1]
    return name if name != device else None
