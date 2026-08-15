"""Windows volume backend, via the Win32 API through ctypes.

Kept dependency-free on purpose: pywin32 is a heavy install for what amounts
to three calls. Ejection asks the volume to flush, lock and eject, which is
the same sequence Explorer's "Safely Remove Hardware" performs.

Note: this backend has not been exercised on a Windows host; it is written
against the documented API behaviour.
"""

from __future__ import annotations

import ctypes
import logging
import string
from ctypes import wintypes
from pathlib import Path

from .base import Volume, VolumeBackend

log = logging.getLogger(__name__)

DRIVE_REMOVABLE = 2
_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value

_FSCTL_LOCK_VOLUME = 0x00090018
_FSCTL_DISMOUNT_VOLUME = 0x00090020
_IOCTL_STORAGE_EJECT_MEDIA = 0x002D4808


class WindowsVolumeBackend(VolumeBackend):
    def __init__(self) -> None:
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def list_volumes(self) -> list[Volume]:
        volumes: list[Volume] = []
        mask = self._k32.GetLogicalDrives()
        for index, letter in enumerate(string.ascii_uppercase):
            if not mask & (1 << index):
                continue
            root = f"{letter}:\\"
            if self._k32.GetDriveTypeW(root) != DRIVE_REMOVABLE:
                continue
            label, serial = self._volume_info(root)
            if label is None:
                # Reader present but no card inserted.
                continue
            volumes.append(
                Volume(
                    label=label,
                    mountpoint=Path(root),
                    device=f"{letter}:",
                    uuid=serial,
                )
            )
        return volumes

    def _volume_info(self, root: str) -> tuple[str | None, str | None]:
        name_buf = ctypes.create_unicode_buffer(261)
        fs_buf = ctypes.create_unicode_buffer(261)
        serial = wintypes.DWORD()
        ok = self._k32.GetVolumeInformationW(
            wintypes.LPCWSTR(root),
            name_buf, ctypes.sizeof(name_buf) // ctypes.sizeof(ctypes.c_wchar),
            ctypes.byref(serial), None, None,
            fs_buf, ctypes.sizeof(fs_buf) // ctypes.sizeof(ctypes.c_wchar),
        )
        if not ok:
            return None, None
        return name_buf.value, f"{serial.value:08X}"

    def eject(self, volume: Volume) -> tuple[bool, str]:
        letter = (volume.device or "").rstrip(":\\")
        if not letter:
            return False, "Lettre de lecteur inconnue ; éjection impossible."

        handle = self._k32.CreateFileW(
            wintypes.LPCWSTR(f"\\\\.\\{letter}:"),
            _GENERIC_READ | _GENERIC_WRITE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE,
            None, _OPEN_EXISTING, 0, None,
        )
        if handle == _INVALID_HANDLE or handle is None:
            return False, (
                f"Impossible d'ouvrir le volume {letter}: "
                f"(erreur {ctypes.get_last_error()})."
            )
        try:
            for code, what in (
                (_FSCTL_LOCK_VOLUME, "verrouillage"),
                (_FSCTL_DISMOUNT_VOLUME, "démontage"),
                (_IOCTL_STORAGE_EJECT_MEDIA, "éjection"),
            ):
                returned = wintypes.DWORD()
                ok = self._k32.DeviceIoControl(
                    handle, code, None, 0, None, 0, ctypes.byref(returned), None
                )
                if not ok:
                    err = ctypes.get_last_error()
                    if code == _FSCTL_LOCK_VOLUME:
                        return False, (
                            "Le volume est encore utilisé par un autre programme ; "
                            "fermez-le puis réessayez."
                        )
                    return False, f"Échec du {what} (erreur {err})."
        finally:
            self._k32.CloseHandle(handle)
        return True, "Carte éjectée ; vous pouvez la retirer."
