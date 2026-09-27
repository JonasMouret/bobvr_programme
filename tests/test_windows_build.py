"""What a frozen Windows build depends on and a source checkout does not.

None of this can be exercised on Windows from here, so the tests pin the
behaviour that differs by platform rather than the platform itself: where
helper tools are looked for, and what an operator is told to type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bobvr.cards.labeling import LabelError, plan_label
from bobvr.render import caps as caps_module
from bobvr.render.caps import app_dir, find_tool

FLEET = {"B": 6, "S": 6, "R": 5}


# ------------------------------------------------------------- finding tools


def test_a_tool_on_the_path_wins(monkeypatch):
    monkeypatch.setattr(caps_module.shutil, "which", lambda n: f"/usr/bin/{n}")
    assert find_tool("ffmpeg") == "/usr/bin/ffmpeg"


@pytest.mark.parametrize(
    "subdir",
    [".", "vendor", "ffmpeg/bin", "_internal/vendor", "_internal/ffmpeg/bin"],
)
def test_a_tool_shipped_beside_the_app_is_found(monkeypatch, tmp_path, subdir):
    """An unzipped build has no PATH: the tools travel with the .exe.

    The ``_internal`` cases are the ones PyInstaller 6 actually produces: it
    files bundled data under ``_internal`` rather than beside the executable.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    tool = tmp_path / subdir / f"ffmpeg{suffix}"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/bin/sh\n")
    monkeypatch.setattr(caps_module.shutil, "which", lambda _n: None)
    monkeypatch.setattr(caps_module, "app_dir", lambda: tmp_path)

    assert find_tool("ffmpeg") == str(tool.resolve())


def test_a_missing_tool_is_reported_as_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(caps_module.shutil, "which", lambda _n: None)
    monkeypatch.setattr(caps_module, "app_dir", lambda: tmp_path)
    assert find_tool("exiftool") is None


def test_a_frozen_build_looks_beside_its_executable(monkeypatch, tmp_path):
    """sys.argv[0] can be a shim or a relative path; sys.executable cannot."""
    exe = tmp_path / "dist" / "BobVr" / "BobVr.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "argv", ["BobVr"])

    assert app_dir() == exe.parent


# ----------------------------------------------------------------- labelling


def test_windows_labels_by_drive_letter(monkeypatch):
    """No fatlabel on Windows -- the built-in `label` does it, mounted."""
    monkeypatch.setattr(sys, "platform", "win32")
    plan = plan_label(device="E:", card_id="b1", fstype="exFAT", fleet=FLEET,
                      mounted_at=Path("E:\\"))

    assert plan.command == ["cmd", "/c", "label", "E:", "B1"]
    assert not plan.needs_unmount, "Windows renomme un volume monté"
    assert "sudo" not in plan.instructions()
    assert "label E: B1" in plan.instructions()


@pytest.mark.parametrize("device", ["E:", "E:\\", "e"])
def test_the_drive_letter_is_accepted_however_it_is_written(monkeypatch, device):
    monkeypatch.setattr(sys, "platform", "win32")
    assert plan_label(device, "B1", "exFAT", FLEET).device == "E:"


def test_something_that_is_not_a_drive_letter_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(LabelError, match="lettre de lecteur"):
        plan_label("/dev/sdd1", "B1", "exFAT", FLEET)


def test_an_unknown_engine_is_still_refused_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(LabelError, match="identifiant d'engin"):
        plan_label("E:", "Z9", "exFAT", FLEET)


# -------------------------------------------------------------- the OpenCL kernel


def test_the_kernel_source_ships_with_the_package():
    """The renderer reads it from disk at run time, so it must be bundled.

    A frozen build that forgot this file starts, finds a GPU, and fails on the
    first render -- which is the worst possible moment to find out.
    """
    from bobvr.render.geometry import KERNEL_SOURCE

    assert KERNEL_SOURCE.is_file()
    assert "gopromax_equirect" in KERNEL_SOURCE.read_text(encoding="utf-8")
    spec = Path(__file__).resolve().parents[1] / "installer" / "bobvr.spec"
    assert spec.is_file(), "le fichier .spec PyInstaller a disparu"
    assert "kernels" in spec.read_text(encoding="utf-8"), (
        "le .spec n'embarque plus le kernel OpenCL"
    )
