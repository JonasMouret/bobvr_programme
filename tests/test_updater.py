"""La logique de mise à jour, éprouvée hors ligne et hors Windows.

Comme pour le reste du portage, on ne peut pas exercer ici le vrai
téléchargement, la vraie bascule de dossier ni Windows : les tests fixent donc
la logique qui décide (comparaison de versions, lecture de la réponse GitHub,
génération du script de bascule), en injectant les réponses réseau plutôt qu'en
les appelant.
"""

from __future__ import annotations

import sys

import pytest

from bobvr import updater
from bobvr.updater import (
    Release,
    build_swap_script,
    check_for_update,
    is_newer,
    parse_release,
    parse_version,
)


# ------------------------------------------------------------- versions


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.2.3", (1, 2, 3)),
        ("v1.2.3", (1, 2, 3)),
        ("V0.1.0", (0, 1, 0)),
        ("2.0", (2, 0)),
        ("1.2.0-rc1", (1, 2, 0)),  # le suffixe de pré-version est ignoré
        ("sans-numero", (0,)),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


@pytest.mark.parametrize(
    "candidate, current, newer",
    [
        ("0.2.0", "0.1.0", True),
        ("v0.2.0", "0.1.0", True),
        ("0.1.1", "0.1.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.1.0", "0.1.0", False),
        ("0.1.0", "0.2.0", False),
        ("0.1", "0.1.0", False),  # 0.1 == 0.1.0
    ],
)
def test_is_newer(candidate, current, newer):
    assert is_newer(candidate, current) is newer


# ----------------------------------------------------------- release JSON


def _payload(**overrides):
    """Une réponse « latest release » plausible, surchargée au besoin."""
    base = {
        "tag_name": "v0.2.0",
        "name": "BobVr 0.2.0",
        "body": "Correctifs et stabilisation.",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "BobVr-0.2.0-win64.zip",
                "browser_download_url": "https://example/BobVr-0.2.0-win64.zip",
                "size": 231_000_000,
            }
        ],
    }
    base.update(overrides)
    return base


def test_parse_release_reads_the_fields():
    release = parse_release(_payload())
    assert release is not None
    assert release.version == "0.2.0"
    assert release.tag == "v0.2.0"
    assert release.asset_name == "BobVr-0.2.0-win64.zip"
    assert release.asset_url.endswith("win64.zip")
    assert release.asset_size == 231_000_000
    assert release.can_install


def test_parse_release_prefers_the_windows_zip():
    release = parse_release(
        _payload(
            assets=[
                {"name": "notes.txt", "browser_download_url": "u1", "size": 1},
                {"name": "BobVr-src.zip", "browser_download_url": "u2", "size": 2},
                {"name": "BobVr-0.2.0-win64.zip", "browser_download_url": "u3", "size": 3},
            ]
        )
    )
    assert release.asset_name == "BobVr-0.2.0-win64.zip"


def test_parse_release_without_a_zip_cannot_install():
    release = parse_release(
        _payload(assets=[{"name": "notes.txt", "browser_download_url": "u", "size": 1}])
    )
    assert release is not None
    assert release.asset_url is None
    assert not release.can_install


def test_parse_release_skips_drafts_and_prereleases():
    assert parse_release(_payload(draft=True)) is None
    assert parse_release(_payload(prerelease=True)) is None


def test_parse_release_without_a_tag_is_nothing():
    assert parse_release(_payload(tag_name="")) is None


# ------------------------------------------------------------ check flow


def test_check_returns_the_release_when_newer():
    release = check_for_update("0.1.0", fetch=_payload)
    assert release is not None
    assert release.version == "0.2.0"


def test_check_returns_none_when_up_to_date():
    assert check_for_update("0.2.0", fetch=_payload) is None
    assert check_for_update("0.3.0", fetch=_payload) is None


def test_check_swallows_network_errors():
    def boom():
        raise OSError("pas de réseau")

    assert check_for_update("0.1.0", fetch=boom) is None


# ------------------------------------------------------ check() (with reason)


def test_check_distinguishes_update_uptodate_and_error():
    from bobvr.updater import check

    release, error = check("0.1.0", fetch=_payload)
    assert release is not None and error is None

    release, error = check("0.2.0", fetch=_payload)
    assert release is None and error is None            # up to date

    def boom():
        raise OSError("pas de réseau")

    release, error = check("0.1.0", fetch=boom)
    assert release is None and error                    # a reason is given


def test_check_names_the_private_repo_404():
    import urllib.error
    from bobvr.updater import check

    def not_found():
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    release, error = check("0.1.0", fetch=not_found)
    assert release is None
    assert "404" in error and "priv" in error.lower()


# ---------------------------------------------------------------- staging


def test_stage_update_finds_the_app_folder_inside_the_zip(tmp_path):
    """Le zip du CI contient un dossier BobVr/ à sa racine ; on le retrouve."""
    import zipfile

    exe = "BobVr.exe" if sys.platform == "win32" else "BobVr"
    zip_path = tmp_path / "BobVr-0.2.0.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"BobVr/{exe}", "")
        archive.writestr("BobVr/_internal/data.bin", "x")

    app = updater.stage_update(zip_path, tmp_path / "staging")
    assert (app / exe).is_file()
    assert app.name == "BobVr"


# --------------------------------------------------------- script de bascule


def test_swap_script_waits_copies_and_restarts(tmp_path):
    staging = tmp_path / "bobvr-update" / "BobVr"
    install = tmp_path / "install" / "BobVr"
    script = build_swap_script(
        pid=4242, staging_app_dir=staging, install_dir=install
    )
    # attend le bon processus
    assert "PID eq 4242" in script
    assert "goto waitloop" in script
    # copie la nouvelle version par-dessus l'ancienne
    assert "robocopy" in script
    assert str(staging) in script
    assert str(install) in script
    assert "/MIR" in script
    # relance puis s'efface
    assert f"{install}\\BobVr.exe" in script
    assert 'del "%~f0"' in script


def test_swap_script_uses_crlf_line_endings():
    """Un .bat à fins de ligne Unix se comporte mal sous cmd.exe."""
    script = build_swap_script(
        pid=1, staging_app_dir=__import__("pathlib").Path("s"),
        install_dir=__import__("pathlib").Path("i"),
    )
    assert "\r\n" in script
    assert "\n" not in script.replace("\r\n", "")


# --------------------------------------------------------- garde-fous OS


def test_self_update_is_refused_from_source(monkeypatch, tmp_path):
    """Depuis les sources (non gelé), on ne se réécrit pas : on lève."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    with pytest.raises(RuntimeError):
        updater.apply_update(tmp_path / "BobVr", spawn=False)


def test_can_self_update_needs_frozen_windows(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert updater.can_self_update() is False

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert updater.can_self_update() is False

    monkeypatch.setattr(sys, "platform", "win32")
    assert updater.can_self_update() is True
