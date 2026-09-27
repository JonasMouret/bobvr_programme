"""Vérifier, télécharger et installer une mise à jour depuis GitHub Releases.

Le poste de la piste reçoit BobVr une première fois par copie de dossier ;
ensuite, tout passe par ici. Aucun serveur à maintenir : GitHub publie les
releases (le workflow CI attache le zip win64), et l'app interroge l'API
publique des releases, compare les versions, télécharge le zip et se remplace
elle-même.

Le module est découpé pour être testable hors ligne et hors Windows, comme le
reste du portage : les fonctions pures (comparaison de versions, lecture de la
réponse JSON, génération du script de bascule) ne touchent ni au réseau ni au
disque, et les fonctions qui le font acceptent une injection pour les tests.

Le remplacement de soi-même est le point délicat : une build « un dossier » ne
peut pas réécrire ``BobVr.exe`` et les DLL de Qt pendant qu'elle tourne, car
Windows les verrouille. La solution est classique : télécharger la nouvelle
version à côté, écrire un petit script détaché qui *attend la fermeture de
BobVr*, remplace le dossier, puis relance l'exécutable. L'app lance ce script
et se ferme aussitôt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __version__
from .render.caps import app_dir, subprocess_kwargs

log = logging.getLogger(__name__)

#: Le dépôt qui publie les releases. Le vérificateur ne lit que celui-ci.
GITHUB_REPO = "JonasMouret/bobvr_programme"

#: L'API publique renvoie la dernière release *stable* (les brouillons et les
#: pré-versions en sont déjà exclus par GitHub).
LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

#: La page à ouvrir dans le navigateur si l'installation automatique n'est pas
#: possible (build non gelée, ou release sans zip attaché).
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

#: GitHub exige un en-tête User-Agent, sinon l'API répond 403.
_USER_AGENT = f"BobVr/{__version__}"

ProgressCallback = Callable[[int, int], None]


# --------------------------------------------------------------- versions


def parse_version(text: str) -> tuple[int, ...]:
    """Réduit ``v1.2.3`` ou ``1.2`` à un tuple d'entiers comparable.

    Volontairement simple : on ne gère que des versions numériques pointées,
    ce qui couvre nos tags. Un éventuel suffixe de pré-version (``-rc1``) est
    ignoré, donc ``1.2.0-rc1`` et ``1.2.0`` se comparent égaux — acceptable
    puisque l'API « latest » ne renvoie jamais de pré-version.
    """
    cleaned = text.strip().lstrip("vV")
    parts: list[int] = []
    for chunk in cleaned.split("."):
        match = re.match(r"\d+", chunk)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts) or (0,)


def is_newer(candidate: str, current: str) -> bool:
    """Vrai si ``candidate`` est une version strictement plus récente."""
    return parse_version(candidate) > parse_version(current)


# ----------------------------------------------------------- release JSON


@dataclass(frozen=True)
class Release:
    """Une release publiée, telle qu'on en a besoin pour décider et installer."""

    version: str
    tag: str
    name: str
    notes: str
    asset_name: str | None
    asset_url: str | None
    asset_size: int

    @property
    def can_install(self) -> bool:
        """Un zip est attaché : l'installation automatique est possible."""
        return bool(self.asset_url)


def _pick_asset(assets: list[dict]) -> dict | None:
    """Choisit le zip Windows parmi les fichiers attachés à la release.

    On préfère un nom qui mentionne Windows (``BobVr-0.2.0-win64.zip``) ; à
    défaut, le premier ``.zip`` fait l'affaire. Les autres fichiers (notes,
    sommes de contrôle) sont ignorés.
    """
    zips = [a for a in assets if str(a.get("name", "")).lower().endswith(".zip")]
    if not zips:
        return None
    for asset in zips:
        if re.search(r"win", str(asset.get("name", "")), re.IGNORECASE):
            return asset
    return zips[0]


def parse_release(payload: dict) -> Release | None:
    """Transforme la réponse JSON de GitHub en :class:`Release`.

    Renvoie ``None`` si l'objet est un brouillon ou une pré-version — l'API
    « latest » n'en renvoie pas, mais on est défensif au cas où on lui donne
    une autre réponse.
    """
    if payload.get("draft") or payload.get("prerelease"):
        return None
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return None

    asset = _pick_asset(payload.get("assets") or [])
    return Release(
        version=parse_version_string(tag),
        tag=tag,
        name=str(payload.get("name") or tag),
        notes=str(payload.get("body") or "").strip(),
        asset_name=str(asset["name"]) if asset else None,
        asset_url=str(asset["browser_download_url"]) if asset else None,
        asset_size=int(asset.get("size", 0)) if asset else 0,
    )


def parse_version_string(tag: str) -> str:
    """Le numéro de version affichable, sans le ``v`` du tag."""
    return tag.strip().lstrip("vV")


# ---------------------------------------------------------------- réseau


def _fetch_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def check_for_update(
    current_version: str | None = None,
    *,
    fetch: Callable[[], dict] | None = None,
    timeout: float = 8.0,
) -> Release | None:
    """La dernière release, uniquement si elle est plus récente qu'ici.

    Ne lève jamais : une vérification qui échoue (pas de réseau, API muette,
    réponse illisible) renvoie ``None`` et se contente d'un message de log. Le
    démarrage de l'app ne doit jamais dépendre de GitHub.

    ``fetch`` est injectable pour les tests ; par défaut, on interroge l'API.
    """
    current = current_version or __version__
    fetch = fetch or (lambda: _fetch_json(LATEST_RELEASE_URL, timeout))
    try:
        payload = fetch()
    except Exception as exc:  # réseau, DNS, timeout, JSON… tout est non fatal
        log.info("vérification des mises à jour impossible : %s", exc)
        return None

    release = parse_release(payload)
    if release is None:
        return None
    if not is_newer(release.version, current):
        log.debug("à jour : %s (dernière publiée : %s)", current, release.version)
        return None
    return release


def download_asset(
    release: Release,
    dest_dir: Path,
    *,
    on_progress: ProgressCallback | None = None,
    timeout: float = 30.0,
) -> Path:
    """Télécharge le zip de la release dans ``dest_dir`` et renvoie son chemin.

    ``on_progress(reçu, total)`` est appelé au fil de l'eau pour alimenter une
    barre de progression ; ``total`` peut être 0 si le serveur ne l'annonce
    pas.
    """
    if not release.asset_url or not release.asset_name:
        raise ValueError("cette release n'a pas de zip à télécharger")

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / release.asset_name
    request = urllib.request.Request(
        release.asset_url, headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or release.asset_size or 0)
        received = 0
        with target.open("wb") as out:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                received += len(chunk)
                if on_progress:
                    on_progress(received, total)
    return target


def stage_update(zip_path: Path, staging_dir: Path) -> Path:
    """Décompresse le zip et renvoie le dossier applicatif prêt à installer.

    Le zip produit par le CI contient le dossier ``BobVr/`` à sa racine (c'est
    ce que produit PyInstaller). On renvoie ce dossier-là — celui qui contient
    ``BobVr.exe`` — et non la racine du zip, pour que la bascule copie le bon
    niveau.
    """
    if staging_dir.exists():
        _remove_tree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging_dir)

    exe = "BobVr.exe" if sys.platform == "win32" else "BobVr"
    for candidate in (staging_dir, *sorted(p for p in staging_dir.iterdir() if p.is_dir())):
        if (candidate / exe).is_file():
            return candidate
    # Rien qui ressemble à l'app : on renvoie ce qu'on a extrait, l'appelant
    # tranchera. Le cas ne devrait pas arriver avec nos propres zips.
    log.warning("aucun %s trouvé dans %s après extraction", exe, staging_dir)
    return staging_dir


# ---------------------------------------------------- bascule (self-update)


def can_self_update() -> bool:
    """L'auto-remplacement n'a de sens que sur une build gelée sous Windows.

    Depuis les sources (développement), on ne se réécrit pas : il n'y a pas de
    dossier autonome à remplacer, et la mise à jour se fait par ``git``.
    """
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def build_swap_script(
    *,
    pid: int,
    staging_app_dir: Path,
    install_dir: Path,
    exe_name: str = "BobVr.exe",
    log_path: Path | None = None,
) -> str:
    """Le contenu du .bat qui remplace le dossier une fois BobVr fermé.

    Le script attend que le processus ``pid`` disparaisse (les DLL de Qt sont
    verrouillées tant qu'il tourne), miroite le nouveau dossier par-dessus
    l'ancien avec ``robocopy``, relance l'exécutable, puis s'efface lui-même.

    ``robocopy /MIR`` supprime dans la cible ce qui n'est plus dans la source,
    ce qui est voulu (une DLL retirée d'une version à l'autre doit partir) et
    sans danger : les réglages et la base vivent dans le profil utilisateur,
    jamais dans le dossier de l'app.
    """
    redirect = f'>> "{log_path}" 2>&1' if log_path else ""
    return "\r\n".join(
        [
            "@echo off",
            "setlocal",
            f"echo Mise a jour de BobVr... {redirect}".rstrip(),
            ":waitloop",
            f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul',
            "if not errorlevel 1 (",
            "  timeout /t 1 /nobreak >nul",
            "  goto waitloop",
            ")",
            f'robocopy "{staging_app_dir}" "{install_dir}" /MIR /NFL /NDL '
            f"/NJH /NJS /R:3 /W:2 {redirect}".rstrip(),
            f'start "" "{install_dir}\\{exe_name}"',
            # On efface la copie source, pas le dossier du script : le .bat vit
            # dans le dossier parent et ne peut pas se supprimer un dossier
            # qu'il occupe. Il s'efface lui-même en dernier.
            f'rmdir /s /q "{staging_app_dir}" {redirect}'.rstrip(),
            'del "%~f0"',
        ]
    ) + "\r\n"


def apply_update(
    staging_app_dir: Path,
    *,
    install_dir: Path | None = None,
    pid: int | None = None,
    spawn: bool = True,
) -> Path:
    """Écrit le script de bascule et le lance en arrière-plan.

    Renvoie le chemin du script écrit. L'appelant doit ensuite quitter l'app
    (fermer la fenêtre) : le script attend précisément cette fermeture avant
    de remplacer les fichiers. ``spawn=False`` écrit le script sans le lancer,
    pour les tests.
    """
    if not can_self_update():
        raise RuntimeError(
            "l'installation automatique n'est disponible que dans la build "
            "Windows. En développement, mettez à jour avec git."
        )

    install = (install_dir or app_dir()).resolve()
    log_path = staging_app_dir.parent / "update.log"
    script_path = staging_app_dir.parent / "apply_update.bat"
    script_path.write_text(
        build_swap_script(
            pid=pid if pid is not None else os.getpid(),
            staging_app_dir=staging_app_dir.resolve(),
            install_dir=install,
            log_path=log_path,
        ),
        encoding="utf-8",
    )

    if spawn:
        import subprocess

        # Détaché : le .bat doit survivre à la fermeture de BobVr, sinon il ne
        # pourra jamais copier par-dessus lui.
        flags = subprocess_kwargs()
        creationflags = flags.get("creationflags", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(  # noqa: S603
            ["cmd", "/c", str(script_path)],
            creationflags=creationflags,
            close_fds=True,
            cwd=str(script_path.parent),
        )
    return script_path


def staging_root() -> Path:
    """Dossier de travail des mises à jour, hors du dossier de l'app.

    Il doit être *à l'extérieur* de l'install : ``robocopy /MIR`` y effacerait
    sinon le zip et le script en pleine bascule.
    """
    return Path(tempfile.gettempdir()) / "bobvr-update"


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
