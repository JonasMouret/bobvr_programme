"""BobVr — import et conversion des vidéos GoPro MAX d'une piste de bobsleigh.

La version vit ici, et nulle part ailleurs. ``pyproject.toml`` la lit depuis
cet attribut (version dynamique), et le vérificateur de mise à jour la compare
à la dernière publiée sur GitHub. On la garde en dur dans le code plutôt que
lue via ``importlib.metadata`` parce que les métadonnées du paquet ne voyagent
pas de façon fiable dans une build PyInstaller gelée : l'attribut, lui, est
compilé avec le reste.
"""

from __future__ import annotations

__version__ = "0.1.2"
