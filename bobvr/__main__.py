"""Command line entry point.

The window is the normal way in, but the same pipeline is reachable from a
terminal -- useful for re-rendering an archive, checking what acceleration a
machine offers, or scripting a batch overnight.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .render.caps import FFmpegMissingError, detect

log = logging.getLogger(__name__)


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="fichier .360 à convertir")
    parser.add_argument("-o", "--output", type=Path, help="fichier MP4 de sortie")
    parser.add_argument("--width", type=int, help="largeur de sortie, en pixels")
    parser.add_argument(
        "--height", type=int,
        help="hauteur de sortie ; l'une des deux suffit, le format reste 2:1",
    )
    parser.add_argument("--quality", type=int, help="qualité 1-51, plus bas = mieux")
    parser.add_argument("--yaw", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument(
        "--fov", type=float, default=None,
        help="champ de vision à l'ouverture, en degrés (40-150 ; 80 = neutre)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="forcer le rendu logiciel"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bobvr", description="Traitement des vidéos GoPro MAX pour la piste de bob."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("ui", help="ouvrir l'interface (par défaut)")
    sub.add_parser("info", help="afficher l'accélération matérielle détectée")

    probe_parser = sub.add_parser("probe", help="analyser un fichier .360")
    probe_parser.add_argument("source", type=Path)

    _add_render_arguments(sub.add_parser("render", help="convertir un .360"))

    label_parser = sub.add_parser(
        "label", help="nommer une carte pour que BobVr la reconnaisse"
    )
    label_parser.add_argument(
        "device", nargs="?", help="ex. /dev/sdd1 ; omettre pour lister les cartes"
    )
    label_parser.add_argument(
        "card_id", nargs="?", help="identifiant de l'engin, ex. B1"
    )
    return parser


def command_label(device: str | None, card_id: str | None) -> int:
    from .cards import make_backend
    from .cards.labeling import LabelError, apply_label, plan_label

    settings = Settings.load()
    backend = make_backend()
    volumes = backend.list_volumes()

    if not device or not card_id:
        if not volumes:
            print("Aucun volume amovible détecté.")
            return 1
        print("Volumes amovibles détectés :\n")
        print(f"  {'Périphérique':<16} {'Système':<8} {'Nom actuel':<14} Monté sur")
        for volume in volumes:
            current = volume.label or "(sans nom)"
            print(
                f"  {volume.device or '?':<16} {volume.fstype or '?':<8} "
                f"{current:<14} {volume.mountpoint}"
            )
        print(
            "\nPour nommer une carte :  bobvr label <périphérique> <engin>\n"
            "  par exemple :          bobvr label /dev/sdd1 B1\n"
            f"  engins attendus :      {', '.join(settings.known_card_ids())}"
        )
        return 0

    volume = next((v for v in volumes if v.device == device), None)
    if volume is None:
        print(
            f"erreur : {device} n'est pas un volume amovible monté. "
            "Lancez « bobvr label » sans argument pour voir la liste.",
            file=sys.stderr,
        )
        return 1

    try:
        plan = plan_label(
            device=device,
            card_id=card_id,
            fstype=volume.fstype or "",
            fleet=settings.fleet,
            mounted_at=volume.mountpoint,
        )
    except LabelError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1

    applied, message = apply_label(plan)
    print(message)
    return 0 if applied else 1


def command_info() -> int:
    caps = detect()
    print(caps.summary())
    print()
    print(f"  ffmpeg          : {caps.ffmpeg}")
    print(f"  ffprobe         : {caps.ffprobe}")
    print(f"  décodage NVDEC  : {'oui' if caps.has_nvdec else 'non'}")
    print(f"  kernel OpenCL   : {'oui' if caps.has_opencl_kernel else 'non'}"
          f"{f' ({caps.opencl_name})' if caps.opencl_name else ''}")
    print(f"  encodeur vidéo  : {caps.video_encoder}")
    print(f"  filtre v360     : {'oui' if caps.has_v360 else 'non'}")
    for gap in caps.explain_gaps():
        print(f"  ! {gap}")
    return 0


def command_probe(source: Path) -> int:
    from .media import NotAMaxVideoError, probe

    caps = detect()
    try:
        info = probe(source, caps)
    except NotAMaxVideoError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1
    print(f"{info.path.name}")
    print(f"  durée        : {info.duration:.2f} s à {info.fps:.3f} img/s")
    print(f"  pistes vidéo : {info.video[0].index} et {info.video[1].index} "
          f"({info.track_width}x{info.track_height}, {info.video[0].codec})")
    print(f"  audio        : piste {info.audio_index}"
          if info.audio_index is not None else "  audio        : aucun")
    print(f"  télémétrie   : {'piste ' + str(info.telemetry_index) if info.has_telemetry else 'absente'}")
    print(f"  taille       : {info.size / 1e9:.2f} Go")
    return 0


def command_render(args) -> int:
    from .media import NotAMaxVideoError, probe
    from .render.geometry import Orientation
    from .render.pipeline import (
        RenderError,
        RenderSettings,
        Renderer,
        inject_spherical_metadata,
    )

    settings = Settings.load()
    config = settings.render
    caps = detect(settings.ffmpeg_path, settings.ffprobe_path)

    try:
        info = probe(args.source, caps)
    except NotAMaxVideoError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 1

    # Either dimension on its own is enough; giving both that disagree is
    # refused by RenderSettings rather than quietly resolved.
    if args.width:
        width, height = args.width, args.height or args.width // 2
    elif args.height:
        width, height = args.height * 2, args.height
    else:
        width, height = config.width, config.height
    render_settings = RenderSettings(
        width=width,
        height=height,
        orientation=Orientation(
            args.yaw if args.yaw is not None else config.yaw,
            args.pitch if args.pitch is not None else config.pitch,
            args.roll if args.roll is not None else config.roll,
        ),
        cubic=config.cubic,
        initial_fov=args.fov if args.fov is not None else config.initial_fov,
        quality=args.quality or config.quality,
        max_bitrate_kbps=config.max_bitrate_kbps,
        force_cpu=args.cpu or config.force_cpu,
    )

    output = args.output or args.source.with_suffix(".mp4")

    from platformdirs import user_cache_dir

    from .config import APP_NAME

    cache = Path(user_cache_dir(APP_NAME, appauthor=False)) / "kernels"
    renderer = Renderer(caps, cache)
    print(f"{args.source.name} -> {output}  [{renderer.describe_route(render_settings)}]")

    def show(progress) -> None:
        eta = progress.eta_seconds
        tail = f"  reste ~{eta:.0f}s" if eta else ""
        print(
            f"\r  {progress.fraction * 100:5.1f}%  {progress.speed:.2f}x{tail}   ",
            end="", flush=True,
        )

    try:
        result = renderer.render(info, output, render_settings, on_progress=show)
    except RenderError as exc:
        print(f"\nerreur : {exc}", file=sys.stderr)
        return 1
    print()
    if not inject_spherical_metadata(result):
        print(
            "attention : les métadonnées 360 n'ont pas pu être écrites ; "
            "installez exiftool pour que les lecteurs proposent la navigation.",
            file=sys.stderr,
        )
    print(f"terminé : {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    command = args.command or "ui"
    try:
        if command == "info":
            return command_info()
        if command == "probe":
            return command_probe(args.source)
        if command == "render":
            return command_render(args)
        if command == "label":
            return command_label(args.device, args.card_id)
        from .ui.app import main as ui_main

        return ui_main([sys.argv[0]])
    except FFmpegMissingError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
