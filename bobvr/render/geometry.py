"""Geometry of the GoPro MAX .360 layout, and OpenCL kernel specialisation.

The constants below were not taken on faith: they were recovered by pushing a
coordinate-encoding image through ffmpeg's own ``v360=eac:e`` filter and
fitting the result, so the kernel's projection matches the reference
implementation face for face. See docs/geometry.md for the derivation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

KERNEL_SOURCE = Path(__file__).parent / "kernels" / "gopromax_equirect.cl"
KERNEL_NAME = "gopromax_equirect"

#: Width of one lens' contribution to a stitch seam, in source pixels.
#: GoPro stores both lenses across a 2*SEAM_HALF strip and expects them to be
#: cross-faded; this is the one value the frame dimensions cannot reveal.
SEAM_HALF = 64

#: The field of view a 360 player opens at, in degrees. Players do not read one
#: from the file -- there is no such field in any spherical-video spec -- they
#: each pick their own, and 80 degrees is what VLC and the common phone viewers
#: use. It is therefore the value at which BobVr's output is a true, untouched
#: 360: ask for more and the sphere is stretched to suit (see the kernel).
NEUTRAL_FOV = 80.0

#: Bounds on the requested field of view. Both are comfort limits rather than
#: arithmetic ones: the mapping below stays sound to either side, but a very
#: wide view spends the sphere's detail on the front and leaves the far side
#: soft, and a very narrow one throws away most of the frame.
MIN_FOV = 40.0
MAX_FOV = 150.0


def view_scale(fov: float) -> float:
    """Tangent-space factor that turns a player's own view into ``fov``.

    A player draws a rectilinear view: a point ``b`` degrees off centre lands
    at ``tan(b)`` on its screen. Stretching the sphere by this factor *in
    tangent space* therefore cancels against that projection exactly, and what
    the player draws is a true rectilinear view of the wider angle -- straight
    lines still straight. Stretching the angles themselves would leave a
    tangent in the composition, which is what curves them.

    1.0 leaves the projection alone.
    """
    wanted = max(MIN_FOV, min(MAX_FOV, float(fov)))
    return math.tan(math.radians(wanted) / 2) / math.tan(math.radians(NEUTRAL_FOV) / 2)


class UnsupportedLayoutError(ValueError):
    """The track dimensions do not match a layout this kernel can unpack."""


@dataclass(frozen=True)
class MaxLayout:
    """Packing of a .360 file's two video tracks.

    ``src_w``/``src_h`` are the dimensions of a single track. Each track holds
    three EAC faces side by side, with a stitch seam buried in the middle of
    the first and third.
    """

    src_w: int
    src_h: int

    @property
    def face(self) -> int:
        """Edge length of one (square) cube face."""
        return self.src_h

    @property
    def eac_w(self) -> int:
        """Width of one track's row once the seams are collapsed."""
        return 3 * self.face

    @property
    def seam_src_w(self) -> int:
        """Width of a seam strip as stored: both lenses, side by side."""
        return 2 * SEAM_HALF

    @property
    def seam_eac_w(self) -> int:
        """Width the seam occupies after collapsing."""
        return self.seam_src_w - self.net_loss

    @property
    def net_loss(self) -> int:
        """Pixels each seam sheds when collapsed."""
        loss = self.src_w - self.eac_w
        return loss // 2

    @property
    def seam0_eac(self) -> int:
        """Start of the first seam, centred in face 0."""
        return (self.face - self.seam_eac_w) // 2

    @property
    def seam1_eac(self) -> int:
        """Start of the second seam, centred in face 2."""
        return 2 * self.face + self.seam0_eac

    @property
    def seam0_src(self) -> int:
        return self.seam0_eac

    @property
    def seam1_src(self) -> int:
        return self.seam1_eac + self.net_loss

    def validate(self) -> None:
        if self.src_w <= 0 or self.src_h <= 0:
            raise UnsupportedLayoutError(f"invalid track size {self.src_w}x{self.src_h}")
        loss = self.src_w - self.eac_w
        if loss <= 0 or loss % 2:
            raise UnsupportedLayoutError(
                f"track {self.src_w}x{self.src_h} does not look like a GoPro MAX "
                f"layout: expected width slightly above 3x height, got a "
                f"{loss}px difference"
            )
        if self.seam_eac_w <= 0 or self.seam_eac_w >= self.face:
            raise UnsupportedLayoutError(
                f"implausible seam width {self.seam_eac_w}px for a {self.face}px face"
            )


#: The layout every GoPro MAX .360 file observed so far uses (5.6K).
MAX_5_6K = MaxLayout(src_w=4096, src_h=1344)


@dataclass(frozen=True)
class Orientation:
    """Camera re-orientation applied before projecting, in degrees."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.yaw == 0.0 and self.pitch == 0.0 and self.roll == 0.0


#: Fixed-point scale for a packed rotation component. The stored quaternion is
#: the ``w >= 0`` representative, so each part is in [-1, 1] and a signed
#: 16-bit value resolves it to 0.0017 degrees -- fifty times finer than the
#: 0.09 degrees one pixel covers at 3840 wide.
STAB_SCALE = 32767

#: Bytes each frame costs in the table: three int16, with w recovered in the
#: kernel from the other three.
STAB_BYTES_PER_FRAME = 6

#: What the table may occupy. An OpenCL device is only obliged to offer 64 KiB
#: of ``__constant`` memory and NVIDIA offers exactly that, so this is the real
#: ceiling rather than a chosen one. It works out at about six minutes of
#: footage at 30 fps -- comfortable for a run, and refused rather than
#: truncated beyond that.
STAB_BUDGET_BYTES = 64 * 1024

MAX_STAB_FRAMES = STAB_BUDGET_BYTES // STAB_BYTES_PER_FRAME


#: How the readout instant varies across a frame, as an expression in ``vr``,
#: the direction in the camera's own frame. ``0`` is the first thing read out
#: and ``1`` the last; the table's entry sits at ``0.5``.
#:
#: This is measured, not assumed. The gradient was fitted face by face on
#: B21.360 (``docs/stabilisation/track_rotation.py``), by splitting tracked
#: points into halves of each face and regressing the difference in fitted
#: rotation against angular velocity -- which separates it from translation
#: parallax, since that follows linear speed instead. Three faces carry enough
#: points and all three agree: FRONT -0.565 along its ``fb``, RIGHT -0.397
#: along ``fb``, BACK +0.398 along ``fa``, and BACK's ``fa`` is ``1 - b``. So
#: every gradient runs along a face's own vertical axis, in both tracks.
#:
#: What that means is that the instant does not depend on the EAC packing at
#: all: the sensors scan along the camera's vertical, so elevation alone fixes
#: it. That is why this is an expression in the direction rather than in the
#: source coordinates the packing produces.
ROLLING_AXES = {
    "up-last":  "(0.5f - asin(clamp(vr.y, -1.0f, 1.0f)) * (float)M_1_PI)",
    "up-first": "(0.5f + asin(clamp(vr.y, -1.0f, 1.0f)) * (float)M_1_PI)",
}

#: How long the scan takes to cross the whole sphere, in frame periods. The
#: half-face gradients above average 0.45 frames across roughly 52 degrees of
#: elevation, which extrapolates to about 1.5 over the full 180 -- larger than
#: the 1.0 that ``bobvr.orientation`` fits for the *average* over a frame,
#: because most of a frame's content sits near the equator and never sees the
#: extremes.
ROLLING_SPAN = 1.5


class TooManyFramesError(ValueError):
    """More frames to steady than the kernel's constant memory can hold."""


def build_kernel_source(
    layout: MaxLayout,
    orientation: Orientation = Orientation(),
    cubic: bool = True,
    fov: float = NEUTRAL_FOV,
    stabilisation: list[tuple[float, float, float, float]] | None = None,
    frame_offset: int = 0,
    rolling: str | None = None,
    rolling_span: float = ROLLING_SPAN,
) -> str:
    """Return the kernel source with its geometry prologue prepended.

    ``stabilisation`` is one rotation per output frame, as ``(w, x, y, z)``.
    It is baked into the source rather than passed as an argument because
    ffmpeg's ``program_opencl`` has no way to hand a filter a value that
    changes with time -- the one thing it does provide is the frame counter,
    which the kernel already receives and can use to index a table.

    ``frame_offset`` is added to that counter. The counter starts at zero for
    whatever reaches the filter, so a render that seeks first -- a preview --
    would otherwise read the table from the beginning while showing the middle
    of the clip.

    ``rolling`` names the sensor's scan order, one of ``ROLLING_AXES``. With it
    the rotation applied to a sample depends on where in the *source* that
    sample lands, because that is what fixes when it was captured -- a frame
    is read out over ``rolling_span`` frame periods rather than in an instant.
    ``None`` treats each frame as a single moment, which leaves the within-frame
    skew in the picture. It needs ``stabilisation``: there is no per-frame
    rotation to vary without it.

    Raises:
        TooManyFramesError: the table would not fit in constant memory.
        ValueError: ``rolling`` is not a known scan order.
    """
    layout.validate()
    if rolling is not None and rolling not in ROLLING_AXES:
        raise ValueError(
            f"Ordre de lecture inconnu : {rolling!r}. "
            f"Attendu l'un de {', '.join(sorted(ROLLING_AXES))}."
        )
    scale = view_scale(fov)
    defines = {
        "SRC_W": layout.src_w,
        "SRC_H": layout.src_h,
        "EAC_FACE": layout.face,
        "SEAM_HALF": SEAM_HALF,
        "SEAM_EAC_W": layout.seam_eac_w,
        "SEAM0_EAC": layout.seam0_eac,
        "SEAM1_EAC": layout.seam1_eac,
        "SEAM0_SRC": layout.seam0_src,
        "SEAM1_SRC": layout.seam1_src,
        "INTERP_CUBIC": 1 if cubic else 0,
        "APPLY_ROTATION": 0 if orientation.is_identity else 1,
        "YAW_RAD": _f(math.radians(orientation.yaw)),
        "PITCH_RAD": _f(math.radians(orientation.pitch)),
        "ROLL_RAD": _f(math.radians(orientation.roll)),
        "APPLY_VIEW_SCALE": 0 if scale == 1.0 else 1,
        "VIEW_SCALE_C": _f(scale),
        "APPLY_STABILISATION": 1 if stabilisation else 0,
        "APPLY_ROLLING": 1 if (stabilisation and rolling) else 0,
        "ROLLING_TAU": ROLLING_AXES.get(rolling or "", "0.5f"),
        "ROLLING_SPAN": _f(rolling_span),
        "STAB_FRAMES": len(stabilisation) if stabilisation else 0,
        "STAB_OFFSET": int(frame_offset),
        "STAB_SCALE_INV": _f(1.0 / STAB_SCALE),
    }
    prologue = "\n".join(f"#define {k} {v}" for k, v in defines.items())
    table = _stabilisation_table(stabilisation) if stabilisation else ""
    body = KERNEL_SOURCE.read_text(encoding="utf-8")
    return f"/* generated -- do not edit */\n{prologue}\n\n{table}{body}"


def _stabilisation_table(quats: list[tuple[float, float, float, float]]) -> str:
    """The per-frame rotations as an OpenCL ``__constant`` array.

    Only the vector part is stored: the caller hands over the ``w >= 0``
    representative of each rotation, so the kernel recovers ``w`` as
    ``sqrt(1 - x^2 - y^2 - z^2)`` and the table costs three shorts a frame
    instead of four floats.
    """
    if len(quats) > MAX_STAB_FRAMES:
        raise TooManyFramesError(
            f"La stabilisation ne peut porter que {MAX_STAB_FRAMES} images "
            f"({MAX_STAB_FRAMES / 30:.0f} s à 30 i/s) : la table tient dans les "
            f"{STAB_BUDGET_BYTES // 1024} Kio de mémoire constante d'un GPU, et "
            f"cette séquence en demande {len(quats)}."
        )
    values = []
    for w, x, y, z in quats:
        if w < 0.0:
            w, x, y, z = -w, -x, -y, -z
        values += [max(-STAB_SCALE, min(STAB_SCALE, int(round(c * STAB_SCALE))))
                   for c in (x, y, z)]
    rows = [", ".join(str(v) for v in values[i:i + 12])
            for i in range(0, len(values), 12)]
    body = ",\n    ".join(rows)
    return (
        "/* Per-frame rotation, three int16 each: x, y, z of a unit\n"
        " * quaternion whose w is non-negative and therefore implied. */\n"
        f"__constant short STAB_TABLE[{len(values)}] = {{\n    {body}\n}};\n\n"
    )


def _f(value: float) -> str:
    return f"{value!r}f"


def materialise_kernel(
    cache_dir: Path,
    layout: MaxLayout,
    orientation: Orientation = Orientation(),
    cubic: bool = True,
    fov: float = NEUTRAL_FOV,
    stabilisation: list[tuple[float, float, float, float]] | None = None,
    frame_offset: int = 0,
    rolling: str | None = None,
    rolling_span: float = ROLLING_SPAN,
) -> Path:
    """Write the specialised kernel to ``cache_dir`` and return its path.

    ffmpeg's program_opencl reads its source from disk, and the content is
    fully determined by the parameters, so the filename doubles as the cache
    key. Note what that means once a stabilisation table is in there: the
    source is specific to one clip, so it is written once and never hit again.
    """
    source = build_kernel_source(layout, orientation, cubic, fov,
                                 stabilisation, frame_offset,
                                 rolling, rolling_span)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{KERNEL_NAME}_{digest}.cl"
    if not path.exists():
        # Write via a temporary file so a concurrent render never observes a
        # half-written kernel under the final name.
        tmp = path.with_suffix(".cl.tmp")
        tmp.write_text(source, encoding="utf-8")
        tmp.replace(path)
    return path
