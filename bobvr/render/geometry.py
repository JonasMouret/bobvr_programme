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


def build_kernel_source(
    layout: MaxLayout,
    orientation: Orientation = Orientation(),
    cubic: bool = True,
    fov: float = NEUTRAL_FOV,
) -> str:
    """Return the kernel source with its geometry prologue prepended."""
    layout.validate()
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
    }
    prologue = "\n".join(f"#define {k} {v}" for k, v in defines.items())
    body = KERNEL_SOURCE.read_text(encoding="utf-8")
    return f"/* generated -- do not edit */\n{prologue}\n\n{body}"


def _f(value: float) -> str:
    return f"{value!r}f"


def materialise_kernel(
    cache_dir: Path,
    layout: MaxLayout,
    orientation: Orientation = Orientation(),
    cubic: bool = True,
    fov: float = NEUTRAL_FOV,
) -> Path:
    """Write the specialised kernel to ``cache_dir`` and return its path.

    ffmpeg's program_opencl reads its source from disk, and the content is
    fully determined by the parameters, so the filename doubles as the cache
    key.
    """
    source = build_kernel_source(layout, orientation, cubic, fov)
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
