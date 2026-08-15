"""The EAC layout constants, and the kernel they specialise."""

from __future__ import annotations

import pytest

import math

from bobvr.render.geometry import (
    MAX_5_6K,
    MAX_FOV,
    MIN_FOV,
    NEUTRAL_FOV,
    MaxLayout,
    Orientation,
    UnsupportedLayoutError,
    build_kernel_source,
    materialise_kernel,
    view_scale,
)


def test_known_layout_matches_measured_constants():
    """These were derived from ffmpeg's own v360 output; pin them down."""
    layout = MAX_5_6K
    assert layout.face == 1344
    assert layout.eac_w == 4032
    assert layout.net_loss == 32
    assert layout.seam_eac_w == 96
    assert layout.seam0_eac == 624
    assert layout.seam1_eac == 3312
    assert layout.seam0_src == 624
    assert layout.seam1_src == 3344


def test_seam_mapping_covers_the_whole_row_exactly():
    """Every source pixel must be accounted for, with no gap or overlap."""
    layout = MAX_5_6K
    consumed = (
        layout.seam0_eac                                                  # face 0 left
        + layout.seam_src_w                                               # seam 0
        + (layout.seam1_eac - (layout.seam0_eac + layout.seam_eac_w))     # middle
        + layout.seam_src_w                                               # seam 1
        + (layout.eac_w - (layout.seam1_eac + layout.seam_eac_w))         # face 2 right
    )
    assert consumed == layout.src_w

    produced = (
        layout.seam0_eac
        + layout.seam_eac_w
        + (layout.seam1_eac - (layout.seam0_eac + layout.seam_eac_w))
        + layout.seam_eac_w
        + (layout.eac_w - (layout.seam1_eac + layout.seam_eac_w))
    )
    assert produced == layout.eac_w


def test_seams_sit_inside_the_outer_faces():
    """The stitch line falls mid-face, not on a face boundary."""
    layout = MAX_5_6K
    assert 0 < layout.seam0_eac < layout.face
    assert layout.seam0_eac + layout.seam_eac_w < layout.face
    assert 2 * layout.face < layout.seam1_eac < 3 * layout.face
    # Centred within its face.
    assert layout.seam0_eac == (layout.face - layout.seam_eac_w) // 2


@pytest.mark.parametrize(
    "width,height",
    [(0, 1344), (4096, 0), (4032, 1344), (3000, 1344), (4097, 1344)],
)
def test_implausible_layouts_are_rejected(width, height):
    with pytest.raises(UnsupportedLayoutError):
        MaxLayout(src_w=width, src_h=height).validate()


def test_kernel_source_carries_its_geometry():
    source = build_kernel_source(MAX_5_6K, Orientation(), cubic=True)
    assert "#define SEAM0_EAC 624" in source
    assert "#define SEAM1_SRC 3344" in source
    assert "#define INTERP_CUBIC 1" in source
    assert "#define APPLY_ROTATION 0" in source
    assert "__kernel void gopromax_equirect" in source


def test_rotation_only_compiled_in_when_asked():
    plain = build_kernel_source(MAX_5_6K, Orientation())
    turned = build_kernel_source(MAX_5_6K, Orientation(yaw=90))
    assert "#define APPLY_ROTATION 0" in plain
    assert "#define APPLY_ROTATION 1" in turned
    assert plain != turned


def test_materialised_kernels_are_cached_per_parameter_set(tmp_path):
    first = materialise_kernel(tmp_path, MAX_5_6K, Orientation(), cubic=True)
    again = materialise_kernel(tmp_path, MAX_5_6K, Orientation(), cubic=True)
    other = materialise_kernel(tmp_path, MAX_5_6K, Orientation(yaw=10), cubic=True)
    wider = materialise_kernel(tmp_path, MAX_5_6K, Orientation(), True, fov=120)
    assert first == again
    assert first != other
    assert first != wider, "le FOV ne fait pas partie de la clé de cache"
    assert first.exists() and other.exists() and wider.exists()
    # No temporary files left behind.
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------- field of view


def test_the_neutral_field_of_view_leaves_the_projection_alone():
    assert view_scale(NEUTRAL_FOV) == 1.0
    source = build_kernel_source(MAX_5_6K, fov=NEUTRAL_FOV)
    assert "#define APPLY_VIEW_SCALE 0" in source


def test_a_wider_field_of_view_is_compiled_in():
    source = build_kernel_source(MAX_5_6K, fov=120.0)
    assert "#define APPLY_VIEW_SCALE 1" in source
    assert f"#define VIEW_SCALE_C {view_scale(120.0)!r}f" in source


@pytest.mark.parametrize("fov", [MIN_FOV, 60.0, NEUTRAL_FOV, 120.0, 130.0, MAX_FOV])
def test_a_player_at_the_neutral_view_ends_up_showing_the_asked_angle(fov):
    """The whole point of the factor, stated as arithmetic.

    A player draws whatever sits at ``tan(angle)`` on its screen. Its edge is
    at ``tan(NEUTRAL_FOV/2)``; multiplying by the factor must land on
    ``tan(fov/2)``, so that its window holds exactly ``fov`` degrees.
    """
    edge = math.tan(math.radians(NEUTRAL_FOV) / 2) * view_scale(fov)
    shown = 2 * math.degrees(math.atan(edge))
    assert math.isclose(shown, fov, rel_tol=1e-9)


def test_out_of_range_requests_are_clamped_not_passed_through():
    assert view_scale(500.0) == view_scale(MAX_FOV)
    assert view_scale(1.0) == view_scale(MIN_FOV)


@pytest.mark.parametrize("fov", [MIN_FOV, 60.0, 120.0, MAX_FOV])
def test_the_sphere_stays_whole_at_every_allowed_field_of_view(fov):
    """Redistributed, never cut: the map must keep its order all the way round.

    ``beta = atan2(c*sin a, cos a)`` holds 0, 90 and 180 degrees in place and
    rises throughout for any positive c, so nothing folds over itself and
    nothing is left with no scene in it.
    """
    c = view_scale(fov)
    steps = 4000
    previous = -1.0
    for i in range(steps + 1):
        a = math.pi * i / steps
        beta = math.atan2(c * math.sin(a), math.cos(a))
        assert beta > previous, f"repli à {math.degrees(a):.1f}° de l'axe"
        previous = beta
    assert math.isclose(previous, math.pi, abs_tol=1e-6)

    for fixed in (0.0, math.pi / 2, math.pi):
        beta = math.atan2(c * math.sin(fixed), math.cos(fixed))
        assert math.isclose(beta, fixed, abs_tol=1e-9)


def test_straight_lines_survive_the_widening():
    """A straight edge in the scene must still be straight on screen.

    Take a line in front of the camera -- points (x, 1, 1) in space, which is
    what a roof beam running away from the bob looks like. Project each point
    through the widening and back out the way a player draws it; the results
    have to stay collinear. An angle-space stretch fails this, which is what
    bent the roof of the tunnel.
    """
    c = view_scale(130.0)
    screen = []
    for x in [i / 10 for i in range(-12, 13)]:
        # Direction of the scene point, as an angle off-axis and a bearing.
        norm = math.hypot(x, 1.0)
        a_scene = math.atan2(norm, 1.0)
        # Invert the widening: which direction in the file carries it?
        a_file = math.atan2(math.tan(a_scene), c) if a_scene < math.pi / 2 else None
        assert a_file is not None
        # The player draws it at tan(a_file), keeping the bearing.
        radius = math.tan(a_file)
        screen.append((radius * x / norm, radius * 1.0 / norm))

    # Collinear: every point sits on the line through the first and last.
    (x0, y0), (x1, y1) = screen[0], screen[-1]
    for px, py in screen:
        area = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)
        assert abs(area) < 1e-9, "la ligne droite s'est courbée"
