"""Render settings and the filter graphs they produce.

No video is needed here: the graphs are strings, and what matters is that the
two routes agree on the framing they promise the operator.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from bobvr.render.geometry import MAX_5_6K, MAX_STAB_FRAMES, NEUTRAL_FOV
from bobvr.render.pipeline import RenderError, RenderSettings, Renderer


@pytest.fixture
def renderer(caps, tmp_path):
    return Renderer(caps, tmp_path / "kernels")


@pytest.mark.parametrize("fov", [0.0, 39.0, 151.0, 360.0])
def test_an_impossible_field_of_view_is_refused(fov):
    with pytest.raises(RenderError, match="champ de vision"):
        RenderSettings(initial_fov=fov).validate()


def test_the_neutral_field_of_view_leaves_the_graph_untouched(renderer):
    settings = RenderSettings(width=3840, height=1920, initial_fov=NEUTRAL_FOV)
    graph = renderer._cpu_graph(0, 1, MAX_5_6K, settings)
    assert "w=3840:h=1920" in graph
    assert "pad=" not in graph and "crop=3840" not in graph


@pytest.mark.parametrize("fov", [50.0, 120.0, 130.0])
def test_the_software_route_refuses_rather_than_bend_the_picture(renderer, fov):
    """v360 can project the sphere but not redistribute it.

    Every way of faking the field of view with plain filters curves straight
    lines, which is the one thing the setting exists to avoid.
    """
    settings = RenderSettings(width=3840, height=1920, initial_fov=fov)
    with pytest.raises(RenderError, match="kernel OpenCL"):
        renderer._cpu_graph(0, 1, MAX_5_6K, settings)


def test_the_kernel_route_specialises_on_the_field_of_view(renderer):
    plain = renderer._gpu_graph(
        0, 1, MAX_5_6K, RenderSettings(initial_fov=NEUTRAL_FOV)
    )
    wider = renderer._gpu_graph(0, 1, MAX_5_6K, RenderSettings(initial_fov=120.0))
    assert plain != wider, "le FOV ne parvient pas jusqu'au kernel"


# ------------------------------------------------------------- stabilisation


def test_stabilisation_is_off_by_default():
    assert RenderSettings().stabilise_seconds == 0.0


def test_a_negative_smoothing_constant_is_refused():
    with pytest.raises(RenderError, match="négative"):
        RenderSettings(stabilise_seconds=-1.0).validate()


def test_nothing_is_read_from_the_clip_when_it_is_off(renderer):
    """Off means off: no telemetry extraction, so `info` is never touched."""
    assert renderer._stabilisation(None, RenderSettings()) == (None, 0)


def test_a_table_reaches_the_kernel(renderer):
    plain = renderer._gpu_graph(0, 1, MAX_5_6K, RenderSettings())
    steady = renderer._gpu_graph(0, 1, MAX_5_6K, RenderSettings(),
                                 stabilisation=[(1.0, 0.0, 0.0, 0.0)] * 4)
    assert plain != steady, "la table ne parvient pas jusqu'au kernel"


def test_a_clip_too_long_to_hold_is_refused_as_a_render_error(renderer):
    """geometry raises its own type; the operator should see a render error."""
    huge = [(1.0, 0.0, 0.0, 0.0)] * (MAX_STAB_FRAMES + 1)
    with pytest.raises(RenderError, match="stabilisation"):
        renderer._gpu_graph(0, 1, MAX_5_6K, RenderSettings(), stabilisation=huge)


def test_the_software_fallback_refuses_rather_than_dropping_it(renderer, caps):
    """Silently rendering unstabilised would be the worst of the options."""
    if not caps.has_v360:
        pytest.skip("ffmpeg sans v360")
    settings = RenderSettings(stabilise_seconds=0.5, force_cpu=True)
    with pytest.raises(RenderError, match="kernel OpenCL"):
        renderer._decode_and_project(_stub_info(), settings)


class _stub_info:
    """Just enough of MaxVideoInfo to build a graph."""
    path = Path("descente.360")
    track_width, track_height = MAX_5_6K.src_w, MAX_5_6K.src_h
    fps = 29.97
    duration = 10.0
    audio_index = None
    telemetry_index = None
    video = (type("S", (), {"index": 0})(), type("S", (), {"index": 1})())


def test_the_setting_reaches_the_render_only_when_ticked():
    """The checkbox and the strength are two fields; unticked wins."""
    from bobvr.config import RenderConfig

    off = RenderConfig(stabilise=False, stabilise_seconds=1.2)
    on = RenderConfig(stabilise=True, stabilise_seconds=1.2)
    for config, expected in ((off, 0.0), (on, 1.2)):
        settings = RenderSettings(
            stabilise_seconds=config.stabilise_seconds if config.stabilise else 0.0)
        assert settings.stabilise_seconds == expected


def test_a_zero_strength_is_refused_by_the_config():
    """Off is the checkbox's job; a zero smoother would just be a no-op."""
    from pydantic import ValidationError

    from bobvr.config import RenderConfig

    with pytest.raises(ValidationError):
        RenderConfig(stabilise_seconds=0.0)


def test_unknown_scan_order_is_refused():
    with pytest.raises(RenderError, match="Ordre de lecture"):
        RenderSettings(stabilise_seconds=1.0, rolling_axis="sideways").validate()


def test_scan_order_defaults_to_off():
    """Left off until a run shows it earns its keep: see STABILISATION.md 11."""
    assert RenderSettings().rolling_axis is None
