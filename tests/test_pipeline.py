"""Render settings and the filter graphs they produce.

No video is needed here: the graphs are strings, and what matters is that the
two routes agree on the framing they promise the operator.
"""

from __future__ import annotations

import pytest

from bobvr.render.geometry import MAX_5_6K, NEUTRAL_FOV
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
