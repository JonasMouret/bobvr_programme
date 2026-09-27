"""Orientation from the gyroscope.

Built on synthetic telemetry with a known answer, so the conventions the
module commits to -- extrinsic composition, world-to-camera quaternions, an
axis frame solved rather than declared -- are pinned down without a camera.
"""

from __future__ import annotations

import math
import struct

import pytest

from bobvr import orientation as ori
from bobvr.telemetry import collect, parse

from test_telemetry import klv, nested


def about(q, expected, tol=1e-6):
    """Compare rotations, not tuples: q and -q are the same rotation."""
    delta = ori.qangle(ori.qmul(q, ori.qconj(expected)))
    assert delta < tol, f"{q} is {math.degrees(delta):.4f} deg from {expected}"


# ---------------------------------------------------------- the algebra


def test_multiplication_composes_rotations():
    quarter_z = ori.from_rotvec((0.0, 0.0, math.pi / 2))
    about(ori.qmul(quarter_z, quarter_z), ori.from_rotvec((0.0, 0.0, math.pi)))


def test_rotvec_round_trips():
    for v in [(0.3, -0.7, 0.2), (0.0, 0.0, 0.0), (math.pi * 0.9, 0.0, 0.0)]:
        got = ori.to_rotvec(ori.from_rotvec(v))
        assert got == pytest.approx(v, abs=1e-9)


def test_angle_is_never_more_than_half_a_turn():
    """A 350 degree turn is a 10 degree turn the other way."""
    q = ori.from_rotvec((0.0, 0.0, math.radians(350)))
    assert math.degrees(ori.qangle(q)) == pytest.approx(10.0, abs=1e-6)


def test_positive_picks_one_of_the_two_representatives():
    q = ori.from_rotvec((0.0, 0.0, math.radians(200)))
    assert ori.positive(q)[0] >= 0
    about(ori.positive(q), q)


def test_slerp_takes_the_short_way_round():
    a = ori.from_rotvec((0.0, 0.0, math.radians(-10)))
    b = ori.from_rotvec((0.0, 0.0, math.radians(10)))
    about(ori.slerp(a, b, 0.5), ori.IDENTITY, tol=1e-9)
    # Negating one input names the same rotation and must not change the path.
    about(ori.slerp(a, tuple(-c for c in b), 0.5), ori.IDENTITY, tol=1e-9)


# ------------------------------------------------------- synthetic clips


GYRO_HZ = 400
CORI_HZ = 30


def build(rates, quats, payloads):
    """A telemetry blob carrying GYRO and CORI, stamped consistently."""
    per_gyro = len(rates) // payloads
    per_cori = len(quats) // payloads
    blocks = []
    for p in range(payloads):
        g = rates[p * per_gyro:(p + 1) * per_gyro]
        c = quats[p * per_cori:(p + 1) * per_cori]
        stamp = klv(b"STMP", b"J", 8, 1, struct.pack(">Q", p * 1_000_000))
        gyro = nested(b"STRM", stamp
                      + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1000))
                      + klv(b"GYRO", b"s", 6, len(g),
                            b"".join(struct.pack(">3h", *(int(round(v * 1000)) for v in s))
                                     for s in g)))
        cori = nested(b"STRM", stamp
                      + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 32767))
                      + klv(b"CORI", b"s", 8, len(c),
                            b"".join(struct.pack(">4h", *(int(round(v * 32767)) for v in s))
                                     for s in c)))
        blocks.append(nested(b"DEVC", gyro + cori))
    return collect(parse(b"".join(blocks)))


#: How fast the fixture's rotation axis swings, per component. This is what
#: sets the residual the solver can reach: it matches rotations *integrated*
#: over half a second, and rotation vectors only add linearly to first order,
#: so an axis that tumbles inside the window costs accuracy. The default is
#: tuned to score like real footage (~4 %); see the fast-tumble test.
WANDER_FREQS = (0.25, 0.4, 0.55)


def wander(seconds, mapping=lambda v: v, scale=0.4, freqs=WANDER_FREQS):
    """A clip that turns about a genuinely changing axis.

    One axis is not enough to pin a frame down: every rotation vector then
    points the same way, the fit is rank-deficient, and there is nothing to
    tell a swap of the other two axes from leaving them alone. A real run
    turns every way, so the fixture does too.

    Returns gyro samples and the matching orientations, the latter being the
    exact integral of the former sampled at CORI's rate.
    """
    rates, quats, q = [], [], ori.IDENTITY
    dt = 1.0 / GYRO_HZ
    for i in range(int(GYRO_HZ * seconds)):
        t = i * dt
        w = (scale * 1.7 * math.sin(freqs[0] * t),
             scale * 2.1 * math.cos(freqs[1] * t + 0.7),
             scale * 1.3 * math.sin(freqs[2] * t + 1.9))
        if i % (GYRO_HZ // CORI_HZ) == 0:
            quats.append(q)
        rates.append(mapping(w))
        q = ori.qnorm(ori.qmul(ori.from_rotvec((w[0] * dt, w[1] * dt, w[2] * dt)), q))
    return rates, quats


def test_integration_reproduces_a_steady_spin():
    rate = math.radians(90)
    rates = [(0.0, 0.0, rate)] * (GYRO_HZ * 2)
    quats = [ori.from_rotvec((0.0, 0.0, rate * i / CORI_HZ))
             for i in range(CORI_HZ * 2)]
    track = ori.integrate(build(rates, quats, 2), ori.FrameSolution(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 0.0, 99))
    # One second in, the camera should have turned 90 degrees.
    about(track.resample([1.0])[0],
          ori.from_rotvec((0.0, 0.0, math.radians(90))), tol=2e-3)


def test_frame_is_solved_from_the_clip():
    """The gyro is stored with y negated; nothing in the file declares it."""
    rates, quats = wander(8.0, mapping=lambda v: (v[0], -v[1], v[2]))
    solution = ori.solve_gyro_frame(build(rates, quats, 8))
    assert solution.matrix == ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    assert solution.residual < 0.06
    assert solution.trustworthy


def test_frame_survives_an_axis_that_tumbles_inside_the_window():
    """Still the right answer when the fit is at its worst.

    A fast-swinging axis breaks the linear approximation the solver leans on,
    so the residual roughly triples -- but it stays well under the bar that
    marks a solution untrustworthy, and the matrix is unchanged.
    """
    rates, quats = wander(8.0, mapping=lambda v: (v[0], -v[1], v[2]),
                          freqs=(0.9, 1.6, 2.3))
    solution = ori.solve_gyro_frame(build(rates, quats, 8))
    assert solution.matrix == ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    assert 0.06 < solution.residual < 0.20
    assert solution.trustworthy


def test_a_reflection_is_solvable_too():
    """Swapping two axes flips the determinant, which is what this camera does."""
    rates, quats = wander(8.0, mapping=lambda v: (v[2], v[1], v[0]))
    solution = ori.solve_gyro_frame(build(rates, quats, 8))
    assert solution.matrix == ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def test_solved_frame_makes_the_integration_agree_with_cori():
    rates, quats = wander(8.0, mapping=lambda v: (v[2], -v[0], v[1]))
    tel = build(rates, quats, 8)
    track = ori.integrate(tel)
    drift = ori.qangle(ori.qmul(track.quats[-1], ori.qconj(quats[-1])))
    assert math.degrees(drift) < 2.0


def test_a_clip_that_barely_turns_is_refused_not_guessed():
    rates, quats = wander(8.0, scale=0.001)
    with pytest.raises(ori.OrientationError, match="ne tourne pas assez"):
        ori.solve_gyro_frame(build(rates, quats, 8))


def test_a_clip_turning_about_one_axis_only_is_refused():
    """Rank-deficient: nothing in the data distinguishes the other two axes."""
    rate = math.radians(120)
    rates = [(0.0, 0.0, rate)] * (GYRO_HZ * 3)
    quats = [ori.from_rotvec((0.0, 0.0, rate * i / CORI_HZ))
             for i in range(CORI_HZ * 3)]
    with pytest.raises(ori.OrientationError, match="dans le même plan"):
        ori.solve_gyro_frame(build(rates, quats, 3))


def test_missing_gyro_says_so():
    stream = nested(b"STRM", klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)))
    with pytest.raises(ori.OrientationError, match="GYRO"):
        ori.solve_gyro_frame(collect(parse(nested(b"DEVC", stream))))


# --------------------------------------------------------- the smoother


def test_smoothing_does_not_lag_a_steady_turn():
    """The point of filtering both ways: a ramp comes back where it was.

    A one-pole run only forwards would leave this trailing by its whole time
    constant -- 45 degrees at 90 deg/s and tau = 0.5 s. Checked away from the
    ends, where each pass is still settling.
    """
    dt = 1 / 30
    quats = [ori.from_rotvec((0.0, 0.0, math.radians(90) * i * dt))
             for i in range(300)]
    out = ori.smooth(quats, dt, 0.5)
    for i in range(120, 180):
        about(out[i], quats[i], tol=math.radians(0.05))


def test_smoothing_removes_a_shake():
    dt = 1 / 30
    quats = [ori.from_rotvec((0.0, 0.0, math.radians(6) * (-1) ** i))
             for i in range(300)]
    out = ori.smooth(quats, dt, 0.5)
    worst = max(math.degrees(ori.qangle(q)) for q in out[100:200])
    assert worst < 0.5, "a 6 degree per frame shake should be flattened"


def test_smoothing_is_a_no_op_at_zero():
    quats = [ori.from_rotvec((0.0, 0.0, 0.1 * i)) for i in range(10)]
    assert ori.smooth(quats, 1 / 30, 0.0) == quats


def test_corrections_carry_the_steadied_frame_onto_the_real_one():
    raw = [ori.from_rotvec((0.0, 0.0, math.radians(12)))]
    smoothed = [ori.from_rotvec((0.0, 0.0, math.radians(2)))]
    correction = ori.corrections(raw, smoothed)[0]
    # Applying it to the smoothed orientation must land on the raw one.
    about(ori.qmul(correction, smoothed[0]), raw[0])
    assert math.degrees(ori.qangle(correction)) == pytest.approx(10.0, abs=1e-6)


def test_stabilise_leaves_a_perfectly_steady_shot_alone():
    rates = [(0.0, 0.0, 0.0)] * 800
    quats = [ori.IDENTITY] * 60
    tel = build(rates, quats, 2)
    frame = ori.FrameSolution(((1, 0, 0), (0, 1, 0), (0, 0, 1)), 0.0, 99)
    out = ori.stabilise(tel, [0.5, 1.0, 1.5], 0.5, frame)
    for q in out:
        assert math.degrees(ori.qangle(q)) < 1e-6
