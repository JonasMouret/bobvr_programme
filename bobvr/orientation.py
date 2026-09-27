"""Camera orientation from the gyroscope, and the rotations that steady it.

The camera writes two things worth having: ``GYRO``, raw angular rate at
802 Hz, and ``CORI``, its own fused orientation at frame rate. Stabilisation
wants the first -- a third of the movement on a bobsleigh run sits above the
15 Hz that 30 Hz quaternions can represent -- but the first is useless until
it is expressed in the same frame as the second, and the file's own
``ORIN``/``ORIO`` declaration does not do that (see ``telemetry``). So the
frame is *solved* here, from the clip, with a residual that doubles as the
confidence check.

Everything is plain Python on 4-tuples ``(w, x, y, z)``. That is deliberate:
the alternative pulls numpy into a Windows build that ships as one folder, and
the algorithms below are all linear in the sample count -- the integration
touches each gyro sample once, and the smoother is two passes of a one-pole
filter rather than a wide convolution.

Two conventions, both measured rather than assumed:

* Orientation composes **extrinsically**, like ``CORI``: the body-frame step
  between two samples is ``q(k+1) * conj(q(k))``, so integrating means
  left-multiplying. Getting this backwards costs a ~57 % residual that no
  amount of axis-swapping recovers.
* A quaternion maps **world to camera**. The rotation the renderer needs is
  therefore ``raw * conj(smoothed)``: it takes a direction in the steadied
  frame back to where the lens actually was.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

from .telemetry import Telemetry

#: A quaternion, ``(w, x, y, z)``.
Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]

IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


class OrientationError(RuntimeError):
    """The clip does not carry what an orientation needs."""


# ------------------------------------------------------------ quaternions


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q: Quat) -> Quat:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-12:
        return IDENTITY
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def qangle(q: Quat) -> float:
    """The rotation's magnitude in radians, always in ``[0, pi]``."""
    v = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return 2.0 * math.atan2(v, abs(q[0]))


def positive(q: Quat) -> Quat:
    """The representative with ``w >= 0``.

    ``q`` and ``-q`` are the same rotation, so this settles which one is
    written down -- which is what lets the packed kernel table drop ``w`` and
    recover it from the other three.
    """
    return q if q[0] >= 0.0 else (-q[0], -q[1], -q[2], -q[3])


def from_rotvec(v: Vec3) -> Quat:
    """Rotation of ``|v|`` radians about ``v``."""
    theta = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if theta < 1e-12:
        return IDENTITY
    s = math.sin(theta / 2.0) / theta
    return (math.cos(theta / 2.0), v[0] * s, v[1] * s, v[2] * s)


def to_rotvec(q: Quat) -> Vec3:
    q = positive(q)
    v = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if v < 1e-12:
        return (2.0 * q[1], 2.0 * q[2], 2.0 * q[3])
    scale = 2.0 * math.atan2(v, q[0]) / v
    return (q[1] * scale, q[2] * scale, q[3] * scale)


def slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Interpolate along the shorter arc between two orientations."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if dot < 0.0:
        b, dot = (-b[0], -b[1], -b[2], -b[3]), -dot
    if dot > 0.9995:
        # Nearly parallel: the arc is indistinguishable from the chord, and
        # the trigonometric form loses precision.
        return qnorm(tuple(x + t * (y - x) for x, y in zip(a, b)))  # type: ignore[return-value]
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return tuple(wa * x + wb * y for x, y in zip(a, b))  # type: ignore[return-value]


# ---------------------------------------------------------- the axis frame


@dataclass(frozen=True)
class FrameSolution:
    """How the gyroscope's axes relate to the orientation quaternions'.

    ``matrix`` is a signed permutation, as rows. ``residual`` is the median
    error left over, relative to the size of the rotation being matched --
    around 4 % is what an honest fit looks like, and anything above about 20 %
    means the clip disagrees with every candidate and should not be trusted.
    """

    matrix: tuple[Vec3, Vec3, Vec3]
    residual: float
    windows: int

    def apply(self, v: Vec3) -> Vec3:
        return (
            self.matrix[0][0] * v[0] + self.matrix[0][1] * v[1] + self.matrix[0][2] * v[2],
            self.matrix[1][0] * v[0] + self.matrix[1][1] * v[1] + self.matrix[1][2] * v[2],
            self.matrix[2][0] * v[0] + self.matrix[2][1] * v[1] + self.matrix[2][2] * v[2],
        )

    @property
    def trustworthy(self) -> bool:
        return self.windows >= 8 and self.residual < 0.20


#: Rotation matched over this long a window when solving the frame. Long
#: enough that integrating low-passes the vibration away, short enough that a
#: rotation vector still adds up linearly.
SOLVE_WINDOW_S = 0.5

#: Windows turning less than this are skipped: their direction is noise.
SOLVE_MIN_TURN_DEG = 5.0


def solve_gyro_frame(tel: Telemetry) -> FrameSolution:
    """Work out the gyroscope-to-``CORI`` axis map from the clip itself.

    Compares rotations *integrated* over half a second, never differentiated
    per frame: a third of this footage's angular power sits above ``CORI``'s
    15 Hz Nyquist, so differentiating folds that back in as noise and buries
    the answer. Over a window the two are being asked the same question --
    "where did the camera end up" -- and the fast stuff averages out.

    Raises:
        OrientationError: the clip has no gyroscope or no orientation stream.
    """
    times, rates = _stream(tel, "GYRO")
    c_times, c_quats = _stream(tel, "CORI")
    if len(c_quats) < 4:
        raise OrientationError("Le flux d'orientation CORI est trop court.")

    step = max(1, int(round(SOLVE_WINDOW_S * len(c_quats) / max(c_times[-1] - c_times[0], 1e-6))))
    threshold = math.radians(SOLVE_MIN_TURN_DEG)

    source: list[Vec3] = []
    target: list[Vec3] = []
    spans: list[tuple[float, float]] = []
    for k in range(0, len(c_quats) - step, step):
        # Extrinsic composition, as CORI uses.
        turn = to_rotvec(qmul(c_quats[k + step], qconj(c_quats[k])))
        if math.sqrt(sum(c * c for c in turn)) < threshold:
            continue
        span = (c_times[k], c_times[k + step])
        got = _integrate_between(times, rates, *span)
        target.append(turn)
        source.append(to_rotvec(got))
        spans.append(span)

    if len(source) < 3:
        raise OrientationError(
            "Cette séquence ne tourne pas assez pour déterminer les axes du "
            "gyroscope ; il faut au moins trois demi-secondes à plus de "
            f"{SOLVE_MIN_TURN_DEG:g}° de rotation."
        )

    matrix = _snap(_least_squares(source, target))
    solution = FrameSolution(matrix, 0.0, len(source))

    # Score it the way the renderer will use it. Mapping the rates and then
    # integrating is not the same as integrating and then mapping the result:
    # they part company at second order, which on this footage is the whole
    # difference between a 4 % residual and an 8 % one. The matrix is the same
    # either way -- but the residual is the confidence signal, so it has to
    # measure what actually gets done.
    errors = []
    for (start, end), want in zip(spans, target):
        got = to_rotvec(_integrate_between(times, rates, start, end, solution))
        size = math.sqrt(sum(c * c for c in want))
        errors.append(math.sqrt(sum((g - w) ** 2 for g, w in zip(got, want))) / size)
    errors.sort()
    return FrameSolution(matrix, errors[len(errors) // 2], len(source))


def _least_squares(source: list[Vec3], target: list[Vec3]) -> list[list[float]]:
    """The 3x3 taking ``source`` onto ``target``, one row at a time.

    Each row is an independent three-unknown fit, so this is three small
    normal-equation solves rather than anything that needs a linear algebra
    package. Left unconstrained on purpose: forcing a proper rotation would
    rule out the answer, which on this camera has a negative determinant.
    """
    gram = [[sum(a[i] * a[j] for a in source) for j in range(3)] for i in range(3)]
    rows = []
    for axis in range(3):
        rhs = [sum(a[i] * b[axis] for a, b in zip(source, target)) for i in range(3)]
        rows.append(_solve3(gram, rhs))
    return rows


def _solve3(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting, on a 3x3."""
    m = [row[:] + [r] for row, r in zip(matrix, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise OrientationError(
                "Les rotations de cette séquence sont toutes dans le même "
                "plan : les axes du gyroscope ne peuvent pas en être déduits."
            )
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def _snap(rows: list[list[float]]) -> tuple[Vec3, Vec3, Vec3]:
    """Round a fitted matrix to the nearest signed permutation.

    The relationship between two sets of sensor axes is one of 48 discrete
    possibilities; the fit is only a way of finding which. Snapping also
    removes the fit's scale error, so the integration downstream neither
    gains nor loses angle.
    """
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    used: set[int] = set()
    order = sorted(range(3), key=lambda r: -max(abs(v) for v in rows[r]))
    for r in order:
        col = max((c for c in range(3) if c not in used),
                  key=lambda c: abs(rows[r][c]))
        used.add(col)
        out[r][col] = 1.0 if rows[r][col] >= 0 else -1.0
    return tuple(tuple(row) for row in out)  # type: ignore[return-value]


# ------------------------------------------------------------ integration


@dataclass(frozen=True)
class Track:
    """An orientation sampled over time, on the camera's sensor clock."""

    times: list[float]
    quats: list[Quat]

    def __len__(self) -> int:
        return len(self.quats)

    def resample(self, wanted: list[float]) -> list[Quat]:
        """Orientations at ``wanted`` instants, interpolated on the sphere."""
        out: list[Quat] = []
        i = 0
        for t in wanted:
            while i + 2 < len(self.times) and self.times[i + 1] < t:
                i += 1
            span = self.times[i + 1] - self.times[i]
            frac = 0.0 if span <= 0 else (t - self.times[i]) / span
            out.append(slerp(self.quats[i], self.quats[i + 1],
                             max(0.0, min(1.0, frac))))
        return out


def integrate(tel: Telemetry, frame: FrameSolution | None = None) -> Track:
    """Integrate the gyroscope into an orientation, one sample at a time.

    Starts from the camera's own first ``CORI`` sample, so the result shares
    its reference rather than starting from an arbitrary identity. Solves the
    axis frame from the clip if one is not supplied.
    """
    times, rates = _stream(tel, "GYRO")
    if len(times) < 2:
        raise OrientationError("Le flux gyroscopique est trop court.")
    if frame is None:
        frame = solve_gyro_frame(tel)

    _, c_quats = _stream(tel, "CORI")
    q: Quat = c_quats[0] if c_quats else IDENTITY

    quats = [q]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        w = frame.apply(rates[i - 1])
        q = qnorm(qmul(from_rotvec((w[0] * dt, w[1] * dt, w[2] * dt)), q))
        quats.append(q)
    return Track(times, quats)


def _integrate_between(times: list[float], rates: list[Vec3],
                       start: float, end: float,
                       frame: "FrameSolution | None" = None) -> Quat:
    """Gyro-only rotation from ``start`` to ``end``.

    The first sample taken is the first at or after ``start``. Reaching one
    sample further back instead adds a whole extra step of rotation to every
    window -- 1.7 degrees where this footage peaks, against windows of about
    25 degrees -- and doubles the residual the solver reports.
    """
    q = IDENTITY
    for i in range(bisect_left(times, start), len(times) - 1):
        if times[i] >= end:
            break
        dt = times[i + 1] - times[i]
        w = rates[i] if frame is None else frame.apply(rates[i])
        q = qnorm(qmul(from_rotvec((w[0] * dt, w[1] * dt, w[2] * dt)), q))
    return q


def _stream(tel: Telemetry, key: str) -> tuple[list[float], list]:
    times = tel.sensor_times(key)
    samples = tel.flatten(key)
    if not times or not samples:
        raise OrientationError(
            f"{key} est absent de la télémétrie, ou n'est pas horodaté ; "
            "l'orientation ne peut pas en être tirée."
        )
    count = min(len(times), len(samples))
    return times[:count], samples[:count]


# -------------------------------------------------------------- smoothing


def smooth(quats: list[Quat], dt: float, seconds: float) -> list[Quat]:
    """Low-pass an orientation without introducing any lag.

    A one-pole filter run forwards and then backwards. Running it both ways
    squares the magnitude response and cancels the phase exactly, so the
    result does not trail the movement -- which a camera stabilising in real
    time cannot do, and which is most of the difference between an image that
    is steady and one that swims.

    ``seconds`` is the time constant: bigger means calmer, and in 360 that
    costs nothing but a larger rotation, since the sphere is turned rather
    than cropped.
    """
    if seconds <= 0.0 or len(quats) < 3:
        return list(quats)
    alpha = 1.0 - math.exp(-dt / seconds)

    forward = [quats[0]]
    for q in quats[1:]:
        forward.append(slerp(forward[-1], q, alpha))

    backward = [forward[-1]]
    for q in reversed(forward[:-1]):
        backward.append(slerp(backward[-1], q, alpha))
    backward.reverse()
    return backward


def corrections(raw: list[Quat], smoothed: list[Quat]) -> list[Quat]:
    """The rotation to apply per frame: steadied frame back to the real lens.

    A viewing direction in the output is expressed in the smoothed camera's
    frame; sampling the source needs it in the real camera's. With both
    quaternions mapping world to camera, that is ``raw * conj(smoothed)``.
    """
    return [positive(qnorm(qmul(r, qconj(s)))) for r, s in zip(raw, smoothed)]


#: How long the sensor takes to read one frame out, as a fraction of the frame
#: period. The sensor does not sample an instant: it reads row after row, so a
#: frame holds the camera's orientation *averaged* over that span, not the
#: orientation at one moment.
#:
#: Measured on B21.360 by fitting picture rotation against gyroscope rotation
#: (see docs): modelling the picture as an average over T lifts agreement on
#: the fast band -- the very band stabilisation exists to remove -- from
#: R2 0.15 at T = 0 to R2 0.50 at T = 1 frame, 33 ms. No metadata field
#: carries this, and assuming an instant instead over-corrects exactly where
#: it hurts.
READOUT_FRAMES = 1.0

#: Sub-samples used to average across that span. Eight is well past the point
#: where the answer stops moving.
READOUT_SAMPLES = 8


def blur(track: "Track", times: list[float], window: float,
         samples: int = READOUT_SAMPLES) -> list[Quat]:
    """Orientation averaged over ``window`` seconds starting at each time.

    Quaternions are averaged componentwise and renormalised, with signs
    aligned first. Over a window this short the rotations involved are a few
    degrees apart, where that is indistinguishable from a proper spherical
    mean and vastly cheaper.
    """
    if window <= 0.0 or samples < 2:
        return track.resample(times)
    step = window / (samples - 1)
    grids = [track.resample([t + i * step for t in times])
             for i in range(samples)]
    out: list[Quat] = []
    for k in range(len(times)):
        ref = grids[0][k]
        acc = [0.0, 0.0, 0.0, 0.0]
        for g in grids:
            q = g[k]
            if sum(a * b for a, b in zip(q, ref)) < 0.0:
                q = (-q[0], -q[1], -q[2], -q[3])
            acc = [a + b for a, b in zip(acc, q)]
        out.append(qnorm(tuple(a / samples for a in acc)))  # type: ignore[arg-type]
    return out


def stabilise(tel: Telemetry, times: list[float], seconds: float,
              frame: FrameSolution | None = None,
              readout: float | None = None) -> list[Quat]:
    """Per-instant rotations that steady the picture at ``times``.

    ``times`` are on the camera's sensor clock, one per video frame.

    ``readout`` is how long the sensor spends reading a frame out, in seconds;
    the orientation the picture actually carries is averaged over that span
    rather than sampled at its start. Passing ``None`` derives it from
    ``READOUT_FRAMES`` and the spacing of ``times``. Pass ``0.0`` for the old
    instantaneous behaviour.

    Note this corrects the *global* consequence of a rolling shutter, not the
    within-frame skew: rows still come from different instants, and undoing
    that needs a rotation that varies down the frame, not one per frame.
    """
    track = integrate(tel, frame)
    if len(times) < 2:
        return [IDENTITY] * len(times)
    dt = (times[-1] - times[0]) / (len(times) - 1)
    if readout is None:
        readout = READOUT_FRAMES * dt
    raw = blur(track, times, readout)
    return corrections(raw, smooth(raw, dt, seconds))
