"""Reading GoPro's GPMF telemetry out of a .360 file.

The camera records speed, position, acceleration and rotation alongside the
video, in a ``gpmd`` track. This module extracts and decodes it into plain
Python values; drawing gauges and a track trace over the video is the next
step, and will consume what comes out of here.

GPMF is a KLV format: a four-character key, a one-byte type, the size of one
element group, and a count -- then that many groups, padded to four bytes. A
``SCAL`` key rescales whatever follows it in the same stream, which is how the
camera stores fixed-point values compactly.

Two further keys matter as soon as the sensors are used together rather than
one at a time, and both are decoded here:

* ``ORIN``/``ORIO`` give the axis order the samples are *stored* in and the
  order they are *meant* to be read in. The MAX writes ``XzY`` -> ``ZXY`` for
  the accelerometer, gyroscope and magnetometer, and nothing at all for the
  orientation quaternions. Both are decoded and reported, but **not applied**:
  measured against ``CORI`` on real footage, the declared pair does not land
  the gyroscope in the quaternions' frame, so applying it would be a step that
  has to be undone. Samples stay in stored order and the declaration is
  reported on the stream, for a caller to act on knowingly.
* ``STMP`` and ``VPTS`` timestamp the payload on the sensor clock and on the
  video clock. Together they place every sample on the video timeline, which
  is what any stabilisation has to have and what "one payload per second"
  only approximates.
"""

from __future__ import annotations

import logging
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .media import MaxVideoInfo
from .render.caps import Capabilities, subprocess_kwargs

log = logging.getLogger(__name__)

#: GPMF type character -> (struct code, bytes). Type 0 marks a nested
#: container and is handled separately.
_TYPES: dict[str, tuple[str, int]] = {
    "b": ("b", 1), "B": ("B", 1),
    "s": ("h", 2), "S": ("H", 2),
    "l": ("i", 4), "L": ("I", 4),
    "q": ("i", 4), "Q": ("q", 8),
    "j": ("q", 8), "J": ("Q", 8),
    "f": ("f", 4), "d": ("d", 8),
}

#: Fixed-point types, and the divisor that turns them back into reals.
#: 'q' is Q15.16 and 'Q' is Q31.32.
_FIXED_POINT: dict[str, float] = {"q": 65536.0, "Q": 4294967296.0}

_STRING_TYPES = frozenset("cFU")
_NESTED = "\x00"


class TelemetryError(RuntimeError):
    """Telemetry could not be read."""


@dataclass
class Item:
    """One decoded KLV entry."""

    key: str
    type: str
    value: Any
    children: list["Item"] = field(default_factory=list)

    def find(self, key: str) -> "Item | None":
        return next((c for c in self.children if c.key == key), None)

    def find_all(self, key: str) -> list["Item"]:
        return [c for c in self.children if c.key == key]


@dataclass
class Stream:
    """A named sensor stream from one payload."""

    name: str
    key: str
    units: str | None
    samples: list[tuple]
    #: ``STMP``: capture time of this block's first sample, in microseconds on
    #: the camera's sensor clock. Absent on streams the camera does not stamp.
    start_us: int | None = None
    #: The ``(ORIN, ORIO)`` pair the camera declared, or None if it declared
    #: none -- which is itself the answer for CORI, GRAV and IORI. ``samples``
    #: are in ORIN order: see the module docstring for why nothing is applied.
    axes: tuple[str, str] | None = None
    #: ``MTRX``, the same remap written out as a square matrix, if present.
    matrix: tuple[float, ...] | None = None

    @property
    def axis_plan(self) -> list[tuple[int, float]] | None:
        """The declared remap as ``(source index, sign)`` per output axis.

        What ``ORIN``/``ORIO`` mean, ready to apply -- for a caller that wants
        the camera's stated convention rather than a measured one.
        """
        if not self.axes:
            return None
        return _axis_remap(*self.axes)

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class Telemetry:
    """Everything decoded from a clip's telemetry track."""

    streams: dict[str, list[Stream]] = field(default_factory=dict)
    #: Payloads are roughly one second of data each, so index ~= second.
    payload_count: int = 0
    #: ``VPTS``: the video presentation timestamp of each payload, in
    #: microseconds. The camera writes it beside the orientation quaternions.
    video_pts_us: list[int] = field(default_factory=list)
    #: Sensor-clock time of the payload each ``video_pts_us`` entry came with,
    #: so the two clocks can be related. Same length as ``video_pts_us``.
    video_pts_stamp_us: list[int] = field(default_factory=list)

    def stream_names(self) -> list[str]:
        return sorted(self.streams)

    def flatten(self, key: str) -> list[tuple]:
        """All samples for ``key``, in capture order."""
        return [s for block in self.streams.get(key, []) for s in block.samples]

    def sample_rate(self, key: str) -> float:
        """Approximate samples per second."""
        if not self.payload_count:
            return 0.0
        return len(self.flatten(key)) / self.payload_count

    # ------------------------------------------------------------- timing

    def sensor_times(self, key: str) -> list[float]:
        """Capture time of every sample of ``key``, in seconds.

        On the camera's own sensor clock, which is shared by every stream --
        so gyroscope and orientation samples taken at the same instant get the
        same number even though they arrive at 802 Hz and 30 Hz. Within a
        payload the samples are assumed evenly spaced, which is what the
        format's one stamp per payload lets anyone assume.

        Empty if the camera stamped nothing.
        """
        blocks = self.streams.get(key, [])
        stamped = [b for b in blocks if b.start_us is not None and b.samples]
        if not stamped:
            return []

        times: list[float] = []
        for i, block in enumerate(stamped):
            start = block.start_us
            count = len(block.samples)
            if i + 1 < len(stamped):
                span = stamped[i + 1].start_us - start
            elif i > 0:
                # No payload follows: reuse the rate the previous one ran at.
                previous = stamped[i - 1]
                span = round((start - previous.start_us) * count / len(previous))
            else:
                span = 0
            step = span / count if count else 0.0
            times.extend((start + j * step) / 1e6 for j in range(count))
        return times

    def _clock_offsets(self) -> list[int]:
        return [
            pts - stamp
            for pts, stamp in zip(self.video_pts_us, self.video_pts_stamp_us)
        ]

    @property
    def clock_offset_us(self) -> int | None:
        """``VPTS - STMP``: what to add to a sensor time to get a video time.

        The MAX writes both stamps on every payload. Taken as a median because
        the last payload is cut short by the end of the recording and its
        stamps land differently; see ``clock_offset_spread_us``. Returns
        ``None`` when the camera wrote no ``VPTS``.

        Note what this does *not* settle: whether ``VPTS`` marks the first or
        the last sample of its payload. That is a one-second ambiguity, and
        the file does not say; it has to be pinned down against the picture.
        """
        offsets = sorted(self._clock_offsets())
        return offsets[len(offsets) // 2] if offsets else None

    def clock_offset_spread_us(self) -> int:
        """How far ``VPTS - STMP`` wanders, in microseconds.

        Zero means the two clocks are rigidly locked and one offset is exact;
        a growing figure means they drift and stabilisation would have to
        track the offset rather than assume it. On the footage this was built
        against it comes to one frame interval, so they are locked and the
        residue is the payload boundary landing between frames.

        The final payload is excluded: recording stops mid-payload, which
        moves its stamps relative to each other and says nothing about drift.
        """
        offsets = self._clock_offsets()[:-1]
        return max(offsets) - min(offsets) if offsets else 0

    @property
    def has_gps(self) -> bool:
        return bool(self.streams.get("GPS5") or self.streams.get("GPS9"))

    def speeds_kmh(self) -> list[float]:
        """Ground speed in km/h, from whichever GPS stream the camera wrote."""
        if self.streams.get("GPS9"):
            # GPS9: lat, lon, alt, 2D speed, 3D speed, days, secs, dop, fix
            return [row[3] * 3.6 for row in self.flatten("GPS9") if len(row) > 3]
        # GPS5: lat, lon, alt, 2D speed, 3D speed
        return [row[3] * 3.6 for row in self.flatten("GPS5") if len(row) > 3]

    def track(self) -> list[tuple[float, float]]:
        """(latitude, longitude) pairs, for drawing the run."""
        key = "GPS9" if self.streams.get("GPS9") else "GPS5"
        return [(row[0], row[1]) for row in self.flatten(key) if len(row) > 1]


# ---------------------------------------------------------------- parsing


def parse(data: bytes) -> list[Item]:
    """Decode a GPMF byte stream into a tree of items."""
    return list(_walk(data, 0, len(data)))


def _walk(data: bytes, start: int, end: int) -> Iterator[Item]:
    offset = start
    while offset + 8 <= end:
        key = data[offset : offset + 4].decode("latin1")
        type_char = chr(data[offset + 4])
        elem_size = data[offset + 5]
        repeat = struct.unpack_from(">H", data, offset + 6)[0]
        payload = elem_size * repeat
        body = offset + 8
        if body + payload > end:
            log.debug("truncated GPMF entry %r at %d", key, offset)
            return

        if type_char == _NESTED:
            item = Item(key=key, type=type_char, value=None)
            item.children = list(_walk(data, body, body + payload))
            yield item
        else:
            yield Item(
                key=key,
                type=type_char,
                value=_decode(data, body, type_char, elem_size, repeat),
            )
        # Payloads are padded out to a four-byte boundary.
        offset = body + (payload + 3) // 4 * 4


def _decode(
    data: bytes, offset: int, type_char: str, elem_size: int, repeat: int
) -> Any:
    if type_char in _STRING_TYPES:
        raw = data[offset : offset + elem_size * repeat]
        text = raw.split(b"\x00")[0].decode("latin1", errors="replace").strip()
        return text

    spec = _TYPES.get(type_char)
    if spec is None:
        return data[offset : offset + elem_size * repeat]

    code, width = spec
    per_group = elem_size // width
    if per_group < 1:
        return []
    values = struct.unpack_from(f">{per_group * repeat}{code}", data, offset)
    divisor = _FIXED_POINT.get(type_char)
    if divisor is not None:
        values = tuple(v / divisor for v in values)
    if per_group == 1:
        return list(values)
    return [
        tuple(values[i * per_group : (i + 1) * per_group]) for i in range(repeat)
    ]


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _flat_numbers(value: Any) -> list[float]:
    """Flatten a decoded value into a plain list of numbers.

    SCAL factors arrive either as one group of N values or as N groups of
    one, depending on how the camera packed them; both mean the same thing.
    """
    numbers: list[float] = []
    stack = [value]
    while stack:
        current = stack.pop(0)
        if isinstance(current, (list, tuple)):
            stack = list(current) + stack
        elif isinstance(current, (int, float)):
            numbers.append(float(current))
    return numbers


def _scale(samples: list, factors: list[float]) -> list[tuple]:
    """Apply SCAL divisors, broadcasting a single factor across components."""
    scaled = []
    for sample in samples:
        row = _as_tuple(sample)
        if len(factors) == 1:
            divisor = factors[0] or 1
            scaled.append(tuple(v / divisor for v in row))
        else:
            scaled.append(
                tuple(
                    v / (factors[i] or 1) if i < len(factors) else v
                    for i, v in enumerate(row)
                )
            )
    return scaled


#: Keys that describe a stream rather than carrying its samples. TMPC is the
#: sensor's own temperature, reported once per payload alongside the real
#: samples; treating it as data would file it under the parent stream's name.
#: VPTS is the payload's video timestamp -- it rides inside the orientation
#: stream, and letting it through would both invent a stream and divide a
#: microsecond count by that stream's SCAL.
_META_KEYS = frozenset(
    {"STMP", "TSMP", "STNM", "MTRX", "ORIN", "ORIO", "SIUN", "UNIT",
     "SCAL", "TYPE", "TIMO", "EMPT", "TICK", "TOCK", "TMPC", "VPTS"}
)


def _axis_remap(orin: str, orio: str) -> list[tuple[int, float]] | None:
    """Plan for reordering samples from ``ORIN`` order into ``ORIO`` order.

    Each character names an axis, upper case for the positive direction and
    lower case for the negative one: the MAX stores its inertial sensors as
    ``XzY`` and asks for them as ``ZXY``. The result is one
    ``(source index, sign)`` per output component.

    Returns None if the two do not describe the same set of axes, in which
    case the samples are better left alone than silently scrambled.
    """
    if len(orin) != len(orio):
        return None
    plan: list[tuple[int, float]] = []
    for wanted in orio:
        source = orin.upper().find(wanted.upper())
        if source < 0:
            return None
        # Cases agree -> the stored axis already points the wanted way.
        sign = 1.0 if orin[source].isupper() == wanted.isupper() else -1.0
        plan.append((source, sign))
    return plan


def remap(samples: list[tuple], plan: list[tuple[int, float]]) -> list[tuple]:
    """Apply an axis plan (from ``Stream.axis_plan``, or a measured one)."""
    return [
        tuple(sample[i] * sign for i, sign in plan)
        if len(sample) == len(plan) else sample
        for sample in samples
    ]


def collect(items: list[Item]) -> Telemetry:
    """Turn a parsed tree into per-key sample streams, correctly scaled."""
    telemetry = Telemetry()
    devices = [i for i in items if i.key == "DEVC"]
    telemetry.payload_count = len(devices)

    for device in devices:
        for stream in device.find_all("STRM"):
            scal = stream.find("SCAL")
            factors = _flat_numbers(scal.value) if scal else []
            if not factors:
                factors = [1.0]
            name_item = stream.find("STNM")
            units_item = stream.find("SIUN") or stream.find("UNIT")
            stamp_item = stream.find("STMP")
            stamp = _first_int(stamp_item.value) if stamp_item else None

            orin_item, orio_item = stream.find("ORIN"), stream.find("ORIO")
            orin = str(orin_item.value) if orin_item else ""
            orio = str(orio_item.value) if orio_item else ""
            mtrx_item = stream.find("MTRX")

            # VPTS rides along inside the orientation stream and carries the
            # payload's place on the video timeline. Pairing it with this
            # stream's STMP is the whole point: it ties the two clocks.
            vpts_item = stream.find("VPTS")
            if vpts_item is not None and stamp is not None:
                vpts = _first_int(vpts_item.value)
                if vpts is not None:
                    telemetry.video_pts_us.append(vpts)
                    telemetry.video_pts_stamp_us.append(stamp)

            for entry in stream.children:
                if entry.key in _META_KEYS or entry.type == _NESTED:
                    continue
                if not isinstance(entry.value, list) or not entry.value:
                    continue
                samples = _scale(entry.value, factors)
                block = Stream(
                    name=str(name_item.value) if name_item else entry.key,
                    key=entry.key,
                    units=str(units_item.value) if units_item else None,
                    samples=samples,
                    start_us=stamp,
                    axes=(orin, orio) if orin and orio else None,
                    matrix=(tuple(_flat_numbers(mtrx_item.value))
                            if mtrx_item is not None else None),
                )
                telemetry.streams.setdefault(entry.key, []).append(block)
    return telemetry


def _first_int(value: Any) -> int | None:
    numbers = _flat_numbers(value)
    return int(numbers[0]) if numbers else None


# -------------------------------------------------------------- extraction


def extract(path: Path, info: MaxVideoInfo, caps: Capabilities) -> Telemetry:
    """Pull the telemetry track out of ``path`` and decode it.

    Raises:
        TelemetryError: the file carries no telemetry, or ffmpeg refused.
    """
    if info.telemetry_index is None:
        raise TelemetryError(
            f"{path.name} ne contient pas de piste de télémétrie."
        )
    cmd = [
        caps.ffmpeg, "-hide_banner", "-v", "error", "-y",
        "-i", str(path),
        "-map", f"0:{info.telemetry_index}",
        "-c", "copy", "-f", "data", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=300, check=True, **subprocess_kwargs()
        )
    except subprocess.CalledProcessError as exc:
        raise TelemetryError(
            f"Extraction de la télémétrie impossible : "
            f"{exc.stderr.decode('utf-8', 'replace').strip()}"
        ) from exc
    except subprocess.SubprocessError as exc:
        raise TelemetryError(f"Extraction de la télémétrie interrompue : {exc}") from exc

    if not proc.stdout:
        raise TelemetryError(f"{path.name} : la piste de télémétrie est vide.")
    return collect(parse(proc.stdout))
