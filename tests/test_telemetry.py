"""Decoding GPMF.

The parser is fed hand-built payloads so the tricky parts -- padding,
fixed-point types, SCAL broadcasting -- are pinned down without needing a
camera.
"""

from __future__ import annotations

import struct

import pytest

from bobvr.telemetry import Item, collect, parse, remap


def klv(key: bytes, type_char: bytes, elem_size: int, repeat: int, payload: bytes) -> bytes:
    header = key + type_char + bytes([elem_size]) + struct.pack(">H", repeat)
    padding = (-len(payload)) % 4
    return header + payload + b"\x00" * padding


def nested(key: bytes, body: bytes) -> bytes:
    return key + b"\x00" + bytes([1]) + struct.pack(">H", len(body)) + body


def test_scalar_stream_is_decoded():
    data = klv(b"TSMP", b"L", 4, 1, struct.pack(">I", 1234))
    items = parse(data)
    assert len(items) == 1
    assert items[0].key == "TSMP"
    assert items[0].value == [1234]


def test_grouped_samples_become_tuples():
    payload = struct.pack(">6h", 1, 2, 3, 4, 5, 6)
    items = parse(klv(b"ACCL", b"s", 6, 2, payload))
    assert items[0].value == [(1, 2, 3), (4, 5, 6)]


def test_payloads_are_padded_to_four_bytes():
    """A 9-byte string is followed by 3 bytes of padding, not by garbage."""
    first = klv(b"DVNM", b"c", 9, 1, b"HERO\x00\x00\x00\x00\x00")
    second = klv(b"TSMP", b"L", 4, 1, struct.pack(">I", 7))
    items = parse(first + second)
    assert [i.key for i in items] == ["DVNM", "TSMP"]
    assert items[0].value == "HERO"
    assert items[1].value == [7]


def test_strings_stop_at_the_first_null():
    items = parse(klv(b"STNM", b"c", 12, 1, b"Gyroscope\x00\x00\x00"))
    assert items[0].value == "Gyroscope"


def test_nested_containers_are_walked():
    inner = klv(b"TSMP", b"L", 4, 1, struct.pack(">I", 9))
    data = nested(b"DEVC", nested(b"STRM", inner))
    items = parse(data)
    assert items[0].key == "DEVC"
    assert items[0].children[0].key == "STRM"
    assert items[0].children[0].children[0].value == [9]


def test_fixed_point_types_are_converted():
    """'q' is Q15.16: 1.5 is stored as 98304."""
    items = parse(klv(b"WRGB", b"q", 4, 1, struct.pack(">i", 98304)))
    assert items[0].value == pytest.approx([1.5])


def test_truncated_entry_is_ignored_not_fatal():
    good = klv(b"TSMP", b"L", 4, 1, struct.pack(">I", 1))
    truncated = b"ACCL" + b"s" + bytes([6]) + struct.pack(">H", 100) + b"\x00\x00"
    items = parse(good + truncated)
    assert [i.key for i in items] == ["TSMP"]


def test_single_scal_factor_applies_to_every_component():
    stream = nested(
        b"STRM",
        klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 100))
        + klv(b"STNM", b"c", 12, 1, b"Accelerometer"[:12])
        + klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 100, 200, 300)),
    )
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert telemetry.flatten("ACCL") == [(1.0, 2.0, 3.0)]


def test_per_component_scal_factors():
    stream = nested(
        b"STRM",
        klv(b"SCAL", b"l", 12, 1, struct.pack(">3i", 10, 100, 1000))
        + klv(b"GPS5", b"l", 12, 1, struct.pack(">3i", 100, 100, 100)),
    )
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert telemetry.flatten("GPS5") == [(10.0, 1.0, 0.1)]


def test_metadata_keys_are_not_mistaken_for_samples():
    """TMPC rides along inside a stream but is the sensor's temperature."""
    stream = nested(
        b"STRM",
        klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"TMPC", b"f", 4, 1, struct.pack(">f", 31.5))
        + klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)),
    )
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert "TMPC" not in telemetry.streams
    assert telemetry.flatten("ACCL") == [(1.0, 2.0, 3.0)]


def test_payload_count_tracks_devc_blocks():
    stream = nested(b"STRM", klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)))
    data = nested(b"DEVC", stream) * 3
    telemetry = collect(parse(data))
    assert telemetry.payload_count == 3
    assert len(telemetry.flatten("ACCL")) == 3
    assert telemetry.sample_rate("ACCL") == pytest.approx(1.0)


def test_speeds_and_track_come_from_gps5():
    stream = nested(
        b"STRM",
        klv(b"SCAL", b"l", 20, 1, struct.pack(">5i", 10000000, 10000000, 1000, 1000, 1000))
        + klv(
            b"GPS5", b"l", 20, 2,
            struct.pack(">10i",
                        455000000, 65000000, 1200000, 25000, 25500,
                        455001000, 65001000, 1201000, 30000, 30500),
        ),
    )
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert telemetry.has_gps
    speeds = telemetry.speeds_kmh()
    assert speeds == pytest.approx([90.0, 108.0])
    track = telemetry.track()
    assert track[0] == pytest.approx((45.5, 6.5))


def sensor_stream(body: bytes) -> bytes:
    return nested(b"DEVC", nested(b"STRM", body))


def test_axis_declaration_is_reported_but_not_applied():
    """Samples stay in stored order; the declaration rides alongside.

    Applying ORIN/ORIO here would be wrong: measured against CORI on real
    footage it does not land the gyro in the quaternions' frame.
    """
    telemetry = collect(parse(sensor_stream(
        klv(b"ORIN", b"c", 3, 1, b"XzY")
        + klv(b"ORIO", b"c", 3, 1, b"ZXY")
        + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"GYRO", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)),
    )))
    assert telemetry.flatten("GYRO") == [(1.0, 2.0, 3.0)]
    assert telemetry.streams["GYRO"][0].axes == ("XzY", "ZXY")


def test_axis_plan_spells_out_the_declaration():
    """ZXY out of XzY: take the axis called Z (stored second, negated),
    then X (stored first), then Y (stored third)."""
    telemetry = collect(parse(sensor_stream(
        klv(b"ORIN", b"c", 3, 1, b"XzY")
        + klv(b"ORIO", b"c", 3, 1, b"ZXY")
        + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"GYRO", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)),
    )))
    stream = telemetry.streams["GYRO"][0]
    assert stream.axis_plan == [(1, -1.0), (0, 1.0), (2, 1.0)]
    assert remap(stream.samples, stream.axis_plan) == [(-2.0, 1.0, 3.0)]


def test_matrix_is_reported_when_present():
    matrix = struct.pack(">9f", 0, -1, 0, 1, 0, 0, 0, 0, 1)
    telemetry = collect(parse(sensor_stream(
        klv(b"MTRX", b"f", 36, 1, matrix)
        + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)),
    )))
    assert telemetry.flatten("ACCL") == [(1.0, 2.0, 3.0)]
    assert telemetry.streams["ACCL"][0].matrix == (0, -1, 0, 1, 0, 0, 0, 0, 1)


def test_streams_without_an_axis_declaration_say_so():
    """CORI declares none, and that absence is itself information."""
    telemetry = collect(parse(sensor_stream(
        klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"CORI", b"s", 8, 1, struct.pack(">4h", 1, 2, 3, 4)),
    )))
    assert telemetry.flatten("CORI") == [(1.0, 2.0, 3.0, 4.0)]
    assert telemetry.streams["CORI"][0].axes is None
    assert telemetry.streams["CORI"][0].axis_plan is None


def test_mismatched_axis_declaration_yields_no_plan():
    telemetry = collect(parse(sensor_stream(
        klv(b"ORIN", b"c", 3, 1, b"XYZ")
        + klv(b"ORIO", b"c", 3, 1, b"ABC")
        + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
        + klv(b"GYRO", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)),
    )))
    assert telemetry.streams["GYRO"][0].axis_plan is None


def test_vpts_is_timing_not_a_stream():
    """VPTS shares the orientation stream, and its SCAL must not touch it."""
    payload = (
        klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 32767))
        + klv(b"STMP", b"J", 8, 1, struct.pack(">Q", 185089))
        + klv(b"VPTS", b"L", 4, 1, struct.pack(">I", 1286189))
        + klv(b"CORI", b"s", 8, 1, struct.pack(">4h", 32767, 0, 0, 0))
    )
    telemetry = collect(parse(nested(b"DEVC", nested(b"STRM", payload))))
    assert "VPTS" not in telemetry.streams
    assert telemetry.video_pts_us == [1286189]
    assert telemetry.clock_offset_us == 1286189 - 185089
    assert telemetry.clock_offset_spread_us() == 0
    assert telemetry.flatten("CORI") == [(1.0, 0.0, 0.0, 0.0)]


def test_sample_times_are_spread_across_the_payload():
    """Two payloads a second apart, two samples each: 0, 0.5, 1.0, 1.5."""
    def block(stamp: int) -> bytes:
        return nested(b"STRM",
                      klv(b"STMP", b"J", 8, 1, struct.pack(">Q", stamp))
                      + klv(b"SCAL", b"s", 2, 1, struct.pack(">h", 1))
                      + klv(b"GYRO", b"s", 6, 2, struct.pack(">6h", 1, 2, 3, 4, 5, 6)))

    data = nested(b"DEVC", block(0)) + nested(b"DEVC", block(1_000_000))
    telemetry = collect(parse(data))
    assert telemetry.sensor_times("GYRO") == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_sample_times_are_empty_without_stamps():
    stream = nested(b"STRM", klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)))
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert telemetry.sensor_times("ACCL") == []
    assert telemetry.clock_offset_us is None


def test_no_gps_is_reported_honestly():
    stream = nested(b"STRM", klv(b"ACCL", b"s", 6, 1, struct.pack(">3h", 1, 2, 3)))
    telemetry = collect(parse(nested(b"DEVC", stream)))
    assert not telemetry.has_gps
    assert telemetry.speeds_kmh() == []
    assert telemetry.track() == []
