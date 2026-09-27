"""Measure what the IMU streams actually look like on a real run."""
import math
import sys
from pathlib import Path

from bobvr import telemetry
from bobvr.media import probe
from bobvr.render.caps import detect

path = Path(sys.argv[1])
caps = detect()
info = probe(path, caps)
print(f"{path.name}: {info.duration:.1f}s @ {info.fps} fps -> "
      f"{info.duration * (info.fps or 30):.0f} frames")

tel = telemetry.extract(path, info, caps)
print("payloads:", tel.payload_count)
for key in tel.stream_names():
    flat = tel.flatten(key)
    blocks = tel.streams[key]
    print(f"  {key:5s} {len(flat):7d} samples  {tel.sample_rate(key):8.2f} Hz  "
          f"width={len(flat[0]) if flat and isinstance(flat[0], tuple) else 1}"
          f"  name={blocks[0].name!r} units={blocks[0].units!r}")

# Accelerometer magnitude: how far from 1 g does the sled actually go?
accl = tel.flatten("ACCL")
if accl:
    mags = sorted(math.sqrt(sum(c * c for c in row)) for row in accl)
    n = len(mags)
    print("\nACCL |a| in g (9.81 m/s2 assumed):")
    for label, idx in (("min", 0), ("p05", n // 20), ("median", n // 2),
                       ("p95", 19 * n // 20), ("max", n - 1)):
        print(f"  {label:6s} {mags[idx] / 9.81:6.2f} g")
    over = sum(1 for m in mags if m / 9.81 > 1.3) / n
    print(f"  fraction above 1.3 g: {over:.1%}")

# Gyro: how fast does the camera actually rotate, and at what frequency?
gyro = tel.flatten("GYRO")
if gyro:
    rates = sorted(math.degrees(math.sqrt(sum(c * c for c in row))) for row in gyro)
    n = len(rates)
    print("\nGYRO |w| in deg/s:")
    for label, idx in (("median", n // 2), ("p95", 19 * n // 20),
                       ("p99", 99 * n // 100), ("max", n - 1)):
        print(f"  {label:6s} {rates[idx]:8.1f}")

# CORI: one per frame?
cori = tel.flatten("CORI")
if cori:
    print(f"\nCORI: {len(cori)} samples, first={tuple(round(c, 4) for c in cori[0])}")
    norms = [math.sqrt(sum(c * c for c in row)) for row in cori[:50]]
    print(f"  |q| over first 50: {min(norms):.4f}..{max(norms):.4f}")
