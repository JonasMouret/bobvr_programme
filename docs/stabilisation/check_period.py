"""Does `index` count frames, or plane invocations?

program_opencl runs the kernel once per plane, so a counter that ticks per
invocation would read the table three times too fast -- and every earlier test
here would still have passed, because a rotation growing three times too
quickly still grows.

A table with period 3 separates the two cleanly. Reading it at k gives
0, 60, 120, 0, 60, 120 degrees; reading it at 3k gives 0 every single time,
because 3k is always a multiple of 3. So: if the frames move, the counter
counts frames. If nothing moves, it counts planes.
"""
import math
import subprocess
import sys
from pathlib import Path

from bobvr.media import probe
from bobvr.render.caps import detect
from bobvr.render.geometry import MaxLayout, materialise_kernel

HERE = Path(__file__).parent
src = Path(sys.argv[1])
caps = detect()
info = probe(src, caps)
layout = MaxLayout(src_w=info.track_width, src_h=info.track_height)
FRAMES = 6


def yaw(degrees):
    half = math.radians(degrees) / 2
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def render(tag, table):
    out = HERE / f"per_{tag}"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    kernel = materialise_kernel(HERE / "kc5", layout, stabilisation=table)
    source = str(kernel).replace("\\", "/").replace(":", r"\:")
    graph = (
        f"[0:{info.video[0].index}]format=yuv420p,hwupload[t0];"
        f"[0:{info.video[1].index}]format=yuv420p,hwupload[t1];"
        f"[t0][t1]program_opencl=source={source}:kernel=gopromax_equirect:"
        f"inputs=2:size=640x320,hwdownload,format=yuv420p[v]"
    )
    result = subprocess.run(
        [caps.ffmpeg, "-hide_banner", "-v", "error", "-y",
         "-init_hw_device", f"opencl=ocl:{caps.opencl_device}",
         "-filter_hw_device", "ocl", "-hwaccel", "cuda",
         "-ss", "40", "-i", str(src),
         "-filter_complex", graph, "-map", "[v]",
         "-frames:v", str(FRAMES), str(out / "%02d.png")], capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode()[-1500:])
        sys.exit(1)
    return out


def psnr(a, b):
    result = subprocess.run(
        [caps.ffmpeg, "-hide_banner", "-i", str(a), "-i", str(b),
         "-filter_complex", "psnr", "-f", "null", "-"],
        capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "average:" in line:
            value = line.split("average:")[1].split()[0]
            return float("inf") if value == "inf" else float(value)
    return None


cycling = render("cycle", [yaw(60.0 * (k % 3)) for k in range(FRAMES)])
flat = render("flat", [yaw(0.0)] * FRAMES)

print("table de période 3, comparée à une table nulle :")
moved = 0
for k in range(1, FRAMES + 1):
    value = psnr(cycling / f"{k:02d}.png", flat / f"{k:02d}.png")
    expect = 60.0 * ((k - 1) % 3)
    same = value == float("inf")
    if not same:
        moved += 1
    print(f"  image {k - 1}: attendu {expect:3.0f}°  "
          f"{'identique' if same else f'{value:.1f} dB'}")

print()
if moved == 0:
    print("index compte les PLANS : la table est lue trois fois trop vite.")
else:
    print("index compte les IMAGES : le décalage n'est pas là.")
