"""Does program_opencl's `index` really count frames?

Renders the same six source frames twice: once with a table that turns further
on every frame, once with a table that holds the first rotation throughout. If
`index` counts, frame 0 matches between the two and the rest diverge. If it is
stuck, every pair matches.
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
    """Rotation of `degrees` about y, as (w, x, y, z)."""
    half = math.radians(degrees) / 2
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def render(tag, table):
    out = HERE / f"idx_{tag}"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    kernel = materialise_kernel(HERE / "kcache", layout, stabilisation=table)
    source = str(kernel).replace("\\", "/").replace(":", r"\:")
    graph = (
        f"[0:{info.video[0].index}]format=yuv420p,hwupload[t0];"
        f"[0:{info.video[1].index}]format=yuv420p,hwupload[t1];"
        f"[t0][t1]program_opencl=source={source}:kernel=gopromax_equirect:"
        f"inputs=2:size=1280x640,hwdownload,format=yuv420p[v]"
    )
    cmd = [caps.ffmpeg, "-hide_banner", "-v", "error", "-y",
           "-init_hw_device", f"opencl=ocl:{caps.opencl_device}",
           "-filter_hw_device", "ocl", "-hwaccel", "cuda",
           "-ss", "40", "-i", str(src),
           "-filter_complex", graph, "-map", "[v]",
           "-frames:v", str(FRAMES), str(out / "%02d.png")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED ({tag}):\n{result.stderr.strip()[-2000:]}")
        sys.exit(1)
    print(f"  {tag}: kernel {kernel.stat().st_size / 1024:.0f} Ko, "
          f"{len(list(out.glob('*.png')))} images")
    return out


def psnr(a, b):
    result = subprocess.run(
        [caps.ffmpeg, "-hide_banner", "-v", "info", "-i", str(a), "-i", str(b),
         "-filter_complex", "psnr", "-f", "null", "-"],
        capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "average:" in line:
            value = line.split("average:")[1].split()[0]
            return float("inf") if value == "inf" else float(value)
    return None


print(f"{src.name}: {info.track_width}x{info.track_height} par piste")
turning = render("turning", [yaw(20.0 * k) for k in range(FRAMES)])
still = render("still", [yaw(0.0)] * FRAMES)

print("\nPSNR entre les deux rendus, image par image (inf = identiques) :")
verdict = []
for k in range(1, FRAMES + 1):
    value = psnr(turning / f"{k:02d}.png", still / f"{k:02d}.png")
    verdict.append(value)
    print(f"  image {k - 1}: table à {20 * (k - 1):3.0f}°  PSNR "
          f"{'identique' if value == float('inf') else f'{value:.1f} dB'}")

if verdict[0] != float("inf"):
    print("\nINATTENDU : l'image 0 devrait être identique, les deux tables y "
          "portent la même rotation.")
elif all(v == float("inf") for v in verdict):
    print("\nindex NE COMPTE PAS : chaque image lit la même entrée.")
else:
    print("\nindex compte bien les images : la table est lue par image.")
