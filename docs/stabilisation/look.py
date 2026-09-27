"""Render the same three moments with and without a lock, to look at them."""
import subprocess
import sys
from pathlib import Path

from bobvr import orientation as ori, telemetry
from bobvr.media import probe
from bobvr.render.caps import detect
from bobvr.render.geometry import MaxLayout, materialise_kernel

HERE = Path(__file__).parent
src = Path(sys.argv[1])
caps = detect()
info = probe(src, caps)
layout = MaxLayout(src_w=info.track_width, src_h=info.track_height)

W, H = 960, 480
START_S, SECONDS = 47.7, 1.5
fps = float(info.fps)
start = round(START_S * fps)
span = int(SECONDS * fps) + 2

tel = telemetry.extract(src, info, caps)
solution = ori.solve_gyro_frame(tel)
quats = ori.integrate(tel, solution).resample(tel.sensor_times("CORI"))
window = quats[start: start + span]
import math
print(f"rotation sur la fenêtre : "
      f"{math.degrees(ori.qangle(ori.qmul(window[-1], ori.qconj(window[0])))):.1f}°")


def render(tag, stab):
    out = HERE / f"look_{tag}"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    kernel = materialise_kernel(HERE / "kcL", layout, stabilisation=stab)
    source = str(kernel).replace("\\", "/").replace(":", r"\:")
    graph = (
        f"[0:{info.video[0].index}]format=yuv420p,hwupload[t0];"
        f"[0:{info.video[1].index}]format=yuv420p,hwupload[t1];"
        f"[t0][t1]program_opencl=source={source}:kernel=gopromax_equirect:"
        f"inputs=2:size={W}x{H},hwdownload,format=yuv420p[v];"
        f"[v]select='eq(n\\,0)+eq(n\\,22)+eq(n\\,43)'[out]"
    )
    r = subprocess.run(
        [caps.ffmpeg, "-hide_banner", "-v", "error", "-y",
         "-init_hw_device", f"opencl=ocl:{caps.opencl_device}",
         "-filter_hw_device", "ocl", "-hwaccel", "cuda",
         "-ss", f"{START_S}", "-i", str(src), "-t", f"{SECONDS}",
         "-filter_complex", graph, "-map", "[out]", "-vsync", "0",
         str(out / "%d.png")], capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode()[-1200:]); sys.exit(1)
    print(f"  {tag}: {sorted(p.name for p in out.glob('*.png'))}")
    return out


render("plain", None)
lock = [ori.positive(ori.qmul(q, ori.qconj(window[0]))) for q in window]
render("lock", lock)
