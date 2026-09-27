"""Search the axis map on residual rotation, not on pixel difference.

Pixel difference was the wrong instrument: the camera turns about 0.6 degrees
a frame, which is 0.8 px at this width, while the sled advances nearly a metre
down a tunnel in the same time. Translation swamps rotation, so removing the
rotation barely moves the number and every resampling artefact makes it worse.

Measuring the output's own rotation instead answers the question directly. A
locked view should show almost none; the unstabilised render is the control.
"""
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

from bobvr import orientation as ori, telemetry
from bobvr.media import probe
from bobvr.render.caps import detect
from bobvr.render.geometry import MaxLayout, materialise_kernel

HERE = Path(__file__).parent
src = Path(sys.argv[1])
caps = detect()
info = probe(src, caps)
layout = MaxLayout(src_w=info.track_width, src_h=info.track_height)

W, H = 512, 256
START_S, SECONDS = 60.0, 4.0
fps = float(info.fps)
start = round(START_S * fps)
count = int(SECONDS * fps) + 2
BAND = 48
DELTA = int(sys.argv[2]) if len(sys.argv) > 2 else -30

tel = telemetry.extract(src, info, caps)
solution = ori.solve_gyro_frame(tel)
quats = ori.integrate(tel, solution).resample(tel.sensor_times("CORI"))
window = quats[start + DELTA: start + DELTA + count]
win2d = np.hanning(2 * BAND)[:, None] * np.ones(W)[None, :]


def residual_yaw(stab):
    """Degrees of yaw per frame still visible in the output."""
    kernel = materialise_kernel(HERE / "kc8", layout, stabilisation=stab)
    source = str(kernel).replace("\\", "/").replace(":", r"\:")
    graph = (
        f"[0:{info.video[0].index}]format=yuv420p,hwupload[t0];"
        f"[0:{info.video[1].index}]format=yuv420p,hwupload[t1];"
        f"[t0][t1]program_opencl=source={source}:kernel=gopromax_equirect:"
        f"inputs=2:size={W}x{H},hwdownload,format=yuv420p[v]"
    )
    out = subprocess.run(
        [caps.ffmpeg, "-hide_banner", "-v", "error",
         "-init_hw_device", f"opencl=ocl:{caps.opencl_device}",
         "-filter_hw_device", "ocl", "-hwaccel", "cuda",
         "-ss", f"{START_S}", "-i", str(src), "-t", f"{SECONDS}",
         "-filter_complex", graph, "-map", "[v]",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True)
    if out.returncode != 0:
        print(out.stderr.decode()[-1200:]); sys.exit(1)
    f = np.frombuffer(out.stdout, dtype=np.uint8)
    f = f[: len(f) // (W * H) * W * H].reshape(-1, H, W).astype(np.float32)
    b = f[:, H // 2 - BAND: H // 2 + BAND, :]
    shifts = []
    for p, q in zip(b, b[1:]):
        fa = np.fft.fft2((p - p.mean()) * win2d)
        fb = np.fft.fft2((q - q.mean()) * win2d)
        cross = fa * np.conj(fb)
        cross /= np.abs(cross) + 1e-9
        corr = np.real(np.fft.ifft2(cross))
        _, dx = np.unravel_index(np.argmax(corr), corr.shape)
        if dx > W // 2:
            dx -= W
        shifts.append(abs(dx) * 360.0 / W)
    return float(np.mean(shifts))


plain = residual_yaw(None)
print(f"sans stabilisation : {plain:.3f}°/image de lacet résiduel\n")

lock = [ori.qmul(q, ori.qconj(window[0])) for q in window]
results = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1.0, -1.0), repeat=3):
        table = []
        for w, x, y, z in lock:
            u = (x, y, z)
            table.append(ori.positive(ori.qnorm(
                (w, *(s * u[p] for p, s in zip(perm, signs))))))
        tag = "".join("xyz"[p].upper() if s > 0 else "xyz"[p]
                      for p, s in zip(perm, signs))
        results.append((residual_yaw(table), tag))

results.sort()
print("lacet résiduel après verrouillage, par carte d'axes :")
for score, tag in results[:8]:
    print(f"  {tag}   {score:.3f}°   {(score - plain) / plain * 100:+6.1f} %")
print("  ...")
for score, tag in results[-2:]:
    print(f"  {tag}   {score:.3f}°   {(score - plain) / plain * 100:+6.1f} %")
