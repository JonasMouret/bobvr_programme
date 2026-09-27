"""Which telemetry axis drives which movement on screen?

The magnitudes already agree -- 0.55 deg per frame measured against CORI's
0.60 -- so the telemetry does describe the picture. What is missing is the
pairing. Rather than assume one and sweep the other, this correlates every
measured component against every telemetry component, at every plausible time
offset, and prints the table.
"""
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
START_S, SECONDS = 35.0, 12.0
fps = float(info.fps)
start_frame = round(START_S * fps)
BAND = 48

kernel = materialise_kernel(HERE / "kc6", layout)
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
    print(out.stderr.decode()[-1500:]); sys.exit(1)

frames = np.frombuffer(out.stdout, dtype=np.uint8)
frames = frames[: len(frames) // (W * H) * W * H].reshape(-1, H, W).astype(np.float32)
band = frames[:, H // 2 - BAND: H // 2 + BAND, :]
win = np.hanning(band.shape[1])[:, None] * np.ones(W)[None, :]

measured = []
for a, b in zip(band, band[1:]):
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-9
    corr = np.real(np.fft.ifft2(cross))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    peak = corr.max() / (corr.std() + 1e-9)
    if dy > corr.shape[0] // 2:
        dy -= corr.shape[0]
    if dx > W // 2:
        dx -= W
    measured.append((-dx * 2 * np.pi / W, -dy * np.pi / H, peak))
measured = np.array(measured)
strong = measured[:, 2] > np.median(measured[:, 2])
print(f"{len(frames)} images, {strong.sum()} pics de corrélation nets retenus")

tel = telemetry.extract(src, info, caps)
solution = ori.solve_gyro_frame(tel)
times = tel.sensor_times("CORI")
quats = ori.integrate(tel, solution).resample(times)
steps = np.array([ori.to_rotvec(ori.qmul(b, ori.qconj(a)))
                  for a, b in zip(quats, quats[1:])])

names = ("horizontal (lacet)", "vertical (tangage)")
print("corrélation du lacet mesuré avec le meilleur axe, selon le décalage :")
curve = []
for off in range(start_frame - 120, start_frame + 121):
    w = steps[off: off + len(measured)]
    if len(w) < len(measured):
        continue
    r = max(abs(np.corrcoef(w[strong, k], measured[strong, 0])[0, 1])
            for k in range(3))
    curve.append((off - start_frame, r))
for shift, r in curve[::10]:
    bar = "#" * int(r * 60)
    print(f"  {shift:+5d} img  {r:+.3f}  {bar}")
peak = max(curve, key=lambda c: c[1])
flat = sum(r for _, r in curve) / len(curve)
print(f"\npic {peak[1]:.3f} à {peak[0]:+d} ; moyenne {flat:.3f} ; "
      f"rapport pic/moyenne {peak[1] / flat:.2f}")
best = (0.0, None)
for offset in range(start_frame - 120, start_frame + 121):
    window = steps[offset: offset + len(measured)]
    if len(window) < len(measured):
        continue
    total = 0.0
    for m in range(2):
        total += max(abs(np.corrcoef(window[strong, k], measured[strong, m])[0, 1])
                     for k in range(3))
    if total > best[0]:
        best = (total, offset)

offset = best[1]
window = steps[offset: offset + len(measured)]
print(f"\nmeilleur décalage : {offset - start_frame:+d} images "
      f"({(offset - start_frame) / fps:+.3f} s)\n")
print("corrélation mesure / télémétrie          x        y        z")
for m in range(2):
    row = [np.corrcoef(window[strong, k], measured[strong, m])[0, 1] for k in range(3)]
    print(f"  {names[m]:32s} " + "  ".join(f"{v:+.3f}" for v in row))

print("\namplitudes (degrés par image, écart-type) :")
print(f"  mesure horizontale {np.degrees(measured[strong, 0]).std():.3f}   "
      f"verticale {np.degrees(measured[strong, 1]).std():.3f}")
print("  télémétrie x/y/z   " + "  ".join(
    f"{np.degrees(window[strong, k]).std():.3f}" for k in range(3)))
