"""Judge a lock by how well it holds the view on the FIRST frame.

Frame-to-frame measures were hopeless: the rotation is under a pixel a frame
and the sled's forward motion swamps it. Over a whole window the rotation is
tens of degrees, which is an enormous signal by comparison -- so the question
becomes "does frame k still show what frame 0 showed", and a correct axis map
should stand out plainly.

Run on a stretch before the sled gets going, where the camera turns but does
not travel, so parallax does not confound the comparison.
"""
import itertools
import math
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
SECONDS = 2.0
fps = float(info.fps)

tel = telemetry.extract(src, info, caps)
solution = ori.solve_gyro_frame(tel)
quats = ori.integrate(tel, solution).resample(tel.sensor_times("CORI"))

# Pick a window that turns a decent amount: a lock has nothing to prove on a
# stretch that was already still.
span = int(SECONDS * fps)
turns = sorted(
    ((math.degrees(ori.qangle(ori.qmul(quats[k + span], ori.qconj(quats[k])))), k)
     for k in range(30, min(len(quats) - span - 2, int(28 * fps)), 15)),
    reverse=True)
best_turn, start = turns[0]
START_S = start / fps
print(f"fenêtre à {START_S:.1f} s : la caméra y tourne de {best_turn:.1f}° "
      f"en {SECONDS} s (avant le lancement)")
window = quats[start: start + span + 2]


def lock_quality(stab):
    """Mean correlation of each frame with the first one."""
    kernel = materialise_kernel(HERE / "kc9", layout, stabilisation=stab)
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
    band = f[:, H // 4: 3 * H // 4, :].reshape(len(f), -1)
    band = band - band.mean(axis=1, keepdims=True)
    ref = band[0]
    denom = np.linalg.norm(ref) * np.linalg.norm(band, axis=1) + 1e-9
    return float((band @ ref / denom)[1:].mean())


plain = lock_quality(None)
print(f"sans stabilisation : corrélation à l'image 0 = {plain:.4f}\n")

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
        results.append((lock_quality(table), tag))

results.sort(reverse=True)
print("qualité du verrouillage (plus haut = mieux tenu) :")
for score, tag in results[:8]:
    print(f"  {tag}   {score:.4f}   {(score - plain) / abs(plain) * 100:+7.1f} %")
print("  ...")
for score, tag in results[-2:]:
    print(f"  {tag}   {score:.4f}   {(score - plain) / abs(plain) * 100:+7.1f} %")
