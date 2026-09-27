"""Does the real table actually steady the picture?

Renders the same three seconds with and without stabilisation and measures how
much consecutive output frames differ. Rotation is most of what changes
between frames on a bobsleigh run, so removing it should drop the figure
plainly. Also sweeps the table offset, which is the open question the file
does not answer: VPTS places the telemetry on the video clock, but not whether
it marks the first or the last sample of its payload.
"""
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

W, H = 640, 320
START_S = 60.0
SECONDS = 3.0
fps = float(info.fps)
start_frame = round(START_S * fps)

tel = telemetry.extract(src, info, caps)
frame = ori.solve_gyro_frame(tel)
print(f"repère {frame.matrix}, résidu {frame.residual:.1%}")

times = tel.sensor_times("CORI")
table = ori.stabilise(tel, times, 0.5, frame)
print(f"table : {len(table)} images")


def render(stab, offset):
    kernel = materialise_kernel(HERE / "kcache2", layout,
                                stabilisation=stab, frame_offset=offset)
    source = str(kernel).replace("\\", "/").replace(":", r"\:")
    graph = (
        f"[0:{info.video[0].index}]format=yuv420p,hwupload[t0];"
        f"[0:{info.video[1].index}]format=yuv420p,hwupload[t1];"
        f"[t0][t1]program_opencl=source={source}:kernel=gopromax_equirect:"
        f"inputs=2:size={W}x{H},hwdownload,format=yuv420p[v]"
    )
    cmd = [caps.ffmpeg, "-hide_banner", "-v", "error",
           "-init_hw_device", f"opencl=ocl:{caps.opencl_device}",
           "-filter_hw_device", "ocl", "-hwaccel", "cuda",
           "-ss", f"{START_S}", "-i", str(src), "-t", f"{SECONDS}",
           "-filter_complex", graph, "-map", "[v]",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        print(out.stderr.decode()[-1500:])
        sys.exit(1)
    return out.stdout


def unsteadiness(raw):
    """Mean absolute difference between consecutive frames, in grey levels."""
    size = W * H
    frames = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    total, count = 0, 0
    for a, b in zip(frames, frames[1:]):
        total += sum(abs(x - y) for x, y in zip(a, b))
        count += size
    return total / count, len(frames)


plain, frames = unsteadiness(render(None, 0))
print(f"\n{frames} images comparées, {W}x{H}")
print(f"  sans stabilisation        {plain:6.2f} niveaux de gris")

identity = [ori.IDENTITY] * len(table)
control, _ = unsteadiness(render(identity, start_frame))
print(f"  table à l'identité        {control:6.2f}   (témoin : doit égaler ci-dessus)")

flipped = [ori.qconj(q) for q in table]
for name, tab in (("q x conj(s)", table), ("conj(q) x s", flipped)):
    score, _ = unsteadiness(render(tab, start_frame))
    delta = (score - plain) / plain * 100
    print(f"  {name:24s}{score:6.2f}   {delta:+5.1f} %")
