"""Gyro-integrated orientation for a .360 run, with the frame solved from the file."""
import json
import sys
from pathlib import Path

import numpy as np

from bobvr import telemetry
from bobvr.media import probe
from bobvr.render.caps import detect

path = Path(sys.argv[1])
out = Path(sys.argv[2])
caps = detect()
info = probe(path, caps)
tel = telemetry.extract(path, info, caps)

t_g = np.array(tel.sensor_times("GYRO"))
gyro = np.array(tel.flatten("GYRO"), dtype=float)
t_c = np.array(tel.sensor_times("CORI"))
cori = np.array(tel.flatten("CORI"), dtype=float)
fps = float(info.fps)


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def rotvec(q):
    q = q * np.sign(q[..., :1] + 1e-30)
    n = np.linalg.norm(q[..., 1:], axis=-1)
    ang = 2 * np.arctan2(n, q[..., 0])
    return q[..., 1:] * np.where(n > 1e-12, ang / np.where(n > 1e-12, n, 1), 2.0)[..., None]


def angle(q):
    return 2 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


def integrate_window(w, lo_t, hi_t):
    lo, hi = np.searchsorted(t_g, (lo_t, hi_t))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for i in range(lo, min(hi, len(w) - 1)):
        th = np.linalg.norm(w[i]) * (t_g[i + 1] - t_g[i])
        if th < 1e-12:
            continue
        ax = w[i] / np.linalg.norm(w[i])
        q = qmul(q, np.concatenate([[np.cos(th / 2)], ax * np.sin(th / 2)]))
    return q / np.linalg.norm(q)


# ------------------------------- solve the gyro -> CORI frame from the clip

N = 15
starts = np.arange(0, len(t_c) - N, N)
# CORI composes extrinsically: the body-frame increment is q_k+N (x) conj(q_k).
target = rotvec(qmul(cori[starts + N], qconj(cori[starts])))
source = np.array([rotvec(integrate_window(gyro, t_c[k], t_c[k + N]))
                   for k in starts])
keep = np.linalg.norm(target, axis=1) > np.radians(5)
m, *_ = np.linalg.lstsq(source[keep], target[keep], rcond=None)
m = m.T

snapped = np.zeros((3, 3))
for r in range(3):
    c = int(np.argmax(np.abs(m[r])))
    snapped[r, c] = np.sign(m[r, c])

for label, mat in (("least squares", m), ("snapped", snapped)):
    err = np.median(np.linalg.norm((mat @ source[keep].T).T - target[keep], axis=1)
                    / np.linalg.norm(target[keep], axis=1))
    print(f"{label:14s} residual {err:.1%}")
    if label == "snapped":
        snapped_resid = err
print("gyro -> CORI frame, snapped to a signed permutation:")
for row in snapped:
    print("   " + "  ".join(f"{v:+.0f}" for v in row))
print(f"declared by ORIN/ORIO: {tel.streams['GYRO'][0].axes} -> "
      f"{tel.streams['GYRO'][0].axis_plan}")

aligned = (snapped @ gyro.T).T

# -------------------------------------------- integrate the whole run

q = np.empty((len(t_g), 4))
q[0] = cori[0]
for i in range(1, len(t_g)):
    w = aligned[i - 1]
    th = np.linalg.norm(w) * (t_g[i] - t_g[i - 1])
    if th < 1e-12:
        q[i] = q[i - 1]
        continue
    ax = w / np.linalg.norm(w)
    dq = np.concatenate([[np.cos(th / 2)], ax * np.sin(th / 2)])
    q[i] = qmul(dq, q[i - 1])          # extrinsic, matching CORI
    q[i] /= np.linalg.norm(q[i])

idx = np.searchsorted(t_g, t_c).clip(0, len(q) - 1)
q_frame = q[idx]
drift = np.degrees(angle(qmul(q_frame, qconj(cori))))
print(f"\ndrift vs CORI: {drift[len(drift) // 2]:.1f} deg at mid-run, "
      f"{drift[-1]:.1f} deg at the end ({drift[-1] / t_c[-1] * 60:.1f} deg/min)")

# ------------------------------------- what a stabiliser would have to cancel


def hemisphere(qq):
    qq = qq.copy()
    flip = np.cumprod(np.where(np.sum(qq[:-1] * qq[1:], axis=-1) < 0, -1.0, 1.0))
    qq[1:] *= flip[:, None]
    return qq


def smooth(quats, passes):
    """Zero-phase binomial smoothing: symmetric, so it introduces no lag."""
    s = hemisphere(quats).copy()
    for _ in range(passes):
        mid = s.copy()
        mid[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
        s = mid / np.linalg.norm(mid, axis=-1, keepdims=True)
    return s


qh = hemisphere(q_frame)
residuals, stats = {}, {}
for label, seconds in (("0.5 s", 0.5), ("1 s", 1.0), ("2 s", 2.0)):
    sm = smooth(qh, max(1, int(round(2 * (seconds * fps) ** 2))))
    r = np.degrees(angle(qmul(sm, qconj(qh))))
    residuals[label] = r
    stats[label] = [float(np.median(r)), float(np.percentile(r, 95)), float(r.max())]
    print(f"residual to cancel, {label} smoother: median {stats[label][0]:.2f} deg, "
          f"p95 {stats[label][1]:.2f}, max {stats[label][2]:.2f}")

# ------------------------------------------- how much lives above CORI's ceiling

wmag = np.linalg.norm(gyro, axis=1)
dt = float(np.median(np.diff(t_g)))
power = np.abs(np.fft.rfft(wmag - wmag.mean())) ** 2
freq = np.fft.rfftfreq(len(wmag), d=dt)
above = power[freq > 15.0].sum() / power[freq > 0].sum()
print(f"\ngyro power above 15 Hz (CORI's Nyquist): {above:.1%}")

octaves, edges = [], [0.5, 2, 5, 10, 15, 30, 60, 120, 400]
for lo, hi in zip(edges, edges[1:]):
    band = (freq >= lo) & (freq < hi)
    octaves.append([lo, hi, float(power[band].sum() / power[freq > 0].sum())])
    print(f"   {lo:5.1f}-{hi:5.1f} Hz : {octaves[-1][2]:6.1%}")


# Euler angles are not usable to plot this run: pitch passes 60 degrees for a
# quarter of it, so the ZYX yaw and roll trade places near gimbal lock and end
# up counting the same physical turn twice (both reach ~1270 deg). Everything
# below is representation-free instead.
print(f"\npitch above 60 deg on {(np.abs(np.degrees(np.arcsin(np.clip(2 * (qh[:, 0] * qh[:, 2] - qh[:, 3] * qh[:, 1]), -1, 1)))) > 60).mean():.1%} "
      "of the run -- why Euler angles are not plotted")

# Angle from the orientation the clip started at: one curve, no convention.
from_start_g = np.degrees(angle(qmul(qh, qconj(qh[0]))))
from_start_c = np.degrees(angle(qmul(hemisphere(cori), qconj(cori[0]))))

# Total turning: the integral of |w|, which only ever grows.
turned = np.degrees(np.cumsum(wmag[:-1] * np.diff(t_g)))

step = 3
BINS = 540  # ~0.3 s per bin: keeps the peaks a mean would flatten


def envelope(values, times, fn):
    edges = np.linspace(times[0], times[-1], BINS + 1)
    idx = np.clip(np.searchsorted(edges, times) - 1, 0, BINS - 1)
    out = np.zeros(BINS)
    for b in range(BINS):
        sel = values[idx == b]
        if len(sel):
            out[b] = fn(sel)
    return (edges[:-1] + edges[1:]) / 2, out


bin_t, w_peak = envelope(np.degrees(wmag), t_g, np.max)
_, w_typ = envelope(np.degrees(wmag), t_g, np.median)

payload = {
    "clip": path.name, "duration": info.duration, "fps": fps,
    "gyro_hz": len(gyro) / (t_g[-1] - t_g[0]),
    "clock_offset_us": tel.clock_offset_us,
    "clock_spread_us": tel.clock_offset_spread_us(),
    "declared_axes": list(tel.streams["GYRO"][0].axes),
    "declared_plan": tel.streams["GYRO"][0].axis_plan,
    "solved_matrix": snapped.tolist(),
    "solved_residual": float(snapped_resid),
    "drift_end": float(drift[-1]), "drift_mid": float(drift[len(drift) // 2]),
    "drift_max": float(drift.max()),
    "above_15hz": float(above), "bands": octaves,
    "residual_stats": stats,
    "total_turn": float(turned[-1]),
    "t": t_c[::step].tolist(),
    "from_start": from_start_g[::step].tolist(),
    "from_start_cori": from_start_c[::step].tolist(),
    "drift": drift[::step].tolist(),
    "res_half": residuals["0.5 s"][::step].tolist(),
    "res_two": residuals["2 s"][::step].tolist(),
    "rate_t": bin_t.tolist(),
    "rate_peak": w_peak.tolist(),
    "rate_typ": w_typ.tolist(),
}
out.write_text(json.dumps(payload))
print(f"\nwrote {out}")
