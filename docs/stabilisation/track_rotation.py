"""Rotation de la camera mesuree par suivi de points + RANSAC sur la sphere.

L'instrument qui a debloque la stabilisation (18 aout). Demande opencv, qui
n'est pas une dependance du projet : `pip install opencv-python-headless` dans
le virtualenv BobVr.

Etalonnage : sur des rotations connues appliquees a une vraie image il rend
1,503 pour 1,5 deg et 5,996 pour 6 deg, avec 91 a 100 % d'inliers. Sur la
video il faut le contraindre au cone avant (35 deg) et un seuil de 0,8 deg,
sinon RANSAC s'accroche a une paroi proche ou la translation ressemble
localement a une rotation -- l'amplitude tombe alors de moitie.

Chaque point suivi donne une correspondance entre deux directions ; une
rotation en explique un sous-ensemble, la parallaxe de translation et le
traineau n'en font pas partie et sont rejetes comme aberrants.
"""
import subprocess
import numpy as np
import cv2

W, H = 960, 480

def frames(path, w=W, h=H):
    raw = subprocess.run(["ffmpeg","-v","error","-i",str(path),
                          "-vf",f"scale={w}:{h},format=gray","-f","rawvideo","-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.uint8)
    n = len(a)//(w*h)
    return a[:n*w*h].reshape(n,h,w).copy()

def to_sphere(pts, w=W, h=H):
    lon = (2*(pts[:,0]+0.5)/w - 1)*np.pi
    lat = (2*(pts[:,1]+0.5)/h - 1)*np.pi/2
    cl = np.cos(lat)
    return np.stack([cl*np.sin(lon), np.sin(lat), cl*np.cos(lon)], 1)

def kabsch(a, b, wgt=None):
    """R tel que R @ a ~ b, pour des vecteurs unitaires."""
    Hm = (a*(1 if wgt is None else wgt[:,None])).T @ b
    U,S,Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1,1,d]) @ U.T

def ransac_rotation(a, b, thresh_deg=0.35, iters=120, rng=None):
    n = len(a)
    if n < 6: return None, None
    rng = rng or np.random.default_rng(0)
    ct = np.cos(np.radians(thresh_deg))
    best_in, best_R = None, None
    idx = rng.integers(0, n, size=(iters, 3))
    for s in idx:
        R = kabsch(a[s], b[s])
        ok = (b*(a@R.T)).sum(1) > ct
        if best_in is None or ok.sum() > best_in.sum():
            best_in, best_R = ok, R
    if best_in is None or best_in.sum() < 6: return None, None
    R = kabsch(a[best_in], b[best_in])
    for _ in range(3):                       # raffinement sur les inliers
        ok = (b*(a@R.T)).sum(1) > ct
        if ok.sum() < 6: break
        R = kabsch(a[ok], b[ok])
    return R, ok

def rotvec(R):
    t = np.clip((np.trace(R)-1)/2, -1, 1)
    ang = np.arccos(t)
    if ang < 1e-9: return np.zeros(3)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return np.degrees(ang * v/(2*np.sin(ang)))

LK = dict(winSize=(21,21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def sequence(fr, mask_fn=None, maxpts=1200):
    """Rend un vecteur rotation par paire d'images, et le taux d'inliers."""
    out, inl = [], []
    for k in range(1, len(fr)):
        m = None if mask_fn is None else mask_fn(k-1)
        p0 = cv2.goodFeaturesToTrack(fr[k-1], maxpts, 0.01, 6, mask=m)
        if p0 is None or len(p0) < 8:
            out.append(np.zeros(3)); inl.append(0.0); continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(fr[k-1], fr[k], p0, None, **LK)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(fr[k], fr[k-1], p1, None, **LK)
        good = (st.ravel()==1) & (st2.ravel()==1) & \
               (np.linalg.norm((p0-p0b).reshape(-1,2), axis=1) < 1.0)
        a = to_sphere(p0.reshape(-1,2)[good]); b = to_sphere(p1.reshape(-1,2)[good])
        R, ok = ransac_rotation(a, b)
        if R is None:
            out.append(np.zeros(3)); inl.append(0.0)
        else:
            out.append(rotvec(R)); inl.append(float(ok.mean()))
    return np.array(out), np.array(inl)
