"""Data: real MotionSense (has running, phone-compatible sensors) with a synthetic fallback."""
import io, os, zipfile, urllib.request
import numpy as np, pandas as pd
from .features import FS, CODES
URL = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"

def load_motionsense(folder="data"):
    """Returns list of (subject, code, array[N,6]) or None if the download is impossible."""
    root = os.path.join(folder, "A_DeviceMotion_data")
    try:
        if not os.path.isdir(root):
            raw = urllib.request.urlopen(URL, timeout=30).read()
            zipfile.ZipFile(io.BytesIO(raw)).extractall(folder)
        if os.path.isdir(os.path.join(root, "A_DeviceMotion_data")): root = os.path.join(root, "A_DeviceMotion_data")
    except Exception as e:
        print("  MotionSense download failed:", repr(e)[:90]); return None
    recs = []
    for d in sorted(os.listdir(root)):
        code = d.split("_")[0]
        if code not in CODES or not os.path.isdir(os.path.join(root, d)): continue
        for f in os.listdir(os.path.join(root, d)):
            df = pd.read_csv(os.path.join(root, d, f))
            acc = df[["gravity.x", "gravity.y", "gravity.z"]].values + df[["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]].values
            gyr = df[["rotationRate.x", "rotationRate.y", "rotationRate.z"]].values
            recs.append((int(f.split("_")[1].split(".")[0]), code, np.hstack([acc, gyr])))
    return recs or None

_P = dict(sit=(0, 0, .004, .01), std=(0, 0, .010, .03), wlk=(1.9, .25, .02, .9),
          ups=(1.6, .30, .03, 1.0), dws=(2.2, .42, .03, 1.3), jog=(2.8, .95, .05, 2.6))

def synthetic(seed=0, subjects=14, seconds=40):
    """Physically-motivated fake data - only for testing the pipeline."""
    rng = np.random.default_rng(seed); recs = []
    for s in range(1, subjects + 1):
        gdir = rng.normal(size=3); gdir /= np.linalg.norm(gdir)     # random phone orientation
        v = rng.normal(size=3); v -= v @ gdir * gdir; v /= np.linalg.norm(v)
        k = rng.uniform(.85, 1.15)
        for c, (f, A, noise, gy) in _P.items():
            t = np.arange(seconds * FS) / FS; f_, ph = f * k * rng.uniform(.95, 1.05), rng.uniform(0, 6.28)
            vert = A * k * (np.sin(2 * np.pi * f_ * t + ph) + .4 * np.sin(4 * np.pi * f_ * t + 2 * ph)) if f else 0 * t
            hor = .35 * A * np.sin(np.pi * f_ * t) if f else 0 * t
            x = gdir[None] + np.outer(vert, gdir) + np.outer(hor, v) + rng.normal(0, noise + .01, (len(t), 3))
            gyro = gy * k * np.sin(2 * np.pi * f_ * t[:, None] + rng.uniform(0, 6, 3)) + rng.normal(0, .02, (len(t), 3))
            recs.append((s, c, np.hstack([x, gyro])))
    return recs
