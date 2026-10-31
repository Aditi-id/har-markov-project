"""Constants + orientation-invariant window features (work for any phone orientation)."""
import numpy as np
FS, WIN, HOP = 50, 128, 50            # 50 Hz, 2.56 s windows, one prediction per second
CODES = ["sit", "std", "wlk", "ups", "dws", "jog"]
NAMES = dict(sit="Sitting", std="Standing", wlk="Walking", ups="Upstairs", dws="Downstairs", jog="Running")
EMOJI = dict(sit="🪑", std="🧍", wlk="🚶", ups="🪜", dws="⬇️", jog="🏃")
COLORS = dict(sit="#6b7280", std="#0ea5e9", wlk="#22c55e", ups="#f59e0b", dws="#a855f7", jog="#ef4444")

def window_features(w):
    """w: (N,6) = accelerometer incl. gravity [g] (3) + gyroscope [rad/s] (3)."""
    a, g = w[:, :3], w[:, 3:6]
    grav = a.mean(0); u = grav / (np.linalg.norm(grav) + 1e-9)
    dyn = a - grav
    vert = dyn @ u
    hor = np.linalg.norm(dyn - np.outer(vert, u), axis=1)
    am, gm = np.linalg.norm(a, axis=1), np.linalg.norm(g, axis=1)
    sp = np.abs(np.fft.rfft(vert - vert.mean()))[1:]
    fr = np.fft.rfftfreq(len(vert), 1 / FS)[1:]
    k = int(sp.argmax())
    return np.array([am.mean(), am.std(), am.max() - am.min(), vert.std(), hor.mean(), fr[k],
                     sp[k] / (sp.sum() + 1e-9), (np.diff(np.sign(vert - vert.mean())) != 0).mean(),
                     gm.mean(), gm.std(), np.abs(np.diff(am)).mean() * FS])

def windows(x, hop=HOP):
    return np.array([window_features(x[i:i + WIN]) for i in range(0, len(x) - WIN + 1, hop)])
