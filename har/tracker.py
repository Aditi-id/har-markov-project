"""Sliding window -> Markov network -> HMM-style temporal filter -> activity log."""
import numpy as np
from .features import WIN, HOP, FS, window_features

class Tracker:
    def __init__(self, model, stay=0.85):
        self.m, self.stay = model, stay; self.reset()

    def reset(self):
        self.buf, self.log, self.belief, self.n_win = [], [], None, 0

    def push(self, samples, truth=None):
        self.buf.extend(np.asarray(samples, float).tolist())
        while len(self.buf) >= WIN:
            w = np.array(self.buf[:WIN]); del self.buf[:HOP]
            p = self.m.proba(window_features(w)[None])[0]; C = len(p)
            if self.belief is None: self.belief = np.ones(C) / C
            T = np.full((C, C), (1 - self.stay) / (C - 1)); np.fill_diagonal(T, self.stay)
            b = (T.T @ self.belief) * p / self.m.prior          # predict, then weight by likelihood
            self.belief = b / b.sum(); t = self.n_win * HOP / FS; self.n_win += 1
            self.log.append(dict(t0=t, t1=t + HOP / FS, raw=self.m.classes[p.argmax()],
                                 smooth=self.m.classes[self.belief.argmax()], conf=float(self.belief.max()),
                                 probs=self.belief.copy(), truth=truth))
