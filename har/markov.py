"""Discrete pairwise Markov network (factor graph) for P(activity | sensor features)."""
import numpy as np
from .features import CODES

class MarkovNet:
    def __init__(self, K=8, alpha=1.0, pair_weight=0.5, temp=1.0):
        self.K, self.alpha, self.pair_weight, self.temp = K, alpha, pair_weight, temp

    def discretize(self, F):
        return np.stack([np.searchsorted(self.cuts[i], F[:, i]) for i in range(F.shape[1])], 1)

    def fit(self, F, y):
        K, a = self.K, self.alpha
        self.classes = [c for c in CODES if c in set(y)]; C = len(self.classes); d = F.shape[1]
        self.cuts = [np.quantile(F[:, i], np.linspace(0, 1, K + 1)[1:-1]) for i in range(d)]
        X, yi = self.discretize(F), np.array([self.classes.index(c) for c in y])
        self.prior = np.bincount(yi, minlength=C) / len(yi)
        self.unary, P1 = [], []                                  # factor phi_i(x_i, Y)
        for i in range(d):
            T = np.stack([np.bincount(X[yi == c, i], minlength=K) + a for c in range(C)])
            T = T / T.sum(1, keepdims=True); P1.append(T); self.unary.append(np.log(T))
        # structure: class-conditional mutual information -> maximum spanning tree (Chow-Liu)
        J = {}
        for i in range(d):
            for j in range(i + 1, d):
                P = np.stack([np.histogram2d(X[yi == c, i], X[yi == c, j], bins=[range(K + 1)] * 2)[0] + a for c in range(C)])
                J[(i, j)] = P / P.sum((1, 2), keepdims=True)
        mi = {e: sum(self.prior[c] * (J[e][c] * np.log(J[e][c] / np.outer(P1[e[0]][c], P1[e[1]][c]))).sum() for c in range(C)) for e in J}
        comp, self.edges = list(range(d)), []
        for e in sorted(mi, key=mi.get, reverse=True):
            if comp[e[0]] != comp[e[1]]:
                old, new = comp[e[1]], comp[e[0]]; comp = [new if c == old else c for c in comp]; self.edges.append(e)
        # pairwise factor phi_ij(x_i, x_j, Y): correction over independent features
        self.pair = {e: np.log(J[e] / (P1[e[0]][:, :, None] * P1[e[1]][:, None, :])).transpose(1, 2, 0) for e in self.edges}
        return self

    def log_scores(self, F):
        X = self.discretize(np.atleast_2d(F))
        S = np.tile(np.log(self.prior), (len(X), 1))
        for i in range(X.shape[1]): S += self.unary[i][:, X[:, i]].T
        for (i, j), c in self.pair.items(): S += self.pair_weight * c[X[:, i], X[:, j]]
        return S

    def proba(self, F):
        S = self.log_scores(F) / self.temp; S = np.exp(S - S.max(1, keepdims=True))
        return S / S.sum(1, keepdims=True)

    def predict(self, F):
        return [self.classes[i] for i in self.proba(F).argmax(1)]

    def gibbs(self, f, n=1000, seed=0):
        """Gibbs sampling of the joint P(Y, X): resample Y | X, then each X_i | Y, neighbours."""
        rng = np.random.default_rng(seed); x = self.discretize(np.atleast_2d(f))[0].copy(); d = len(x)
        nb = {i: [] for i in range(d)}
        for (i, j), c in self.pair.items(): nb[i].append((j, c, 0)); nb[j].append((i, c, 1))
        out = np.zeros(len(self.classes))
        for _ in range(n):
            S = self.log_scores(x[None])[0] / self.temp; p = np.exp(S - S.max()); p /= p.sum(); y = rng.choice(len(p), p=p); out[y] += 1
            for i in range(d):
                lp = self.unary[i][y].copy()
                for j, c, side in nb[i]:
                    lp += self.pair_weight * (c[:, x[j], y] if side == 0 else c[x[j], :, y])
                p = np.exp(lp - lp.max()); x[i] = rng.choice(self.K, p=p / p.sum())
        return dict(zip(self.classes, out / n))
