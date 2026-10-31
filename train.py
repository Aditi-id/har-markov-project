"""python train.py            -> real MotionSense if downloadable, else synthetic
   python train.py --synthetic -> force synthetic (pipeline test only)"""
import os, sys, time, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from har import data as D
from har.features import windows, CODES
from har.markov import MarkovNet

def _with(m, T):
    import copy; c = copy.copy(m); c.temp = T; return c

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv; t0 = time.time()
    os.makedirs("models", exist_ok=True)
    recs = None if "--synthetic" in argv else D.load_motionsense()
    synthetic = recs is None
    if synthetic:
        print("!! Using SYNTHETIC data (pipeline test). Run with internet for real MotionSense.")
        recs = D.synthetic()

    # ---- split by person (no leakage)
    subs = sorted({r[0] for r in recs}); rng = np.random.default_rng(0); rng.shuffle(subs)
    nt, nv = max(2, len(subs) // 4), max(2, len(subs) // 6)
    sp = dict(test=subs[:nt], val=subs[nt:nt + nv], train=subs[nt + nv:])

    def build(S):
        F, y = [], []
        for s, c, x in recs:
            if s in S and len(x) >= 128:
                f = windows(x); F.append(f); y += [c] * len(f)
        return np.vstack(F), np.array(y)

    (Ftr, ytr), (Fva, yva), (Fte, yte) = build(sp["train"]), build(sp["val"]), build(sp["test"])
    print(f"windows train {len(ytr)} val {len(yva)} test {len(yte)} | subjects split by person")

    def nll(m, F, y):
        P = m.proba(F)
        return -np.mean(np.log(P[np.arange(len(y)), [m.classes.index(c) for c in y]] + 1e-9))

    # ---- tune pairwise weight + temperature on validation people
    best = None
    for w in (.25, .5, .75, 1.):                  # w=0 would be plain naive Bayes
        m = MarkovNet(pair_weight=w).fit(Ftr, ytr)
        ll, T = min(((nll(_with(m, T_), Fva, yva), T_) for T_ in (1, 2, 4, 8, 16, 32)), key=lambda r: r[0])
        print(f"  pairwise weight {w:.2f}: val acc {accuracy_score(yva, m.predict(Fva))*100:5.1f}%  "
              f"calibrated log-loss {ll:.3f} (T={T})")
        if best is None or ll < best[0]: best = (ll, w, T)

    Fall, yall = np.vstack([Ftr, Fva]), np.concatenate([ytr, yva])
    mn = MarkovNet(pair_weight=best[1], temp=best[2]).fit(Fall, yall)
    nb = MarkovNet(pair_weight=0, temp=best[2]).fit(Fall, yall)
    rf = RandomForestClassifier(200, random_state=0).fit(Ftr, ytr)

    res = {}
    for n, p in (("Bayesian network (naive Bayes)", nb.predict(Fte)),
                 ("Markov network", mn.predict(Fte)),
                 ("Random forest (reference)", rf.predict(Fte))):
        res[n] = dict(acc=accuracy_score(yte, p), f1=f1_score(yte, p, average="macro"))
        print(f"  {n:32s} acc {res[n]['acc']*100:5.1f}%  macro-F1 {res[n]['f1']:.3f}")
    print(classification_report(yte, mn.predict(Fte), digits=3))

    # ---- demo stream: up to 25 s of every activity from one unseen subject
    demo = []
    for c in np.random.default_rng(1).permutation(CODES):
        for subj in sp["test"] + sp["val"]:
            hit = [x for s, cc, x in recs if s == subj and cc == c and len(x) >= 500]
            if hit:
                seg = hit[0][:1250]                                   # may be shorter than 1250
                demo.append(np.hstack([seg, np.full((len(seg), 1), CODES.index(c))]))
                break

    mn.meta = dict(source="synthetic" if synthetic else "MotionSense", results=res,
                   edges=[(int(i), int(j)) for i, j in mn.edges], pair_weight=best[1])
    joblib.dump(dict(model=mn, demo=np.vstack(demo)), "models/har_model.joblib")
    print(f"saved models/har_model.joblib ({time.time()-t0:.0f}s) | pairwise weight {best[1]}")

if __name__ == "__main__":
    main()