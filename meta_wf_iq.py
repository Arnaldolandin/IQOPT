# meta_wf_iq.py - Walk-forward OPERABLE (gap=1) del meta-labeling en IQ.
# 5 folds temporales (ventana expansiva): train -> val (fija umbral) -> test ciego.
# Si el WR>53.2% (BE) aguanta en las ventanas, el edge operable es real.
import os, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]; t = d.get("times", list(range(len(c))))
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def build(V, Vmtf):
    closes = [v[4] for v in V]; N = len(V); out = []; mep = [v[0] for v in Vmtf]
    for i in range(max(PERIOD, 60), N - NCON - GAP - 1):
        w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
        if sd <= 0:
            continue
        z = (closes[i] - sma) / sd
        side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = win[-1][0]
        k = bisect.bisect_right(mep, ep); cmtf = Vmtf[max(0, k - 60):k] if k >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        base = closes[i + GAP]; fut = closes[i + GAP + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        out.append((ep, fv, won))
    return out


def fold(X, y, tr1, v1, te1):
    m = HistGradientBoostingClassifier(**PARAMS).fit(X[:tr1], y[:tr1])
    pv = m.predict_proba(X[tr1:v1])[:, 1]; yv = y[tr1:v1]
    thr = None
    for t in np.arange(0.50, 0.72, 0.01):
        mm = pv >= t; n = int(mm.sum())
        if n >= 300 and yv[mm].mean() >= BE + 0.005:
            thr = round(float(t), 2); break
    if thr is None:
        return None
    pt = m.predict_proba(X[v1:te1])[:, 1]; yt = y[v1:te1]; mm = pt >= thr
    return thr, (yt[mm].mean() if mm.sum() else 0), int(mm.sum())


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    rows = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if len(d.get("close", [])) < 400:
            continue
        V = velas_de(d); rows.extend(build(V, mtf(V)))
    rows.sort(key=lambda r: r[0])
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); n = len(X)
    print(f"OPERABLE (gap=1) | {n} senales | BE {BE*100:.1f}% | walk-forward 5 folds\n")
    res = []
    for k in range(5):
        tr1 = int(n * (0.40 + 0.10 * k)); v1 = int(n * (0.50 + 0.10 * k)); te1 = int(n * (0.60 + 0.10 * k))
        r = fold(X, y, tr1, v1, te1)
        if r is None:
            print(f"fold {k+1}: sin umbral valido (no edge en val)"); res.append(None)
        else:
            thr, wr, nn = r
            print(f"fold {k+1}: umbral {thr} -> TEST {wr*100:.2f}% ({nn} ops) EV {ev(wr)*100:+.1f}% [{'>BE' if wr>BE else '<BE'}]")
            res.append((wr, nn))
    ok = [r for r in res if r]
    if ok:
        ws = np.array([w for w, _ in ok]); ns = np.array([n2 for _, n2 in ok])
        wp = float((ws * ns).sum() / ns.sum()); N = int(ns.sum())
        se = (wp * (1 - wp) / N) ** 0.5; z = (wp - BE) / se
        print(f"\nRESUMEN: {int((ws>BE).sum())}/{len(ok)} folds >BE | WR pond {wp*100:.2f}% ({N} ops) EV {ev(wp)*100:+.1f}%")
        print(f"z vs BE = {z:.2f} ->", "SIGNIFICATIVO (edge operable real)" if z > 1.64 else "no significativo")
    else:
        print("\nNingun fold con edge -> el +7% no aguanta en el tiempo.")


if __name__ == "__main__":
    main()
