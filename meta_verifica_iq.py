# meta_verifica_iq.py - Verifica si el ~69% del meta-labeling IQ es REAL u operable
# o un artefacto de misma-vela (close[i]). Compara gap=0 (entra en close[i]) vs
# gap=1 (OPERABLE, entra en close[i+1]) + control barajado. Mismo dato y features.
import os, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]; t = d.get("times", list(range(len(c))))
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def build(V, Vmtf, gap):
    closes = [v[4] for v in V]; N = len(V); out = []; mep = [v[0] for v in Vmtf]
    for i in range(max(PERIOD, 60), N - NCON - gap - 1):
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
        base = closes[i + gap]; fut = closes[i + gap + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        out.append((ep, fv, won))
    return out


def oof(X, y):
    o = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o


def rep(tag, X, y):
    o = oof(X, y); mask = ~np.isnan(o); yo, po = y[mask], o[mask]
    print(f"\n[{tag}] baseline {yo.mean()*100:.1f}% n={len(yo)}")
    for t in (0.55, 0.60, 0.65):
        mm = po >= t; n = int(mm.sum())
        if n:
            print(f"   P>={t:.2f}: WR {yo[mm].mean()*100:.1f}% n={n} EV {ev(yo[mm].mean())*100:+.1f}%")
    return o, yo, po


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    R0, R1 = [], []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if len(d.get("close", [])) < 400:
            continue
        V = velas_de(d); Vm = mtf(V)
        R0.extend(build(V, Vm, 0)); R1.extend(build(V, Vm, 1))
    for R in (R0, R1):
        R.sort(key=lambda r: r[0])
    X0 = np.array([r[1] for r in R0]); y0 = np.array([r[2] for r in R0])
    X1 = np.array([r[1] for r in R1]); y1 = np.array([r[2] for r in R1])
    print(f"BE {BE*100:.1f}% | gap0 {len(y0)} senales | gap1 {len(y1)} senales")
    rep("gap=0 (entra close[i], NO operable)", X0, y0)
    rep("gap=1 (OPERABLE, entra close[i+1])", X1, y1)
    rng = np.random.default_rng(0); ysh = y0.copy(); rng.shuffle(ysh)
    o = oof(X0, ysh); m = ~np.isnan(o); print(f"\n[control barajado] {(o[m]>0.5).astype(int).__eq__(ysh[m]).mean()*100:.1f}% (debe ~50%)")
    print("\n=== VEREDICTO ===")
    o1 = oof(X1, y1); m1 = ~np.isnan(o1); yo1, po1 = y1[m1], o1[m1]
    sel = po1 >= 0.60
    if sel.sum() > 200:
        wr = yo1[sel].mean()
        print(f"OPERABLE P>=0.60: WR {wr*100:.1f}% EV {ev(wr)*100:+.1f}% ->",
              "EDGE REAL, confirmar walk-forward" if wr > BE + 0.005 else "cae al break-even (era misma-vela)")


if __name__ == "__main__":
    main()
