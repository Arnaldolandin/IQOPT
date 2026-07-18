# meta_srtrend_iq.py - Prueba si agregar TENDENCIA/S-R dinamicas al meta-modelo mejora.
# Nuevas features: canal de regresion (canal_z, canal_slope) + S/R por swings (dist a
# resistencia/soporte mas cercanos). Compara 32 vs 36 features, OOF, operable (gap=1).
# Regulares (cache_ohlc_5m). BE 53.2%.
import os, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]; t = d["times"]
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def canal(closes, N=50):
    if len(closes) < N:
        return 0.0, 0.0
    c = np.asarray(closes[-N:], float); x = np.arange(N); xb = x.mean(); den = ((x - xb) ** 2).sum()
    slope = float(((x - xb) / den * c).sum())
    centro = c.mean() + slope * ((N - 1) / 2.0); sN = c.std()
    z = (c[-1] - centro) / sN if sN > 0 else 0.0
    px = max(abs(c[-1]), 1e-12)
    return float(z), float(slope / px * 1000)


def sr(highs, lows, closes, look=100, k=3):
    h = highs[-look:]; l = lows[-look:]; px = closes[-1]
    hi = [h[i] for i in range(k, len(h) - k) if h[i] == max(h[i - k:i + k + 1])]
    lo = [l[i] for i in range(k, len(l) - k) if l[i] == min(l[i - k:i + k + 1])]
    res = [s for s in hi if s > px]; sup = [s for s in lo if s < px]
    d_res = (min(res) - px) / px * 100 if res else 0.5
    d_sup = (px - max(sup)) / px * 100 if sup else 0.5
    return float(d_res), float(d_sup)


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
        cw = [v[4] for v in win]; hw = [v[2] for v in win]; lw = [v[3] for v in win]
        cz, cs = canal(cw); dr, ds = sr(hw, lw, cw)
        extra = np.array([cz, cs, dr, ds])
        base = closes[i + GAP]; fut = closes[i + GAP + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        out.append((ep, fv, extra, won))
    return out


def oof(X, y):
    o = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o


def rep(nombre, X, y):
    o = oof(X, y); m = ~np.isnan(o); yo, po = y[m], o[m]
    best = None
    print(f"\n[{nombre}]  base {yo.mean()*100:.1f}%  ({len(yo)})")
    for u in (0.55, 0.60, 0.65):
        sel = po >= u; n = int(sel.sum())
        if n:
            wr = yo[sel].mean(); e = ev(wr)
            print(f"   P>={u}: WR {wr*100:.1f}% EV {e*100:+.1f}% n={n}")
            if best is None or (n >= 500 and e > best[1]):
                best = (u, e, wr, n)
    return best


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    rows = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        V = velas_de(d)
        if len(V) < 400:
            continue
        rows.extend(build(V, mtf(V)))
    rows.sort(key=lambda r: r[0])
    y = np.array([r[3] for r in rows])
    X32 = np.array([r[1] for r in rows])
    X36 = np.array([np.concatenate([r[1], r[2]]) for r in rows])
    print(f"OPERABLE gap=1 | {len(rows)} señales regulares | BE {BE*100:.1f}%")
    b32 = rep("32 features (actual)", X32, y)
    b36 = rep("36 features (+canal +S/R)", X36, y)
    print("\n=== VEREDICTO ===")
    if b32 and b36:
        d = (b36[1] - b32[1]) * 100
        print(f"EV mejor  SIN {b32[1]*100:+.1f}%  ->  CON {b36[1]*100:+.1f}%   (delta {d:+.1f} pts)")
        print("Las features tendencia/S-R", "AYUDAN" if d > 0.5 else "NO ayudan (dentro de ruido)" if abs(d) <= 0.5 else "EMPEORAN")


if __name__ == "__main__":
    main()
