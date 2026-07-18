# meta_backtest_3m_iq.py - Backtest meta-labeling, ultimos 3 meses, a varios umbrales,
# separado en REGULARES / OTC / AMBOS. Entrada OPERABLE (gap=1). P out-of-fold.
# Usa cache_ohlc (tiene reales + OTC, ~3 meses). Payout IQ ~88% -> BE 53.2%.
import os, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1
DIAS = 90; UMBRALES = [0.70, 0.65, 0.60, 0.55]
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]; t = d["times"]
    V = [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]
    if V:
        corte = V[-1][0] - DIAS * 86400
        V = [v for v in V if v[0] >= corte]
    return V


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def build(V, Vmtf, tag):
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
        out.append((ep, fv, won, tag))
    return out


def oof(X, y):
    o = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o


def tabla(nombre, rows):
    if len(rows) < 2000:
        print(f"\n[{nombre}] muy pocas señales ({len(rows)})"); return
    rows = sorted(rows, key=lambda r: r[0])
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    o = oof(X, y); m = ~np.isnan(o); yo, po = y[m], o[m]
    print(f"\n[{nombre}]  base {yo.mean()*100:.1f}%  ({len(yo)} señales, operable)")
    print(f"  {'umbral':>7} {'WR':>7} {'EV':>8} {'ops':>8} {'%':>6}")
    for u in UMBRALES:
        sel = po >= u; n = int(sel.sum())
        if n:
            wr = yo[sel].mean()
            fl = "  RENTA" if wr > BE else "  pierde"
            print(f"  {u:>7.2f} {wr*100:6.1f}% {ev(wr)*100:+7.1f}% {n:8d} {n/len(yo)*100:5.1f}%{fl}")
        else:
            print(f"  {u:>7.2f}  sin señales")


def main():
    print(f"BACKTEST META-LABELING | ultimos {DIAS}d (~3 meses) | OPERABLE gap=1 | BE {BE*100:.1f}%")
    files = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    reg, otc = [], []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        V = velas_de(d)
        if len(V) < 400:
            continue
        es_otc = "-OTC" in os.path.basename(f)
        sig = build(V, mtf(V), "OTC" if es_otc else "REG")
        (otc if es_otc else reg).extend(sig)
    print(f"señales: regulares={len(reg)} OTC={len(otc)}")
    tabla("SOLO REGULARES", reg)
    tabla("SOLO OTC", otc)
    tabla("AMBOS", reg + otc)
    print("\n[!] Operable (gap=1). OTC es feed RNG de la casa (~50%). En walk-forward IQ")
    print("    el meta-labeling queda en break-even. Un umbral 'RENTA' aqui es in-sample-ish.")


if __name__ == "__main__":
    main()
