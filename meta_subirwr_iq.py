# meta_subirwr_iq.py - Busca subir el WR: prueba primario STOCH (%K extremo) ademas de
# bbrev, y el combo (mr_combo). Barrido de umbral. Operable (gap=1), OOF. Regulares.
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


def stoch_k(highs, lows, closes, i, p=14):
    hh = max(highs[i - p + 1:i + 1]); ll = min(lows[i - p + 1:i + 1])
    return 100 * (closes[i] - ll) / (hh - ll) if hh > ll else 50.0


def build(V, Vmtf, modo):
    """modo: 'bbrev', 'stoch' o 'combo' (bbrev y si no dispara, stoch)."""
    closes = [v[4] for v in V]; highs = [v[2] for v in V]; lows = [v[3] for v in V]
    N = len(V); out = []; mep = [v[0] for v in Vmtf]
    for i in range(max(PERIOD, 60), N - NCON - GAP - 1):
        side = None
        if modo in ("bbrev", "combo"):
            w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
            if sd > 0:
                z = (closes[i] - sma) / sd
                side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None and modo in ("stoch", "combo"):
            kk = stoch_k(highs, lows, closes, i)
            side = "CALL" if kk < 20 else "PUT" if kk > 80 else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = win[-1][0]
        kk = bisect.bisect_right(mep, ep); cmtf = Vmtf[max(0, kk - 60):kk] if kk >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        base = closes[i + GAP]; fut = closes[i + GAP + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        out.append((ep, fv, won))
    return out


def oof(X, y):
    o = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    return o


def rep(nombre, rows):
    rows = sorted(rows, key=lambda r: r[0])
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    o = oof(X, y); m = ~np.isnan(o); yo, po = y[m], o[m]
    print(f"\n[{nombre}]  base {yo.mean()*100:.1f}%  ({len(yo)} señales)")
    print(f"   {'umbral':>7} {'WR':>7} {'EV':>8} {'ops':>8} {'%':>6}")
    for u in (0.55, 0.58, 0.60, 0.62, 0.65, 0.70):
        sel = po >= u; n = int(sel.sum())
        if n >= 50:
            wr = yo[sel].mean()
            print(f"   {u:>7.2f} {wr*100:6.1f}% {ev(wr)*100:+7.1f}% {n:8d} {n/len(yo)*100:5.1f}%")


def cargar(modo):
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
        rows.extend(build(V, mtf(V), modo))
    return rows


def main():
    print(f"SUBIR WR | operable gap=1 | BE {BE*100:.1f}% | regulares")
    rep("bbrev (actual)", cargar("bbrev"))
    rep("stoch (%K extremo)", cargar("stoch"))
    rep("combo bbrev+stoch (mr_combo)", cargar("combo"))
    print("\n[!] El WR sube con el umbral, pero baja el volumen. Recorda: en walk-forward")
    print("    IQ el meta queda en break-even; el WR alto a umbral alto es in-sample-ish.")


if __name__ == "__main__":
    main()
