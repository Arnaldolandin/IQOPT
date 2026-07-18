# horarios_meta_iq.py - WR de las senales bbrev+meta por HORA (UTC), DIA de semana
# y por PAR. Usa P out-of-fold (TimeSeriesSplit, honesto) + entrada OPERABLE (gap=1).
# Subconjunto operado = P>=UMBRAL. Payout IQ ~88% -> BE 53.2%.
import os, json, glob, bisect, warnings
from datetime import datetime, timezone
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1; UMBRAL = 0.55
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)
DIAS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]; t = d.get("times", list(range(len(c))))
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def build(V, Vmtf, par):
    closes = [v[4] for v in V]; N = len(V); out = []; mep = [v[0] for v in Vmtf]
    for i in range(max(PERIOD, 60), N - NCON - GAP - 1):
        w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
        if sd <= 0:
            continue
        z = (closes[i] - sma) / sd
        side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = int(win[-1][0])
        k = bisect.bisect_right(mep, ep); cmtf = Vmtf[max(0, k - 60):k] if k >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        base = closes[i + GAP]; fut = closes[i + GAP + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        out.append((ep, fv, won, par))
    return out


def tabla(nombre, claves, keyfn, yo, po, sel, meta):
    print(f"\n=== {nombre} (subconjunto operado P>={UMBRAL}) ===")
    print(f"  {'':12} {'WR':>7} {'EV':>8} {'n':>7}")
    for k in claves:
        m = sel & np.array([keyfn(x) == k for x in meta])
        n = int(m.sum())
        if n < 30:
            continue
        wr = yo[m].mean()
        flag = "  <-- rentable" if wr > BE else ""
        print(f"  {str(k):12} {wr*100:6.1f}% {ev(wr)*100:+7.1f}% {n:7d}{flag}")


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
        par = os.path.basename(f)[:-5]
        V = velas_de(d); rows.extend(build(V, mtf(V), par))
    rows.sort(key=lambda r: r[0])
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    meta = [(r[0], r[3]) for r in rows]                       # (epoch, par)
    n = len(X)
    print(f"OPERABLE gap=1 | {n} senales | BE {BE*100:.1f}% | umbral P>={UMBRAL}")
    oof = np.full(n, np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    yo = y[mask]; po = oof[mask]; metao = [meta[i] for i in range(n) if mask[i]]
    sel = po >= UMBRAL
    print(f"operadas (P>={UMBRAL}): {int(sel.sum())} | WR global {yo[sel].mean()*100:.1f}% EV {ev(yo[sel].mean())*100:+.1f}%")

    def hora(x): return datetime.fromtimestamp(x[0], timezone.utc).hour
    def dia(x): return datetime.fromtimestamp(x[0], timezone.utc).weekday()
    def parf(x): return x[1]
    tabla("POR HORA (UTC)", list(range(24)), hora, yo, po, sel, metao)
    tabla("POR DIA DE SEMANA", list(range(7)), lambda x: DIAS[dia(x)], yo, po, sel,
          [(e, DIAS[datetime.fromtimestamp(e, timezone.utc).weekday()]) for e, _ in metao])
    pares = sorted(set(p for _, p in metao))
    tabla("POR PAR", pares, parf, yo, po, sel, metao)
    print("\n[!] Estas tablas son historicas/in-sample-ish: seleccionar la mejor hora/par")
    print("    SOBREAJUSTA. Cualquier 'rentable' hay que validarlo OOS antes de confiar.")


if __name__ == "__main__":
    main()
