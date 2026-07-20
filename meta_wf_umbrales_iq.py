# meta_wf_umbrales_iq.py - Walk-forward OPERABLE (gap=1) del meta-labeling en IQ con
# umbrales FIJOS (no elegidos en validacion). Responde: la meseta 0.54-0.58 que aparece
# en meta_backtest_3m_iq (out-of-fold) sobrevive mes a mes fuera de muestra?
#
# Diferencia clave con meta_wf_iq.py: alli el umbral se ELIGE en una ventana de validacion
# (y el fold se descarta si ninguno pasa). Aqui el umbral se fija de antemano, asi que no
# hay seleccion post-hoc: cada fold entrena con el pasado y se mide en el futuro ciego.
# Solo REGULARES (los -OTC son feed RNG y ademas contaminan el umbral: ver AMBOS del sweep).
import os, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1
UMBRALES = [0.54, 0.56, 0.58]
NFOLDS = 5
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


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
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
    print(f"WALK-FORWARD UMBRAL FIJO | {CACHE} | solo REGULARES ({len(files)} pares)")
    print(f"OPERABLE gap=1 | {n} senales | BE {BE*100:.1f}% | {NFOLDS} folds ciegos\n")

    # Cada fold: entrena con todo lo anterior, mide en el 10% siguiente (nunca visto).
    pred, real = [], []
    for k in range(NFOLDS):
        tr1 = int(n * (0.40 + 0.10 * k)); te1 = int(n * (0.50 + 0.10 * k))
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[:tr1], y[:tr1])
        pred.append(m.predict_proba(X[tr1:te1])[:, 1]); real.append(y[tr1:te1])
        print(f"fold {k+1}: train {tr1} -> test [{tr1}:{te1}] ({te1-tr1} senales, base {y[tr1:te1].mean()*100:.1f}%)")

    for u in UMBRALES:
        print(f"\n=== umbral {u} ===")
        ws, ns = [], []
        for k in range(NFOLDS):
            mm = pred[k] >= u; nn = int(mm.sum())
            if nn == 0:
                print(f"  fold {k+1}: sin senales"); continue
            wr = float(real[k][mm].mean())
            ws.append(wr); ns.append(nn)
            print(f"  fold {k+1}: {wr*100:6.2f}% ({nn:6d} ops) EV {ev(wr)*100:+6.1f}%  [{'>BE' if wr > BE else '<BE'}]")
        if not ns:
            continue
        ws = np.array(ws); ns = np.array(ns)
        wp = float((ws * ns).sum() / ns.sum()); N = int(ns.sum())
        se = (wp * (1 - wp) / N) ** 0.5; z = (wp - BE) / se
        print(f"  RESUMEN: {int((ws > BE).sum())}/{len(ns)} folds >BE | WR pond {wp*100:.2f}% ({N} ops) EV {ev(wp)*100:+.1f}%")
        print(f"  z vs BE = {z:.2f} ->", "SIGNIFICATIVO" if z > 1.64 else "NO significativo")

    print("\n[!] Umbral fijado de antemano: sin seleccion post-hoc. Si la meseta 0.54-0.58")
    print("    no aguanta aqui, el +2% del sweep out-of-fold era sobreajuste de seleccion.")


if __name__ == "__main__":
    main()
