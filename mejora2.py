# mejora2.py - Segundo intento de mejorar la prediccion con features NUEVAS:
#  microestructura de vela, rezagos, volatilidad realizada, hora/dia (ciclicas).
#  Primario GBM enriquecido + meta-labeling enriquecido, validado con walk-forward 5 folds.
#  Si el agregado no supera 53.5% de forma significativa, el techo es estructural (payout).
import warnings, glob, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import predictor_opt as P
import mejora_prediccion as MP
warnings.filterwarnings("ignore")
CACHE = "cache_ohlc_5m"; WARM = 210; BE = 53.5


def feats_ricas(o, h, l, c, t):
    """Features del PRIMARIO: continuas + microestructura + rezagos + vol realizada."""
    n = len(c)
    F = MP.feats_continuas(o, h, l, c)          # 10 continuas base (macd, emas, rsi, bb, roc, canal...)
    rng = np.maximum(h - l, 1e-12)
    body = (c - o) / rng                          # cuerpo (signo=direccion, magnitud=fuerza)
    uw = (h - np.maximum(c, o)) / rng             # mecha superior (rechazo arriba)
    lw = (np.minimum(c, o) - l) / rng             # mecha inferior (rechazo abajo)
    r = np.zeros(n); r[1:] = np.diff(c) / np.maximum(np.abs(c[:-1]), 1e-12)
    rv = np.array([r[max(0, i - 19):i + 1].std() for i in range(n)])   # vol realizada 20
    # rezagos de retorno y de la 1a continua (macd hist normalizado)
    def lag(x, k):
        y = np.zeros(n); y[k:] = x[:-k]; return y
    micro = np.column_stack([body, uw, lw, r, lag(r, 1), lag(r, 2), rv,
                             lag(F[:, 0], 1), lag(F[:, 0], 2), lag(F[:, 5], 1)])
    return np.nan_to_num(np.column_stack([F, micro]))


def feats_meta_extra(t, atrpct):
    """Features EXTRA para el meta: hora/dia ciclicas + regimen de vol."""
    hora = (t / 3600.0) % 24
    dow = (t / 86400.0) % 7
    return np.nan_to_num(np.column_stack([
        np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), atrpct]))


def cargar():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    XV = []; XR = []; XE = []; T = []; YG = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); o = np.asarray(d.get("open", d["close"]), float)
            h = np.asarray(d.get("high", d["close"]), float); l = np.asarray(d.get("low", d["close"]), float)
            t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < WARM + 400: continue
        V, _ = P.indicadores(o, h, l, c); FR = feats_ricas(o, h, l, c, t)
        tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atrp = np.array([tr[max(0, i - 13):i + 1].mean() for i in range(n)]) / np.maximum(np.abs(c), 1e-12)
        FE = feats_meta_extra(t, atrp)
        H = 2; g = np.zeros(n); g[:n - 1 - H] = (c[1 + H:] > c[1:n - H]).astype(float)
        hi = n - 2 - H
        XV.append(np.nan_to_num(V[WARM:hi])); XR.append(FR[WARM:hi]); XE.append(FE[WARM:hi])
        T.append(t[WARM:hi]); YG.append(g[WARM:hi])
    XV = np.vstack(XV); XR = np.vstack(XR); XE = np.vstack(XE); T = np.concatenate(T); YG = np.concatenate(YG)
    order = np.argsort(T)
    return XV[order], XR[order], XE[order], YG[order]


def main():
    print("Cargando (features ricas)..."); XV, XR, XE, yg = cargar()
    n = len(XV)
    print(f"muestras {n} | primario {XR.shape[1]} feats | meta +{XE.shape[1]} | BE {BE}%")
    print("walk-forward 5 folds (ventana expansiva), primario=GBM rico, meta=GBM enriquecido\n")

    def do_fold(tr1, v1, te1):
        Xr_tr, Xe_tr, y_tr = XR[:tr1], XE[:tr1], yg[:tr1]
        prim = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=5,
                                              l2_regularization=1.0).fit(Xr_tr, y_tr)
        p_tr = prim.predict_proba(Xr_tr)[:, 1]; pred_tr = (p_tr > 0.5).astype(float)
        ac = (pred_tr == y_tr).astype(float)
        Mtr = np.column_stack([Xr_tr, Xe_tr, np.abs(p_tr - 0.5)])
        meta = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4).fit(Mtr, ac)

        def ev(s, e):
            p = prim.predict_proba(XR[s:e])[:, 1]; pred = (p > 0.5).astype(float)
            M = np.column_stack([XR[s:e], XE[s:e], np.abs(p - 0.5)]); pm = meta.predict_proba(M)[:, 1]
            return pred, yg[s:e], pm
        pv, yv, pmv = ev(tr1, v1)
        thr = None
        for tt in np.arange(0.50, 0.72, 0.01):
            m = pmv >= tt
            if m.sum() >= 300 and (pv[m] == yv[m]).mean() * 100 >= BE + 0.5:
                thr = round(float(tt), 2); break
        if thr is None: return None
        pt, yt, pmt = ev(v1, te1); m = pmt >= thr
        return thr, ((pt[m] == yt[m]).mean() * 100 if m.sum() else 0), int(m.sum())

    res = []
    for k in range(5):
        tr1 = int(n * (0.40 + 0.10 * k)); v1 = int(n * (0.50 + 0.10 * k)); te1 = int(n * (0.60 + 0.10 * k))
        r = do_fold(tr1, v1, te1)
        if r is None:
            print(f"fold {k+1}: sin umbral valido (no edge)"); res.append(None)
        else:
            thr, wt, nn = r; print(f"fold {k+1}: umbral {thr} -> TEST {wt:.2f}%  ({nn} ops)  [{'>BE' if wt>BE else '<BE'}]")
            res.append((wt, nn))
    ok = [r for r in res if r]
    if ok:
        ws = np.array([w for w, _ in ok]); ns = np.array([n2 for _, n2 in ok])
        wp = float((ws * ns).sum() / ns.sum()); N = int(ns.sum())
        se = (wp / 100 * (1 - wp / 100) / N) ** 0.5 * 100; z = (wp - BE) / se
        print(f"\nRESUMEN: {int((ws>BE).sum())}/{len(ok)} folds >BE | WR pond {wp:.2f}% ({N} ops) | z={z:.2f} ->",
              "SIGNIFICATIVO" if z > 1.64 else "no significativo")
    else:
        print("\nNingun fold con edge -> features nuevas no ayudan. Techo estructural.")


if __name__ == "__main__":
    main()
