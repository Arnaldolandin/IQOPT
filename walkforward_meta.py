# walkforward_meta.py - Test DECISIVO del meta-labeling: walk-forward en 5 folds
# temporales (ventana expansiva). En cada fold: train -> val (fija umbral) -> test ciego.
# Si el WR>53.5% aguanta en las 5 ventanas, es edge real; si oscila, es azar.
import warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import mejora_prediccion as MP
warnings.filterwarnings("ignore")
BE = 53.5


def fold(XV, XC, yg, tr0, tr1, v0, v1, te0, te1):
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(XV[tr0:tr1], yg[tr0:tr1])
    p_tr = clf.predict_proba(XV[tr0:tr1])[:, 1]; pred_tr = (p_tr > 0.5).astype(float)
    ac = (pred_tr == yg[tr0:tr1]).astype(float)
    Xm_tr = np.column_stack([XC[tr0:tr1], np.abs(p_tr - 0.5)])
    meta = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=4).fit(Xm_tr, ac)

    def ev(s, e):
        p = clf.predict_proba(XV[s:e])[:, 1]; pred = (p > 0.5).astype(float)
        Xm = np.column_stack([XC[s:e], np.abs(p - 0.5)]); pm = meta.predict_proba(Xm)[:, 1]
        return pred, yg[s:e], pm
    pv, yv, pmv = ev(v0, v1)
    thr = None
    for t in np.arange(0.50, 0.72, 0.01):
        m = pmv >= t
        if m.sum() >= 300 and (pv[m] == yv[m]).mean() * 100 >= BE + 0.5:
            thr = round(float(t), 2); break
    if thr is None:
        return None
    pt, yt, pmt = ev(te0, te1); m = pmt >= thr
    wt = (pt[m] == yt[m]).mean() * 100 if m.sum() else 0
    return thr, wt, int(m.sum())


def main():
    print("Cargando..."); XV, XC, trend, Y, Hs = MP.cargar()
    yg = Y[2]; n = len(XV)
    print(f"muestras {n} | BE {BE}%  | 5 folds walk-forward (ventana expansiva)\n")
    # fold k: train 0..(30+10k)%, val siguiente 10%, test siguiente 10%
    res = []
    for k in range(5):
        tr1 = int(n * (0.40 + 0.10 * k)); v1 = int(n * (0.50 + 0.10 * k)); te1 = int(n * (0.60 + 0.10 * k))
        r = fold(XV, XC, yg, 0, tr1, tr1, v1, v1, te1)
        if r is None:
            print(f"fold {k+1}: sin umbral valido en val (no edge en esta ventana)")
            res.append(None)
        else:
            thr, wt, nn = r
            flag = "OK >BE" if wt > BE else "<BE"
            print(f"fold {k+1}: umbral {thr} -> TEST {wt:.2f}%  ({nn} ops)  [{flag}]")
            res.append((wt, nn))
    ok = [r for r in res if r]
    if ok:
        ws = np.array([w for w, _ in ok]); ns = np.array([n for _, n in ok])
        wprom = float((ws * ns).sum() / ns.sum())
        superan = int((ws > BE).sum())
        print(f"\nRESUMEN: {superan}/{len(ok)} folds sobre BE | WR ponderado {wprom:.2f}% | total {int(ns.sum())} ops")
        # significancia del agregado
        p = wprom / 100; N = int(ns.sum()); se = (p * (1 - p) / N) ** 0.5 * 100
        z = (wprom - BE) / se
        print(f"agregado: {wprom:.2f}% +-{se:.2f} (1sd) | z vs BE = {z:.2f}  -> ",
              "SIGNIFICATIVO" if z > 1.64 else "no significativo (podria ser azar)")
    else:
        print("\nNingun fold produjo edge -> confirmado: no hay senal, era sobreajuste de un split.")


if __name__ == "__main__":
    main()
