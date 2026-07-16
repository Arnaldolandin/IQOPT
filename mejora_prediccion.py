# mejora_prediccion.py - Intenta MEJORAR la prediccion con 5 palancas, cada una
# medida con OOS riguroso (train 70% / test 30% por tiempo, entrada OPERABLE +1 barra).
# Objetivo: superar el break-even 53.5%. 6 meses de 5m, pares reales (sin OTC).
#   .venv314\Scripts\python.exe mejora_prediccion.py
import json, glob, os, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import predictor_opt as P
warnings.filterwarnings("ignore")

CACHE = "cache_ohlc_5m"
WARM = 210
BE = 53.5


def ema(c, s):
    a = 2.0 / (s + 1); o = np.copy(c)
    for i in range(1, len(c)):
        o[i] = a * c[i] + (1 - a) * o[i - 1]
    return o


def feats_continuas(o, h, l, c):
    """Features CONTINUAS (magnitud, no votos) para GBM/meta."""
    n = len(c)
    ml = ema(c, 6) - ema(c, 13); sg = ema(ml, 5); hist = ml - sg
    e50 = ema(c, 50); e200 = ema(c, 200)
    d = np.zeros(n); d[1:] = np.diff(c); up = np.where(d > 0, d, 0.); dn = np.where(d < 0, -d, 0.)
    def rma(x, p):
        oo = np.copy(x)
        for i in range(1, len(x)): oo[i] = (oo[i - 1] * (p - 1) + x[i]) / p
        return oo
    rs = rma(up, 14) / np.where(rma(dn, 14) == 0, np.nan, rma(dn, 14)); rsi = np.nan_to_num(100 - 100 / (1 + rs), nan=50)
    m20 = P.roll(c, 20); s20 = np.array([c[max(0, i - 19):i + 1].std() for i in range(n)])
    bbz = np.where(s20 > 0, (c - np.nan_to_num(m20, nan=c)) / s20, 0)
    roc = np.zeros(n); roc[10:] = c[10:] / np.maximum(c[:-10], 1e-12) - 1
    # canal
    N = 50; x = np.arange(N); xb = x.mean(); den = ((x - xb) ** 2).sum()
    slope = np.zeros(n)
    if n >= N: slope[N - 1:] = np.convolve(c, ((x - xb) / den)[::-1], mode="valid")
    ybar = P.roll(c, N); centro = np.nan_to_num(ybar, nan=c) + slope * ((N - 1) / 2.0)
    sN = np.array([c[max(0, i - N + 1):i + 1].std() for i in range(n)])
    czr = np.where(sN > 0, (c - centro) / sN, 0)
    px = np.maximum(np.abs(c), 1e-12)
    F = np.column_stack([
        hist / px, (hist - np.concatenate([[0], hist[:-1]])) / px,
        (c - e50) / px, (c - e200) / px, (e50 - e200) / px,
        (rsi - 50) / 50, bbz, roc * 100, czr, slope / px * 1000,
    ])
    return np.nan_to_num(F)


def cargar():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    XV = []; XC = []; T = []; Y = {}
    Hs = [1, 2, 3, 6]
    for H in Hs: Y[H] = []
    trend = []
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
        V, _ = P.indicadores(o, h, l, c); Fc = feats_continuas(o, h, l, c)
        e200 = ema(c, 200); tr = np.sign(c - e200)   # tendencia (precio vs EMA200)
        hi = n - 2 - 6
        XV.append(np.nan_to_num(V[WARM:hi])); XC.append(Fc[WARM:hi]); T.append(t[WARM:hi]); trend.append(tr[WARM:hi])
        for H in Hs:
            g = np.zeros(n); g[:n - 1 - H] = (c[1 + H:] > c[1:n - H]).astype(float)
            Y[H].append(g[WARM:hi])
    XV = np.vstack(XV); XC = np.vstack(XC); T = np.concatenate(T); trend = np.concatenate(trend)
    for H in Hs: Y[H] = np.concatenate(Y[H])
    order = np.argsort(T)
    XV, XC, trend = XV[order], XC[order], trend[order]
    for H in Hs: Y[H] = Y[H][order]
    return XV, XC, trend, Y, Hs


def wr(pred, y, m):
    return ((pred[m] == y[m]).mean() * 100 if m.sum() else 0.0, int(m.sum()))


def main():
    print("Cargando 6 meses de pares reales..."); XV, XC, trend, Y, Hs = cargar()
    n = len(XV); sp = int(n * 0.70)
    yg = Y[2]  # target operable, horizonte 2 (10m) por defecto
    print(f"muestras {n} | train {sp} | test {n-sp} | break-even {BE}%\n")

    # baseline logistica sobre votos
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(XV[:sp], yg[:sp])
    p = clf.predict_proba(XV[sp:])[:, 1]; predL = (p > 0.5).astype(float); conf = np.abs(p - 0.5)

    print("[1] BARRIDO DE CONFIANZA (logistica votos)")
    for mg in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        w, nn = wr(predL, yg[sp:], conf >= mg)
        print(f"    conf>={mg:.2f}: {w:.2f}%  ({nn} ops)")

    print("\n[2] FILTRO DE TENDENCIA (precio vs EMA200)")
    tt = trend[sp:]
    afav = np.sign(predL * 2 - 1) == tt         # prediccion a favor de la tendencia
    for nm, mask in (("a favor", afav), ("en contra", ~afav)):
        w, nn = wr(predL, yg[sp:], mask & (conf >= 0.02))
        print(f"    {nm}: {w:.2f}%  ({nn} ops)")

    print("\n[3] GRADIENT BOOSTING (features continuas)")
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                        max_depth=4, l2_regularization=1.0).fit(XC[:sp], yg[:sp])
    pg = gb.predict_proba(XC[sp:])[:, 1]; predG = (pg > 0.5).astype(float); cg = np.abs(pg - 0.5)
    for mg in (0.0, 0.05, 0.10, 0.15):
        w, nn = wr(predG, yg[sp:], cg >= mg)
        print(f"    conf>={mg:.2f}: {w:.2f}%  ({nn} ops)")

    print("\n[4] META-LABELING (2do modelo predice si el 1ro acierta)")
    # primario = logistica votos; etiqueta meta = acierto del primario en TRAIN
    p_tr = clf.predict_proba(XV[:sp])[:, 1]; pred_tr = (p_tr > 0.5).astype(float)
    acierto_tr = (pred_tr == yg[:sp]).astype(float)
    # features del meta = continuas + confianza del primario
    Xm_tr = np.column_stack([XC[:sp], np.abs(p_tr - 0.5)])
    Xm_te = np.column_stack([XC[sp:], np.abs(p - 0.5)])
    meta = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4).fit(Xm_tr, acierto_tr)
    pm = meta.predict_proba(Xm_te)[:, 1]   # prob de que el primario acierte
    for thr in (0.50, 0.55, 0.60, 0.65):
        w, nn = wr(predL, yg[sp:], pm >= thr)
        print(f"    meta>={thr:.2f}: {w:.2f}%  ({nn} ops)")

    print("\n[5] BARRIDO DE HORIZONTE (logistica votos, conf>=0.02)")
    for H in Hs:
        yh = Y[H]
        cl = LogisticRegression(C=1.0, max_iter=1000).fit(XV[:sp], yh[:sp])
        ph = cl.predict_proba(XV[sp:])[:, 1]; prh = (ph > 0.5).astype(float); ch = np.abs(ph - 0.5)
        w, nn = wr(prh, yh[sp:], ch >= 0.02)
        print(f"    H={H} ({H*5}m): {w:.2f}%  ({nn} ops)")

    print(f"\n>>> break-even = {BE}%. Todo lo que quede debajo pierde con el payout.")


if __name__ == "__main__":
    main()
