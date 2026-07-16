# predictor_opt.py - Predictor EXPANDIDO y OPTIMIZADO. ~12 indicadores. Dos modos:
#  1) Confluencia (votos), barriendo el umbral min_score.
#  2) Pesos APRENDIDOS (regresion logistica sobre los votos) - optimiza la combinacion.
# 6 meses de 5m, train/test OOS, entrada estandar y OPERABLE (+1 barra, sin rebote).
#   python predictor_opt.py [cache_dir]
import json, glob, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
H = 2
BE = 1.0 / 1.87
TEST_DAYS = 60
WARM = 210


def ema(c, s):
    a = 2.0 / (s + 1); o = np.copy(c)
    for i in range(1, len(c)):
        o[i] = a * c[i] + (1 - a) * o[i - 1]
    return o


def rma(x, p):
    o = np.copy(x)
    for i in range(1, len(x)):
        o[i] = (o[i - 1] * (p - 1) + x[i]) / p
    return o


def roll(x, w, fn="mean"):
    n = len(x); out = np.full(n, np.nan); cs = np.cumsum(x)
    if fn == "mean":
        out[w - 1:] = (cs[w - 1:] - np.concatenate([[0], cs[:-w]])) / w
    return out


def indicadores(o, h, l, c):
    """Devuelve matriz de VOTOS (+1/-1/0) por indicador y sus nombres."""
    n = len(c); px = np.maximum(np.abs(c), 1e-12)
    votos = []; nombres = []
    def add(v, nm):
        votos.append(np.nan_to_num(v).astype(float)); nombres.append(nm)
    # 1 MACD posicion, 2 momentum
    ml = ema(c, 6) - ema(c, 13); sg = ema(ml, 5); hist = ml - sg
    add(np.sign(hist), "macd_pos")
    dh = np.zeros(n); dh[1:] = np.diff(hist); add(np.sign(dh), "macd_mom")
    # 3 EMA50, 4 EMA200
    add(np.sign(c - ema(c, 50)), "ema50")
    add(np.sign(c - ema(c, 200)), "ema200")
    # 5 ultimas 3 velas
    v3 = np.zeros(n)
    v3[2:] = np.where((c[2:] > c[1:-1]) & (c[1:-1] > c[:-2]), 1, np.where((c[2:] < c[1:-1]) & (c[1:-1] < c[:-2]), -1, 0))
    add(v3, "ult3")
    # 6 RSI(14) extremo
    d = np.zeros(n); d[1:] = np.diff(c); up = np.where(d > 0, d, 0.); dn = np.where(d < 0, -d, 0.)
    rs = rma(up, 14) / np.where(rma(dn, 14) == 0, np.nan, rma(dn, 14)); rsi = np.nan_to_num(100 - 100 / (1 + rs), nan=50)
    add(np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)), "rsi_ext")
    # 7 Stochastic(14,3)
    hh = np.array([h[max(0, i - 13):i + 1].max() for i in range(n)])
    ll = np.array([l[max(0, i - 13):i + 1].min() for i in range(n)])
    k = np.where(hh - ll > 0, 100 * (c - ll) / (hh - ll), 50); dstoc = roll(k, 3)
    add(np.where(k < 20, 1, np.where(k > 80, -1, 0)), "stoch_ext")
    add(np.sign(k - np.nan_to_num(dstoc, nan=50)), "stoch_kd")
    # 8 Williams %R(14)
    wr = np.where(hh - ll > 0, -100 * (hh - c) / (hh - ll), -50)
    add(np.where(wr < -80, 1, np.where(wr > -20, -1, 0)), "williamsR")
    # 9 CCI(20)
    tp = (h + l + c) / 3; sma = roll(tp, 20)
    md = np.array([np.abs(tp[max(0, i - 19):i + 1] - (sma[i] if not np.isnan(sma[i]) else tp[i])).mean() for i in range(n)])
    cci = np.where(md > 0, (tp - np.nan_to_num(sma, nan=tp)) / (0.015 * md), 0)
    add(np.where(cci < -100, 1, np.where(cci > 100, -1, 0)), "cci_ext")
    # 10 ADX/DI direccion
    upmove = np.zeros(n); dnmove = np.zeros(n)
    upmove[1:] = h[1:] - h[:-1]; dnmove[1:] = l[:-1] - l[1:]
    pdm = np.where((upmove > dnmove) & (upmove > 0), upmove, 0.); ndm = np.where((dnmove > upmove) & (dnmove > 0), dnmove, 0.)
    tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = rma(tr, 14); pdi = 100 * rma(pdm, 14) / np.where(atr > 0, atr, 1); ndi = 100 * rma(ndm, 14) / np.where(atr > 0, atr, 1)
    add(np.sign(pdi - ndi), "di_dir")
    # 11 ROC(10)
    roc = np.zeros(n); roc[10:] = c[10:] / np.maximum(c[:-10], 1e-12) - 1
    add(np.sign(roc), "roc")
    # 12 Bollinger z extremo
    m20 = roll(c, 20); s20 = np.array([c[max(0, i - 19):i + 1].std() for i in range(n)])
    bbz = np.where(s20 > 0, (c - np.nan_to_num(m20, nan=c)) / s20, 0)
    add(np.where(bbz < -2, 1, np.where(bbz > 2, -1, 0)), "bb_ext")
    # 13-14 CANAL DE TENDENCIA DINAMICA (regresion lineal movil N=50)
    N = 50; x = np.arange(N); xb = x.mean(); den = ((x - xb) ** 2).sum(); kern = (x - xb) / den
    slope = np.full(n, np.nan)
    if n >= N:
        slope[N - 1:] = np.convolve(c, kern[::-1], mode="valid")
    ybar = roll(c, N)
    centro = np.nan_to_num(ybar, nan=c) + np.nan_to_num(slope) * ((N - 1) / 2.0)   # linea al extremo
    sN = np.array([c[max(0, i - N + 1):i + 1].std() for i in range(n)])
    czr = np.where(sN > 0, (c - centro) / sN, 0)
    add(np.where(czr < -1, 1, np.where(czr > 1, -1, 0)), "canal_pos")   # extremos del canal (reversion)
    add(np.sign(np.nan_to_num(slope)), "canal_slope")                   # direccion del canal (tendencia)
    return np.column_stack(votos), nombres


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    X = []; ys = []; yg = []; T = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); o = np.asarray(d.get("open", d["close"]), float)
            h = np.asarray(d.get("high", d["close"]), float); l = np.asarray(d.get("low", d["close"]), float)
            t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < WARM + 400:
            continue
        V, nombres = indicadores(o, h, l, c)
        s = np.zeros(n); s[:n - H] = (c[H:] > c[:n - H]).astype(float)
        g = np.zeros(n); g[:n - 1 - H] = (c[1 + H:] > c[1:n - H]).astype(float)
        hi = n - 2 - H
        X.append(V[WARM:hi]); ys.append(s[WARM:hi]); yg.append(g[WARM:hi]); T.append(t[WARM:hi])
    X = np.nan_to_num(np.vstack(X)); ys = np.concatenate(ys); yg = np.concatenate(yg); T = np.concatenate(T)
    order = np.argsort(T); X = X[order]; ys = ys[order]; yg = yg[order]
    n = len(X); split = int(n * 0.70)
    _, nombres = indicadores(np.ones(300), np.ones(300), np.ones(300), np.ones(300))
    print("=" * 88)
    print(f"PREDICTOR OPTIMIZADO | {CACHE} | {n} muestras | {X.shape[1]} indicadores | BE {BE*100:.1f}%")
    print(f"indicadores: {', '.join(nombres)}")
    print("=" * 88)

    def wr(pred, y, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        return (pred[m] == y[m]).mean() * 100 if m.sum() else 0, int(m.sum())

    # 1) CONFLUENCIA: net vote, barrer umbral, OOS operable
    print("\n[1] CONFLUENCIA (voto neto) — OOS operable (entrada +1)")
    net = X.sum(axis=1)
    for thr in (0, 2, 3, 4, 5, 6, 8):
        mask = np.abs(net[split:]) >= thr
        pred = (net[split:] > 0).astype(float)
        w, nn = wr(pred, yg[split:], mask)
        print(f"  |voto|>={thr:>2}: OOS {w:.2f}%  ({nn} ops)")

    # 2) PESOS APRENDIDOS: logistica sobre los votos, train->test
    print("\n[2] PESOS APRENDIDOS (logistica sobre los votos)")
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X[:split], ys[:split])
    p = clf.predict_proba(X[split:])[:, 1]
    for margin in (0.0, 0.02, 0.05, 0.08):
        mask = np.abs(p - 0.5) >= margin
        pred = (p > 0.5).astype(float)
        w, nn = wr(pred, yg[split:], mask)   # evaluado en target OPERABLE
        ws, _ = wr(pred, ys[split:], mask)   # tambien estandar (referencia)
        print(f"  confianza>={margin:.2f}: OOS operable {w:.2f}%  | estandar {ws:.2f}%  ({nn} ops)")
    # pesos aprendidos (que indicador pesa mas)
    pesos = sorted(zip(nombres, clf.coef_[0]), key=lambda x: -abs(x[1]))
    print("  pesos (top): " + " ".join(f"{nm}={w:+.2f}" for nm, w in pesos[:8]))

    print(f"\nbreak-even {BE*100:.1f}%. Recordatorio: el GBM con 29 features dio 50.8% operable.")


if __name__ == "__main__":
    main()
