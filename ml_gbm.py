# ml_gbm.py - Gradient Boosting (no lineal) sobre ~6 meses de 5m para predecir direccion.
# ~30 features. Walk-forward temporal (varios folds). Reporta accuracy OOS estandar
# Y con entrada retrasada 1 barra (sin rebote bid-ask = accuracy OPERABLE).
#   python ml_gbm.py [cache_dir] [factor]

import json, glob, os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
FACTOR = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MODE = sys.argv[3] if len(sys.argv) > 3 else "real"   # real | otc
H = 2
BE = 1.0 / 1.87
FOLDS = 5


def resample(a, f, agg="last"):
    if f == 1:
        return np.asarray(a, float)
    n = len(a) // f; b = np.asarray(a[:n * f], float).reshape(n, f)
    return b[:, -1] if agg == "last" else (b.max(1) if agg == "max" else (b.min(1) if agg == "min" else b[:, 0]))


def ema(c, s):
    a = 2.0 / (s + 1); o = np.copy(c)
    for i in range(1, len(c)):
        o[i] = a * c[i] + (1 - a) * o[i - 1]
    return o


def roll(x, w, fn):
    n = len(x); out = np.full(n, np.nan)
    if fn == "mean":
        cs = np.cumsum(x); out[w:] = (cs[w:] - cs[:-w]) / w
    elif fn == "std":
        cs = np.cumsum(x); cs2 = np.cumsum(x * x)
        out[w:] = np.sqrt(np.clip((cs2[w:] - cs2[:-w]) / w - ((cs[w:] - cs[:-w]) / w) ** 2, 0, None))
    return out


def rsi(c, p):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = roll(up, p, "mean"); ad = roll(dn, p, "mean")
    rs = au / np.where(ad == 0, np.nan, ad)
    return np.nan_to_num(100 - 100 / (1 + rs), nan=50.0)


def feats(o, h, l, c, t):
    n = len(c); px = np.maximum(np.abs(c), 1e-12)
    r = np.zeros(n); r[1:] = np.diff(np.log(px))
    cols = []
    for k in (1, 2, 3, 4, 5, 6, 8):
        cols.append(np.roll(r, k - 1))
    for k in (3, 5, 10, 20, 50):
        m = np.zeros(n); m[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1; cols.append(m)
    cols.append((rsi(c, 7) - 50) / 50); cols.append((rsi(c, 14) - 50) / 50)
    ml = ema(c, 6) - ema(c, 13); hist = ml - ema(ml, 5); cols.append(hist / px)
    m20 = roll(c, 20, "mean"); s20 = roll(c, 20, "std")
    cols.append(np.nan_to_num((c - m20) / np.where(s20 == 0, np.nan, s20), nan=0.0))
    tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    cols.append(roll(tr, 14, "mean") / px)
    cols.append(roll(r, 10, "std")); cols.append(roll(r, 30, "std"))
    for w in (20, 50):
        mn = np.full(n, np.nan); mx = np.full(n, np.nan)
        for i in range(w, n):
            seg = c[i - w:i]; mn[i] = seg.min(); mx[i] = seg.max()
        cols.append(np.nan_to_num((c - mn) / np.where(mx - mn == 0, np.nan, mx - mn), nan=0.5))
    hour = (t % 86400) // 3600; dow = (t // 86400) % 7
    cols.append(np.sin(2 * np.pi * hour / 24)); cols.append(np.cos(2 * np.pi * hour / 24))
    cols.append(np.sin(2 * np.pi * dow / 7)); cols.append(np.cos(2 * np.pi * dow / 7))
    cols.append((h - l) / px); cols.append(np.abs(c - o) / px)
    cols.append((c - ema(c, 50)) / px); cols.append((c - ema(c, 200)) / px)
    return np.column_stack(cols)


def main():
    allf = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    files = [f for f in allf if ("-OTC" in os.path.basename(f)) == (MODE == "otc")]
    X = []; y_std = []; y_gap = []; T = []
    WARM = 210
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = resample(d["close"], FACTOR); o = resample(d.get("open", d["close"]), FACTOR, "first")
            h = resample(d.get("high", d["close"]), FACTOR, "max"); l = resample(d.get("low", d["close"]), FACTOR, "min")
            t = resample(d["times"], FACTOR)
        except Exception:
            continue
        n = len(c)
        if n < WARM + 400:
            continue
        F = feats(o, h, l, c, t)
        ys = np.zeros(n); ys[:n - H] = (c[H:] > c[:n - H]).astype(float)          # entrada en t
        yg = np.zeros(n); yg[:n - 1 - H] = (c[1 + H:] > c[1:n - H]).astype(float)  # entrada en t+1 (sin rebote)
        hi = n - 2 - H
        X.append(F[WARM:hi]); y_std.append(ys[WARM:hi]); y_gap.append(yg[WARM:hi]); T.append(t[WARM:hi])
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0)
    y_std = np.concatenate(y_std); y_gap = np.concatenate(y_gap); T = np.concatenate(T)
    order = np.argsort(T); X = X[order]; y_std = y_std[order]; y_gap = y_gap[order]
    N = len(X)
    print("=" * 80)
    print(f"GRADIENT BOOSTING  |  {CACHE} factor {FACTOR}  |  {N} muestras, {X.shape[1]} features  |  BE {BE*100:.1f}%")
    print(f"Walk-forward expandido, {FOLDS} folds")
    print("=" * 80)

    def run(y, etiqueta):
        accs = []
        fold = N // (FOLDS + 1)
        for k in range(1, FOLDS + 1):
            tr_end = fold * k; te_end = fold * (k + 1)
            Xtr, ytr = X[:tr_end], y[:tr_end]
            Xte, yte = X[tr_end:te_end], y[tr_end:te_end]
            clf = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=0.05,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 validation_fraction=0.1, random_state=0)
            clf.fit(Xtr, ytr)
            acc = clf.score(Xte, yte) * 100
            accs.append(acc)
        base = max(y.mean(), 1 - y.mean()) * 100
        print(f"\n[{etiqueta}] baseline(clase mayoritaria) {base:.2f}%")
        print("  folds OOS: " + " ".join(f"{a:.2f}%" for a in accs))
        print(f"  MEDIA OOS: {np.mean(accs):.2f}%   ->  " +
              ("SUPERA break-even" if np.mean(accs) > BE * 100 + 0.3 else "NO supera break-even"))
        return np.mean(accs)

    a_std = run(y_std, "ESTANDAR (entrada en t)")
    a_gap = run(y_gap, "OPERABLE (entrada t+1, sin rebote)")
    print("\n" + "=" * 80)
    print(f"VEREDICTO: estandar {a_std:.2f}% | operable {a_gap:.2f}% | break-even {BE*100:.1f}%")
    if a_gap <= BE * 100 + 0.3:
        print("  El modelo NO predice direccion operable por encima del break-even. Sin edge real.")
    if a_std - a_gap > 0.7:
        print(f"  (La diferencia estandar-operable = {a_std-a_gap:.1f}pt es microestructura no capturable.)")


if __name__ == "__main__":
    main()
