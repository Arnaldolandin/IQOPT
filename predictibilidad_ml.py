# predictibilidad_ml.py - Kitchen-sink: regresion logistica L2 (numpy) con ~16 features para
# predecir el SIGNO del movimiento a H barras. Walk-forward temporal, OOS. Pooled sobre pares reales.
# Si un clasificador flexible con muchas features no supera 53.5% OOS, no hay edge que ninguna regla capture.
#   python predictibilidad_ml.py [cache_dir] [factor]

import json, glob, os, sys, math
import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
FACTOR = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H = 2
BE = 1.0 / (1.0 + 0.87)


def resample(a, f, agg="last"):
    if f == 1:
        return np.asarray(a, float)
    n = len(a) // f; b = np.asarray(a[:n * f], float).reshape(n, f)
    return b[:, -1] if agg == "last" else (b.max(1) if agg == "max" else (b.min(1) if agg == "min" else b[:, 0]))


def ema(c, span):
    a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def rsi(c, p=14):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    cs = np.cumsum(up); cd = np.cumsum(dn)
    au = np.full(n, np.nan); ad = np.full(n, np.nan)
    au[p:] = (cs[p:] - cs[:-p]) / p; ad[p:] = (cd[p:] - cd[:-p]) / p
    rs = au / np.where(ad == 0, np.nan, ad); r = 100 - 100 / (1 + rs)
    return np.nan_to_num(r, nan=50.0)


def features_matrix(c, h, l, t):
    n = len(c)
    r = np.zeros(n); r[1:] = np.diff(np.log(np.maximum(c, 1e-12)))
    # rolling std de r (vol)
    vol = np.zeros(n)
    w = 20; cs = np.cumsum(r); cs2 = np.cumsum(r * r)
    vol[w:] = np.sqrt(np.clip((cs2[w:] - cs2[:-w]) / w - ((cs[w:] - cs[:-w]) / w) ** 2, 0, None))
    ml = ema(c, 6) - ema(c, 13); sig = ema(ml, 5); hist = ml - sig
    rr = rsi(c, 14)
    # BB z-score
    w2 = 20; m = np.full(n, np.nan); s = np.full(n, np.nan)
    csa = np.cumsum(c); cs2a = np.cumsum(c * c)
    m[w2:] = (csa[w2:] - csa[:-w2]) / w2
    s[w2:] = np.sqrt(np.clip((cs2a[w2:] - cs2a[:-w2]) / w2 - m[w2:] ** 2, 0, None))
    bbz = np.nan_to_num((c - m) / np.where(s == 0, np.nan, s), nan=0.0)
    # ATR%
    tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    catr = np.cumsum(tr); atr = np.zeros(n); atr[14:] = (catr[14:] - catr[:-14]) / 14
    atrp = atr / np.maximum(np.abs(c), 1e-12)
    hour = ((t % 86400) // 3600)
    hs = np.sin(2 * np.pi * hour / 24); hc = np.cos(2 * np.pi * hour / 24)
    rng = (h - l) / np.maximum(np.abs(c), 1e-12)
    def mom(k):
        mm = np.zeros(n); mm[k:] = c[k:] / np.maximum(c[:-k], 1e-12) - 1; return mm
    feats = np.column_stack([
        r, np.roll(r, 1), np.roll(r, 2), np.roll(r, 3), np.roll(r, 4),
        vol, hist / np.maximum(np.abs(c), 1e-12), (rr - 50) / 50.0, bbz, atrp,
        mom(5), mom(10), mom(20), hs, hc, rng,
    ])
    return feats


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(X, y, l2=1.0, lr=0.1, epochs=300):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    Xall = []; yall = []; tall = []
    WARM = 60
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = resample(d["close"], FACTOR); h = resample(d.get("high", d["close"]), FACTOR, "max")
            l = resample(d.get("low", d["close"]), FACTOR, "min"); t = resample(d["times"], FACTOR)
        except Exception:
            continue
        n = len(c)
        if n < WARM + 500:
            continue
        F = features_matrix(c, h, l, t)
        y = np.zeros(n); y[:n - H] = (c[H:] > c[:n - H]).astype(float)
        lo, hi = WARM, n - H
        Xall.append(F[lo:hi]); yall.append(y[lo:hi]); tall.append(t[lo:hi])
    X = np.vstack(Xall); y = np.concatenate(yall); tt = np.concatenate(tall)
    # ordenar por tiempo y split temporal 70/30
    order = np.argsort(tt); X = X[order]; y = y[order]; tt = tt[order]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # submuestreo temporal (preserva el orden) para que la regresion sea rapida
    if len(X) > 300000:
        stride = len(X) // 300000 + 1
        X = X[::stride]; y = y[::stride]; tt = tt[::stride]
    split = int(len(X) * 0.70)
    mu = X[:split].mean(0); sd = X[:split].std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xtr, ytr = Xs[:split], y[:split]
    Xte, yte = Xs[split:], y[split:]

    print("=" * 84)
    print(f"ML kitchen-sink (regresion logistica L2, 16 features)  |  {CACHE} factor {FACTOR}")
    print(f"muestras train {len(Xtr)}  test {len(Xte)}  |  break-even {BE*100:.1f}%")
    print("=" * 84)
    base = max(yte.mean(), 1 - yte.mean()) * 100
    print(f"Baseline (clase mayoritaria en test): {base:.2f}%")

    best = None
    for l2 in (1.0, 10.0, 100.0):
        w, b = train_logreg(Xtr, ytr, l2=l2, lr=0.3, epochs=200)
        ptr = (sigmoid(Xtr @ w + b) > 0.5).astype(float)
        pte = (sigmoid(Xte @ w + b) > 0.5).astype(float)
        acc_tr = (ptr == ytr).mean() * 100
        acc_te = (pte == yte).mean() * 100
        print(f"  L2={l2:>6}: train {acc_tr:.2f}%  |  TEST(OOS) {acc_te:.2f}%")
        if best is None or acc_te > best[1]:
            best = (l2, acc_te)
    # confianza alta: solo apostar cuando |p-0.5| grande
    w, b = train_logreg(Xtr, ytr, l2=best[0], lr=0.3, epochs=200)
    pte_prob = sigmoid(Xte @ w + b)
    for thr in (0.52, 0.55, 0.60):
        conf = np.abs(pte_prob - 0.5) > (thr - 0.5)
        if conf.sum() > 50:
            pc = (pte_prob[conf] > 0.5).astype(float)
            acc = (pc == yte[conf]).mean() * 100
            print(f"  Solo con confianza |p-0.5|>{thr-0.5:.2f}: {conf.sum()} trades, acc {acc:.2f}%")

    print("-" * 84)
    print(f"MEJOR OOS: {best[1]:.2f}%  (break-even {BE*100:.1f}%)  ->  " +
          ("SUPERA break-even (revisar!)" if best[1] > BE * 100 + 0.3 else "NO supera break-even. Sin edge."))


if __name__ == "__main__":
    main()
