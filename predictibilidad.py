# predictibilidad.py - ¿Existe ALGUNA predictibilidad real en los precios? (modelo-libre)
# No usa estrategias. Mide la estructura estadistica cruda: autocorrelacion de retornos,
# variance ratio (random walk vs tendencia vs reversion), y accuracy direccional con
# hueco de 1 barra (para descartar el rebote bid-ask). Todo por timeframe, pooled y por hora.
#   python predictibilidad.py

import json, math, glob, os
import numpy as np

TFS = [("cache_ohlc_1m", 1, "1m"), ("cache_ohlc_5m", 1, "5m"),
       ("cache_ohlc_5m", 3, "15m"), ("cache_ohlc_5m", 6, "30m")]


def resample_last(a, f):
    if f == 1:
        return np.asarray(a, float)
    n = len(a) // f
    return np.asarray(a[:n * f], float).reshape(n, f)[:, -1]


def cargar(cache, factor):
    files = [f for f in sorted(glob.glob(os.path.join(cache, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    out = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = resample_last(d["close"], factor)
            t = resample_last(d["times"], factor)
        except Exception:
            continue
        if len(c) > 2000:
            out.append((os.path.basename(f)[:-5], c, t))
    return out


def autocorr(x, k):
    if len(x) < k + 30:
        return np.nan
    a = x[:-k]; b = x[k:]
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def variance_ratio(r, q):
    # VR(q) con bloques NO solapados
    n = len(r) // q
    if n < 20:
        return np.nan
    v1 = r.var()
    if v1 == 0:
        return np.nan
    block = r[:n * q].reshape(n, q).sum(axis=1)
    return float(block.var() / (q * v1))


def main():
    print("=" * 92)
    print("AUDITORIA DE PREDICTIBILIDAD (modelo-libre) — ¿hay edge crudo antes de cualquier estrategia?")
    print("=" * 92)
    print(f"{'TF':>5} {'pares':>6} {'N ret':>9} | {'AC(lag1)':>9} {'AC(lag2)':>9} | {'VR(2)':>7} {'VR(5)':>7} | "
          f"{'signAcc':>8} {'signGap1':>9}")
    print("-" * 92)
    # AC(lag1): incluye rebote bid-ask (negativo espurio).  AC(lag2): salta el rebote -> estructura mas real.
    # VR<1 reversion, >1 tendencia, =1 random walk.
    # signAcc: acierto de 'la proxima barra repite signo de la anterior' (momentum). 50% = sin señal.
    # signGap1: predecir signo de r_{t+1} desde r_{t-1} (hueco de 1) -> sin contaminacion de microestructura.
    resumen = {}
    for cache, factor, tfname in TFS:
        data = cargar(cache, factor)
        if not data:
            continue
        ac1s, ac2s, vr2s, vr5s = [], [], [], []
        sign_ok = sign_tot = 0
        gap_ok = gap_tot = 0
        Ntot = 0
        se_pairs_sig = 0
        for name, c, t in data:
            r = np.diff(np.log(np.maximum(c, 1e-12)))
            r = r[np.isfinite(r)]
            if len(r) < 500:
                continue
            Ntot += len(r)
            a1 = autocorr(r, 1); a2 = autocorr(r, 2)
            ac1s.append(a1); ac2s.append(a2)
            vr2s.append(variance_ratio(r, 2)); vr5s.append(variance_ratio(r, 5))
            # significancia individual del lag1 (SE ~ 1/sqrt(N))
            if not np.isnan(a1) and abs(a1) > 2.0 / math.sqrt(len(r)):
                se_pairs_sig += 1
            # sign momentum: sign(r_t)==sign(r_{t-1})
            s = np.sign(r)
            m = s[1:] == s[:-1]
            valid = (s[1:] != 0) & (s[:-1] != 0)
            sign_ok += int((m & valid).sum()); sign_tot += int(valid.sum())
            # gap-1: predecir sign(r_t) desde sign(r_{t-2})  (salta la barra intermedia)
            g = s[2:] == s[:-2]
            gv = (s[2:] != 0) & (s[:-2] != 0)
            gap_ok += int((g & gv).sum()); gap_tot += int(gv.sum())
        f = lambda L: np.nanmean(L) if L else np.nan
        signacc = sign_ok / sign_tot * 100 if sign_tot else 0
        gapacc = gap_ok / gap_tot * 100 if gap_tot else 0
        resumen[tfname] = (f(ac1s), f(ac2s), f(vr2s), f(vr5s), signacc, gapacc, se_pairs_sig, len(data))
        print(f"{tfname:>5} {len(data):>6} {Ntot:>9} | {f(ac1s):>+9.4f} {f(ac2s):>+9.4f} | "
              f"{f(vr2s):>7.3f} {f(vr5s):>7.3f} | {signacc:>7.2f}% {gapacc:>8.2f}%")

    print("\nLECTURA:")
    print("  - AC(lag1) muy negativo pero AC(lag2)~0  => rebote bid-ask (microestructura), NO edge operable.")
    print("  - VR ~ 1.0 => random walk (sin memoria). VR<<1 reversion real, VR>>1 tendencia real.")
    print("  - signGap1 debe ser >53.5% para ser operable en binarias (con hueco, sin microestructura).")
    print(f"  - Umbral de significancia por par (lag1): |AC| > 2/sqrt(N).")

    print("\n" + "=" * 92)
    print("POR HORA (5m) — AC(lag1) y accuracy con hueco, para ver si ALGUNA hora tiene estructura real")
    print("=" * 92)
    data = cargar("cache_ohlc_5m", 1)
    ac_h = {h: [] for h in range(24)}
    gap_h = {h: [0, 0] for h in range(24)}
    for name, c, t in data:
        r = np.diff(np.log(np.maximum(c, 1e-12)))
        hr = ((t[1:] % 86400) // 3600).astype(int)
        s = np.sign(r)
        for h in range(24):
            mask = hr == h
            rh = r[mask]
            if len(rh) > 200:
                ac_h[h].append(autocorr(rh, 1))
            # gap-1 por hora
            idx = np.where(mask)[0]
            idx = idx[idx >= 2]
            if len(idx):
                pred = s[idx - 2]; act = s[idx]
                v = (pred != 0) & (act != 0)
                gap_h[h][0] += int((pred[v] == act[v]).sum()); gap_h[h][1] += int(v.sum())
    print(f"{'UTC':>3} {'CL':>4} | {'AC(lag1)':>9} | {'signGap1':>9} {'ops':>7}")
    print("-" * 92)
    for h in range(24):
        a = np.nanmean(ac_h[h]) if ac_h[h] else np.nan
        g = gap_h[h][0] / gap_h[h][1] * 100 if gap_h[h][1] else 0
        mk = "  <-- >53.5%" if g > 53.5 and gap_h[h][1] > 500 else ""
        print(f"{h:>3} {(h-4)%24:>3}h | {a:>+9.4f} | {g:>8.2f}% {gap_h[h][1]:>7}{mk}")


if __name__ == "__main__":
    main()
