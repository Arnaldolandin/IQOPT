# backtest_adx.py - Estrategias con ADX/DI. Testea:
#  A) Cruce DI+/DI- (entrada de tendencia), opcional filtro ADX>thr
#  B) MACD crossover SOLO con ADX>thr (momentum en tendencia fuerte)
#  C) Reversion Bollinger SOLO con ADX<thr (rango)
# Barre umbrales de ADX. WR/EV train/test, entrada estandar y OPERABLE (+1 barra).
#   python backtest_adx.py [cache_dir]

import json, glob, os, sys
import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
H = 2
BE = 1.0 / 1.87
TEST_DAYS = 60
ADXP = 14


def wilder(x, p):
    n = len(x); out = np.zeros(n)
    if n <= p:
        return out
    out[p] = x[1:p + 1].sum()
    for i in range(p + 1, n):
        out[i] = out[i - 1] - out[i - 1] / p + x[i]
    return out


def adx_di(h, l, c, p=14):
    n = len(c)
    up = np.zeros(n); dn = np.zeros(n); tr = np.zeros(n)
    up[1:] = h[1:] - h[:-1]; dn[1:] = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    str_ = wilder(tr, p); spdm = wilder(pdm, p); sndm = wilder(ndm, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * np.where(str_ > 0, spdm / str_, 0)
        ndi = 100 * np.where(str_ > 0, sndm / str_, 0)
        dx = 100 * np.abs(pdi - ndi) / np.where(pdi + ndi > 0, pdi + ndi, 1)
    adx = np.zeros(n)
    if n > 2 * p:
        adx[2 * p] = dx[p + 1:2 * p + 1].mean()
        for i in range(2 * p + 1, n):
            adx[i] = (adx[i - 1] * (p - 1) + dx[i]) / p
    return adx, pdi, ndi


def ema(c, s):
    a = 2.0 / (s + 1); o = np.copy(c)
    for i in range(1, len(c)):
        o[i] = a * c[i] + (1 - a) * o[i - 1]
    return o


def bollinger(c, p=20, k=2.0):
    n = len(c); cs = np.cumsum(c); cs2 = np.cumsum(c * c)
    m = np.full(n, np.nan); s = np.full(n, np.nan)
    mm = (cs[p:] - cs[:-p]) / p; m2 = (cs2[p:] - cs2[:-p]) / p
    s[p:] = np.sqrt(np.clip(m2 - mm * mm, 0, None)); m[p:] = mm
    return m - k * s, m + k * s


def cross_up(a, b):
    n = len(a); m = np.zeros(n, bool); m[1:] = (a[:-1] <= b[:-1]) & (a[1:] > b[1:]); return m


def cross_below_arr(x, lv):
    n = len(x); m = np.zeros(n, bool); m[1:] = (x[:-1] >= lv[:-1]) & (x[1:] < lv[1:]); return m


def cross_above_arr(x, lv):
    n = len(x); m = np.zeros(n, bool); m[1:] = (x[:-1] <= lv[:-1]) & (x[1:] > lv[1:]); return m


WARM = 60


def evaluar(files, gen, nombre):
    tr = [0, 0, 0]; te = [0, 0, 0]
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); h = np.asarray(d.get("high", d["close"]), float)
            l = np.asarray(d.get("low", d["close"]), float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < 400:
            continue
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        cu, cd = gen(c, h, l)
        cu[:WARM] = False; cd[:WARM] = False
        idx = np.where(cu | cd)[0]
        for i in idx:
            if i + 1 + H >= n:
                continue
            call = cu[i]
            gs = (c[i + H] > c[i]) if call else (c[i + H] < c[i])
            gg = (c[i + 1 + H] > c[i + 1]) if call else (c[i + 1 + H] < c[i + 1])
            box = tr if i < split else te
            box[0] += 1; box[1] += int(gs); box[2] += int(gg)
    wr = lambda x, k: x[k] / x[0] * 100 if x[0] else 0
    ev = lambda w: (w / 100 * 0.87 - (1 - w / 100)) * 100
    print(f"  {nombre:<34} | tr {tr[0]:>6} | TEST {te[0]:>6} std {wr(te,1):.2f}% "
          f"| operable {wr(te,2):.2f}% (EV {ev(wr(te,2)):+.1f}%)")
    return wr(te, 2)


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    print("=" * 96)
    print(f"ESTRATEGIAS ADX  |  {CACHE}  |  {len(files)} pares REALES  |  break-even {BE*100:.1f}%  |  operable = entrada +1")
    print("=" * 96)

    def gen_di(thr):
        def g(c, h, l):
            adx, pdi, ndi = adx_di(h, l, c, ADXP)
            cu = cross_up(pdi, ndi); cd = cross_up(ndi, pdi)
            if thr:
                f = adx >= thr; cu = cu & f; cd = cd & f
            return cu, cd
        return g

    def gen_macd_adx(thr):
        def g(c, h, l):
            adx, _, _ = adx_di(h, l, c, ADXP)
            ml = ema(c, 6) - ema(c, 13); sl = ema(ml, 5); diff = ml - sl
            cu = np.zeros(len(c), bool); cd = np.zeros(len(c), bool)
            cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0); cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
            f = adx >= thr; return cu & f, cd & f
        return g

    def gen_bb_lowadx(thr):
        def g(c, h, l):
            adx, _, _ = adx_di(h, l, c, ADXP)
            lo, up = bollinger(c, 20, 2.0)
            cu = cross_below_arr(c, lo); cd = cross_above_arr(c, up)
            f = adx <= thr; return cu & f, cd & f
        return g

    print("\n[A] Cruce DI+/DI- (entrada de tendencia)")
    for thr in (0, 20, 25, 30):
        evaluar(files, gen_di(thr), f"DI cross, ADX>={thr}" if thr else "DI cross (sin filtro)")
    print("\n[B] MACD crossover con filtro ADX alto (momentum en tendencia)")
    for thr in (20, 25, 30, 35):
        evaluar(files, gen_macd_adx(thr), f"MACD + ADX>={thr}")
    print("\n[C] Reversion Bollinger con ADX bajo (rango)")
    for thr in (30, 25, 20, 15):
        evaluar(files, gen_bb_lowadx(thr), f"BB rev + ADX<={thr}")
    print(f"\nbreak-even {BE*100:.1f}%. Recordatorio: techo modelo-libre ~49.5% operable.")


if __name__ == "__main__":
    main()
