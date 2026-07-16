# backtest_patrones.py - Pinbar, Engulfing y Alligator+Fractals (Williams), con definiciones objetivas.
# 5m, 6 meses, train/test OOS, entrada estandar y OPERABLE (+1 barra, sin rebote).
#   python backtest_patrones.py [cache_dir]

import json, glob, os, sys
import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
H = 2
BE = 1.0 / 1.87
TEST_DAYS = 60
WARM = 30


def sma(x, p):
    n = len(x); cs = np.cumsum(x); out = np.full(n, np.nan)
    out[p - 1:] = (cs[p - 1:] - np.concatenate([[0], cs[:-p]])) / p
    return out


# ---------- generadores: devuelven (call_mask, put_mask) ----------
def gen_pinbar(o, h, l, c):
    n = len(c); rng = h - l; body = np.abs(c - o)
    uw = h - np.maximum(o, c); lw = np.minimum(o, c) - l
    ok = rng > 0
    bull = ok & (lw >= 0.6 * rng) & (body <= 0.3 * rng)   # martillo -> CALL
    bear = ok & (uw >= 0.6 * rng) & (body <= 0.3 * rng)   # estrella -> PUT
    return bull, bear


def gen_engulfing(o, h, l, c):
    n = len(c); cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    green = c > o; red = c < o
    cu[1:] = green[1:] & red[:-1] & (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])   # alcista -> CALL
    cd[1:] = red[1:] & green[:-1] & (c[1:] <= o[:-1]) & (o[1:] >= c[:-1])   # bajista -> PUT
    return cu, cd


def gen_alligator_fractals(o, h, l, c):
    n = len(c); med = (h + l) / 2.0
    jaw = np.full(n, np.nan); teeth = np.full(n, np.nan); lips = np.full(n, np.nan)
    j = sma(med, 13); te = sma(med, 8); li = sma(med, 5)
    jaw[8:] = j[:-8]; teeth[5:] = te[:-5]; lips[3:] = li[:-3]     # shifts de Williams
    up_align = (lips > teeth) & (teeth > jaw)
    dn_align = (lips < teeth) & (teeth < jaw)
    # fractales confirmados (con 2 barras de lag): high mayor que 2 vecinos a cada lado
    up_fr = np.zeros(n, bool); dn_fr = np.zeros(n, bool)
    for i in range(2, n - 2):
        if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
            up_fr[i] = True
        if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
            dn_fr[i] = True
    # nivel del ultimo fractal confirmado (disponible i+2)
    last_up = np.full(n, np.nan); last_dn = np.full(n, np.nan)
    lu = ld = np.nan
    for i in range(n):
        if i - 2 >= 0 and up_fr[i - 2]:
            lu = h[i - 2]
        if i - 2 >= 0 and dn_fr[i - 2]:
            ld = l[i - 2]
        last_up[i] = lu; last_dn[i] = ld
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    # entrada: alineado + ruptura del ultimo fractal
    br_up = (c > last_up); br_dn = (c < last_dn)
    cu[1:] = up_align[1:] & br_up[1:] & ~br_up[:-1] & np.isfinite(last_up[1:])
    cd[1:] = dn_align[1:] & br_dn[1:] & ~br_dn[:-1] & np.isfinite(last_dn[1:])
    return cu, cd


def evaluar(files, gen, nombre):
    tr = [0, 0, 0]; te = [0, 0, 0]
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); o = np.asarray(d.get("open", d["close"]), float)
            h = np.asarray(d.get("high", d["close"]), float); l = np.asarray(d.get("low", d["close"]), float)
            t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < 400:
            continue
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        cu, cd = gen(o, h, l, c)
        cu[:WARM] = False; cd[:WARM] = False
        for i in np.where(cu | cd)[0]:
            if i + 1 + H >= n:
                continue
            call = cu[i]
            gs = (c[i + H] > c[i]) if call else (c[i + H] < c[i])
            gg = (c[i + 1 + H] > c[i + 1]) if call else (c[i + 1 + H] < c[i + 1])
            box = tr if i < split else te
            box[0] += 1; box[1] += int(gs); box[2] += int(gg)
    wr = lambda x, k: x[k] / x[0] * 100 if x[0] else 0
    ev = lambda w: (w / 100 * 0.87 - (1 - w / 100)) * 100
    ope = wr(te, 2)
    print(f"  {nombre:<26} | train {tr[0]:>6} | TEST {te[0]:>6} | std {wr(te,1):.2f}% | "
          f"operable {ope:.2f}% (EV {ev(ope):+.1f}%) {'>BE' if ope>BE*100 else 'pierde'}")
    return ope


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    print("=" * 92)
    print(f"PATRONES  |  {CACHE}  |  {len(files)} pares REALES  |  H={H} | break-even {BE*100:.1f}% | operable=+1 barra")
    print("=" * 92)
    evaluar(files, gen_pinbar, "Pinbar (martillo/estrella)")
    evaluar(files, gen_engulfing, "Engulfing (envolvente)")
    evaluar(files, gen_alligator_fractals, "Alligator + Fractals")
    # tambien la version FADE de cada una (por si el patron predice al reves)
    print("\n  --- version invertida (fade del patron) ---")
    inv = lambda g: (lambda o, h, l, c: (g(o, h, l, c)[1], g(o, h, l, c)[0]))
    evaluar(files, inv(gen_pinbar), "Pinbar-fade")
    evaluar(files, inv(gen_engulfing), "Engulfing-fade")
    evaluar(files, inv(gen_alligator_fractals), "Alligator+Fractals-fade")
    print(f"\nbreak-even {BE*100:.1f}% | techo modelo-libre ~49.5% operable.")


if __name__ == "__main__":
    main()
