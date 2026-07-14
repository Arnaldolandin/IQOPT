# backtest_valid.py - Validacion estadistica formal de configs candidatas (OFFLINE, cache).
# Para cada config: WR OOS (test) agregado sobre reales, IC Wilson 95%, p-valor (binomial
# 1-cola vs break-even), EV al payout, y estabilidad en 4 sub-periodos cronologicos del test.
import json, os, glob, math
import numpy as np

CACHE_DIR = "cache_closes"; SPLIT = 0.70; WARMUP = 210
PAYOUT = 0.87
BE = 100.0 / (1.0 + PAYOUT)   # break-even WR %
TFS = {"5m": 1, "15m": 3, "30m": 6}

# (nombre, tf, macd, exp_barras, ema_period(0=off), direccion(1 normal / -1 invert))
CANDS = [
    ("baseline 5m normal",        "5m",  (12, 26, 9), 1, 0,   1),
    ("lider: 30m (19,39,9) INV",  "30m", (19, 39, 9), 2, 200, -1),
    ("30m (8,17,9) INV EMA200",   "30m", (8, 17, 9),  1, 200, -1),
    ("30m (8,17,9) INV sinEMA",   "30m", (8, 17, 9),  1, 0,   -1),
    ("30m (12,26,9) INV sinEMA",  "30m", (12, 26, 9), 1, 0,   -1),
]


def ema(c, span):
    a = 2.0/(span+1); out = np.copy(c)
    for i in range(1, len(c)): out[i] = a*c[i] + (1-a)*out[i-1]
    return out


def resample(c5, f):
    if f == 1: return c5
    b = len(c5)//f
    return c5[:b*f].reshape(b, f)[:, -1]


def wilson(w, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = w/n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-hw), 100*(c+hw))


def binom_p_greater(w, n, p0):
    # P(X >= w) bajo Binom(n,p0), aprox normal con correccion de continuidad. 1-cola.
    if n == 0: return 1.0
    mu = n*p0; sd = math.sqrt(n*p0*(1-p0))
    if sd == 0: return 1.0
    z = (w - 0.5 - mu)/sd
    return 0.5*math.erfc(z/math.sqrt(2))


def evaluar(cand):
    _, tf, (fa, sl, si), exp, ema_p, direc = cand
    f = TFS[tf]
    # recolecta (t_index_global_frac, win) del TEST sobre reales -> para estabilidad temporal
    wins = []   # lista de (orden, win)
    for fp in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json"))):
        name = os.path.basename(fp)[:-5]
        if "-OTC" in name: continue  # solo reales
        try:
            d = json.load(open(fp, encoding="utf-8")); c5 = np.asarray(d["closes"], float)
            times = d.get("times", [])
        except Exception:
            continue
        if len(c5) < 400: continue
        closes = resample(c5, f); n = len(closes)
        if n < WARMUP + 30: continue
        cut = int(n*SPLIT)
        ml = ema(closes, fa) - ema(closes, sl); sg = ema(ml, si)
        emat = ema(closes, ema_p) if ema_p else None
        for i in range(max(WARMUP, cut), n - exp):
            lado = None
            if ml[i-1] <= sg[i-1] and ml[i] > sg[i]: lado = "call"
            elif ml[i-1] >= sg[i-1] and ml[i] < sg[i]: lado = "put"
            if lado is None: continue
            ex = lado if direc == 1 else ("put" if lado == "call" else "call")
            if ema_p:
                if (ex == "call" and closes[i] <= emat[i]) or (ex == "put" and closes[i] >= emat[i]):
                    continue
            win = 1 if ((closes[i+exp] > closes[i]) if ex == "call" else (closes[i+exp] < closes[i])) else 0
            # orden temporal aprox: fraccion dentro de su serie (para trocear en cuartiles)
            wins.append(((i - cut) / (n - cut), win))
    return wins


def main():
    print(f"Payout {PAYOUT:.0%} -> break-even WR = {BE:.2f}% | reales, TEST OOS agregado\n")
    print(f"{'Config':30s} {'n':>6s} {'WR%':>7s} {'IC95 Wilson':>16s} {'p(>BE)':>8s} {'EV/trade':>9s}")
    print("-"*90)
    for cand in CANDS:
        wins = evaluar(cand)
        n = len(wins); w = sum(x for _, x in wins)
        if n == 0:
            print(f"{cand[0]:30s}  sin trades"); continue
        wr = 100*w/n
        lo, hi = wilson(w, n)
        p = binom_p_greater(w, n, BE/100)
        ev = (w/n)*PAYOUT - (1 - w/n)   # EV por trade (unidad de stake)
        sig = " <--" if lo > BE else ""   # IC entero sobre BE = significativo
        print(f"{cand[0]:30s} {n:6d} {wr:6.2f}% [{lo:5.1f},{hi:5.1f}] {p:8.3f} {ev*100:+7.1f}%{sig}")

    # Estabilidad temporal del lider (cuartiles del test)
    print("\nEstabilidad temporal del LIDER (cuartiles cronologicos del TEST):")
    wins = evaluar(CANDS[1])
    for q in range(4):
        seg = [x for fr, x in wins if q/4 <= fr < (q+1)/4]
        if seg:
            wr = 100*sum(seg)/len(seg)
            print(f"  Q{q+1}: WR {wr:5.1f}% (n={len(seg):4d})")
    print(f"\nRegla: solo es edge real si el IC95 queda ENTERO por encima de {BE:.1f}% (p<0.05).")


if __name__ == "__main__":
    main()
