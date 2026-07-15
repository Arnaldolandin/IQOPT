# backtest_reversion.py - Prueba estrategias de REVERSION (contra-tendencia) + filtro ATR.
# Motivacion: a 1m el MACD-momentum pierde (~47%), lo que sugiere que revertir podria ganar.
# Estrategias: fade-MACD (contra el cruce), RSI extremos, Bollinger. Todas x umbral ATR.
# Train/test OOS con control de azar.
#   python backtest_reversion.py [cache_dir] [test_days]

import json, math, os, glob, sys
import numpy as np

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
H = 2
WARMUP = 60
ATRS = [0.0, 0.001, 0.002, 0.003, 0.005]

CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_1m"
TEST_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def roll_sum(x, p):
    cs = np.cumsum(x); n = len(x); out = np.full(n, np.nan)
    out[p:] = cs[p:] - cs[:-p]
    out[p - 1] = cs[p - 1]
    return out


def rsi(c, period=14):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = roll_sum(up, period) / period; ad = roll_sum(dn, period) / period
    rs = au / np.where(ad == 0, np.nan, ad)
    r = 100 - 100 / (1 + rs)
    r[(ad == 0) & (au > 0)] = 100.0
    r[(ad == 0) & (au == 0)] = 50.0
    return r


def bollinger(c, period=20, k=2.0):
    n = len(c); cs = np.cumsum(c); cs2 = np.cumsum(c * c)
    mean = np.full(n, np.nan); std = np.full(n, np.nan)
    m = (cs[period:] - cs[:-period]) / period
    m2 = (cs2[period:] - cs2[:-period]) / period
    var = np.clip(m2 - m * m, 0, None)
    mean[period:] = m; std[period:] = np.sqrt(var)
    return mean - k * std, mean + k * std


def atr_pct_series(h, l, c, period=14):
    n = len(c); tr = np.empty(n); tr[0] = h[0] - l[0]; pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    cs = np.cumsum(tr); atr = np.full(n, np.nan); atr[period:] = (cs[period:] - cs[:-period]) / period
    pr = np.abs(c); pr[pr == 0] = 1.0
    return np.nan_to_num(atr / pr, nan=0.0)


def cross_below(x, level):
    n = len(x); m = np.zeros(n, bool)
    lv = level if np.ndim(level) else np.full(n, level, float)
    m[1:] = (x[:-1] >= lv[:-1]) & (x[1:] < lv[1:])
    return m


def cross_above(x, level):
    n = len(x); m = np.zeros(n, bool)
    lv = level if np.ndim(level) else np.full(n, level, float)
    m[1:] = (x[:-1] <= lv[:-1]) & (x[1:] > lv[1:])
    return m


def signals(name, c, h, l):
    """Devuelve (call_mask, put_mask) para la estrategia de reversion `name`."""
    n = len(c)
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    if name == "fade_macd":
        ml = ema(c, 6) - ema(c, 13); sl = ema(ml, 5); diff = ml - sl
        up = np.zeros(n, bool); dn = np.zeros(n, bool)
        up[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)   # cruce alcista
        dn[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)   # cruce bajista
        cu = dn.copy()   # FADE: cruce bajista -> CALL
        cd = up.copy()   # FADE: cruce alcista -> PUT
    elif name == "rsi_25_75":
        r = rsi(c, 14)
        cu = cross_below(r, 25)   # sobreventa -> CALL (rebote)
        cd = cross_above(r, 75)   # sobrecompra -> PUT
    elif name == "rsi_20_80":
        r = rsi(c, 14)
        cu = cross_below(r, 20); cd = cross_above(r, 80)
    elif name == "bollinger":
        lo, up = bollinger(c, 20, 2.0)
        cu = cross_below(c, lo)   # bajo banda inferior -> CALL
        cd = cross_above(c, up)   # sobre banda superior -> PUT
    cu[:WARMUP] = False; cd[:WARMUP] = False
    return cu, cd


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def pval(w, n, p0=BREAK_EVEN):
    if n == 0:
        return 1.0
    sd = math.sqrt(n * p0 * (1 - p0))
    return 1.0 if sd == 0 else 0.5 * math.erfc(((w - 0.5 - n * p0) / sd) / math.sqrt(2))


STRATS = ["fade_macd", "rsi_25_75", "rsi_20_80", "bollinger"]


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if not files:
        print(f"No hay datos REALES en {CACHE_DIR}/"); return

    pool = {(s, a): {"tr": [0, 0], "te": [0, 0]} for s in STRATS for a in ATRS}
    procesados = 0
    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
            c = np.asarray(d["close"], float)
            h = np.asarray(d.get("high", d["close"]), float)
            l = np.asarray(d.get("low", d["close"]), float)
            t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < WARMUP + 300:
            continue
        procesados += 1
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        wc = np.zeros(n, bool); wp = np.zeros(n, bool)
        wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
        ap = atr_pct_series(h, l, c, 14)
        for s in STRATS:
            cu, cd = signals(s, c, h, l)
            for a in ATRS:
                vok = ap >= a if a > 0 else np.ones(n, bool)
                cuf = cu & vok; cdf = cd & vok
                for seg, lo, hi in (("tr", WARMUP, max(WARMUP, split - H)), ("te", split, n - H)):
                    rng = np.zeros(n, bool); rng[lo:hi] = True
                    cs = cuf & rng; ps = cdf & rng
                    tr = int(cs.sum() + ps.sum()); wins = int((wc & cs).sum() + (wp & ps).sum())
                    pool[(s, a)][seg][0] += tr; pool[(s, a)][seg][1] += wins

    def wr(x):
        return x[1] / x[0] if x[0] else 0.0

    W = 96
    print("=" * W)
    print(f"REVERSION + ATR  |  {CACHE_DIR}  |  {procesados} pares REALES  |  H={H}  |  "
          f"payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})  |  OOS = ultimos {TEST_DAYS}d")
    print("=" * W)
    print(f"{'ESTRATEGIA':>12} {'ATR>=':>7} | {'TRAINtr':>8} {'wr':>6} | {'TESTtr':>8} {'wr':>6} {'ev':>7} | {'p(test)':>8}")
    print("-" * W)
    best = None
    for s in STRATS:
        for a in ATRS:
            tr = pool[(s, a)]["tr"]; te = pool[(s, a)]["te"]
            ewr = wr(te); p = pval(te[1], te[0])
            mark = ""
            if te[0] >= 100 and ewr > BREAK_EVEN:
                mark = "  <<BE+" if p < 0.05 else "  ~BE"
            print(f"{s:>12} {a:>7.3f} | {tr[0]:>8} {wr(tr)*100:>5.1f}% | {te[0]:>8} {ewr*100:>5.1f}% "
                  f"{ev_de_wr(ewr)*100:>+6.1f}% | {p:>8.3f}{mark}")
            if te[0] >= 100 and (best is None or ev_de_wr(ewr) > best[1]):
                best = ((s, a), ev_de_wr(ewr), ewr, te[0], p)
        print("-" * W)

    if best:
        (s, a), ev, wrb, ntr, p = best
        print(f"\nMEJOR: {s} ATR>={a:.3f}  ->  test WR {wrb*100:.1f}% EV {ev*100:+.1f}% ({ntr} trades, p={p:.3f})")
        if ev > 0 and p < 0.05:
            print("  => SUPERA break-even OOS con significancia. Vale la pena profundizar.")
        elif ev > 0:
            print("  => EV>0 pero sin significancia fuerte. Hipotesis, validar con mas datos.")
        else:
            print("  => Sigue -EV. La reversion simple tampoco vence el payout.")


if __name__ == "__main__":
    main()
