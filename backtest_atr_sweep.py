# backtest_atr_sweep.py - Mide si el filtro ATR (volatilidad) mejora la estrategia.
# Barre umbrales de ATR/precio sobre el MACD crossover. Train/test OOS.
# Uso:  python backtest_atr_sweep.py [cache_dir] [test_days]
#   por defecto cache_ohlc_1m (TF actual 1m), test = ultimos 30 dias.

import json, math, os, glob, sys
import numpy as np

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
H = 2
WARMUP = 210
MACD = (6, 13, 5)
ATR_PERIOD = 14
ATRS = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005]

CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_1m"
TEST_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_lines(c):
    ml = ema(c, MACD[0]) - ema(c, MACD[1])
    return ml, ema(ml, MACD[2])


def atr_pct_series(h, l, c, period):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    csum = np.cumsum(tr)
    atr = np.full(n, np.nan)
    atr[period:] = (csum[period:] - csum[:-period]) / period
    pr = np.abs(c); pr[pr == 0] = 1.0
    return atr / pr


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def pval(w, n, p0=BREAK_EVEN):
    if n == 0:
        return 1.0
    sd = math.sqrt(n * p0 * (1 - p0))
    return 1.0 if sd == 0 else 0.5 * math.erfc(((w - 0.5 - n * p0) / sd) / math.sqrt(2))


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    files = [f for f in files if "-OTC" not in os.path.basename(f)]
    if not files:
        print(f"No hay datos REALES en {CACHE_DIR}/")
        return

    pool = {a: {"tr": [0, 0], "te": [0, 0]} for a in ATRS}
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
        ml, sl = macd_lines(c)
        diff = ml - sl
        cu = np.zeros(n, bool); cd = np.zeros(n, bool)
        cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
        cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
        cu[:WARMUP] = False; cd[:WARMUP] = False
        ap = atr_pct_series(h, l, c, ATR_PERIOD)
        ap = np.nan_to_num(ap, nan=0.0)

        for a in ATRS:
            volok = ap >= a if a > 0 else np.ones(n, bool)
            cuf = cu & volok; cdf = cd & volok
            for seg, lo, hi in (("tr", WARMUP, max(WARMUP, split - H)), ("te", split, n - H)):
                rng = np.zeros(n, bool); rng[lo:hi] = True
                cs = cuf & rng; ps = cdf & rng
                tr = int(cs.sum() + ps.sum())
                wins = int((wc & cs).sum() + (wp & ps).sum())
                pool[a][seg][0] += tr; pool[a][seg][1] += wins

    def wr(s):
        return s[1] / s[0] if s[0] else 0.0

    W = 92
    print("=" * W)
    print(f"SWEEP ATR  |  {CACHE_DIR}  |  {procesados} pares REALES  |  MACD{MACD}  |  ATR({ATR_PERIOD})  |  "
          f"payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})  |  OOS = ultimos {TEST_DAYS}d")
    print("=" * W)
    print(f"{'ATR>=':>9} {'(%)':>7} | {'TRAINtr':>9} {'wr':>6} {'ev':>7} | {'TESTtr':>9} {'wr':>6} {'ev':>7} | {'p(test)':>8}")
    print("-" * W)
    for a in ATRS:
        twr = wr(pool[a]["tr"]); ewr = wr(pool[a]["te"])
        te = pool[a]["te"]
        p = pval(te[1], te[0])
        mark = "  <BE+" if (te[0] >= 100 and ewr > BREAK_EVEN) else ""
        print(f"{a:>9.4f} {a*100:>6.2f}% | {pool[a]['tr'][0]:>9} {twr*100:>5.1f}% {ev_de_wr(twr)*100:>+6.1f}% | "
              f"{te[0]:>9} {ewr*100:>5.1f}% {ev_de_wr(ewr)*100:>+6.1f}% | {p:>8.3f}{mark}")

    # mejor umbral por EV test (con muestra)
    cand = [(a, wr(pool[a]["te"])) for a in ATRS if pool[a]["te"][0] >= 100]
    if cand:
        best = max(cand, key=lambda x: x[1])
        base_ev = ev_de_wr(wr(pool[0.0]["te"]))
        print("-" * W)
        print(f"Base (ATR off): EV test {base_ev*100:+.1f}%  |  Mejor umbral ATR={best[0]:.4f} "
              f"({best[0]*100:.2f}%): WR {best[1]*100:.1f}% EV {ev_de_wr(best[1])*100:+.1f}%")
        mejora = (ev_de_wr(best[1]) - base_ev) * 100
        print(f"Mejora del ATR vs sin filtro: {mejora:+.1f} pts de EV  ->  " +
              ("AYUDA" if mejora > 0.5 and ev_de_wr(best[1]) > 0 else
               "reduce perdida pero sigue -EV" if mejora > 0.5 else "no ayuda"))


if __name__ == "__main__":
    main()
