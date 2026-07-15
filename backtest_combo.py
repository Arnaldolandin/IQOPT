# backtest_combo.py - Compara usar RSI y BB por separado vs combinadas (union / confluencia).
# MACD queda fuera (es momentum, pierde y cancela la reversion).
#   python backtest_combo.py [cache_dir] [test_days]
import json, math, os, glob, sys
import numpy as np

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
H = 2
WARMUP = 60
CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_30m"
TEST_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RSI_OS, RSI_OB = 25, 75
BB_P, BB_K = 20, 2.0
ATR_MIN = 0.0   # se puede subir para exigir volatilidad


def roll_sum(x, p):
    cs = np.cumsum(x); n = len(x); out = np.full(n, np.nan)
    out[p:] = cs[p:] - cs[:-p]; out[p - 1] = cs[p - 1]; return out


def rsi(c, period=14):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = roll_sum(up, period) / period; ad = roll_sum(dn, period) / period
    rs = au / np.where(ad == 0, np.nan, ad); r = 100 - 100 / (1 + rs)
    r[(ad == 0) & (au > 0)] = 100.0; r[(ad == 0) & (au == 0)] = 50.0; return r


def bollinger(c, period=20, k=2.0):
    n = len(c); cs = np.cumsum(c); cs2 = np.cumsum(c * c)
    mean = np.full(n, np.nan); std = np.full(n, np.nan)
    m = (cs[period:] - cs[:-period]) / period; m2 = (cs2[period:] - cs2[:-period]) / period
    var = np.clip(m2 - m * m, 0, None); mean[period:] = m; std[period:] = np.sqrt(var)
    return mean - k * std, mean + k * std


def atr_pct(h, l, c, period=14):
    n = len(c); tr = np.empty(n); tr[0] = h[0] - l[0]; pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    cs = np.cumsum(tr); a = np.full(n, np.nan); a[period:] = (cs[period:] - cs[:-period]) / period
    pr = np.abs(c); pr[pr == 0] = 1.0; return np.nan_to_num(a / pr, nan=0.0)


def cross_into(state):
    m = np.zeros(len(state), bool); m[1:] = (~state[:-1]) & state[1:]; return m


def modos(c, h, l):
    r = rsi(c, 14); lo, up = bollinger(c, BB_P, BB_K)
    rsi_call_state = r < RSI_OS; rsi_put_state = r > RSI_OB
    bb_call_state = c < lo; bb_put_state = c > up
    rc = cross_into(rsi_call_state); rp = cross_into(rsi_put_state)
    bc = cross_into(bb_call_state); bp = cross_into(bb_put_state)
    out = {}
    out["RSI"] = (rc, rp)
    out["BB"] = (bc, bp)
    out["UNION"] = (rc | bc, rp | bp)                              # cualquiera dispara
    out["CONFLUENCIA"] = (cross_into(rsi_call_state & bb_call_state),
                          cross_into(rsi_put_state & bb_put_state))  # ambas a la vez
    return out


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    names = ["RSI", "BB", "UNION", "CONFLUENCIA"]
    pool = {m: {"tr": [0, 0], "te": [0, 0]} for m in names}
    procesados = 0
    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
            c = np.asarray(d["close"], float); h = np.asarray(d.get("high", d["close"]), float)
            l = np.asarray(d.get("low", d["close"]), float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < WARMUP + 300:
            continue
        procesados += 1
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        wc = np.zeros(n, bool); wp = np.zeros(n, bool)
        wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
        vok = atr_pct(h, l, c, 14) >= ATR_MIN if ATR_MIN > 0 else np.ones(n, bool)
        mm = modos(c, h, l)
        for m in names:
            cu, cd = mm[m]
            cu = cu & vok; cd = cd & vok
            cu[:WARMUP] = False; cd[:WARMUP] = False
            for seg, lo_, hi_ in (("tr", WARMUP, max(WARMUP, split - H)), ("te", split, n - H)):
                rng = np.zeros(n, bool); rng[lo_:hi_] = True
                cs = cu & rng; ps = cd & rng
                tr = int(cs.sum() + ps.sum()); wins = int((wc & cs).sum() + (wp & ps).sum())
                pool[m][seg][0] += tr; pool[m][seg][1] += wins

    def wr(s):
        return s[1] / s[0] if s[0] else 0.0

    def ev(w):
        return w * PAYOUT - (1 - w)

    W = 88
    print("=" * W)
    print(f"COMBO RSI/BB  |  {CACHE_DIR}  |  {procesados} pares REALES  |  ATRmin={ATR_MIN}  |  "
          f"BE {BE:.1%}  |  OOS {TEST_DAYS}d")
    print("=" * W)
    print(f"{'MODO':>12} | {'TRAINtr':>8} {'wr':>6} | {'TESTtr':>8} {'wr':>6} {'ev':>7}  {'vs BE':>6}")
    print("-" * W)
    for m in names:
        twr = wr(pool[m]["tr"]); ewr = wr(pool[m]["te"])
        mark = "  +EV" if ev(ewr) > 0 else ""
        print(f"{m:>12} | {pool[m]['tr'][0]:>8} {twr*100:>5.1f}% | {pool[m]['te'][0]:>8} {ewr*100:>5.1f}% "
              f"{ev(ewr)*100:>+6.1f}%  {(ewr-BE)*100:>+5.1f}{mark}")


if __name__ == "__main__":
    main()
