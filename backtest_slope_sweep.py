# backtest_slope_sweep.py - Barrido del umbral min_macd_slope, OFFLINE desde cache.
# Cadena: cruce MACD(config) -> a favor de EMA(config) -> slope normalizada >= umbral.
# Muestra WR de TEST y nº de trades para cada umbral, real vs OTC. Split 70/30.
import json, os, glob
import numpy as np

CACHE_DIR = "cache_closes"; SPLIT = 0.70; WARMUP = 120
THRESHOLDS = [0.0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002]


def ema(c, span):
    a = 2.0/(span+1); out = np.copy(c)
    for i in range(1, len(c)): out[i] = a*c[i] + (1-a)*out[i-1]
    return out


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    fa, sl, si = cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]
    op = cfg["operacion"]; h = max(1, round(op["expiry_min"]*60/op["timeframe_seg"]))
    ema_p = op.get("ema_trend", 50)

    # por grupo: lista de (slope_signed, lado, win) de test tras pasar EMA
    data = {"real": [], "otc": []}
    for fp in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json"))):
        name = os.path.basename(fp)[:-5]; grp = "otc" if "-OTC" in name else "real"
        try: closes = np.asarray(json.load(open(fp, encoding="utf-8"))["closes"], float)
        except Exception: continue
        n = len(closes)
        if n < WARMUP+30: continue
        cut = int(n*SPLIT)
        macd_l = ema(closes, fa)-ema(closes, sl); sig_l = ema(macd_l, si); hist = macd_l-sig_l
        emat = ema(closes, ema_p)
        for i in range(max(WARMUP, cut), n-h):
            lado = None
            if macd_l[i-1] <= sig_l[i-1] and macd_l[i] > sig_l[i]: lado = "call"
            elif macd_l[i-1] >= sig_l[i-1] and macd_l[i] < sig_l[i]: lado = "put"
            if lado is None: continue
            precio = closes[i]
            if not ((lado == "call" and precio > emat[i]) or (lado == "put" and precio < emat[i])): continue
            slope = (hist[i]-hist[i-1])/(abs(precio) or 1.0)
            win = 1 if ((closes[i+h] > closes[i]) if lado == "call" else (closes[i+h] < closes[i])) else 0
            data[grp].append((slope, lado, win))

    print(f"MACD({fa},{sl},{si}) + EMA{ema_p} a-favor + slope>=umbral | exp h={h}\n")
    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        arr = data[g]
        print("=" * 58)
        print(f"GRUPO {g.upper()} (BE ~{be}%) | {len(arr)} senales tras EMA (test)")
        print("=" * 58)
        print(f"{'umbral':>10s} {'WR test':>9s} {'n':>8s} {'%pasa':>7s}")
        for t in THRESHOLDS:
            sel = [w for (s, l, w) in arr
                   if (l == "call" and s >= t) or (l == "put" and s <= -t)]
            nn = len(sel)
            wr = (100.0*sum(sel)/nn) if nn else float("nan")
            mark = " *" if nn and wr >= be else ""
            print(f"{t:>9.5%} {wr:8.2f}% {nn:8d} {100*nn/len(arr) if arr else 0:6.1f}%{mark}")
        print()


if __name__ == "__main__":
    main()
