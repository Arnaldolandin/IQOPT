# backtest_actual.py - Backtestea la config EXACTA que opera el bot ahora, OFFLINE desde cache.
# Filtros: (1) cruce MACD, (2) a favor de EMA(ema_trend), (3) pendiente histograma normalizada
# >= min_macd_slope. Expiry = h velas (turbo). WR test agregado, real vs OTC. Split 70/30.
#
#   .venv314\Scripts\python.exe backtest_actual.py
import json, os, glob
import numpy as np

CACHE_DIR = "cache_closes"
SPLIT = 0.70
WARMUP = 120


def ema(c, span):
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    fa, sl, si = cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]
    op = cfg["operacion"]
    tf_seg = op["timeframe_seg"]
    h = max(1, round(op["expiry_min"] * 60 / tf_seg))
    ema_p = op.get("ema_trend", 50)
    min_slope = op.get("min_macd_slope", 0.0)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    print(f"Config viva: MACD({fa},{sl},{si}) | EMA{ema_p} a-favor | slope>={min_slope:.4%} | exp {op['expiry_min']}m (h={h})")
    print(f"Cache: {len(files)} pares\n")

    # agg[grp] = {"base":[w,n], "ema":[w,n], "ema_slope":[w,n]} en test
    agg = {g: {"base": [0, 0], "ema": [0, 0], "ema_slope": [0, 0]} for g in ("real", "otc")}
    n_real = n_otc = 0

    for fp in files:
        name = os.path.basename(fp)[:-5]
        grp = "otc" if "-OTC" in name else "real"
        try:
            closes = np.asarray(json.load(open(fp, encoding="utf-8"))["closes"], float)
        except Exception:
            continue
        n = len(closes)
        if n < WARMUP + 30:
            continue
        if grp == "real": n_real += 1
        else: n_otc += 1
        cut = int(n * SPLIT)

        macd_l = ema(closes, fa) - ema(closes, sl)
        sig_l = ema(macd_l, si)
        hist = macd_l - sig_l
        ema_t = ema(closes, ema_p)

        for i in range(WARMUP, n - h):
            if i < cut:
                continue
            # cruce
            lado = None
            if macd_l[i-1] <= sig_l[i-1] and macd_l[i] > sig_l[i]:
                lado = "call"
            elif macd_l[i-1] >= sig_l[i-1] and macd_l[i] < sig_l[i]:
                lado = "put"
            if lado is None:
                continue
            gano = (closes[i + h] > closes[i]) if lado == "call" else (closes[i + h] < closes[i])
            g = 1 if gano else 0

            # (1) baseline
            agg[grp]["base"][1] += 1; agg[grp]["base"][0] += g
            # (2) + filtro EMA a-favor
            precio = closes[i]
            ema_ok = (lado == "call" and precio > ema_t[i]) or (lado == "put" and precio < ema_t[i])
            if not ema_ok:
                continue
            agg[grp]["ema"][1] += 1; agg[grp]["ema"][0] += g
            # (3) + filtro pendiente normalizada
            slope = (hist[i] - hist[i-1]) / (abs(precio) or 1.0)
            slope_ok = (lado == "call" and slope >= min_slope) or (lado == "put" and slope <= -min_slope)
            if not slope_ok:
                continue
            agg[grp]["ema_slope"][1] += 1; agg[grp]["ema_slope"][0] += g

    print(f"Pares: {n_real} reales + {n_otc} OTC\n")
    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        print("=" * 62)
        print(f"GRUPO {g.upper()} | WR test agregado (BE ~{be}%)")
        print("=" * 62)
        for k, lab in [("base", "solo MACD"),
                       ("ema", "MACD + EMA a-favor"),
                       ("ema_slope", "MACD + EMA + slope (config viva)")]:
            w, nn = agg[g][k]
            wr = (100.0 * w / nn) if nn else float("nan")
            mark = " *SUPERA BE" if nn and wr >= be else ""
            print(f"  {lab:34s} WR {wr:6.2f}%  (n={nn:6d}){mark}")
        print()

    json.dump(agg, open("backtest_actual.json", "w", encoding="utf-8"), indent=2)
    print("Guardado backtest_actual.json")


if __name__ == "__main__":
    main()
