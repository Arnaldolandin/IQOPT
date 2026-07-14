# backtest_tf_sweep.py - Barrido de TIMEFRAME de la senal (5m/15m/30m) x MACD x expiry.
# OFFLINE sobre cache_closes/ (velas 5m). Resamplea 5m -> 15m (x3) / 30m (x6), calcula el MACD
# en esas velas y mide WR de TEST agregado (ponderado por nº trades), baseline, real vs OTC.
# Para cada timeframe evalua expiry de 1 y 2 barras (ej. 15m -> exp 15m y 30m). Split 70/30.
#
#   .venv314\Scripts\python.exe backtest_tf_sweep.py
import json, os, glob
import numpy as np

CACHE_DIR = "cache_closes"
SPLIT = 0.70
MACD_CONFIGS = [(8, 17, 9), (12, 26, 9), (19, 39, 9), (6, 13, 5), (24, 52, 18)]
TIMEFRAMES = {"5m": 1, "15m": 3, "30m": 6}   # factor sobre velas de 5m
EXP_BARS = [1, 2]                            # expiry en nº de barras del timeframe


def ema(c, span):
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_cruces(closes, fast, slow, sig_p):
    if len(closes) < slow + sig_p + 2:
        return np.zeros(len(closes), dtype=int)
    macd_l = ema(closes, fast) - ema(closes, slow)
    sig_l = ema(macd_l, sig_p)
    out = np.zeros(len(closes), dtype=int)
    for i in range(1, len(closes)):
        if macd_l[i-1] <= sig_l[i-1] and macd_l[i] > sig_l[i]:
            out[i] = 1
        elif macd_l[i-1] >= sig_l[i-1] and macd_l[i] < sig_l[i]:
            out[i] = -1
    return out


def resamplear(closes5, factor):
    if factor == 1:
        return closes5
    b = len(closes5) // factor
    return closes5[:b * factor].reshape(b, factor)[:, -1]


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    print(f"Cache: {len(files)} pares | TF {list(TIMEFRAMES)} x {len(MACD_CONFIGS)} MACD x exp {EXP_BARS} barras (offline)\n")

    # agg[grp][tf][macd][h] = [w, n]
    agg = {g: {tf: {f"{m}": {h: [0, 0] for h in EXP_BARS} for m in MACD_CONFIGS}
               for tf in TIMEFRAMES} for g in ("real", "otc")}
    n_real = n_otc = 0

    for fp in files:
        name = os.path.basename(fp)[:-5]
        grp = "otc" if "-OTC" in name else "real"
        try:
            closes5 = np.asarray(json.load(open(fp, encoding="utf-8"))["closes"], float)
        except Exception:
            continue
        if len(closes5) < 400:
            continue
        if grp == "real": n_real += 1
        else: n_otc += 1

        for tf, factor in TIMEFRAMES.items():
            closes = resamplear(closes5, factor)
            n = len(closes)
            if n < 200:
                continue
            cut = int(n * SPLIT)
            for (fa, sl, si) in MACD_CONFIGS:
                start = max(fa, sl) + si + 5
                sigs = macd_cruces(closes, fa, sl, si)
                acc = agg[grp][tf][f"{(fa, sl, si)}"]
                for i in range(start, n - max(EXP_BARS)):
                    s = sigs[i]
                    if s == 0 or i < cut:
                        continue
                    for h in EXP_BARS:
                        gano = (closes[i + h] > closes[i]) if s == 1 else (closes[i + h] < closes[i])
                        acc[h][1] += 1; acc[h][0] += 1 if gano else 0

    print(f"Pares usados: {n_real} reales + {n_otc} OTC\n")

    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        print("=" * 74)
        print(f"GRUPO {g.upper()} | WR test agregado (BE ~{be}%)  [* = supera BE]")
        print("=" * 74)
        for tf, factor in TIMEFRAMES.items():
            tfm = 5 * factor
            print(f"  -- Timeframe {tf} (senal en velas de {tfm}m) --")
            hdr = f"  {'MACD':14s}"
            for h in EXP_BARS:
                hdr += f"{'exp'+str(tfm*h)+'m':>17s}"
            print(hdr)
            for m in MACD_CONFIGS:
                acc = agg[g][tf][f"{m}"]
                row = f"  {str(m):14s}"
                for h in EXP_BARS:
                    w, nn = acc[h]
                    wr = (100.0 * w / nn) if nn else float("nan")
                    mark = "*" if nn and wr >= be else " "
                    row += f"{wr:6.2f}%({nn:6d}){mark}"
                print(row)
            print()

    json.dump(agg, open("backtest_tf_sweep.json", "w", encoding="utf-8"), indent=2)
    print("Guardado backtest_tf_sweep.json")


if __name__ == "__main__":
    main()
