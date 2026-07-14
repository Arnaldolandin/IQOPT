# backtest_expiry_sweep.py - Barrido de EXPIRY (5m/10m/15m) x MACD, OFFLINE sobre cache_closes/.
#
# No conecta a IQ: reusa las velas 5m ya cacheadas. Evalua cada senal MACD a distintos
# horizontes (h velas de 5m -> 5/10/15 min) y mide el WR de TEST agregado (ponderado por
# nº trades), baseline (sin filtro), real vs OTC. Split train/test 70/30.
#
#   .venv314\Scripts\python.exe backtest_expiry_sweep.py
import json, os, glob
import numpy as np

CACHE_DIR = "cache_closes"
SPLIT = 0.70
WARMUP = 250
MIN_SIG_TRAIN = 30
MACD_CONFIGS = [(8, 17, 9), (12, 26, 9), (19, 39, 9), (6, 13, 5), (24, 52, 18)]
EXPIRIES = {"5m": 1, "10m": 2, "15m": 3}   # h velas de 5m


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


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    print(f"Cache: {len(files)} pares | {len(MACD_CONFIGS)} MACD x {len(EXPIRIES)} expiries (offline)")

    # agg[grp][macd_str][exp] = [wins, n]  (en test)
    agg = {g: {f"{m}": {e: [0, 0] for e in EXPIRIES} for m in MACD_CONFIGS}
           for g in ("real", "otc")}
    n_real = n_otc = 0

    for fp in files:
        name = os.path.basename(fp)[:-5]  # sin .json
        grp = "otc" if "-OTC" in name else "real"
        try:
            d = json.load(open(fp, encoding="utf-8"))
            closes = np.asarray(d["closes"], float)
        except Exception:
            continue
        n = len(closes)
        if n < WARMUP + 50:
            continue
        if grp == "real": n_real += 1
        else: n_otc += 1
        cut = int(n * SPLIT)

        for (fa, sl, si) in MACD_CONFIGS:
            sigs = macd_cruces(closes, fa, sl, si)
            acc = agg[grp][f"{(fa, sl, si)}"]
            idxs = [i for i in range(WARMUP, n - max(EXPIRIES.values()))
                    if sigs[i] != 0 and i >= cut]
            for e, h in EXPIRIES.items():
                w = nn = 0
                for i in idxs:
                    s = sigs[i]
                    gano = (closes[i + h] > closes[i]) if s == 1 else (closes[i + h] < closes[i])
                    nn += 1; w += 1 if gano else 0
                acc[e][0] += w; acc[e][1] += nn

    print(f"Pares usados: {n_real} reales + {n_otc} OTC\n")

    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        print("=" * 66)
        print(f"GRUPO {g.upper()} | WR test agregado (BE ~{be}%)  [n]")
        print("=" * 66)
        print(f"{'MACD':14s} " + "".join(f"{'exp'+e:>16s}" for e in EXPIRIES))
        print("-" * 66)
        for m in MACD_CONFIGS:
            acc = agg[g][f"{m}"]
            row = f"{str(m):14s} "
            for e in EXPIRIES:
                w, nn = acc[e]
                wr = (100.0 * w / nn) if nn else float("nan")
                mark = "*" if nn and wr >= be else " "
                row += f"{wr:6.2f}%({nn:6d}){mark}"
            print(row)
        print()

    json.dump(agg, open("backtest_expiry_sweep.json", "w", encoding="utf-8"), indent=2)
    print("Guardado backtest_expiry_sweep.json  (* = supera break-even)")


if __name__ == "__main__":
    main()
