# backtest_optim.py - Busqueda de configuraciones optimas, OFFLINE desde cache.
# Rejilla: timeframe (5/15/30m) x MACD x expiry (1-2 barras) x EMA-tendencia (no/50/200) x
# direccion (normal/invertida). Mide WR pooled train/test 70/30 por grupo (real/otc).
# HONESTO: elige la mejor en TRAIN y reporta su TEST (OOS); ademas top por TEST marcando n bajo.
import json, os, glob
import numpy as np

CACHE_DIR = "cache_closes"; SPLIT = 0.70; WARMUP = 210
TFS = {"5m": 1, "15m": 3, "30m": 6}
MACDS = [(8, 17, 9), (12, 26, 9), (19, 39, 9), (6, 13, 5), (24, 52, 18)]
EXPS = [1, 2]
EMAS = [0, 50, 200]        # 0 = sin filtro
DIRS = [1, -1]             # 1 normal, -1 invertida
MIN_TR = 500               # minimo de trades en train para considerar la config
MIN_TE = 300               # minimo en test para no reportar ruido


def ema(c, span):
    a = 2.0/(span+1); out = np.copy(c)
    for i in range(1, len(c)): out[i] = a*c[i] + (1-a)*out[i-1]
    return out


def resample(c5, f):
    if f == 1: return c5
    b = len(c5)//f
    return c5[:b*f].reshape(b, f)[:, -1]


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    # agg[grp][key] = {"tr":[w,n], "te":[w,n]}
    agg = {g: {} for g in ("real", "otc")}
    def bump(grp, key, seg, w):
        d = agg[grp].setdefault(key, {"tr": [0, 0], "te": [0, 0]})
        d[seg][1] += 1; d[seg][0] += w

    for fp in files:
        name = os.path.basename(fp)[:-5]; grp = "otc" if "-OTC" in name else "real"
        try: c5 = np.asarray(json.load(open(fp, encoding="utf-8"))["closes"], float)
        except Exception: continue
        if len(c5) < 400: continue
        for tf, f in TFS.items():
            closes = resample(c5, f); n = len(closes)
            if n < WARMUP + 30: continue
            cut = int(n*SPLIT)
            emas = {p: ema(closes, p) for p in EMAS if p > 0}
            for (fa, sl, si) in MACDS:
                ml = ema(closes, fa) - ema(closes, sl); sg = ema(ml, si)
                cross = []
                for i in range(WARMUP, n - max(EXPS)):
                    if ml[i-1] <= sg[i-1] and ml[i] > sg[i]: cross.append((i, "call"))
                    elif ml[i-1] >= sg[i-1] and ml[i] < sg[i]: cross.append((i, "put"))
                if not cross: continue
                for exp in EXPS:
                    for ema_p in EMAS:
                        for direc in DIRS:
                            key = (tf, f"{(fa,sl,si)}", exp, ema_p, direc)
                            for (i, lado) in cross:
                                ex = lado if direc == 1 else ("put" if lado == "call" else "call")
                                if ema_p:
                                    e = emas[ema_p][i]; pr = closes[i]
                                    if (ex == "call" and pr <= e) or (ex == "put" and pr >= e):
                                        continue
                                win = 1 if ((closes[i+exp] > closes[i]) if ex == "call"
                                            else (closes[i+exp] < closes[i])) else 0
                                bump(grp, key, "tr" if i < cut else "te", win)

    def wr(p): w, nn = p; return (100.0*w/nn) if nn else float("nan")
    lab_dir = {1: "normal", -1: "INVERT"}

    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        rows = [(k, v) for k, v in agg[g].items()
                if v["tr"][1] >= MIN_TR and v["te"][1] >= MIN_TE]
        print("\n" + "=" * 78)
        print(f"GRUPO {g.upper()} (BE ~{be}%) | {len(rows)} configs con muestra suficiente")
        print("=" * 78)
        # 1) mejor en TRAIN -> su TEST (honesto, OOS)
        best_tr = max(rows, key=lambda kv: wr(kv[1]["tr"]))
        k, v = best_tr
        print("OPTIMA POR TRAIN (honesto, se mira su TEST):")
        print(f"  {k}  ->  train {wr(v['tr']):.2f}% (n={v['tr'][1]}) | TEST {wr(v['te']):.2f}% (n={v['te'][1]})"
              f"  {'*SUPERA BE' if wr(v['te'])>=be else 'bajo BE'}")
        # 2) top por TEST (ojo overfitting/multiple-testing)
        print("\nTOP 8 POR TEST (con muestra; posibles artefactos de multiple-testing):")
        for k, v in sorted(rows, key=lambda kv: -wr(kv[1]["te"]))[:8]:
            tf, mc, exp, ep, dr = k
            mark = " *" if wr(v["te"]) >= be else ""
            print(f"  {tf:>3s} {mc:>12s} exp{exp} EMA{ep:<3d} {lab_dir[dr]:6s} | "
                  f"train {wr(v['tr']):5.1f}% | TEST {wr(v['te']):5.1f}% (n={v['te'][1]:5d}){mark}")

    print("\n(Regla: si la optima-por-train no supera BE en test, no hay config robusta.)")


if __name__ == "__main__":
    main()
