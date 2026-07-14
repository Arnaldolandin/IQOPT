# backtest_macd_adx.py - Busca estrategia MACD + ADX rentable, OFFLINE sobre cache_ohlc/.
# ADX filtra por fuerza de tendencia: solo se opera el cruce MACD si ADX >= umbral.
# Rejilla: TF x MACD x ADX(periodo,umbral) x expiry x direccion(normal/invert). Train/test 70/30.
# Seleccion HONESTA: mejor en train -> su TEST; y top por test con IC Wilson 95% + p-valor vs BE.
#   .venv314\Scripts\python.exe backtest_macd_adx.py
import json, os, glob, math
import numpy as np

CACHE_DIR = "cache_ohlc"; SPLIT = 0.70; WARMUP = 260
PAYOUT_REAL, PAYOUT_OTC = 0.87, 0.85
TFS = {"5m": 1, "15m": 3, "30m": 6}
MACDS = [(12, 26, 9), (8, 17, 9), (19, 39, 9), (6, 13, 5)]
ADX_PERIOD = 14
ADX_THRS = [0, 20, 25, 30]     # 0 = sin filtro ADX
EXPS = [1, 2]
DIRS = [1, -1]
MIN_TR, MIN_TE = 500, 300


def ema(c, span):
    a = 2.0/(span+1); out = np.copy(c)
    for i in range(1, len(c)): out[i] = a*c[i] + (1-a)*out[i-1]
    return out


def resample_ohlc(o, h, l, c, f):
    if f == 1: return o, h, l, c
    b = len(c)//f
    o = o[:b*f].reshape(b, f); h = h[:b*f].reshape(b, f)
    l = l[:b*f].reshape(b, f); c = c[:b*f].reshape(b, f)
    return o[:, 0], h.max(1), l.min(1), c[:, -1]


def adx_wilder(high, low, close, period=14):
    n = len(close); out = np.full(n, np.nan)
    if n < 2*period + 2: return out
    tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = high[i]-high[i-1]; dn = low[i-1]-low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = tr[1:period+1].sum(); pv = pdm[1:period+1].sum(); mv = mdm[1:period+1].sum()
    dx = np.zeros(n)
    for i in range(period+1, n):
        atr = atr - atr/period + tr[i]
        pv = pv - pv/period + pdm[i]; mv = mv - mv/period + mdm[i]
        pdi = 100*pv/atr if atr else 0.0; mdi = 100*mv/atr if atr else 0.0
        s = pdi + mdi
        dx[i] = 100*abs(pdi-mdi)/s if s else 0.0
    first = 2*period
    av = np.mean(dx[period+1:first+1])
    out[first] = av
    for i in range(first+1, n):
        av = (av*(period-1) + dx[i])/period
        out[i] = av
    return out


def wilson(w, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = w/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-hw), 100*(c+hw))


def binom_p(w, n, p0):
    if n == 0: return 1.0
    mu = n*p0; sd = math.sqrt(n*p0*(1-p0))
    if sd == 0: return 1.0
    return 0.5*math.erfc(((w-0.5-mu)/sd)/math.sqrt(2))


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    if not files:
        print(f"No hay datos en {CACHE_DIR}/ todavia."); return
    agg = {g: {} for g in ("real", "otc")}
    def bump(g, k, seg, w):
        d = agg[g].setdefault(k, {"tr": [0, 0], "te": [0, 0]}); d[seg][1] += 1; d[seg][0] += w

    for fp in files:
        name = os.path.basename(fp)[:-5]; grp = "otc" if "-OTC" in name else "real"
        try:
            d = json.load(open(fp, encoding="utf-8"))
            o = np.asarray(d["open"], float); h = np.asarray(d["high"], float)
            l = np.asarray(d["low"], float); c = np.asarray(d["close"], float)
        except Exception:
            continue
        if len(c) < 500: continue
        for tf, f in TFS.items():
            oo, hh, ll, cc = resample_ohlc(o, h, l, c, f); n = len(cc)
            if n < WARMUP + 30: continue
            cut = int(n*SPLIT)
            adx = adx_wilder(hh, ll, cc, ADX_PERIOD)
            for (fa, sl, si) in MACDS:
                ml = ema(cc, fa)-ema(cc, sl); sg = ema(ml, si)
                cross = []
                for i in range(WARMUP, n-max(EXPS)):
                    if ml[i-1] <= sg[i-1] and ml[i] > sg[i]: cross.append((i, "call"))
                    elif ml[i-1] >= sg[i-1] and ml[i] < sg[i]: cross.append((i, "put"))
                if not cross: continue
                for thr in ADX_THRS:
                    for exp in EXPS:
                        for direc in DIRS:
                            k = (tf, f"{(fa,sl,si)}", thr, exp, direc)
                            for (i, lado) in cross:
                                if thr and not (adx[i] >= thr): continue
                                ex = lado if direc == 1 else ("put" if lado == "call" else "call")
                                win = 1 if ((cc[i+exp] > cc[i]) if ex == "call" else (cc[i+exp] < cc[i])) else 0
                                bump(grp, k, "tr" if i < cut else "te", win)

    def wr(p): w, nn = p; return (100.0*w/nn) if nn else float("nan")
    ld = {1: "normal", -1: "INVERT"}
    for g in ("real", "otc"):
        payout = PAYOUT_REAL if g == "real" else PAYOUT_OTC
        be = 100.0/(1.0+payout)
        rows = [(k, v) for k, v in agg[g].items() if v["tr"][1] >= MIN_TR and v["te"][1] >= MIN_TE]
        print("\n" + "=" * 92)
        print(f"GRUPO {g.upper()} | payout {payout:.0%} BE {be:.2f}% | {len(rows)} configs con muestra")
        print("=" * 92)
        if not rows:
            print("  (sin configs con muestra suficiente)"); continue
        bt = max(rows, key=lambda kv: wr(kv[1]["tr"]))
        k, v = bt; w, nn = v["te"]; lo, hi = wilson(w, nn); p = binom_p(w, nn, be/100)
        print("OPTIMA POR TRAIN -> TEST (honesto):")
        print(f"  {k} -> train {wr(v['tr']):.2f}% | TEST {wr(v['te']):.2f}% (n={nn}) "
              f"IC95[{lo:.1f},{hi:.1f}] p={p:.3f} {'SIGNIF>BE' if lo>be else 'no signif'}")
        print("\nTOP 8 POR TEST (IC95 Wilson, p vs BE; * si IC entero > BE):")
        for k, v in sorted(rows, key=lambda kv: -wr(kv[1]["te"]))[:8]:
            tf, mc, thr, exp, dr = k; w, nn = v["te"]; lo, hi = wilson(w, nn); p = binom_p(w, nn, be/100)
            ev = (w/nn)*payout - (1-w/nn)
            mark = " *" if lo > be else ""
            print(f"  {tf:>3s} {mc:>12s} ADX>={thr:<2d} exp{exp} {ld[dr]:6s} | "
                  f"tr {wr(v['tr']):5.1f}% | TEST {wr(v['te']):5.1f}% n={nn:5d} "
                  f"IC[{lo:4.1f},{hi:4.1f}] p={p:.3f} EV{ev*100:+5.1f}%{mark}")
    print(f"\n(Edge real solo si IC95 entero > BE, con p<0.05.)")


if __name__ == "__main__":
    main()
