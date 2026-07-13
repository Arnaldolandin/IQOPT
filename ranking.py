# ranking.py - Ranking honesto de estrategias de binarias sobre forex real (IQ).
# Bateria de ~14 estrategias x 5 pares x 3 horizontes. Split 70/30 train/test.
# Se rankea por EV en TRAIN y se reporta el EV en TEST (out-of-sample) para ver si aguanta.
#
# OJO: rankear el top de 210 combos es multiple-testing -> los mejores del TEST pueden ser
# suerte. Toda candidata que salga de aqui hay que validarla con walkforward.py sobre
# histórico largo y folds NO adyacentes antes de creersela. El 2026-07-13 la mejor
# (GBPJPY Bollinger-2.5σ) parecia p<0.05 aqui y se cayo a break-even en walk-forward de 40 dias.
#
#   .venv314\Scripts\python.exe ranking.py
import json, math, time

import numpy as np
from iqoptionapi.stable_api import IQ_Option

PARES = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "GBPJPY"]
HORIZONTES = [5, 10, 15]
N_VELAS = 16000
PAYOUT = 0.84
BE = 1.0 / (1.0 + PAYOUT)         # 0.5435
SPLIT = 0.70
MIN_TR = 40                        # minimo de trades en train para considerar
MIN_TE = 20                        # minimo en test


def rsi_series(c, p=14):
    c = np.asarray(c, float); n = len(c); out = np.full(n, np.nan)
    if n < p + 1: return out
    d = np.diff(c); g = np.where(d > 0, d, 0.); l = np.where(d < 0, -d, 0.)
    ag, al = g[:p].mean(), l[:p].mean()
    for i in range(p, n):
        if i > p: ag = (ag*(p-1)+g[i-1])/p; al = (al*(p-1)+l[i-1])/p
        out[i] = 100 - 100/(1 + (ag/al if al > 0 else 999))
    return out


def ema(c, span):
    c = np.asarray(c, float); a = 2/(span+1); out = np.copy(c)
    for i in range(1, len(c)): out[i] = a*c[i] + (1-a)*out[i-1]
    return out


def señales(H, L, C):
    """Devuelve {nombre: array 'call'/'put'/'' por indice}. H,L,C = high, low, close."""
    c = np.asarray(C, float); n = len(c)
    hi = np.asarray(H, float); lo = np.asarray(L, float)
    rsi = rsi_series(c)
    ef, es = ema(c, 5), ema(c, 20)
    e50 = ema(c, 50)
    # MACD 12/26/9
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    # medias moviles / bandas
    sma = np.full(n, np.nan); sd = np.full(n, np.nan)
    for i in range(19, n):
        w = c[i-19:i+1]; sma[i] = w.mean(); sd[i] = w.std()
    # Stochastic %K(14)
    K = np.full(n, np.nan)
    for i in range(13, n):
        hh = hi[i-13:i+1].max(); ll = lo[i-13:i+1].min()
        K[i] = 100*(c[i]-ll)/(hh-ll) if hh > ll else 50
    # Williams %R(14) = -(hh-c)/(hh-ll)*100  (-100..0)
    WR = np.full(n, np.nan)
    for i in range(13, n):
        hh = hi[i-13:i+1].max(); ll = lo[i-13:i+1].min()
        WR[i] = -100*(hh-c[i])/(hh-ll) if hh > ll else -50

    S = {k: np.array([""]*n, dtype=object) for k in [
        "RSI_rev_30/70", "RSI_rev_35/65", "RSI_rev_25/75",
        "Boll_rev_2sd", "Boll_rev_2.5sd", "Stoch_rev_20/80",
        "WilliamsR_rev", "Consec3_rev", "Consec4_rev",
        "EMA_cross_mom", "MACD_cross_mom", "Breakout_mom_20",
        "RSI_pullback_mom", "SMAfar_rev"]}

    for i in range(50, n):
        r = rsi[i]
        if r < 30: S["RSI_rev_30/70"][i] = "call"
        elif r > 70: S["RSI_rev_30/70"][i] = "put"
        if r < 35: S["RSI_rev_35/65"][i] = "call"
        elif r > 65: S["RSI_rev_35/65"][i] = "put"
        if r < 25: S["RSI_rev_25/75"][i] = "call"
        elif r > 75: S["RSI_rev_25/75"][i] = "put"
        if sd[i] > 0:
            if c[i] < sma[i] - 2*sd[i]: S["Boll_rev_2sd"][i] = "call"
            elif c[i] > sma[i] + 2*sd[i]: S["Boll_rev_2sd"][i] = "put"
            if c[i] < sma[i] - 2.5*sd[i]: S["Boll_rev_2.5sd"][i] = "call"
            elif c[i] > sma[i] + 2.5*sd[i]: S["Boll_rev_2.5sd"][i] = "put"
            # desviacion "grande" de la SMA (>1.5 sd) como reversion suave
            if c[i] < sma[i] - 1.5*sd[i]: S["SMAfar_rev"][i] = "call"
            elif c[i] > sma[i] + 1.5*sd[i]: S["SMAfar_rev"][i] = "put"
        if K[i] < 20: S["Stoch_rev_20/80"][i] = "call"
        elif K[i] > 80: S["Stoch_rev_20/80"][i] = "put"
        if WR[i] < -80: S["WilliamsR_rev"][i] = "call"
        elif WR[i] > -20: S["WilliamsR_rev"][i] = "put"
        if c[i] < c[i-1] < c[i-2] < c[i-3]: S["Consec3_rev"][i] = "call"
        elif c[i] > c[i-1] > c[i-2] > c[i-3]: S["Consec3_rev"][i] = "put"
        if c[i] < c[i-1] < c[i-2] < c[i-3] < c[i-4]: S["Consec4_rev"][i] = "call"
        elif c[i] > c[i-1] > c[i-2] > c[i-3] > c[i-4]: S["Consec4_rev"][i] = "put"
        if ef[i] > es[i] and ef[i-1] <= es[i-1]: S["EMA_cross_mom"][i] = "call"
        elif ef[i] < es[i] and ef[i-1] >= es[i-1]: S["EMA_cross_mom"][i] = "put"
        if macd[i] > sig[i] and macd[i-1] <= sig[i-1]: S["MACD_cross_mom"][i] = "call"
        elif macd[i] < sig[i] and macd[i-1] >= sig[i-1]: S["MACD_cross_mom"][i] = "put"
        if c[i] == hi[i-19:i+1].max(): S["Breakout_mom_20"][i] = "call"
        elif c[i] == lo[i-19:i+1].min(): S["Breakout_mom_20"][i] = "put"
        # pullback en tendencia: RSI bajo pero sobre EMA50 -> call (compra la caida en tendencia alcista)
        if r < 40 and c[i] > e50[i]: S["RSI_pullback_mom"][i] = "call"
        elif r > 60 and c[i] < e50[i]: S["RSI_pullback_mom"][i] = "put"
    return S


def evaluar(sig, closes, h, start, end):
    wins = tr = 0; i = max(start, 50)
    while i < end - h:
        s = sig[i]
        if not s: i += 1; continue
        gano = (closes[i+h] > closes[i]) if s == "call" else (closes[i+h] < closes[i])
        tr += 1; wins += 1 if gano else 0; i += h
    return tr, wins


def pval(w, n, p0=BE):
    if n == 0: return 1.0
    mu = n*p0; sd = math.sqrt(n*p0*(1-p0))
    if sd == 0: return 1.0
    return 0.5*math.erfc(((w-0.5-mu)/sd)/math.sqrt(2))


def ev(wr):
    return wr*PAYOUT - (1-wr)


def bajar(api, par, total):
    todas = {}; et = time.time()
    while len(todas) < total:
        lote = api.get_candles(par, 60, 1000, et)
        if not lote: break
        for v in lote: todas[v["from"]] = v
        et = min(v["from"] for v in lote) - 1
        if len(lote) < 2: break
    return [todas[k] for k in sorted(todas)]


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"]); api.connect(); api.change_balance("PRACTICE")
    print(f"Payout {PAYOUT:.0%} -> break-even {BE:.1%}. Split {SPLIT:.0%} train / resto test.\n")
    filas = []
    for par in PARES:
        velas = bajar(api, par, N_VELAS)
        if len(velas) < 2000:
            print(f"{par}: pocas velas ({len(velas)})"); continue
        C = [float(v["close"]) for v in velas]
        Hg = [float(v["max"]) for v in velas]
        Lw = [float(v["min"]) for v in velas]
        n = len(C); cut = int(n*SPLIT)
        S = señales(Hg, Lw, C)
        for nombre, sig in S.items():
            for h in HORIZONTES:
                ttr, tw = evaluar(sig, C, h, 50, cut)          # train
                etr, ew = evaluar(sig, C, h, cut, n)           # test
                if ttr < MIN_TR or etr < MIN_TE: continue
                tr_wr = tw/ttr; te_wr = ew/etr
                filas.append({
                    "par": par, "estr": nombre, "h": h,
                    "tr_n": ttr, "tr_wr": tr_wr, "tr_ev": ev(tr_wr),
                    "te_n": etr, "te_wr": te_wr, "te_ev": ev(te_wr),
                    "te_p": pval(ew, etr),
                })
    # Ranking por EV en TRAIN (lo que un optimizador elegiria), mostrando el TEST al lado
    filas.sort(key=lambda x: x["tr_ev"], reverse=True)
    print("TOP 15 por EV in-sample (train) — la columna TEST dice si aguanta fuera de muestra:\n")
    print(f"{'#':>2} {'par':7} {'estrategia':16} {'h':>3} | {'TRAIN n/WR/EV':>20} | {'TEST n/WR/EV/p':>26}")
    for k, f in enumerate(filas[:15], 1):
        print(f"{k:>2} {f['par']:7} {f['estr']:16} {f['h']:>2}m | "
              f"{f['tr_n']:>4} {f['tr_wr']*100:5.1f}% {f['tr_ev']*100:+5.1f}% | "
              f"{f['te_n']:>4} {f['te_wr']*100:5.1f}% {f['te_ev']*100:+6.1f}% p={f['te_p']:.3f}")

    # Ranking honesto: mejores por EV en TEST (out-of-sample) con muestra decente
    print("\nTOP 5 por EV OUT-OF-SAMPLE (test), n_test>=30 — el ranking que de verdad importa:\n")
    val = [f for f in filas if f["te_n"] >= 30]
    val.sort(key=lambda x: x["te_ev"], reverse=True)
    print(f"{'#':>2} {'par':7} {'estrategia':16} {'h':>3} {'te_n':>5} {'te_WR':>7} {'te_EV':>7} {'p':>7} {'tr_WR':>7}")
    for k, f in enumerate(val[:5], 1):
        print(f"{k:>2} {f['par']:7} {f['estr']:16} {f['h']:>2}m {f['te_n']:>5} "
              f"{f['te_wr']*100:6.1f}% {f['te_ev']*100:+6.1f}% {f['te_p']:7.3f} {f['tr_wr']*100:6.1f}%")

    pos = [f for f in val if f["te_ev"] > 0 and f["te_p"] < 0.05]
    print(f"\nEstrategias con EV_test>0 Y p<0.05 (edge real fuera de muestra): {len(pos)}")
    for f in pos:
        print(f"  {f['par']} {f['estr']} {f['h']}m: te_WR {f['te_wr']*100:.1f}% n={f['te_n']} p={f['te_p']:.3f}")


if __name__ == "__main__":
    main()
