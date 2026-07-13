# backtest_sweep.py - Barrido empirico de estrategias de BINARIAS sobre forex real (IQ).
# Prueba reversion y momentum x pares x horizontes; reporta WR vs break-even con p-valor.
# Sirve para Deriv (BE 52.4% @ payout 90%) e IQ (BE 54.3% @ payout 84%): mismo forex real.
import json, math, time
import numpy as np
from iqoptionapi.stable_api import IQ_Option

PARES = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "GBPJPY"]
HORIZONTES = [5, 10, 15]
N_VELAS = 7000
BE_IQ, BE_DERIV = 0.543, 0.524


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


def señales(closes):
    """Devuelve dict {nombre_estrategia: array de 'call'/'put'/'' por indice i (decision en vela i)}."""
    c = np.asarray(closes, float); n = len(c)
    rsi = rsi_series(c)
    e_f, e_s = ema(c, 5), ema(c, 20)
    # Bollinger 20,2
    m = np.full(n, np.nan); sd = np.full(n, np.nan)
    for i in range(19, n):
        w = c[i-19:i+1]; m[i] = w.mean(); sd[i] = w.std()
    S = {k: np.array([""]*n, dtype=object) for k in
         ["RSI_rev_30/70","RSI_rev_35/65","RSI_rev_25/75","Boll_rev","Consec3_rev","Consec3_mom","EMA_cross_mom"]}
    for i in range(20, n):
        # reversion RSI
        if rsi[i] < 30: S["RSI_rev_30/70"][i] = "call"
        elif rsi[i] > 70: S["RSI_rev_30/70"][i] = "put"
        if rsi[i] < 35: S["RSI_rev_35/65"][i] = "call"
        elif rsi[i] > 65: S["RSI_rev_35/65"][i] = "put"
        if rsi[i] < 25: S["RSI_rev_25/75"][i] = "call"
        elif rsi[i] > 75: S["RSI_rev_25/75"][i] = "put"
        # bollinger reversion
        if not np.isnan(m[i]) and sd[i] > 0:
            if c[i] < m[i] - 2*sd[i]: S["Boll_rev"][i] = "call"
            elif c[i] > m[i] + 2*sd[i]: S["Boll_rev"][i] = "put"
        # 3 velas consecutivas
        if c[i] < c[i-1] < c[i-2] < c[i-3]:
            S["Consec3_rev"][i] = "call"; S["Consec3_mom"][i] = "put"
        elif c[i] > c[i-1] > c[i-2] > c[i-3]:
            S["Consec3_rev"][i] = "put"; S["Consec3_mom"][i] = "call"
        # EMA cross momentum
        if e_f[i] > e_s[i] and e_f[i-1] <= e_s[i-1]: S["EMA_cross_mom"][i] = "call"
        elif e_f[i] < e_s[i] and e_f[i-1] >= e_s[i-1]: S["EMA_cross_mom"][i] = "put"
    return S


def evaluar(closes, sig, h):
    """WR des-solapado de una serie de señales a horizonte h."""
    c = closes; n = len(c); wins = tr = 0; i = 21
    while i < n - h:
        s = sig[i]
        if not s: i += 1; continue
        gano = (c[i+h] > c[i]) if s == "call" else (c[i+h] < c[i])
        tr += 1; wins += 1 if gano else 0; i += h
    return tr, wins


def pval(w, n, p0):
    if n == 0: return 1.0
    mu = n*p0; sd = math.sqrt(n*p0*(1-p0))
    if sd == 0: return 1.0
    return 0.5*math.erfc(((w-0.5-mu)/sd)/math.sqrt(2))


def bajar(api, par, total):
    todas = {}; et = time.time()
    while len(todas) < total:
        lote = api.get_candles(par, 60, 1000, et)
        if not lote: break
        for v in lote: todas[v["from"]] = v
        et = min(v["from"] for v in lote) - 1
        if len(lote) < 2: break
    return [float(todas[k]["close"]) for k in sorted(todas)]


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"]); api.connect(); api.change_balance("PRACTICE")
    print(f"Break-even: IQ {BE_IQ:.1%} | Deriv {BE_DERIV:.1%}. Marca POS si WR>BE_IQ y p<0.05 y n>=80.\n")
    ganadoras = []
    for par in PARES:
        closes = bajar(api, par, N_VELAS)
        if len(closes) < 500:
            print(f"{par}: pocas velas\n"); continue
        S = señales(closes)
        print(f"===== {par} ({len(closes)} velas 1m) =====")
        for nombre, sig in S.items():
            for h in HORIZONTES:
                tr, w = evaluar(closes, sig, h)
                if tr < 20: continue
                wr = w/tr*100; p = pval(w, tr, BE_IQ)
                pos = wr/100 > BE_IQ and p < 0.05 and tr >= 80
                flag = "  <== POS" if pos else ""
                if pos or (wr/100 > BE_DERIV and tr >= 80):
                    print(f"  {nombre:14} {h:2}m  n={tr:4} WR={wr:5.1f}%  p={p:.3f}{flag}")
                if pos: ganadoras.append((par, nombre, h, tr, wr))
        print()
    print("===== GANADORAS (superan BE_IQ con significancia) =====")
    if ganadoras:
        for g in ganadoras: print(f"  {g[0]} {g[1]} {g[2]}m: n={g[3]} WR={g[4]:.1f}%")
    else:
        print("  NINGUNA estrategia supera el break-even de IQ con significancia.")


if __name__ == "__main__":
    main()
