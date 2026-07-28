# -*- coding: utf-8 -*-
"""Fuera del esquema: el feed de IQ es el mercado real, o va con retraso / es sintetico?

Compara el cierre 5m de IQ (cache) contra Binance (precio real) para cripto. Si:
  - corr contemporanea de retornos ~1  -> IQ sigue el mercado real (no RNG).
  - ret_binance[t] predice ret_iq[t+1] -> IQ va con LAG de nivel-vela: arbitrable.
  - corr ~0                            -> feed sintetico/RNG (como los OTC): inutil.

Nota: el arbitraje de latencia real es sub-minuto (necesita ticks). Con velas de 5m esto
es un PRIMER FILTRO: dice si el activo es real y si hay lag groso; el lag fino se mide en
vivo despues, si esto da verde.
"""
import json, os, sys, urllib.request
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

def binance_5m(symbol, t0, t1):
    """Klines 5m [t0,t1] en segundos. Devuelve dict ts->close."""
    out = {}
    ms = t0 * 1000
    while ms < t1 * 1000:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m"
               f"&startTime={ms}&limit=1000")
        r = urllib.request.urlopen(url, timeout=20)
        d = json.load(r)
        if not d:
            break
        for k in d:
            out[k[0] // 1000] = float(k[4])   # close
        ms = d[-1][0] + 5 * 60 * 1000
        if len(d) < 1000:
            break
    return out

def analiza(par_iq, symbol_bin, n_velas=6000):
    d = json.load(open(f"cache_ohlc_5m_v2/{par_iq}.json", encoding="utf-8"))
    t = np.asarray(d["times"]); c = np.asarray(d["close"], float)
    sl = slice(max(0, len(t) - n_velas), len(t))
    t, c = t[sl], c[sl]
    print(f"{par_iq}: {len(t)} velas IQ, bajando Binance {symbol_bin}...", flush=True)
    bin_close = binance_5m(symbol_bin, int(t[0]), int(t[-1]) + 300)
    # alinear por timestamp
    ci, cb = [], []
    for ts, close in zip(t, c):
        if int(ts) in bin_close:
            ci.append(close); cb.append(bin_close[int(ts)])
    ci = np.array(ci); cb = np.array(cb)
    print(f"  velas alineadas: {len(ci)}")
    if len(ci) < 200:
        print("  muy pocas alineadas"); return
    ri = np.diff(ci) / ci[:-1]           # retornos IQ
    rb = np.diff(cb) / cb[:-1]           # retornos Binance
    print(f"  corr CONTEMPORANEA ret(IQ,Binance): {np.corrcoef(ri, rb)[0,1]:.4f}  "
          f"(~1 = feed real; ~0 = sintetico)")
    # Binance adelanta a IQ? ret_bin[t] vs ret_iq[t+1]
    if len(ri) > 2:
        c_lag = np.corrcoef(rb[:-1], ri[1:])[0,1]     # binance(t) -> iq(t+1)
        c_lag2 = np.corrcoef(ri[:-1], rb[1:])[0,1]    # iq(t) -> binance(t+1)
        print(f"  corr LAG binance(t)->IQ(t+1): {c_lag:.4f}   IQ(t)->binance(t+1): {c_lag2:.4f}")
        print(f"    (si binance->IQ >> IQ->binance, IQ va DETRAS: arbitrable)")
        # prueba de trading: signo de rb[t] predice signo de ri[t+1]?
        sig = np.sign(rb[:-1]); real = np.sign(ri[1:])
        m = sig != 0
        if m.sum() > 100:
            wr = (sig[m] == real[m]).mean()
            print(f"    signo binance(t) acierta signo IQ(t+1): {100*wr:.2f}% (n={int(m.sum())}, BE turbo 54.64%)")

if __name__ == "__main__":
    for par, sym in [("ETHUSD", "ETHUSDT"), ("XRPUSD", "XRPUSDT")]:
        try:
            analiza(par, sym)
        except Exception as e:
            print(f"{par}: fallo {type(e).__name__}: {str(e)[:100]}")
        print()
