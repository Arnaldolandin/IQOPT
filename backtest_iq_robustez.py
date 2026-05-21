# backtest_iq_robustez.py - ¿El edge RSI-reversion en USDJPY (IQ) es estable o una racha?
# Parte la serie en ventanas des-solapadas y reporta el WR por ventana, a 10m y 15m.
import json
import sys
import time

import numpy as np
from iqoptionapi.stable_api import IQ_Option

PAR = "USDJPY"
RSI_PERIOD = 14
RSI_LOW, RSI_HIGH = 35, 65
HORIZONTES = [10, 15]
GRAN = 60
N_VELAS = 8000
N_VENTANAS = 6
PAYOUT_IQ = 0.84
BREAK_EVEN = 1.0 / (1.0 + PAYOUT_IQ) * 100


def rsi_series(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, n):
        if i > period:
            ag = (ag * (period - 1) + gain[i - 1]) / period
            al = (al * (period - 1) + loss[i - 1]) / period
        rs = ag / al if al > 0 else 999
        out[i] = 100 - 100 / (1 + rs)
    return out


def bt(closes, h):
    rsi = rsi_series(closes, RSI_PERIOD)
    n = len(closes)
    wins = trades = 0
    i = RSI_PERIOD + 1
    while i < n - h:
        r = rsi[i]
        if np.isnan(r):
            i += 1; continue
        lado = "CALL" if r < RSI_LOW else ("PUT" if r > RSI_HIGH else None)
        if lado is None:
            i += 1; continue
        gano = (closes[i + h] > closes[i]) if lado == "CALL" else (closes[i + h] < closes[i])
        trades += 1; wins += 1 if gano else 0
        i += h
    return trades, wins


def bajar(api, par, total, gran):
    todas = {}; endtime = time.time()
    while len(todas) < total:
        lote = api.get_candles(par, gran, 1000, endtime)
        if not lote: break
        for v in lote: todas[v["from"]] = v
        endtime = min(v["from"] for v in lote) - 1
        if len(lote) < 2: break
    return [todas[k] for k in sorted(todas)]


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    print("Conectando..."); ok, r = api.connect()
    if not ok: print("NO CONECTO:", r); sys.exit(1)
    api.change_balance("PRACTICE")
    velas = bajar(api, PAR, N_VELAS, GRAN)
    closes = [float(v["close"]) for v in velas]
    print(f"{PAR}: {len(closes)} velas | break-even {BREAK_EVEN:.1f}% WR | {N_VENTANAS} ventanas des-solapadas\n")
    tam = len(closes) // N_VENTANAS
    for h in HORIZONTES:
        print(f"=== Expiracion {h}m ===")
        wrs = []
        for w in range(N_VENTANAS):
            seg = closes[w * tam:(w + 1) * tam]
            tr, win = bt(seg, h)
            if tr == 0: continue
            wr = win / tr * 100; wrs.append(wr)
            ok = "OK" if wr > BREAK_EVEN else ".."
            print(f"  ventana {w+1}: {tr:4} trades  WR {wr:5.1f}%  [{ok}]")
        if wrs:
            pos = sum(1 for x in wrs if x > BREAK_EVEN)
            print(f"  -> WR medio {np.mean(wrs):.1f}% | positivo en {pos}/{len(wrs)} ventanas\n")


if __name__ == "__main__":
    main()
