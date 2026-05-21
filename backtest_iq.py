# backtest_iq.py - Backtest de RSI-reversion sobre el feed REAL de IQ Option.
#
# Replica el unico edge validado (Deriv frxUSDJPY): RSI(14) sobre velas 1-min CERRADAS,
#   CALL si RSI<35, PUT si RSI>65. Des-solapado (una posicion a la vez). Empate = perdida.
# Evalua varios horizontes de expiracion (en minutos) y compara el WR contra el break-even
# real de IQ (payout ~83-84% en pares reales -> break-even ~54.3-54.6%).
#
#   .venv314\Scripts\python.exe backtest_iq.py
import json
import sys
import time

import numpy as np
from iqoptionapi.stable_api import IQ_Option

PARES = ["USDJPY", "EURUSD", "GBPJPY"]   # subyacente real (sin -OTC); el option es "<par>-op"
RSI_PERIOD = 14
RSI_LOW, RSI_HIGH = 35, 65
HORIZONTES = [1, 3, 5, 10, 15]      # minutos de expiracion a probar
GRAN = 60                            # velas de 1 minuto
N_VELAS = 8000                       # cuantas velas 1-min bajar por par
PAYOUT_IQ = 0.84                     # payout tipico par real -> break-even
BREAK_EVEN = 1.0 / (1.0 + PAYOUT_IQ) * 100


def rsi_series(closes, period=14):
    """RSI de Wilder en cada punto (array, NaN hasta tener datos)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = gain[:period].mean()
    al = loss[:period].mean()
    for i in range(period, n):
        if i > period:
            ag = (ag * (period - 1) + gain[i - 1]) / period
            al = (al * (period - 1) + loss[i - 1]) / period
        rs = ag / al if al > 0 else 999
        out[i] = 100 - 100 / (1 + rs)
    return out


def bajar_velas(api, par, total, gran):
    """Baja `total` velas de `gran`s hacia atras, en lotes (la API limita ~1000)."""
    todas = {}
    endtime = time.time()
    while len(todas) < total:
        lote = api.get_candles(par, gran, 1000, endtime)
        if not lote:
            break
        for v in lote:
            todas[v["from"]] = v
        endtime = min(v["from"] for v in lote) - 1
        if len(lote) < 2:
            break
    velas = [todas[k] for k in sorted(todas)]
    return velas


def backtest_par(closes, horizonte):
    """RSI-reversion des-solapado a `horizonte` velas. Devuelve (n, wins)."""
    rsi = rsi_series(closes, RSI_PERIOD)
    n = len(closes)
    wins = trades = 0
    i = RSI_PERIOD + 1
    while i < n - horizonte:
        r = rsi[i]
        if np.isnan(r):
            i += 1
            continue
        lado = None
        if r < RSI_LOW:
            lado = "CALL"
        elif r > RSI_HIGH:
            lado = "PUT"
        if lado is None:
            i += 1
            continue
        entrada = closes[i]
        salida = closes[i + horizonte]
        gano = (salida > entrada) if lado == "CALL" else (salida < entrada)
        trades += 1
        if gano:
            wins += 1
        i += horizonte   # des-solapado: saltar hasta que expire la posicion
    return trades, wins


def main():
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    api = IQ_Option(cfg["email"], cfg["password"])
    print("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        print(f"NO CONECTO: {reason}")
        sys.exit(1)
    api.change_balance("PRACTICE")
    print(f"Conectado (DEMO). Payout asumido {PAYOUT_IQ:.0%} -> break-even {BREAK_EVEN:.1f}% WR\n")

    for par in PARES:
        print(f"===== {par} =====")
        velas = bajar_velas(api, par, N_VELAS, GRAN)
        if len(velas) < 200:
            print(f"  pocas velas ({len(velas)}), salto\n")
            continue
        closes = [float(v["close"]) for v in velas]
        dias = (velas[-1]["from"] - velas[0]["from"]) / 86400
        print(f"  {len(closes)} velas 1-min (~{dias:.1f} dias)")
        print(f"  {'EXPIR':>6}  {'TRADES':>7}  {'WR':>7}  {'vs BE':>8}  {'EV/trade':>9}")
        for h in HORIZONTES:
            tr, w = backtest_par(closes, h)
            if tr == 0:
                continue
            wr = w / tr * 100
            ev = (wr / 100) * PAYOUT_IQ - (1 - wr / 100)   # fraccion del stake
            marca = "  <== POSITIVO" if wr > BREAK_EVEN else ""
            print(f"  {h:>4}m  {tr:>7}  {wr:6.1f}%  {wr-BREAK_EVEN:+7.1f}  {ev*100:+8.1f}%{marca}")
        print()


if __name__ == "__main__":
    main()
