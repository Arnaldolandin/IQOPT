# bt_horas_extraer.py - Extrae TODAS las señales meta (>=umbral) sobre cache_ohlc_5m/
# reutilizando main.predecir_meta() tal cual, para que backtest y bot no puedan divergir.
#
#   .venv314\Scripts\python.exe bt_horas_extraer.py [--pares N] [--umbral 0.54] [--jobs 8]
#
# Salida: bt_horas_senales.json (una fila por señal, con hora UTC y resultado)
#
# Convencion: la decision se toma sobre la vela CERRADA i. El bot compra de
# inmediato -> entrada = close[i]. Expiry 10m = 2 velas de 5m -> liquida en close[i+2].
#
# CONTINUIDAD: se exige ts[i+2]-ts[i] == 600s. Si hay hueco (cierre de mercado,
# fin de semana, feriado) la operacion no existe y se descarta. Sin este filtro el
# backtest contaria gaps overnight como opciones de 10 minutos e inflaria el WR.
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "cache_ohlc_5m"
VENTANA = 300          # velas de contexto que se pasan a predecir_meta
EXP_VELAS = 2          # 10 min de expiry / 5 min por vela
PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT) * 100


def cargar(par):
    with open(os.path.join(CACHE, par + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    n = len(d["times"])
    return [{"from": d["times"][i], "open": d["open"][i], "max": d["high"][i],
             "min": d["low"][i], "close": d["close"][i]} for i in range(n)]


def procesar(par, umbral):
    """Devuelve la lista de señales de un par. Se ejecuta en el worker."""
    import main as bot
    if not bot.CFG:
        bot.CFG = json.load(open("config.json", encoding="utf-8"))
    velas = cargar(par)
    n = len(velas)
    out = []
    if n < VENTANA + EXP_VELAS + 5:
        return out
    for i in range(VENTANA, n - EXP_VELAS):
        lado, p, _ = bot.predecir_meta(velas[i - VENTANA + 1: i + 2])
        if lado is None or p < umbral:
            continue
        if velas[i + EXP_VELAS]["from"] - velas[i]["from"] != EXP_VELAS * 300:
            continue
        entrada = velas[i]["close"]
        salida = velas[i + EXP_VELAS]["close"]
        if salida == entrada:
            res = "empate"
        elif (lado == "call") == (salida > entrada):
            res = "win"
        else:
            res = "loss"
        ts = velas[i]["from"]
        dt = datetime.fromtimestamp(ts, timezone.utc)
        out.append({"par": par, "ts": ts, "hora_utc": dt.hour, "dow": dt.weekday(),
                    "lado": lado, "p": round(p, 4), "res": res})
    return out


def _worker(args):
    return procesar(*args)


def _guardar(filas, salida, t0):
    filas.sort(key=lambda f: f["ts"])
    json.dump(filas, open(salida, "w", encoding="utf-8"))
    dec = [f for f in filas if f["res"] != "empate"]
    wins = sum(1 for f in dec if f["res"] == "win")
    print(f"\nTOTAL {len(filas)} señales en {time.time()-t0:.0f}s -> {salida}")
    if dec:
        print(f"WR global {100*wins/len(dec):.2f}%  (n={len(dec)}, "
              f"empates={len(filas)-len(dec)})")
        print(f"break-even con payout {PAYOUT:.0%} = {BREAK_EVEN:.2f}%")


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pares", type=int, default=0, help="limitar a N pares (0 = todos)")
    ap.add_argument("--umbral", type=float, default=0.54)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--salida", default="bt_horas_senales.json")
    a = ap.parse_args()

    import main as bot
    bot.CFG = json.load(open("config.json", encoding="utf-8"))
    op = bot.CFG["operacion"]
    print(f"estrategia={op.get('estrategia')} bb_std={op.get('bb_std')} "
          f"bb_period={op.get('bb_period')} modelo={op.get('bb_ml_model')} "
          f"umbral={a.umbral} jobs={a.jobs}")

    pares = [f[:-5] for f in sorted(os.listdir(CACHE)) if f.endswith(".json")]
    if a.pares:
        pares = pares[:a.pares]

    filas, t0 = [], time.time()
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for k, (par, sub) in enumerate(
                    zip(pares, ex.map(_worker, [(p, a.umbral) for p in pares])), 1):
                filas.extend(sub)
                print(f"[{k}/{len(pares)}] {par}: {len(sub)} señales "
                      f"(acum {len(filas)}, {time.time()-t0:.0f}s)", flush=True)
    else:
        for k, par in enumerate(pares, 1):
            sub = procesar(par, a.umbral)
            filas.extend(sub)
            print(f"[{k}/{len(pares)}] {par}: {len(sub)} señales "
                  f"(acum {len(filas)}, {time.time()-t0:.0f}s)", flush=True)
    _guardar(filas, a.salida, t0)


if __name__ == "__main__":
    main_()
