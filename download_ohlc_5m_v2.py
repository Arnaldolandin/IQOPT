# download_ohlc_5m_v2.py - Velas 5m con VOLUMEN y span largo -> cache_ohlc_5m_v2/
#
#   .venv314\Scripts\python.exe download_ohlc_5m_v2.py [--dias 300] [--jobs 1]
#
# Diferencias con download_ohlc_5m.py:
#   - guarda 'volume', que la caché vieja descartaba. IQ lo devuelve en cada vela y es
#     informacion real sobre la conviccion detras del movimiento.
#   - 300 dias por defecto (~10 meses) en vez de 185. El modelo tiene 10.801 parametros
#     y se entrenaba con 20.480 muestras: menos de 2 por parametro. Duplicar la muestra
#     es la mejora mas fiable disponible, porque no inventa señal, reduce el ruido.
#
# Escribe en un directorio NUEVO para no pisar cache_ohlc_5m, que es lo que usan los
# modelos en produccion. REANUDABLE: salta lo ya bajado.
import argparse
import json
import os
import threading
import time
from datetime import datetime

from iqoptionapi.stable_api import IQ_Option

GRAN = 300
CACHE_DIR = "cache_ohlc_5m_v2"
CANDLE_TIMEOUT = 30
RECONNECT_CADA = 20


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def conectar(cfg):
    api = IQ_Option(cfg["email"], cfg["password"])
    ok, r = api.connect()
    if not ok:
        raise Exception(f"connect: {r}")
    api.change_balance("PRACTICE")
    done = [False]

    def _u():
        try:
            api.get_ALL_Binary_ACTIVES_OPCODE()
        except Exception:
            pass
        done[0] = True

    threading.Thread(target=_u, daemon=True).start()
    t0 = time.time()
    while not done[0] and time.time() - t0 < 45:
        time.sleep(1)
    time.sleep(1)
    return api


def get_lote(api, par, end):
    res = [None]

    def _c():
        try:
            res[0] = api.get_candles(par, GRAN, 1000, end)
        except Exception:
            res[0] = None

    t = threading.Thread(target=_c, daemon=True)
    t.start()
    t.join(timeout=CANDLE_TIMEOUT)
    if t.is_alive():
        return "TIMEOUT"
    return res[0]


def bajar_span(api, par, dias):
    todas = {}
    end = time.time()
    limite = time.time() - dias * 86400
    vac = calls = 0
    while calls < 800:
        lote = get_lote(api, par, end)
        calls += 1
        if lote == "TIMEOUT":
            return None
        if not lote:
            vac += 1
            if vac >= 3:
                break
            time.sleep(0.6)
            continue
        vac = 0
        for v in lote:
            todas[v["from"]] = v
        oldest = min(v["from"] for v in lote)
        if oldest <= limite:
            break
        newend = oldest - 1
        if newend >= end:
            break
        end = newend
        if len(lote) < 2:
            break
    return [todas[k] for k in sorted(todas)]


def guardar(p, vv):
    json.dump({"tf_seg": GRAN,
               "open": [x["open"] for x in vv],
               "high": [x.get("max", x.get("high", x["close"])) for x in vv],
               "low": [x.get("min", x.get("low", x["close"])) for x in vv],
               "close": [x["close"] for x in vv],
               "volume": [x.get("volume", 0) for x in vv],
               "times": [x["from"] for x in vv]},
              open(p, "w", encoding="utf-8"))


def span_ok(p, dias):
    try:
        d = json.load(open(p, encoding="utf-8"))
        t = d.get("times", [])
        return ("volume" in d and len(t) >= 1000
                and (t[-1] - t[0]) / 86400 >= dias * 0.85)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=300)
    ap.add_argument("--pares", default="", help="coma; vacio = los de config")
    a = ap.parse_args()

    cfg = json.load(open("config.json", encoding="utf-8"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    if a.pares:
        pares = [p.strip() for p in a.pares.split(",") if p.strip()]
    else:
        pares = [p for p in dict.fromkeys(cfg.get("pares_binarios", []))
                 if "-OTC" not in p]
    log(f"Objetivo: {a.dias} dias @5m CON VOLUMEN | {len(pares)} pares -> {CACHE_DIR}/")

    api = conectar(cfg)
    hecho = ya = fail = 0
    for i, name in enumerate(pares):
        p = os.path.join(CACHE_DIR, name.replace("/", "_") + ".json")
        if span_ok(p, a.dias):
            ya += 1
            continue
        vv = bajar_span(api, name, a.dias)
        if vv is None:
            log(f"[{i+1}/{len(pares)}] {name} TIMEOUT, reconectando...")
            try:
                del api
            except Exception:
                pass
            time.sleep(3)
            api = conectar(cfg)
            vv = bajar_span(api, name, a.dias)
        if not vv or len(vv) < 1000:
            fail += 1
            log(f"[{i+1}/{len(pares)}] {name} sin datos ({len(vv) if vv else 0})")
            continue
        guardar(p, vv)
        hecho += 1
        span = (vv[-1]["from"] - vv[0]["from"]) / 86400
        convol = sum(1 for x in vv if x.get("volume"))
        log(f"[{i+1}/{len(pares)}] {name}: {len(vv)} velas, {span:.0f} dias, "
            f"{100*convol/len(vv):.0f}% con volumen | ok {hecho} ya {ya} fail {fail}")
        if hecho and hecho % RECONNECT_CADA == 0:
            try:
                del api
            except Exception:
                pass
            time.sleep(2)
            api = conectar(cfg)
        time.sleep(0.2)

    log(f"LISTO: {hecho} bajados, {ya} ya estaban, {fail} sin datos.")


if __name__ == "__main__":
    main()
