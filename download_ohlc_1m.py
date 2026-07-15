# download_ohlc_1m.py - Baja ~2.6 meses de velas 1m para los 231 activos -> cache_ohlc_1m/.
# Para backtest de estrategia 1m / expiracion corta (turbo 2min).
# Paginado por SPAN de tiempo, resumible, con opcode init + reconexion.
#   .venv314\Scripts\python.exe download_ohlc_1m.py
import json, os, time, threading
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option

DIAS = 80                 # ~2.6 meses -> permite >=1 mes train y 1 mes test (ver backtest)
GRAN = 60                 # 1m
CACHE_DIR = "cache_ohlc_1m"
CANDLE_TIMEOUT = 30
RECONNECT_CADA = 25


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def conectar(cfg):
    api = IQ_Option(cfg["email"], cfg["password"])
    ok, r = api.connect()
    if not ok:
        raise Exception(f"connect: {r}")
    api.change_balance("PRACTICE")
    # Poblar ACTIVES/opcode (sin esto, get_candles falla en la mayoria de OTC).
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
    vac = 0
    calls = 0
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


def path(name):
    return os.path.join(CACHE_DIR, name.replace("/", "_") + ".json")


def span_ok(p, dias):
    try:
        d = json.load(open(p, encoding="utf-8"))
        t = d.get("times", [])
        return len(t) >= 1000 and (t[-1] - t[0]) / 86400 >= dias * 0.9
    except Exception:
        return False


def guardar(p, vv):
    json.dump({"tf_seg": GRAN,
               "open": [x["open"] for x in vv],
               "high": [x.get("max", x.get("high", x["close"])) for x in vv],
               "low": [x.get("min", x.get("low", x["close"])) for x in vv],
               "close": [x["close"] for x in vv],
               "times": [x["from"] for x in vv]},
              open(p, "w", encoding="utf-8"))


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    pares = list(dict.fromkeys(cfg.get("pares_binarios", [])))
    log(f"Objetivo: {DIAS} dias @1m | {len(pares)} activos -> {CACHE_DIR}/")

    api = conectar(cfg)
    hecho = ya = fail = 0
    for i, name in enumerate(pares):
        p = path(name)
        if span_ok(p, DIAS):
            ya += 1
            if (i + 1) % 25 == 0:
                log(f"[{i+1}/{len(pares)}] {name} (ya cubre span)")
            continue

        vv = bajar_span(api, name, DIAS)
        if vv is None:
            log(f"[{i+1}/{len(pares)}] {name} TIMEOUT, reconectando...")
            try:
                del api
            except Exception:
                pass
            time.sleep(3)
            api = conectar(cfg)
            vv = bajar_span(api, name, DIAS)
        if not vv or len(vv) < 1000:
            fail += 1
            log(f"[{i+1}/{len(pares)}] {name} sin datos ({len(vv) if vv else 0})")
            continue

        guardar(p, vv)
        hecho += 1
        span = (vv[-1]["from"] - vv[0]["from"]) / 86400
        log(f"[{i+1}/{len(pares)}] {name}: {len(vv)} velas 1m, {span:.0f} dias | ok {hecho} ya {ya} fail {fail}")

        if hecho and hecho % RECONNECT_CADA == 0:
            log("  [RECONNECT] instancia nueva...")
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
