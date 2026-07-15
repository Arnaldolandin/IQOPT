# download_ohlc_30m.py - Baja ~6.5 meses de velas 30m para los pares REALES -> cache_ohlc_30m/.
# Para validar el timeframe 30m con OOS largo. 30m directo (GRAN=1800), por span, resumible.
#   .venv314\Scripts\python.exe download_ohlc_30m.py
import json, os, time, threading
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option

DIAS = 200                # ~6.5 meses
GRAN = 1800               # 30m
CACHE_DIR = "cache_ohlc_30m"
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
    vac = 0
    calls = 0
    while calls < 400:
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
        # 30m: aceptar si ya cubre >= 85% del objetivo (IQ puede topar antes)
        return len(t) >= 500 and (t[-1] - t[0]) / 86400 >= dias * 0.85
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
    # SOLO pares reales (sin -OTC): son donde el TF 30m mostro señal
    pares = [p for p in dict.fromkeys(cfg.get("pares_binarios", [])) if "-OTC" not in p]
    log(f"Objetivo: {DIAS} dias @30m | {len(pares)} pares REALES -> {CACHE_DIR}/")

    api = conectar(cfg)
    hecho = ya = fail = 0
    for i, name in enumerate(pares):
        p = path(name)
        if span_ok(p, DIAS):
            ya += 1
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
        if not vv or len(vv) < 500:
            fail += 1
            log(f"[{i+1}/{len(pares)}] {name} sin datos ({len(vv) if vv else 0})")
            continue
        guardar(p, vv)
        hecho += 1
        span = (vv[-1]["from"] - vv[0]["from"]) / 86400
        log(f"[{i+1}/{len(pares)}] {name}: {len(vv)} velas 30m, {span:.0f} dias | ok {hecho} ya {ya} fail {fail}")
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
