# download_ohlc.py - Descarga OHLC 5m (open/high/low/close) y cachea en cache_ohlc/.
# Necesario para indicadores de rango (ADX, ATR). Cache por par = resumible; robusto a caidas.
#   .venv314\Scripts\python.exe download_ohlc.py
import json, os, time, threading
from datetime import datetime
import numpy as np
from iqoptionapi.stable_api import IQ_Option

N_VELAS_1M = 45000; GRAN = 60; CACHE_DIR = "cache_ohlc"


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def bajar(api, par, total):
    todas = {}; end = time.time(); vac = 0
    while len(todas) < total:
        try: lote = api.get_candles(par, GRAN, 1000, end)
        except Exception: break
        if not lote:
            vac += 1
            if vac >= 3: break
            time.sleep(1); continue
        vac = 0
        for v in lote: todas[v["from"]] = v
        end = min(v["from"] for v in lote) - 1
        if len(lote) < 2: break
    return [todas[k] for k in sorted(todas)]


def bajar_timeout(api, par, total, t=150):
    r = {"v": None, "d": False}
    def w():
        try: r["v"] = bajar(api, par, total)
        except Exception: r["v"] = None
        r["d"] = True
    th = threading.Thread(target=w, daemon=True); th.start(); th.join(t)
    return r["v"] if r["d"] else None


def resample_ohlc(v1, tf):
    b = {}
    for v in v1:
        k = v["from"] - (v["from"] % tf)
        hi = v.get("max", v.get("high", v["close"])); lo = v.get("min", v.get("low", v["close"]))
        if k not in b:
            b[k] = {"from": k, "open": v["open"], "high": hi, "low": lo, "close": v["close"]}
        else:
            b[k]["high"] = max(b[k]["high"], hi); b[k]["low"] = min(b[k]["low"], lo)
            b[k]["close"] = v["close"]
    return [b[k] for k in sorted(b)]


def descubrir_otc(pr):
    o = set()
    for k in pr:
        if "-OTC" in k: o.add(k[:-3] if k.endswith("-op") else k)
    return sorted(o)


def path(name): return os.path.join(CACHE_DIR, name.replace("/", "_") + ".json")


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    tf = cfg["operacion"]["timeframe_seg"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    api = IQ_Option(cfg["email"], cfg["password"])
    log("Conectando..."); ok, r = api.connect()
    if not ok: log(f"NO CONECTO: {r}"); return
    api.change_balance("PRACTICE")
    d = [False]
    def u():
        try: api.get_ALL_Binary_ACTIVES_OPCODE()
        except Exception: pass
        d[0] = True
    threading.Thread(target=u, daemon=True).start()
    t0 = time.time()
    while not d[0] and time.time()-t0 < 45: time.sleep(1)
    pr = api.get_all_profit() or {}
    pares = list(dict.fromkeys(cfg.get("pares_binarios", []) + descubrir_otc(pr)))
    log(f"{len(pares)} pares. Cache OHLC en {CACHE_DIR}/")
    cache = baj = fail = 0
    for i, name in enumerate(pares):
        p = path(name)
        if os.path.exists(p):
            try:
                dd = json.load(open(p, encoding="utf-8"))
                if dd.get("tf_seg") == tf and len(dd.get("close", [])) >= 300:
                    cache += 1
                    if (i+1) % 20 == 0: log(f"[{i+1}/{len(pares)}] {name} (cache)")
                    continue
            except Exception: pass
        v1 = bajar_timeout(api, name, N_VELAS_1M)
        if v1 is None:
            log(f"[{i+1}/{len(pares)}] {name} TIMEOUT -> reconecta")
            try: api.connect(); api.change_balance("PRACTICE")
            except Exception: pass
            v1 = bajar_timeout(api, name, N_VELAS_1M)
        if not v1 or len(v1) < 500:
            fail += 1; log(f"[{i+1}/{len(pares)}] {name} sin datos"); continue
        vv = resample_ohlc(v1, tf)
        json.dump({"tf_seg": tf,
                   "open": [x["open"] for x in vv], "high": [x["high"] for x in vv],
                   "low": [x["low"] for x in vv], "close": [x["close"] for x in vv],
                   "times": [x["from"] for x in vv]},
                  open(p, "w", encoding="utf-8"))
        baj += 1
        if (i+1) % 10 == 0: log(f"[{i+1}/{len(pares)}] {name} ({len(vv)} velas) | cache {cache} baj {baj}")
    log(f"LISTO: {cache} cacheados, {baj} descargados, {fail} sin datos. OHLC en {CACHE_DIR}/")


if __name__ == "__main__":
    main()
