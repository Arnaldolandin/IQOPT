# actualizar_cache_5m.py - Refresca cache_ohlc_5m con las velas NUEVAS (incremental):
# por cada archivo existente, baja las ultimas ~1000 velas 5m y las mergea (dedup por
# timestamp). Rapido (no re-baja los 6 meses). Uso: .venv314\...\python.exe actualizar_cache_5m.py
import json, glob, os, time, threading
from iqoptionapi.stable_api import IQ_Option

GRAN = 300; CACHE = "cache_ohlc_5m"; TIMEOUT = 25


def conectar():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    ok, _ = api.connect()
    if not ok:
        print("no conecto"); return None
    api.change_balance("PRACTICE"); time.sleep(2)
    r = [None]
    t = threading.Thread(target=lambda: r.__setitem__(0, api.get_ALL_Binary_ACTIVES_OPCODE()), daemon=True)
    t.start(); t.join(timeout=TIMEOUT)
    return api


def fetch(api, par):
    r = [None]
    def _g():
        try: r[0] = api.get_candles(par, GRAN, 1000, time.time())
        except Exception: r[0] = None
    t = threading.Thread(target=_g, daemon=True); t.start(); t.join(timeout=TIMEOUT)
    return r[0]


def main():
    api = conectar()
    if not api:
        return
    files = sorted(glob.glob(os.path.join(CACHE, "*.json")))
    print(f"Actualizando {len(files)} archivos de {CACHE}/...")
    tot_new = 0
    for i, f in enumerate(files):
        par = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        nuevas = fetch(api, par)
        if not nuevas:
            print(f"[{i+1}/{len(files)}] {par}: sin datos (timeout/no ofrecido)"); continue
        # merge por 'from' timestamp
        by_t = {int(t): (o, h, l, c) for t, o, h, l, c in
                zip(d["times"], d["open"], d["high"], d["low"], d["close"])}
        antes = len(by_t)
        for v in nuevas:
            by_t[int(v["from"])] = (float(v["open"]), float(v["max"]), float(v["min"]), float(v["close"]))
        ts = sorted(by_t)
        d["times"] = ts
        d["open"] = [by_t[t][0] for t in ts]
        d["high"] = [by_t[t][1] for t in ts]
        d["low"] = [by_t[t][2] for t in ts]
        d["close"] = [by_t[t][3] for t in ts]
        json.dump(d, open(f, "w", encoding="utf-8"))
        nuevos = len(by_t) - antes
        tot_new += nuevos
        print(f"[{i+1}/{len(files)}] {par}: +{nuevos} velas (total {len(ts)})", flush=True)
    print(f"\nListo. {tot_new} velas nuevas en total.")


if __name__ == "__main__":
    main()
