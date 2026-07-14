# backtest_macd_ema_sweep.py - Barrido MACD x EMA-tendencia con medicion honesta agregada.
#
# Para cada par (real + OTC), cada config MACD y cada periodo de EMA, mide el WR de TEST
# agregado (ponderado por nº de trades), SIN seleccion por hora:
#   - baseline  : todas las senales del MACD (no depende de la EMA)
#   - a-favor   : senal alineada con la EMA (close>EMA -> solo CALL; close<EMA -> solo PUT)
#   - en-contra : senal opuesta a la EMA (reversion)
# Split train/test 70/30. Cachea las velas 5m en cache_closes/ para reusar en futuros barridos.
#
#   .venv314\Scripts\python.exe backtest_macd_ema_sweep.py
import json, os, time, threading
from datetime import datetime, timezone
import numpy as np
from iqoptionapi.stable_api import IQ_Option

N_VELAS_1M = 45000
GRAN = 60
SPLIT = 0.70
WARMUP = 250                  # ventana comun de arranque (>= max EMA y max slow+signal)
MIN_SIG_TRAIN = 30
PAYOUT_FALLBACK = 0.85
CACHE_DIR = "cache_closes"

MACD_CONFIGS = [(8, 17, 9), (12, 26, 9), (19, 39, 9), (6, 13, 5), (24, 52, 18)]
EMA_TRENDS = [50, 100, 200]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_cruces(closes, fast, slow, sig_p):
    c = np.asarray(closes, dtype=float)
    if len(c) < slow + sig_p + 2:
        return np.zeros(len(c), dtype=int)
    macd_l = ema(c, fast) - ema(c, slow)
    sig_l = ema(macd_l, sig_p)
    out = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if macd_l[i-1] <= sig_l[i-1] and macd_l[i] > sig_l[i]:
            out[i] = 1
        elif macd_l[i-1] >= sig_l[i-1] and macd_l[i] < sig_l[i]:
            out[i] = -1
    return out


def resample(velas_1m, tf_seg):
    if not velas_1m:
        return []
    buckets = {}
    for v in velas_1m:
        t = v["from"]; b0 = t - (t % tf_seg)
        if b0 not in buckets:
            buckets[b0] = {"from": b0, "close": v["close"]}
        buckets[b0]["close"] = v["close"]
    return [buckets[k] for k in sorted(buckets)]


def bajar_velas(api, par, total):
    todas = {}; endtime = time.time(); vacios = 0
    while len(todas) < total:
        try:
            lote = api.get_candles(par, GRAN, 1000, endtime)
        except Exception:
            break
        if not lote:
            vacios += 1
            if vacios >= 3:
                break
            time.sleep(1); continue
        vacios = 0
        for v in lote:
            todas[v["from"]] = v
        endtime = min(v["from"] for v in lote) - 1
        if len(lote) < 2:
            break
    return [todas[k] for k in sorted(todas)]


def bajar_con_timeout(api, name, total, timeout=150):
    res = {"velas": None, "done": False}
    def worker():
        try: res["velas"] = bajar_velas(api, name, total)
        except Exception: res["velas"] = None
        res["done"] = True
    th = threading.Thread(target=worker, daemon=True)
    th.start(); th.join(timeout)
    return res["velas"] if res["done"] else None


def descubrir_otc(profits):
    otc = set()
    for k in profits:
        if "-OTC" in k:
            otc.add(k[:-3] if k.endswith("-op") else k)
    return sorted(otc)


def cache_path(name):
    safe = name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def get_closes(api, name, tf_seg):
    """Devuelve np.array de closes 5m; usa cache si existe, si no descarga y cachea."""
    p = cache_path(name)
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding="utf-8"))
            if d.get("tf_seg") == tf_seg and len(d.get("closes", [])) >= 500:
                return np.asarray(d["closes"], float), True
        except Exception:
            pass
    velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
    if velas_1m is None:
        log(f"  {name}: TIMEOUT -> reconectando...")
        try:
            api.connect(); api.change_balance("PRACTICE")
        except Exception:
            pass
        velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
    if not velas_1m or len(velas_1m) < 500:
        return None, False
    velas = resample(velas_1m, tf_seg)
    closes = [v["close"] for v in velas]
    try:
        json.dump({"tf_seg": tf_seg, "closes": closes,
                   "times": [v["from"] for v in velas]},
                  open(p, "w", encoding="utf-8"))
    except Exception:
        pass
    return np.asarray(closes, float), False


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    tf_seg = cfg["operacion"]["timeframe_seg"]
    expiry_min = cfg["operacion"]["expiry_min"]
    h = max(1, round(expiry_min * 60 / tf_seg))
    os.makedirs(CACHE_DIR, exist_ok=True)

    api = IQ_Option(cfg["email"], cfg["password"])
    log("Conectando...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}"); return
    api.change_balance("PRACTICE")

    log("Actualizando opcode...")
    done = [False]
    def _upd():
        try: api.get_ALL_Binary_ACTIVES_OPCODE()
        except Exception: pass
        done[0] = True
    threading.Thread(target=_upd, daemon=True).start()
    t0 = time.time()
    while not done[0] and time.time() - t0 < 45:
        time.sleep(1)

    profits = api.get_all_profit() or {}
    nombres = list(dict.fromkeys(cfg.get("pares_binarios", []) + descubrir_otc(profits)))
    pares = [(n, "-OTC" in n) for n in nombres]
    log(f"Barrido {len(MACD_CONFIGS)} MACD x {len(EMA_TRENDS)} EMA | exp {expiry_min}m (h={h}) | "
        f"{sum(1 for _,o in pares if not o)} reales + {sum(1 for _,o in pares if o)} OTC")

    def payout_de(name, es_otc):
        key = name if es_otc else f"{name}-op"
        info = profits.get(key, {}) or {}
        p = info.get("turbo") or info.get("binary")
        return float(p) if p else PAYOUT_FALLBACK

    # agg[grp][macd_str] = {"base":[w,n], "favor":{ema:[w,n]}, "contra":{ema:[w,n]}}
    def nuevo_macd_acc():
        return {"base": [0, 0],
                "favor": {e: [0, 0] for e in EMA_TRENDS},
                "contra": {e: [0, 0] for e in EMA_TRENDS}}
    agg = {g: {f"{m}": nuevo_macd_acc() for m in MACD_CONFIGS} for g in ("real", "otc")}

    n_cache = n_baj = 0
    for idx, (name, es_otc) in enumerate(pares):
        grp = "otc" if es_otc else "real"
        closes, from_cache = get_closes(api, name, tf_seg)
        if closes is None:
            log(f"[{idx+1}/{len(pares)}] {name} ({grp}): sin datos, salto"); continue
        n_cache += int(from_cache); n_baj += int(not from_cache)
        n = len(closes); cut = int(n * SPLIT)
        emas = {e: ema(closes, e) for e in EMA_TRENDS}
        log(f"[{idx+1}/{len(pares)}] {name} ({grp}, {'cache' if from_cache else 'baj'}, {n} velas)")

        for (fa, sl, si) in MACD_CONFIGS:
            sigs = macd_cruces(closes, fa, sl, si)
            acc = agg[grp][f"{(fa, sl, si)}"]
            ntr = 0
            for i in range(WARMUP, n - h):
                s = sigs[i]
                if s == 0:
                    continue
                if i < cut:
                    ntr += 1
                    continue  # train solo cuenta para el minimo; medimos WR en test
                gano = (closes[i + h] > closes[i]) if s == 1 else (closes[i + h] < closes[i])
                acc["base"][1] += 1; acc["base"][0] += 1 if gano else 0
                for e in EMA_TRENDS:
                    up = closes[i] > emas[e][i]
                    a_favor = (s == 1 and up) or (s == -1 and not up)
                    tgt = acc["favor"][e] if a_favor else acc["contra"][e]
                    tgt[1] += 1; tgt[0] += 1 if gano else 0

        if (idx + 1) % 10 == 0:
            _guardar(agg, cfg, h)

    _guardar(agg, cfg, h)
    log(f"Cache: {n_cache} reusados, {n_baj} descargados.")
    _resumen(agg)


def _guardar(agg, cfg, h):
    out = {"agregado": agg,
           "meta": {"macd_configs": [list(m) for m in MACD_CONFIGS], "ema_trends": EMA_TRENDS,
                    "expiry_min": cfg["operacion"]["expiry_min"], "h": h, "split": SPLIT}}
    json.dump(out, open("backtest_macd_ema_sweep.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


def _resumen(agg):
    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        print("\n" + "=" * 78)
        print(f"GRUPO {g.upper()} | WR test agregado (BE ~{be}%)  [n entre parentesis]")
        print("=" * 78)
        hdr = f"{'MACD':14s} {'baseline':>14s}"
        for e in EMA_TRENDS:
            hdr += f" {'favEMA'+str(e):>13s}"
        print(hdr)
        print("-" * 78)
        for m in MACD_CONFIGS:
            acc = agg[g][f"{m}"]
            w, n = acc["base"]
            row = f"{str(m):14s} {(100*w/n if n else 0):6.2f}% ({n:5d})"
            for e in EMA_TRENDS:
                w2, n2 = acc["favor"][e]
                row += f" {(100*w2/n2 if n2 else 0):5.1f}%({n2:5d})"
            print(row)
    print("\nGuardado backtest_macd_ema_sweep.json")


if __name__ == "__main__":
    main()
