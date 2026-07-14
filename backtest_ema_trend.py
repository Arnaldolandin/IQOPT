# backtest_ema_trend.py - MACD-crossover con filtro de tendencia EMA(100).
#
# Pregunta: filtrar las senales para operar SOLO a favor de la tendencia (CALL si close>EMA100,
# PUT si close<EMA100) da edge? Se mide el WR de test AGREGADO (ponderado por nº de trades),
# SIN seleccion por hora (para no colar sobreajuste). Compara:
#   - baseline   : todas las senales MACD
#   - a favor    : solo senales alineadas con EMA100 (tendencia)
#   - en contra  : solo senales opuestas a EMA100 (reversion)
# Separado real vs OTC. Split train/test 70/30 cronologico.
#
#   .venv314\Scripts\python.exe backtest_ema_trend.py
import json, time, threading
from datetime import datetime, timezone
import numpy as np
from iqoptionapi.stable_api import IQ_Option

N_VELAS_1M = 45000
GRAN = 60
SPLIT = 0.70
EMA_TREND = 100
MIN_SIG_TRAIN = 30      # minimo de senales en train para incluir el par en el agregado
PAYOUT_FALLBACK = 0.85


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
        return []
    macd_l = ema(c, fast) - ema(c, slow)
    sig_l = ema(macd_l, sig_p)
    out = [0] * len(c)
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
        t = v["from"]
        b0 = t - (t % tf_seg)
        if b0 not in buckets:
            buckets[b0] = {"from": b0, "close": v["close"]}
        buckets[b0]["close"] = v["close"]
    return [buckets[k] for k in sorted(buckets)]


def bajar_velas(api, par, total):
    todas = {}
    endtime = time.time()
    vacios = 0
    while len(todas) < total:
        try:
            lote = api.get_candles(par, GRAN, 1000, endtime)
        except Exception:
            break
        if not lote:
            vacios += 1
            if vacios >= 3:
                break
            time.sleep(1)
            continue
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
        try:
            res["velas"] = bajar_velas(api, name, total)
        except Exception:
            res["velas"] = None
        res["done"] = True
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    return res["velas"] if res["done"] else None


def descubrir_otc(profits):
    otc = set()
    for k in profits:
        if "-OTC" in k:
            otc.add(k[:-3] if k.endswith("-op") else k)
    return sorted(otc)


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    fast = cfg["macd"]["fast"]; slow = cfg["macd"]["slow"]; sig_p = cfg["macd"]["signal"]
    tf_seg = cfg["operacion"]["timeframe_seg"]
    expiry_min = cfg["operacion"]["expiry_min"]
    h = max(1, round(expiry_min * 60 / tf_seg))

    api = IQ_Option(cfg["email"], cfg["password"])
    log("Conectando...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}")
        return
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
    log(f"MACD({fast},{slow},{sig_p}) TF {tf_seg//60}m exp {expiry_min}m (h={h}) | filtro EMA({EMA_TREND}) | "
        f"{sum(1 for _,o in pares if not o)} reales + {sum(1 for _,o in pares if o)} OTC")

    def payout_de(name, es_otc):
        key = name if es_otc else f"{name}-op"
        info = profits.get(key, {}) or {}
        p = info.get("turbo") or info.get("binary")
        return float(p) if p else PAYOUT_FALLBACK

    # acumuladores agregados por grupo y variante: [wins, n] en test
    agg = {g: {v: [0, 0] for v in ("base", "favor", "contra")}
           for g in ("real", "otc")}
    por_par = {}

    for idx, (name, es_otc) in enumerate(pares):
        payout = payout_de(name, es_otc)
        grp = "otc" if es_otc else "real"
        log(f"[{idx+1}/{len(pares)}] {name} ({grp})...")

        velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
        if velas_1m is None:
            log(f"  {name}: TIMEOUT -> reconectando...")
            try:
                api.connect(); api.change_balance("PRACTICE")
            except Exception:
                pass
            velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
            if velas_1m is None:
                log(f"  {name}: sigue colgado, salto."); continue
        if not velas_1m or len(velas_1m) < 500:
            log(f"  {name}: sin datos suficientes ({len(velas_1m) if velas_1m else 0})"); continue

        velas = resample(velas_1m, tf_seg)
        closes = np.asarray([v["close"] for v in velas], float)
        n = len(closes)
        cut = int(n * SPLIT)
        sigs = macd_cruces(closes, fast, slow, sig_p)
        if not sigs:
            continue
        ema_t = ema(closes, EMA_TREND)

        # por par: [wins, n] en train y test, por variante
        loc = {v: {"tr": [0, 0], "te": [0, 0]} for v in ("base", "favor", "contra")}
        for i in range(EMA_TREND, n - h):
            s = sigs[i]
            if s == 0:
                continue
            up = closes[i] > ema_t[i]
            a_favor = (s == 1 and up) or (s == -1 and not up)
            gano = (closes[i + h] > closes[i]) if s == 1 else (closes[i + h] < closes[i])
            seg = "tr" if i < cut else "te"
            for v in ("base", "favor" if a_favor else "contra"):
                loc[v][seg][1] += 1
                loc[v][seg][0] += 1 if gano else 0

        if loc["base"]["tr"][1] < MIN_SIG_TRAIN:
            continue

        def wr(pair):
            w, nn = pair
            return (100.0 * w / nn) if nn else float("nan")
        por_par[name] = {"otc": es_otc, "payout": payout,
                         "base": {"tr_wr": wr(loc["base"]["tr"]), "tr_n": loc["base"]["tr"][1],
                                  "te_wr": wr(loc["base"]["te"]), "te_n": loc["base"]["te"][1]},
                         "favor": {"tr_wr": wr(loc["favor"]["tr"]), "tr_n": loc["favor"]["tr"][1],
                                   "te_wr": wr(loc["favor"]["te"]), "te_n": loc["favor"]["te"][1]},
                         "contra": {"tr_wr": wr(loc["contra"]["tr"]), "tr_n": loc["contra"]["tr"][1],
                                    "te_wr": wr(loc["contra"]["te"]), "te_n": loc["contra"]["te"][1]}}
        for v in ("base", "favor", "contra"):
            agg[grp][v][0] += loc[v]["te"][0]
            agg[grp][v][1] += loc[v]["te"][1]

        if (idx + 1) % 10 == 0:
            _guardar(agg, por_par, cfg, h)

    _guardar(agg, por_par, cfg, h)
    _resumen(agg)


def _guardar(agg, por_par, cfg, h):
    out = {"agregado": agg, "por_par": por_par,
           "meta": {"ema_trend": EMA_TREND, "expiry_min": cfg["operacion"]["expiry_min"],
                    "h": h, "split": SPLIT, "macd": [cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]]}}
    with open("backtest_ema_trend.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _resumen(agg):
    print("\n" + "=" * 72)
    print(f"WR de TEST agregado (ponderado por nº trades) | filtro EMA({EMA_TREND})")
    print("=" * 72)
    print(f"{'Grupo':6s} {'variante':8s} {'trades':>8s} {'WR test':>9s}")
    print("-" * 40)
    for g in ("real", "otc"):
        for v in ("base", "favor", "contra"):
            w, n = agg[g][v]
            wr = (100.0 * w / n) if n else float("nan")
            etiqueta = {"base": "baseline", "favor": "a-favor", "contra": "en-contra"}[v]
            print(f"{g:6s} {etiqueta:8s} {n:8d} {wr:8.2f}%")
        print("-" * 40)
    print("Break-even: real ~53.5% (payout 87%), OTC ~54.1% (payout 85%).")
    print("Guardado backtest_ema_trend.json")


if __name__ == "__main__":
    main()
