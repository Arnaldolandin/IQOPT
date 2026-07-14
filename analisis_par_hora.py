# analisis_par_hora.py - Backtest par x hora-del-dia (UTC) para reajustar filtro_hora.
#
# Evalua el resultado REAL del trade (close[i+h] vs close[i]), con h derivado del expiry TURBO
# del config. Split train/test 70/30 cronologico: solo se conservan horas rentables en train Y
# que sobreviven (>= break-even) en test.
#
# Incluye activos REALES (config.pares_binarios) y TODOS los OTC que ofrezca IQ (auto-descubiertos
# de get_all_profit, claves con "-OTC"). Cada combo queda etiquetado otc=True/False.
# OJO: con cientos de pares x 24 horas hay miles de combos -> muchos "ganadores" en test seran
# ruido (multiple-testing). Leer con cautela; esto NO demuestra rentabilidad en vivo.
#
#   .venv314\Scripts\python.exe analisis_par_hora.py
import json, time, threading
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from iqoptionapi.stable_api import IQ_Option

N_VELAS_1M = 45000          # ~31 dias de velas de 1m por par (>= 1 mes)
GRAN = 60
SPLIT = 0.70
MIN_TRADES_TRAIN = 10
MIN_TRADES_TEST = 5
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
    """Ejecuta bajar_velas en un hilo; si get_candles se cuelga, devuelve None al vencer timeout."""
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
    """Nombres de velas OTC: claves de get_all_profit que contienen '-OTC' (sin sufijo -op)."""
    otc = set()
    for k in profits:
        if "-OTC" in k:
            otc.add(k[:-3] if k.endswith("-op") else k)
    return sorted(otc)


def main():
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    fast = cfg["macd"]["fast"]; slow = cfg["macd"]["slow"]; sig_p = cfg["macd"]["signal"]
    tf_seg = cfg["operacion"]["timeframe_seg"]
    expiry_min = cfg["operacion"]["expiry_min"]
    h = max(1, round(expiry_min * 60 / tf_seg))
    min_wr = cfg.get("filtro_hora", {}).get("min_wr", 55.0)

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
    t = threading.Thread(target=_upd, daemon=True)
    t.start()
    t.join(timeout=45)

    profits = api.get_all_profit() or {}

    # Union deduplicada de pares del config + OTC descubiertos. Se clasifica por el sufijo
    # "-OTC" en el nombre (evita procesar dos veces un OTC que ya este en pares_binarios).
    nombres = list(dict.fromkeys(cfg.get("pares_binarios", []) + descubrir_otc(profits)))
    pares = [(n, "-OTC" in n) for n in nombres]
    reales = [n for n, o in pares if not o]
    otc = [n for n, o in pares if o]
    log(f"MACD({fast},{slow},{sig_p}) | TF {tf_seg//60}m | expiry {expiry_min}m -> h={h} | "
        f"{len(reales)} reales + {len(otc)} OTC = {len(pares)} pares | "
        f"umbral train WR>={min_wr}% y test>=break-even")

    def payout_de(name, es_otc):
        key = name if es_otc else f"{name}-op"
        info = profits.get(key, {}) or {}
        p = info.get("turbo") or info.get("binary")
        return float(p) if p else PAYOUT_FALLBACK

    all_combos = []
    horas_por_par = {}
    rango = {}       # name -> [primera_vela_iso, ultima_vela_iso, dias]

    def guardar():
        out = {"horas_por_par": horas_por_par, "combos": all_combos, "rango_fechas": rango,
               "meta": {"macd": [fast, slow, sig_p], "tf_seg": tf_seg, "expiry_min": expiry_min,
                        "h": h, "split": SPLIT, "min_wr": min_wr,
                        "n_reales": len(reales), "n_otc": len(otc)}}
        with open("analisis_par_hora.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    for idx, (name, es_otc) in enumerate(pares):
        payout = payout_de(name, es_otc)
        be_pct = 100.0 / (1.0 + payout)
        tag = "OTC" if es_otc else "real"

        log(f"[{idx+1}/{len(pares)}] {name} ({tag}, payout {payout*100:.0f}%, BE {be_pct:.1f}%)...")
        velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
        if velas_1m is None:
            log(f"  {name}: TIMEOUT (websocket colgado) -> reconectando...")
            try:
                api.connect(); api.change_balance("PRACTICE")
            except Exception as e:
                log(f"  reconexion fallo: {e}")
            velas_1m = bajar_con_timeout(api, name, N_VELAS_1M)
            if velas_1m is None:
                log(f"  {name}: sigue colgado tras reconectar, salto.")
                continue
        if not velas_1m or len(velas_1m) < 200:
            log(f"  {name}: sin datos suficientes ({len(velas_1m) if velas_1m else 0})")
            continue

        t0 = datetime.fromtimestamp(velas_1m[0]["from"], tz=timezone.utc)
        t1 = datetime.fromtimestamp(velas_1m[-1]["from"], tz=timezone.utc)
        dias = (velas_1m[-1]["from"] - velas_1m[0]["from"]) / 86400
        rango[name] = [t0.isoformat(), t1.isoformat(), round(dias, 1)]

        velas = resample(velas_1m, tf_seg)
        closes = [v["close"] for v in velas]
        times = [v["from"] for v in velas]
        n = len(closes)
        cut = int(n * SPLIT)

        sigs = macd_cruces(closes, fast, slow, sig_p)
        if not sigs:
            continue

        data = defaultdict(lambda: {"train": [], "test": []})
        for i in range(1, n - h):
            if sigs[i] == 0:
                continue
            gano = (closes[i + h] > closes[i]) if sigs[i] == 1 else (closes[i + h] < closes[i])
            hora = datetime.fromtimestamp(times[i], tz=timezone.utc).hour
            data[hora]["train" if i < cut else "test"].append(1 if gano else 0)

        buenas = []
        for hora in sorted(data.keys()):
            tr, te = data[hora]["train"], data[hora]["test"]
            if len(tr) < MIN_TRADES_TRAIN or len(te) < MIN_TRADES_TEST:
                continue
            tr_wr = 100.0 * sum(tr) / len(tr)
            te_wr = 100.0 * sum(te) / len(te)
            te_ev = te_wr / 100 * payout - (1 - te_wr / 100)
            combo = {"par": name, "otc": es_otc, "hora": hora, "payout": payout, "be": be_pct,
                     "tr_n": len(tr), "tr_wr": tr_wr, "te_n": len(te), "te_wr": te_wr,
                     "te_ev": te_ev, "brecha": tr_wr - te_wr}
            all_combos.append(combo)
            if tr_wr >= min_wr and te_wr >= be_pct:
                buenas.append(combo)

        buenas.sort(key=lambda c: -c["te_ev"])
        horas_sel = sorted(c["hora"] for c in buenas)
        if horas_sel:
            horas_por_par[name] = horas_sel
            log(f"  {name}: {len(horas_sel)} horas validadas ({tag}, ~{dias:.0f}d) -> {horas_sel}")

        if (idx + 1) % 10 == 0:
            guardar()

    # ── Resumen ─────────────────────────────────────────────────────────────
    validadas = [c for c in all_combos if c["tr_wr"] >= min_wr and c["te_wr"] >= c["be"]]
    validadas.sort(key=lambda c: -c["te_ev"])

    print(f"\n{'='*80}")
    print("TOP 50 par x hora que sobreviven train+test (ordenado por EV test)")
    print(f"{'='*80}")
    print(f"{'Par':16s} {'T':>3s} {'H':>3s} {'trWR':>6s} {'trN':>4s} {'teWR':>6s} {'teN':>4s} {'EV':>7s} {'brecha':>7s}")
    print("-" * 70)
    for c in validadas[:50]:
        print(f"{c['par']:16s} {'OTC' if c['otc'] else 'rea':>3s} {c['hora']:3d} "
              f"{c['tr_wr']:5.1f}% {c['tr_n']:4d} {c['te_wr']:5.1f}% {c['te_n']:4d} "
              f"{c['te_ev']*100:+5.1f}% {c['brecha']:+6.1f}")

    n_real = sum(1 for k in horas_por_par if not k.endswith("-OTC"))
    n_otc = sum(1 for k in horas_por_par if k.endswith("-OTC"))
    ph_real = sum(len(v) for k, v in horas_por_par.items() if not k.endswith("-OTC"))
    ph_otc = sum(len(v) for k, v in horas_por_par.items() if k.endswith("-OTC"))
    print(f"\nPares con >=1 hora validada: {n_real} reales ({ph_real} pares-hora) + "
          f"{n_otc} OTC ({ph_otc} pares-hora).")

    if all_combos:
        b_real = [c["brecha"] for c in all_combos if not c["otc"]]
        b_otc = [c["brecha"] for c in all_combos if c["otc"]]
        if b_real:
            print(f"Brecha media train-test REAL: {np.mean(b_real):+.1f} pts (sobre {len(b_real)} combos)")
        if b_otc:
            print(f"Brecha media train-test OTC:  {np.mean(b_otc):+.1f} pts (sobre {len(b_otc)} combos)")
        print("(brecha ~0 = sin sesgo sistematico; el ruido por multiple-testing se ve en la cola de EV alto)")

    guardar()
    log(f"Guardado analisis_par_hora.json ({len(all_combos)} combos, "
        f"{len(horas_por_par)} pares con horas). Revisa antes de aplicar nada al config.")


if __name__ == "__main__":
    main()
