# backtest_sweep_macd_ema.py - Barrido MACD params x EMA periods con analisis par x hora.
#
# Descarga velas 1x por par, luego evalua cientos de configuraciones MACD x EMA
# en CPU puro (sin mas descargas). Reporta top combos por WR/EV y par x hora.
#
#   .venv314\Scripts\python.exe -u backtest_sweep_macd_ema.py
import json, time, threading, math, itertools
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
from iqoptionapi.stable_api import IQ_Option

N_VELAS_1M = 30000
GRAN = 60
TF_SEG = 300
EXPIRY_MIN = 10
H = EXPIRY_MIN * 60 // TF_SEG
SPLIT = 0.70
MIN_TRADES_COMBO = 8

MACD_FASTS   = [5, 8, 10, 12, 15]
MACD_SLOWS   = [20, 26, 30, 35]
MACD_SIGNALS = [7, 9, 12]
EMA_TRENDS   = [0, 50, 100, 150, 200]

TOTAL_COMBOS = len(MACD_FASTS) * len(MACD_SLOWS) * len(MACD_SIGNALS) * len(EMA_TRENDS)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ema_np(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def resample_1m_to_5m(closes_1m, ts_1m):
    n = len(closes_1m)
    blocks = n // 5
    rec = closes_1m[:blocks * 5].reshape(blocks, 5)[:, -1]
    ts = ts_1m[:blocks * 5].reshape(blocks, 5)[:, -1]
    return rec, ts


def evaluar(senales, closes, timestamps, h, payout):
    wins = tr = 0
    i = 0
    while i < len(closes) - h:
        s = senales[i]
        if not s:
            i += 1
            continue
        gano = (closes[i + h] > closes[i]) if s == "call" else (closes[i + h] < closes[i])
        tr += 1
        wins += 1 if gano else 0
        i += h
    return tr, wins


def evaluar_par_hora(senales, closes, timestamps, h, payout):
    stats = defaultdict(lambda: {"tr": 0, "w": 0})
    i = 0
    while i < len(closes) - h:
        s = senales[i]
        if not s:
            i += 1
            continue
        hora = int(datetime.fromtimestamp(timestamps[i], tz=timezone.utc).hour)
        gano = (closes[i + h] > closes[i]) if s == "call" else (closes[i + h] < closes[i])
        stats[hora]["tr"] += 1
        stats[hora]["w"] += 1 if gano else 0
        i += h
    return stats


def ev(wr, payout):
    return wr * payout - (1 - wr)


def pval(w, n, p0=0.535):
    if n == 0:
        return 1.0
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - mu) / sd) / math.sqrt(2))


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
            time.sleep(0.5)
            continue
        vacios = 0
        for v in lote:
            todas[v["from"]] = v
        endtime = min(v["from"] for v in lote) - 1
        if len(lote) < 2:
            break
    return [todas[k] for k in sorted(todas)]


def macd_senales(closes_5m, fast, slow, sig):
    n = len(closes_5m)
    if n < slow + sig + 2:
        return np.array([""] * n, dtype=object)
    ema_f = ema_np(closes_5m, fast)
    ema_s = ema_np(closes_5m, slow)
    macd_l = ema_f - ema_s
    sig_l = ema_np(macd_l, sig)
    out = np.array([""] * n, dtype=object)
    for i in range(1, n):
        if np.isnan(macd_l[i]) or np.isnan(sig_l[i]):
            continue
        if np.isnan(macd_l[i - 1]) or np.isnan(sig_l[i - 1]):
            continue
        if macd_l[i - 1] <= sig_l[i - 1] and macd_l[i] > sig_l[i]:
            out[i] = "call"
        elif macd_l[i - 1] >= sig_l[i - 1] and macd_l[i] < sig_l[i]:
            out[i] = "put"
    return out


def filtrar_trend(senales, closes_5m, ema_period):
    if ema_period <= 0:
        return senales
    ema_t = ema_np(closes_5m, ema_period)
    out = senales.copy()
    for i in range(len(out)):
        if out[i] == "":
            continue
        up = closes_5m[i] > ema_t[i]
        if (out[i] == "call" and not up) or (out[i] == "put" and up):
            out[i] = ""
    return out


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    log("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}")
        return
    api.change_balance("PRACTICE")

    log("Actualizando opcode...")
    done = [False]
    def _upd():
        try:
            api.get_ALL_Binary_ACTIVES_OPCODE()
        except:
            pass
        done[0] = True
    threading.Thread(target=_upd, daemon=True).start()
    t0 = time.time()
    while not done[0] and time.time() - t0 < 45:
        time.sleep(1)

    profits = api.get_all_profit() or {}
    pares_cfg = cfg.get("pares_binarios", [])
    activos = []
    for par in pares_cfg:
        if "-OTC" in par:
            key = par
        else:
            key = f"{par}-op"
        info = profits.get(key, {})
        p = info.get("turbo") or info.get("binary")
        if p and p > 0:
            activos.append((par, float(p)))
    activos.sort(key=lambda x: -x[1])

    log(f"Backtest sweep: {len(activos)} pares | "
        f"{TOTAL_COMBOS} configs MACD x EMA | ~{N_VELAS_1M//1440}d velas")
    log(f"MACD fast={MACD_FASTS} slow={MACD_SLOWS} signal={MACD_SIGNALS}")
    log(f"EMA trend={EMA_TRENDS} | Expiracion {EXPIRY_MIN}m (h={H})")

    datos = {}
    for idx, (par, payout) in enumerate(activos):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{idx+1}/{len(activos)}] Descargando {par} ({payout*100:.0f}%)...", end=" ", flush=True)
        velas = bajar_velas(api, par, N_VELAS_1M)
        if not velas or len(velas) < 500:
            print(f"solo {len(velas) if velas else 0} velas, salto")
            continue
        closes_1m = np.array([float(v["close"]) for v in velas], dtype=float)
        ts_1m = np.array([v["from"] for v in velas], dtype=float)
        closes_5m, ts_5m = resample_1m_to_5m(closes_1m, ts_1m)
        datos[par] = (closes_5m, ts_5m, payout)
        print(f"{len(velas)} velas -> {len(closes_5m)} velas 5m")
        time.sleep(0.2)

    log(f"\nDescargados {len(datos)} pares. Iniciando barrido de {TOTAL_COMBOS} configs...")

    ranking = []
    best_by_config = {}
    t_start = time.time()

    for fi, fast in enumerate(MACD_FASTS):
        for si, slow in enumerate(MACD_SLOWS):
            if fast >= slow:
                continue
            for sgi, sig in enumerate(MACD_SIGNALS):
                if slow + sig + 2 > len(list(datos.values())[0][0]) if datos else True:
                    continue
                for ei, ema_p in enumerate(EMA_TRENDS):
                    agg_w = agg_n = 0
                    par_hora_all = defaultdict(lambda: {"tr": 0, "w": 0})
                    for par, (closes_5m, ts_5m, payout) in datos.items():
                        senales = macd_senales(closes_5m, fast, slow, sig)
                        senales_f = filtrar_trend(senales, closes_5m, ema_p)
                        tr, w = evaluar(senales_f, closes_5m, ts_5m, H, payout)
                        agg_w += w
                        agg_n += tr
                        stats_ph = evaluar_par_hora(senales_f, closes_5m, ts_5m, H, payout)
                        for hora, s in stats_ph.items():
                            par_hora_all[hora]["tr"] += s["tr"]
                            par_hora_all[hora]["w"] += s["w"]

                    if agg_n < 50:
                        continue
                    wr = agg_w / agg_n
                    avg_payout = np.mean([p for _, _, p in datos.values()])
                    ev_val = ev(wr, avg_payout)
                    pv = pval(agg_w, agg_n)

                    config_key = f"MACD({fast},{slow},{sig}) EMA({ema_p})"
                    entry = {
                        "config": config_key,
                        "fast": fast, "slow": slow, "sig": sig, "ema": ema_p,
                        "tr": agg_n, "wins": agg_w,
                        "wr": wr, "ev": ev_val, "p": pv,
                    }

                    top_horas = []
                    for h in range(24):
                        s = par_hora_all[h]
                        if s["tr"] >= MIN_TRADES_COMBO:
                            wr_h = s["w"] / s["tr"]
                            ev_h = ev(wr_h, avg_payout)
                            top_horas.append({"hora": h, "tr": s["tr"], "wr": wr_h, "ev": ev_h})
                    top_horas.sort(key=lambda x: -x["ev"])
                    entry["top_horas"] = top_horas[:5]
                    ranking.append(entry)

                    n_done = len(ranking)
                    if n_done % 50 == 0:
                        elapsed = time.time() - t_start
                        log(f"  {n_done}/{TOTAL_COMBOS} configs evaluadas ({elapsed:.0f}s)")

    elapsed = time.time() - t_start
    log(f"\nBarrido completado: {len(ranking)} configs en {elapsed:.0f}s")

    ranking.sort(key=lambda x: -x["ev"])

    W = 120
    print(f"\n{'='*W}")
    print(f"TOP 50 CONFIGS POR EV (test completo, {len(datos)} pares, {len(ranking)} configs evaluadas)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'CONFIG':30s} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    for k, e in enumerate(ranking[:50], 1):
        marca = " ***" if e["p"] < 0.05 and e["ev"] > 0 else ""
        print(f"{k:>2} {e['config']:30s} {e['tr']:7} {e['wr']*100:6.1f}% "
              f"{e['ev']*100:+7.1f}% {e['p']:6.3f}{marca}")

    print(f"\n{'='*W}")
    print(f"TOP 50 CONFIGS POR WR (n >= 200)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'CONFIG':30s} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    top_wr = sorted([e for e in ranking if e["tr"] >= 200], key=lambda x: -x["wr"])
    for k, e in enumerate(top_wr[:50], 1):
        marca = " ***" if e["p"] < 0.05 and e["ev"] > 0 else ""
        print(f"{k:>2} {e['config']:30s} {e['tr']:7} {e['wr']*100:6.1f}% "
              f"{e['ev']*100:+7.1f}% {e['p']:6.3f}{marca}")

    print(f"\n{'='*W}")
    print("TOP 20 CONSIGNIFICATIVAS (p < 0.10, WR > 53.5%, EV > 0)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'CONFIG':30s} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7} {'TOP HORA'}")
    print("-" * W)
    significativas = [e for e in ranking
                      if e["p"] < 0.10 and e["wr"] > 0.535 and e["ev"] > 0]
    for k, e in enumerate(significativas[:20], 1):
        top_h = e["top_horas"][0] if e["top_horas"] else None
        h_str = f"h{top_h['hora']:02d} WR{top_h['wr']*100:.0f}%" if top_h else ""
        print(f"{k:>2} {e['config']:30s} {e['tr']:7} {e['wr']*100:6.1f}% "
              f"{e['ev']*100:+7.1f}% {e['p']:6.3f} {h_str}")

    if not significativas:
        print("\nNinguna config supera significancia. Relajando (p < 0.20, WR > 52%):")
        relajadas = [e for e in ranking
                     if e["p"] < 0.20 and e["wr"] > 0.52 and e["ev"] > 0]
        for k, e in enumerate(relajadas[:20], 1):
            top_h = e["top_horas"][0] if e["top_horas"] else None
            h_str = f"h{top_h['hora']:02d} WR{top_h['wr']*100:.0f}%" if top_h else ""
            print(f"  {e['config']:30s} {e['tr']:7} {e['wr']*100:6.1f}% "
                  f"{e['ev']*100:+7.1f}% {e['p']:6.3f} {h_str}")

    resultado = {
        "fecha": datetime.now().isoformat(),
        "meta": {
            "n_pares": len(datos),
            "n_configs": len(ranking),
            "n_velas_1m": N_VELAS_1M,
            "expiry_min": EXPIRY_MIN,
            "h": H,
            "macd_fasts": MACD_FASTS,
            "macd_slows": MACD_SLOWS,
            "macd_signals": MACD_SIGNALS,
            "ema_trends": EMA_TRENDS,
        },
        "ranking_wr": [
            {"config": e["config"], "tr": e["tr"], "wr": e["wr"], "ev": e["ev"], "p": e["p"],
             "top_horas": e["top_horas"]}
            for e in top_wr[:100]
        ],
        "ranking_ev": [
            {"config": e["config"], "tr": e["tr"], "wr": e["wr"], "ev": e["ev"], "p": e["p"],
             "top_horas": e["top_horas"]}
            for e in ranking[:100]
        ],
        "significativas": [
            {"config": e["config"], "tr": e["tr"], "wr": e["wr"], "ev": e["ev"], "p": e["p"],
             "top_horas": e["top_horas"]}
            for e in significativas[:50]
        ],
    }
    with open("backtest_sweep_macd_ema.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log("Resultados guardados en backtest_sweep_macd_ema.json")


if __name__ == "__main__":
    main()
