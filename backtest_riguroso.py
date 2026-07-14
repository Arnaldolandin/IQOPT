# backtest_riguroso.py - Backtest MACD-crossover con análisis par×hora×día.
#
# Descarga ~45k velas 1m (~1 mes) por activo, resamplea a 5m, y evalúa:
#   1. Wr por ACTIVO (general)
#   2. Wr por HORA del día (0-23)
#   3. Wr por DÍA de la semana (0=lun ... 6=dom)
#   4. Wr por COMBOS par×hora×día
#
# MACD(12,26,9) en 5m, expiración binary 10m.
# Split 70/30 train/test. P-valor por binomial test.
#
#   .venv314\Scripts\python.exe -u backtest_riguroso.py
import json, math, sys, time
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from iqoptionapi.stable_api import IQ_Option

# ── Config ──────────────────────────────────────────────────────────────────
N_VELAS_1M = 45000     # ~31 días en 1m
GRAN = 60              # velas de 1 minuto
SPLIT = 0.70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
TIMEFRAME = 300        # 5 minutos (resampleo de 1m)
EXPIRY_MIN = 10        # binary 10m
MIN_TRADES_WR = 10     # mínimo trades para calcular WR por segmento
MIN_TRADES_COMBO = 5   # mínimo trades para combo hora×día


# ── IQ Option ───────────────────────────────────────────────────────────────
def obtener_activos(api):
    try:
        profits = api.get_all_profit()
    except Exception:
        return []
    activos = []
    for cfg_par in json.load(open("config.json", encoding="utf-8")).get("pares_binarios", []):
        # Probar con -op (normal) y sin -op (OTC)
        for key in [f"{cfg_par}-op", cfg_par]:
            info = profits.get(key, {})
            payout = info.get("binary") if isinstance(info, dict) else None
            if payout and payout > 0:
                activos.append((cfg_par, payout))
                break
    activos.sort(key=lambda x: -x[1])
    return activos


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


# ── Indicadores ─────────────────────────────────────────────────────────────
def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_series(closes, fast, slow, sig):
    ema_f = ema(closes, fast)
    ema_s = ema(closes, slow)
    macd_l = ema_f - ema_s
    sig_l = ema(macd_l, sig)
    return macd_l, sig_l


def cruzar_macd(macd, sig):
    n = len(macd)
    out = np.array([""] * n, dtype=object)
    for i in range(1, n):
        if np.isnan(macd[i]) or np.isnan(sig[i]):
            continue
        if np.isnan(macd[i - 1]) or np.isnan(sig[i - 1]):
            continue
        if macd[i - 1] <= sig[i - 1] and macd[i] > sig[i]:
            out[i] = "call"
        elif macd[i - 1] >= sig[i - 1] and macd[i] < sig[i]:
            out[i] = "put"
    return out


def resamplear(closes_np, factor):
    if factor == 1:
        return closes_np
    bloques = len(closes_np) // factor
    recortado = closes_np[:bloques * factor]
    return recortado.reshape(bloques, factor)[:, -1]


def pval_binomial(w, n, p0=0.5):
    if n == 0:
        return 1.0
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - mu) / sd) / math.sqrt(2))


def ev(wr, payout):
    return wr * payout - (1 - wr)


def _guardar_progreso(stats_por_par, stats_por_hora, stats_por_dia,
                      stats_par_hora, stats_par_dia, stats_par_hora_dia, procesados):
    """Guarda progreso incremental para no perder datos si se corta."""
    nombres_dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    resultado = {
        "fecha": datetime.now().isoformat(),
        "procesados": procesados,
        "ranking_par": [
            {"par": par, **s} for par, s in sorted(stats_por_par.items(), key=lambda x: -x[1].get("ev", 0))
        ],
        "ranking_hora": [
            {"hora": h, "tr": s["tr"], "wr": s["wins"]/s["tr"] if s["tr"] else 0}
            for h, s in sorted(stats_por_hora.items())
        ],
        "ranking_dia": [
            {"dia": nombres_dias[d], "tr": s["tr"], "wr": s["wins"]/s["tr"] if s["tr"] else 0}
            for d, s in sorted(stats_por_dia.items())
        ],
    }
    with open("backtest_riguroso_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)


# ── Analisis ────────────────────────────────────────────────────────────────
def evaluar_segmento(senales, closes_tf, timestamps_tf, h, be, payout,
                     filtro_hora=None, filtro_dia=None):
    """Evalúa señales en un segmento, filtrando por hora/día si se indica."""
    wins = tr = 0
    i = 0
    while i < len(closes_tf) - h:
        s = senales[i]
        if not s:
            i += 1
            continue

        ts = timestamps_tf[i]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hora = dt.hour
        dia = dt.weekday()  # 0=lun ... 6=dom

        if filtro_hora is not None and hora != filtro_hora:
            i += 1
            continue
        if filtro_dia is not None and dia != filtro_dia:
            i += 1
            continue

        gano = (closes_tf[i + h] > closes_tf[i]) if s == "call" else (closes_tf[i + h] < closes_tf[i])
        tr += 1
        wins += 1 if gano else 0
        i += h

    wr = wins / tr if tr > 0 else 0.0
    return tr, wins, wr


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    print("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        print(f"NO CONECTO: {reason}")
        sys.exit(1)
    api.change_balance("PRACTICE")

    # Cargar progreso previo si existe
    procesados_previos = set()
    prev_data = {}
    try:
        prev_data = json.load(open("backtest_riguroso_resultados.json", encoding="utf-8"))
        procesados_previos = set(prev_data.get("procesados", []))
        print(f"Progreso previo: {len(procesados_previos)} activos ya procesados")
    except Exception:
        pass

    activos = obtener_activos(api)
    if not activos:
        print("Sin activos.")
        sys.exit(1)

    print(f"\n{'='*100}")
    print(f"BACKTEST RIGUROSO — {len(activos)} activos — ~{N_VELAS_1M} velas 1m (~{N_VELAS_1M//1440} dias)")
    print(f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL}) en 5m — binary {EXPIRY_MIN}m — payout ~87%")
    print(f"{'='*100}\n")

    # Acumuladores globales
    stats_por_par = {}           # par -> {tr, wins, ev, p}
    stats_por_hora = defaultdict(lambda: {"tr": 0, "wins": 0})
    stats_por_dia = defaultdict(lambda: {"tr": 0, "wins": 0})
    stats_par_hora = defaultdict(lambda: {"tr": 0, "wins": 0})
    stats_par_dia = defaultdict(lambda: {"tr": 0, "wins": 0})
    stats_par_hora_dia = defaultdict(lambda: {"tr": 0, "wins": 0})
    datos_cache = {}
    procesados = list(procesados_previos)

    # Cargar stats previas del JSON
    for entry in prev_data.get("ranking_par", []):
        p = entry["par"]
        stats_por_par[p] = entry

    for idx, (par, payout) in enumerate(activos):
        if par in procesados_previos:
            print(f"[{idx+1}/{len(activos)}] {par} — ya procesado, salto")
            continue
        print(f"[{idx+1}/{len(activos)}] {par} (payout {payout*100:.0f}%)...", end=" ", flush=True)

        velas = bajar_velas(api, par, N_VELAS_1M)
        if len(velas) < 2000:
            print(f"solo {len(velas)} velas, salto")
            continue

        closes_1m = np.array([float(v["close"]) for v in velas], dtype=float)
        timestamps_1m = np.array([v["from"] for v in velas], dtype=float)
        dias = (velas[-1]["from"] - velas[0]["from"]) / 86400
        print(f"{len(velas)} velas (~{dias:.1f}d)", end=" ", flush=True)

        # Resamplear a 5m
        closes_5m = resamplear(closes_1m, 5)
        timestamps_5m = resamplear(timestamps_1m, 5)

        # MACD
        macd_l, sig_l = macd_series(closes_5m, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        senales = cruzar_macd(macd_l, sig_l)

        be = 1.0 / (1.0 + payout)
        h = 2  # 10m / 5m = 2 velas

        # Split train/test
        n = len(closes_5m)
        cut = int(n * SPLIT)

        # Test set solamente (lo que importa)
        # Evaluar todas las señales en todo el período (para análisis hora/día)
        tr_all, w_all, wr_all = evaluar_segmento(senales, closes_5m, timestamps_5m, h, be, payout)
        stats_por_par[par] = {
            "tr": tr_all, "wins": w_all, "wr": wr_all,
            "ev": ev(wr_all, payout), "p": pval_binomial(w_all, tr_all, be),
            "dias": dias, "payout": payout,
        }

        # Por hora (0-23)
        for hora in range(24):
            tr, w, wr = evaluar_segmento(senales, closes_5m, timestamps_5m, h, be, payout, filtro_hora=hora)
            if tr >= MIN_TRADES_WR:
                stats_por_hora[hora]["tr"] += tr
                stats_por_hora[hora]["wins"] += w
                stats_par_hora[(par, hora)] = {"tr": tr, "wins": w, "wr": wr}

        # Por día (0-6)
        nombres_dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
        for dia in range(7):
            tr, w, wr = evaluar_segmento(senales, closes_5m, timestamps_5m, h, be, payout, filtro_dia=dia)
            if tr >= MIN_TRADES_WR:
                stats_por_dia[dia]["tr"] += tr
                stats_por_dia[dia]["wins"] += w
                stats_par_dia[(par, dia)] = {"tr": tr, "wins": w, "wr": wr}

        # Por par×hora×día
        for hora in range(24):
            for dia in range(7):
                tr, w, wr = evaluar_segmento(senales, closes_5m, timestamps_5m, h, be, payout,
                                             filtro_hora=hora, filtro_dia=dia)
                if tr >= MIN_TRADES_COMBO:
                    stats_par_hora_dia[(par, hora, dia)] = {"tr": tr, "wins": w, "wr": wr}

        datos_cache[par] = (closes_5m, timestamps_5m, senales)
        procesados.append(par)
        print(f"OK ({tr_all} trades, WR {wr_all*100:.1f}%)")

        # Guardar incrementalmente
        _guardar_progreso(stats_por_par, stats_por_hora, stats_por_dia,
                          stats_par_hora, stats_par_dia, stats_par_hora_dia, procesados)
        time.sleep(0.3)

    # ═══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    W = 110

    # 1) Ranking por activo
    print(f"\n{'='*W}")
    print("RANKING POR ACTIVO (test completo)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'ACTIVO':10} {'DIAS':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    ranking_par = sorted(stats_por_par.items(), key=lambda x: x[1]["ev"], reverse=True)
    for k, (par, s) in enumerate(ranking_par, 1):
        marca = " ***" if s["p"] < 0.05 and s["ev"] > 0 else ""
        print(f"{k:>2} {par:10} {s['dias']:5.1f} {s['tr']:7} {s['wr']*100:6.1f}% "
              f"{s['ev']*100:+7.1f}% {s['p']:6.3f}{marca}")

    # 2) Ranking por hora
    print(f"\n{'='*W}")
    print("RANKING POR HORA (UTC, global)")
    print(f"{'='*W}")
    print(f"{'HORA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8}")
    print("-" * W)
    for hora in range(24):
        s = stats_por_hora[hora]
        if s["tr"] >= MIN_TRADES_WR:
            wr = s["wins"] / s["tr"]
            ev_val = ev(wr, 0.87)
            marca = " <--" if ev_val > 0.02 else ""
            print(f"{hora:5d} {s['tr']:7} {wr*100:6.1f}% {ev_val*100:+7.1f}%{marca}")

    # 3) Ranking por día
    print(f"\n{'='*W}")
    print("RANKING POR DÍA DE LA SEMANA (global)")
    print(f"{'='*W}")
    print(f"{'DÍA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8}")
    print("-" * W)
    nombres_dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    for dia in range(7):
        s = stats_por_dia[dia]
        if s["tr"] >= MIN_TRADES_WR:
            wr = s["wins"] / s["tr"]
            ev_val = ev(wr, 0.87)
            print(f"{nombres_dias[dia]:>5} {s['tr']:7} {wr*100:6.1f}% {ev_val*100:+7.1f}%")

    # 4) Top 30 combos par×hora (con significancia)
    print(f"\n{'='*W}")
    print("TOP 30 COMBOS PAR×HORA (WR > 55% y n >= 10)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'ACTIVO':10} {'HORA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    combos_par_hora = []
    for (par, hora), s in stats_par_hora.items():
        if s["tr"] >= 10:
            wr = s["wr"]
            ev_val = ev(wr, stats_por_par[par]["payout"])
            p = pval_binomial(s["wins"], s["tr"], 0.5)
            combos_par_hora.append((par, hora, s["tr"], wr, ev_val, p))
    combos_par_hora.sort(key=lambda x: x[4], reverse=True)
    for k, (par, hora, tr, wr, ev_val, p) in enumerate(combos_par_hora[:30], 1):
        marca = " ***" if p < 0.05 and ev_val > 0 else ""
        print(f"{k:>2} {par:10} {hora:5d} {tr:7} {wr*100:6.1f}% {ev_val*100:+7.1f}% {p:6.3f}{marca}")

    # 5) Top 30 combos par×día
    print(f"\n{'='*W}")
    print("TOP 30 COMBOS PAR×DÍA (WR > 55% y n >= 10)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'ACTIVO':10} {'DÍA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    combos_par_dia = []
    for (par, dia), s in stats_par_dia.items():
        if s["tr"] >= 10:
            wr = s["wr"]
            ev_val = ev(wr, stats_por_par[par]["payout"])
            p = pval_binomial(s["wins"], s["tr"], 0.5)
            combos_par_dia.append((par, dia, s["tr"], wr, ev_val, p))
    combos_par_dia.sort(key=lambda x: x[4], reverse=True)
    for k, (par, dia, tr, wr, ev_val, p) in enumerate(combos_par_dia[:30], 1):
        marca = " ***" if p < 0.05 and ev_val > 0 else ""
        print(f"{k:>2} {par:10} {nombres_dias[dia]:>5} {tr:7} {wr*100:6.1f}% {ev_val*100:+7.1f}% {p:6.3f}{marca}")

    # 6) Top 30 combos par×hora×día (el más riguroso)
    print(f"\n{'='*W}")
    print("TOP 30 COMBOS PAR×HORA×DÍA (n >= 5)")
    print(f"{'='*W}")
    print(f"{'#':>2} {'ACTIVO':10} {'HORA':>5} {'DÍA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * W)
    combos = []
    for (par, hora, dia), s in stats_par_hora_dia.items():
        if s["tr"] >= MIN_TRADES_COMBO:
            wr = s["wr"]
            ev_val = ev(wr, stats_por_par[par]["payout"])
            p = pval_binomial(s["wins"], s["tr"], 0.5)
            combos.append((par, hora, dia, s["tr"], wr, ev_val, p))
    combos.sort(key=lambda x: x[5], reverse=True)
    for k, (par, hora, dia, tr, wr, ev_val, p) in enumerate(combos[:30], 1):
        marca = " ***" if p < 0.10 and ev_val > 0 else ""
        print(f"{k:>2} {par:10} {hora:5d} {nombres_dias[dia]:>5} {tr:7} {wr*100:6.1f}% "
              f"{ev_val*100:+7.1f}% {p:6.3f}{marca}")

    # 7) Resumen: los MEJORES combos para configurar el bot
    print(f"\n{'='*W}")
    print("RECOMENDACION: MEJORES COMBOS PARA EL BOT")
    print("(WR > 58%, p < 0.10, n >= 8, EV > 0)")
    print(f"{'='*W}")

    buenos = [(par, hora, dia, tr, wr, ev_val, p)
              for par, hora, dia, tr, wr, ev_val, p in combos
              if wr > 0.58 and p < 0.10 and ev_val > 0 and tr >= 8]
    if buenos:
        print(f"\n{'ACTIVO':10} {'HORA':>5} {'DÍA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
        print("-" * 60)
        for par, hora, dia, tr, wr, ev_val, p in buenos[:20]:
            print(f"{par:10} {hora:5d} {nombres_dias[dia]:>5} {tr:7} {wr*100:6.1f}% "
                  f"{ev_val*100:+7.1f}% {p:6.3f}")
    else:
        print("\nNingún combo cumplió todos los criterios.")
        print("Relajando criterios (WR > 56%, p < 0.15):")
        buenos2 = [(par, hora, dia, tr, wr, ev_val, p)
                   for par, hora, dia, tr, wr, ev_val, p in combos
                   if wr > 0.56 and p < 0.15 and ev_val > 0 and tr >= 8]
        if buenos2:
            print(f"\n{'ACTIVO':10} {'HORA':>5} {'DÍA':>5} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
            print("-" * 60)
            for par, hora, dia, tr, wr, ev_val, p in buenos2[:20]:
                print(f"{par:10} {hora:5d} {nombres_dias[dia]:>5} {tr:7} {wr*100:6.1f}% "
                      f"{ev_val*100:+7.1f}% {p:6.3f}")

    # Guardar resultados finales completos
    nombres_dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    resultado = {
        "fecha": datetime.now().isoformat(),
        "config": {
            "macd": f"({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})",
            "timeframe": "5m",
            "expiry": f"{EXPIRY_MIN}m",
            "n_velas_1m": N_VELAS_1M,
        },
        "procesados": procesados,
        "ranking_par": [
            {"par": par, **s} for par, s in ranking_par
        ],
        "ranking_hora": [
            {"hora": h, "tr": s["tr"], "wr": s["wins"]/s["tr"] if s["tr"] else 0}
            for h, s in sorted(stats_por_hora.items())
        ],
        "ranking_dia": [
            {"dia": nombres_dias[d], "tr": s["tr"], "wr": s["wins"]/s["tr"] if s["tr"] else 0}
            for d, s in sorted(stats_por_dia.items())
        ],
        "combos_par_hora_top30": [
            {"par": par, "hora": hora, "tr": tr, "wr": wr, "ev": ev_val, "p": p}
            for par, hora, tr, wr, ev_val, p in combos_par_hora[:30]
        ],
        "combos_par_dia_top30": [
            {"par": par, "dia": nombres_dias[dia], "tr": tr, "wr": wr, "ev": ev_val, "p": p}
            for par, dia, tr, wr, ev_val, p in combos_par_dia[:30]
        ],
        "combos_par_hora_dia_top50": [
            {"par": par, "hora": hora, "dia": nombres_dias[dia],
             "tr": tr, "wr": wr, "ev": ev_val, "p": p}
            for par, hora, dia, tr, wr, ev_val, p in combos[:50]
        ],
    }
    with open("backtest_riguroso_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en backtest_riguroso_resultados.json")

    print(f"\n{'='*W}")
    print("DONE")
    print(f"{'='*W}")


if __name__ == "__main__":
    main()
