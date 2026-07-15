# backtest_horas_ema_slope.py
# Backtest RIGUROSO sobre la cache local (cache_ohlc/, velas 5m, ~1 mes).
#
# Objetivos:
#   1) Mejor combinacion EMA x pendiente-MACD (replicando EXACTO la logica de main.py).
#   2) Mejores horas diarias (UTC) por activo.
#
# Metodologia:
#   - Replica main.py: cruce MACD, filtro EMA-tendencia, filtro pendiente del HISTOGRAMA
#     normalizada por precio  slope = (diff - diff_prev)/|precio|, diff = macd - signal.
#   - Train/test 70/30 por tiempo: se ELIGE en train, se REPORTA en test (out-of-sample).
#     Todo lo que solo brilla in-sample se marca como sobreajuste.
#   - Payout fijo 0.87 -> break-even = 1/1.87 = 53.48% WR.
#   - Horizonte de expiracion h=2 velas de 5m = 10 min (coincide con "binary 10m").
#
# NO usa la API: solo lee cache_ohlc/*.json. No interfiere con el bot ni otros backtests.

import json
import math
import os
import glob
import numpy as np
from collections import defaultdict

# ── Parametros ───────────────────────────────────────────────────────────────
CFG = json.load(open("config.json", encoding="utf-8"))
MACD_FAST = CFG["macd"]["fast"]
MACD_SLOW = CFG["macd"]["slow"]
MACD_SIGNAL = CFG["macd"]["signal"]

EMAS = [0, 50, 100, 200]                       # 0 = sin filtro EMA
EMA_BOT = CFG.get("operacion", {}).get("ema_trend", 100)  # EMA lenta que usa el bot
if EMA_BOT not in EMAS:
    EMAS.append(EMA_BOT)
# NUEVAS CONDICIONES: doble EMA apilada (como main.py con ema_trend_fast).
# CALL: precio>EMA_fast>EMA_slow ; PUT: precio<EMA_fast<EMA_slow.
STACK_FAST = CFG.get("operacion", {}).get("ema_trend_fast", 50) or 50
STACK_SLOW = EMA_BOT
SLOPES = [0.0, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

# Timeframe/expiracion tomados del config del bot (se adaptan a 1m/2min o 5m/10m).
TIMEFRAME_SEG = CFG["operacion"].get("timeframe_seg", 300)
EXPIRY_SEG = CFG["operacion"].get("expiry_min", 5) * 60
H = max(1, EXPIRY_SEG // TIMEFRAME_SEG)          # velas adelante = expiry/timeframe
# Cache segun timeframe: 1m -> cache_ohlc_1m/ ; 5m -> cache_ohlc/
CACHE_DIR = "cache_ohlc_1m" if TIMEFRAME_SEG == 60 else "cache_ohlc"

PAYOUT = 0.87
TEST_DAYS = 30                                   # ventana OOS FIJA de 1 mes (garantiza >=1 mes test)
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)                # 0.5348

MIN_TR_SEG = 40        # min trades por segmento para considerar un combo/activo
MIN_TR_HORA = 15       # min trades en una hora (train) para postularla
MIN_TE_HORA = 10       # min trades en una hora (test) para validarla OOS
OUT_JSON = "backtest_horas_ema_slope.json"


# ── Utilidades numericas ─────────────────────────────────────────────────────
def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_series(closes):
    ema_f = ema(closes, MACD_FAST)
    ema_s = ema(closes, MACD_SLOW)
    macd_l = ema_f - ema_s
    sig_l = ema(macd_l, MACD_SIGNAL)
    return macd_l, sig_l


def pval_binomial(w, n, p0):
    if n == 0:
        return 1.0
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    # one-sided: P(X >= w) bajo H0 = p0 (aprox normal con correccion de continuidad)
    return 0.5 * math.erfc(((w - 0.5 - mu) / sd) / math.sqrt(2))


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


# ── Generacion de senales (vectorizada, identica a main.py) ──────────────────
def senales_masks(closes, macd, sig, ema_period, min_slope, warmup):
    """Devuelve (call_mask, put_mask) booleanos por indice, con filtros EMA y slope."""
    n = len(closes)
    diff = macd - sig
    cu = np.zeros(n, bool)
    cd = np.zeros(n, bool)
    # cruce: prev_diff <=0 y diff>0 (call) ; prev_diff>=0 y diff<0 (put)  (como main.py)
    cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
    cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)

    # filtro EMA-tendencia: call solo si precio > EMA ; put solo si precio < EMA
    if ema_period and ema_period > 0:
        ema_arr = ema(closes, ema_period)
        cu &= closes > ema_arr
        cd &= closes < ema_arr

    # filtro pendiente del histograma normalizada por precio (main.py)
    if min_slope and min_slope > 0:
        price_ref = np.abs(closes)
        price_ref[price_ref == 0] = 1.0
        slope = np.zeros(n)
        slope[1:] = (diff[1:] - diff[:-1]) / price_ref[1:]
        cu &= slope >= min_slope
        cd &= slope <= -min_slope

    # warmup: descartar indices iniciales (EMA/MACD sin estabilizar)
    cu[:warmup] = False
    cd[:warmup] = False
    return cu, cd


def senales_masks_stack(closes, macd, sig, e_fast, e_slow, min_slope, warmup):
    """Doble EMA apilada (como main.py con ema_trend_fast):
    CALL: precio>EMA_fast>EMA_slow ; PUT: precio<EMA_fast<EMA_slow."""
    n = len(closes)
    diff = macd - sig
    cu = np.zeros(n, bool)
    cd = np.zeros(n, bool)
    cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
    cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)

    ema_f = ema(closes, e_fast)
    ema_s = ema(closes, e_slow)
    cu &= (closes > ema_f) & (ema_f > ema_s)   # apilado alcista
    cd &= (closes < ema_f) & (ema_f < ema_s)   # apilado bajista

    if min_slope and min_slope > 0:
        price_ref = np.abs(closes)
        price_ref[price_ref == 0] = 1.0
        slope = np.zeros(n)
        slope[1:] = (diff[1:] - diff[:-1]) / price_ref[1:]
        cu &= slope >= min_slope
        cd &= slope <= -min_slope

    cu[:warmup] = False
    cd[:warmup] = False
    return cu, cd


def evaluar(cu, cd, win_call, win_put, lo, hi, hour_arr=None, hour=None):
    rng = np.zeros(len(cu), bool)
    rng[lo:hi] = True
    call_sel = cu & rng
    put_sel = cd & rng
    if hour is not None:
        hm = hour_arr == hour
        call_sel &= hm
        put_sel &= hm
    tr = int(call_sel.sum() + put_sel.sum())
    wins = int((win_call & call_sel).sum() + (win_put & put_sel).sum())
    wr = wins / tr if tr else 0.0
    return tr, wins, wr


# ── Carga de un activo desde cache ───────────────────────────────────────────
def cargar(path):
    d = json.load(open(path, encoding="utf-8"))
    closes = np.asarray(d["close"], dtype=float)
    times = np.asarray(d["times"], dtype=float)
    return closes, times


def nombre_par(path):
    return os.path.basename(path)[:-5].replace("_", "/")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    archivos = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    if not archivos:
        print(f"No hay cache en {CACHE_DIR}/")
        return

    warmup = max(MACD_SLOW + MACD_SIGNAL + 2, max(EMAS) + 5)

    # Acumuladores globales por combo (pooled), separando REAL vs OTC.
    # combo_pool[(ema,slope)]["real"|"otc"]["train"|"test"] = [tr, wins]
    combo_pool = {(e, s): {"real": {"train": [0, 0], "test": [0, 0]},
                           "otc": {"train": [0, 0], "test": [0, 0]}}
                  for e in EMAS for s in SLOPES}
    combo_sig = {(e, s): {"real": 0, "otc": 0} for e in EMAS for s in SLOPES}

    # Acumuladores de la doble EMA apilada (nuevas condiciones), por slope.
    stack_pool = {s: {"real": {"train": [0, 0], "test": [0, 0]},
                      "otc": {"train": [0, 0], "test": [0, 0]}} for s in SLOPES}
    stack_sig = {s: {"real": 0, "otc": 0} for s in SLOPES}

    por_activo = {}          # par -> resultados por combo + mejores horas
    hour_pool_real = {h: {"train": [0, 0], "test": [0, 0]} for h in range(24)}

    procesados = 0
    for path in archivos:
        par = nombre_par(path)
        es_otc = "-OTC" in par
        grupo = "otc" if es_otc else "real"
        try:
            closes, times = cargar(path)
        except Exception as e:
            print(f"[SKIP] {par}: {e}")
            continue
        n = len(closes)
        if n < 2000:
            continue

        split = int(np.searchsorted(times, times[-1] - TEST_DAYS * 86400))
        # outcomes precomputados
        win_call = np.zeros(n, bool)
        win_put = np.zeros(n, bool)
        win_call[:n - H] = closes[H:] > closes[:n - H]
        win_put[:n - H] = closes[H:] < closes[:n - H]
        hour_arr = ((times % 86400) // 3600).astype(int)

        macd_l, sig_l = macd_series(closes)

        # rangos train/test (sin fuga alrededor del split)
        tr_lo, tr_hi = warmup, max(warmup, split - H)
        te_lo, te_hi = split, n - H

        res_combos = {}
        for e in EMAS:
            for s in SLOPES:
                cu, cd = senales_masks(closes, macd_l, sig_l, e, s, warmup)
                trn = evaluar(cu, cd, win_call, win_put, tr_lo, tr_hi)
                tst = evaluar(cu, cd, win_call, win_put, te_lo, te_hi)
                res_combos[(e, s)] = {"train": trn, "test": tst}
                # pool
                cp = combo_pool[(e, s)][grupo]
                cp["train"][0] += trn[0]; cp["train"][1] += trn[1]
                cp["test"][0] += tst[0]; cp["test"][1] += tst[1]
                # significancia OOS por activo
                if tst[0] >= MIN_TR_SEG:
                    p = pval_binomial(tst[1], tst[0], BREAK_EVEN)
                    if p < 0.05 and ev_de_wr(tst[2]) > 0:
                        combo_sig[(e, s)][grupo] += 1

        # NUEVAS CONDICIONES: doble EMA apilada (STACK_FAST/STACK_SLOW) x slope
        res_stack = {}
        for s in SLOPES:
            cu, cd = senales_masks_stack(closes, macd_l, sig_l, STACK_FAST, STACK_SLOW, s, warmup)
            trn = evaluar(cu, cd, win_call, win_put, tr_lo, tr_hi)
            tst = evaluar(cu, cd, win_call, win_put, te_lo, te_hi)
            res_stack[s] = {"train": trn, "test": tst}
            sp = stack_pool[s][grupo]
            sp["train"][0] += trn[0]; sp["train"][1] += trn[1]
            sp["test"][0] += tst[0]; sp["test"][1] += tst[1]
            if tst[0] >= MIN_TR_SEG:
                if pval_binomial(tst[1], tst[0], BREAK_EVEN) < 0.05 and ev_de_wr(tst[2]) > 0:
                    stack_sig[s][grupo] += 1

        por_activo[par] = {"grupo": grupo, "n": n, "combos": res_combos, "stack": res_stack}
        procesados += 1
        if procesados % 25 == 0:
            print(f"  ...{procesados} activos procesados")

    # ── Elegir mejor combo global (por EV pooled en TEST, solo REAL) ─────────
    def pooled_wr(seg):
        tr, w = seg
        return (w / tr) if tr else 0.0

    combos_rank = []
    for e in EMAS:
        for s in SLOPES:
            rp = combo_pool[(e, s)]["real"]
            tr_wr = pooled_wr(rp["train"])
            te_wr = pooled_wr(rp["test"])
            combos_rank.append({
                "ema": e, "slope": s,
                "train_tr": rp["train"][0], "train_wr": tr_wr, "train_ev": ev_de_wr(tr_wr),
                "test_tr": rp["test"][0], "test_wr": te_wr, "test_ev": ev_de_wr(te_wr),
                "sig_real": combo_sig[(e, s)]["real"], "sig_otc": combo_sig[(e, s)]["otc"],
            })
    # Seleccion HONESTA: se elige por EV en TRAIN (con muestra minima) y se
    # reporta en TEST. Ordenar por test seria peeking sobre la validacion.
    validos = [c for c in combos_rank if c["train_tr"] >= 300]
    validos.sort(key=lambda c: c["train_ev"], reverse=True)
    mejor = validos[0] if validos else max(combos_rank, key=lambda c: c["train_ev"])
    best_combo = (mejor["ema"], mejor["slope"])

    # ── Analisis POR PAR: mejor pendiente y horarios optimos (validados OOS) ──
    # Pendiente por par: se evalua con el EMA del bot (EMA_BOT). Horas por par:
    # se evaluan con EMA_BOT y slope=0 (mas senales -> mas potencia por hora).
    horas_por_par = {}          # par -> [horas UTC] que aguantan en train Y test
    por_par = {}
    for path in archivos:
        par = nombre_par(path)
        info = por_activo.get(par)
        if not info:
            continue

        # (a) mejor pendiente por par: elegir por EV en TRAIN (con muestra), reportar TEST
        best_s_rec = None
        best_train_ev = -1e9
        for s in SLOPES:
            trn = info["combos"][(EMA_BOT, s)]["train"]   # (tr, wins, wr)
            tst = info["combos"][(EMA_BOT, s)]["test"]
            if trn[0] < MIN_TR_SEG:
                continue
            tev = ev_de_wr(trn[2])
            if tev > best_train_ev:
                best_train_ev = tev
                best_s_rec = {
                    "slope": s,
                    "train_tr": trn[0], "train_wr": round(trn[2], 4), "train_ev": round(ev_de_wr(trn[2]), 4),
                    "test_tr": tst[0], "test_wr": round(tst[2], 4), "test_ev": round(ev_de_wr(tst[2]), 4),
                }

        # (b) horarios por par: recomputar por-hora en EMA_BOT, slope=0
        try:
            closes, times = cargar(path)
        except Exception:
            continue
        n = len(closes)
        split = int(np.searchsorted(times, times[-1] - TEST_DAYS * 86400))
        win_call = np.zeros(n, bool); win_put = np.zeros(n, bool)
        win_call[:n - H] = closes[H:] > closes[:n - H]
        win_put[:n - H] = closes[H:] < closes[:n - H]
        hour_arr = ((times % 86400) // 3600).astype(int)
        macd_l, sig_l = macd_series(closes)
        cu, cd = senales_masks(closes, macd_l, sig_l, EMA_BOT, 0.0, warmup)
        tr_lo, tr_hi = warmup, max(warmup, split - H)
        te_lo, te_hi = split, n - H

        horas_train = []       # buenas en train
        horas_robustas = []    # buenas en train Y validadas en test (OOS)
        detalle = []
        for h in range(24):
            t_tr, w_tr, wr_tr = evaluar(cu, cd, win_call, win_put, tr_lo, tr_hi, hour_arr, h)
            t_te, w_te, wr_te = evaluar(cu, cd, win_call, win_put, te_lo, te_hi, hour_arr, h)
            detalle.append({"h": h, "train_tr": t_tr, "train_wr": round(wr_tr, 4),
                            "test_tr": t_te, "test_wr": round(wr_te, 4)})
            buena_train = t_tr >= MIN_TR_HORA and wr_tr > BREAK_EVEN
            if buena_train:
                horas_train.append(h)
                if t_te >= MIN_TE_HORA and wr_te > BREAK_EVEN:
                    horas_robustas.append(h)
            if info["grupo"] == "real":
                hp = hour_pool_real[h]
                hp["train"][0] += t_tr; hp["train"][1] += w_tr
                hp["test"][0] += t_te; hp["test"][1] += w_te

        por_par[par] = {
            "grupo": info["grupo"], "n": info["n"],
            "mejor_pendiente": best_s_rec,
            "horas_train": horas_train,
            "horas_robustas_oos": horas_robustas,
            "hora_detalle": detalle,
        }
        if horas_robustas:
            horas_por_par[par] = horas_robustas

    # ── Guardar JSON ─────────────────────────────────────────────────────────
    salida = {
        "config": {"macd": [MACD_FAST, MACD_SLOW, MACD_SIGNAL], "ema_bot": EMA_BOT,
                   "timeframe_seg": TIMEFRAME_SEG, "expiry_seg": EXPIRY_SEG, "h_velas": H,
                   "payout": PAYOUT, "break_even": round(BREAK_EVEN, 4),
                   "test_days": TEST_DAYS, "cache_dir": CACHE_DIR, "emas": EMAS, "slopes": SLOPES,
                   "activos_procesados": procesados},
        "mejor_combo_global_real": mejor,
        "sweep_ema_slope_real": sorted(combos_rank, key=lambda c: c["train_ev"], reverse=True),
        "horas_pooled_real": {
            str(h): {
                "train_tr": hour_pool_real[h]["train"][0],
                "train_wr": round(pooled_wr(hour_pool_real[h]["train"]), 4),
                "test_tr": hour_pool_real[h]["test"][0],
                "test_wr": round(pooled_wr(hour_pool_real[h]["test"]), 4),
            } for h in range(24)
        },
        "por_par": por_par,
        "horas_por_par_sugeridas": horas_por_par,
        "stack_ema_apilada_real": {
            "fast": STACK_FAST, "slow": STACK_SLOW,
            "por_slope": {
                str(s): {
                    "train_tr": stack_pool[s]["real"]["train"][0],
                    "train_wr": round(pooled_wr(stack_pool[s]["real"]["train"]), 4),
                    "test_tr": stack_pool[s]["real"]["test"][0],
                    "test_wr": round(pooled_wr(stack_pool[s]["real"]["test"]), 4),
                    "test_ev": round(ev_de_wr(pooled_wr(stack_pool[s]["real"]["test"])), 4),
                    "sig_real": stack_sig[s]["real"],
                } for s in SLOPES
            },
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # ── Reporte en consola ───────────────────────────────────────────────────
    W = 100
    print("\n" + "=" * W)
    print(f"BACKTEST RIGUROSO  |  MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})  EMA {EMA_BOT}  |  "
          f"velas {TIMEFRAME_SEG}s  exp {EXPIRY_SEG}s (H={H})  |  payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})  |  "
          f"{procesados} activos  |  test OOS = ultimos {TEST_DAYS}d")
    print("=" * W)

    # ── [0] COMPARATIVA DE FILTROS (incluye la doble EMA apilada) ────────────
    def resume_filtro(nombre, get_pool):
        base = get_pool(0.0)
        best = None
        for s in SLOPES:
            g = get_pool(s)
            if g["train"][0] < 200:
                continue
            tev = ev_de_wr(pooled_wr(g["train"]))
            if best is None or tev > best[1]:
                best = (s, tev, g)
        b_s, _, b_g = best if best else (0.0, 0.0, base)
        return (nombre, base, b_s, b_g)

    filtros = [
        resume_filtro("sin filtro", lambda s: combo_pool[(0, s)]["real"]),
        resume_filtro(f"EMA{EMA_BOT} unica", lambda s: combo_pool[(EMA_BOT, s)]["real"]),
    ]
    if 50 in EMAS:
        filtros.append(resume_filtro("EMA50 unica", lambda s: combo_pool[(50, s)]["real"]))
    filtros.append(resume_filtro(f"STACK {STACK_FAST}>{STACK_SLOW}", lambda s: stack_pool[s]["real"]))

    print("\n[0] COMPARATIVA DE FILTROS (REAL agregado) — base = slope 0 ; mejor = mejor slope (train)")
    print(f"{'FILTRO':>16} | {'baseTESTtr':>10} {'wr':>6} {'ev':>7} | {'mejorSlope':>10} {'TESTtr':>7} {'wr':>6} {'ev':>7}")
    print("-" * W)
    for nombre, base, b_s, b_g in filtros:
        bw = pooled_wr(base["test"]); bev = ev_de_wr(bw)
        mw = pooled_wr(b_g["test"]); mev = ev_de_wr(mw)
        print(f"{nombre:>16} | {base['test'][0]:>10} {bw*100:>5.1f}% {bev*100:>+6.1f}% | "
              f"{b_s:>10.5f} {b_g['test'][0]:>7} {mw*100:>5.1f}% {mev*100:>+6.1f}%")
    print(f"  (STACK = precio>EMA{STACK_FAST}>EMA{STACK_SLOW} para CALL, al reves para PUT — las 'nuevas condiciones')")

    print("\n[1] MEJOR PENDIENTE GLOBAL (EMA x slope, REAL agregado; se ELIGE en train, se REPORTA test)")
    print(f"{'EMA':>4} {'SLOPE':>9} | {'TRAINtr':>7} {'wr':>6} {'ev':>7} | {'TESTtr':>7} {'wr':>6} {'ev':>7} | {'sig.OOS':>7}")
    print("-" * W)
    for c in sorted(combos_rank, key=lambda c: c["train_ev"], reverse=True)[:12]:
        print(f"{c['ema']:>4} {c['slope']:>9.5f} | {c['train_tr']:>7} {c['train_wr']*100:>5.1f}% {c['train_ev']*100:>+6.1f}% | "
              f"{c['test_tr']:>7} {c['test_wr']*100:>5.1f}% {c['test_ev']*100:>+6.1f}% | {c['sig_real']:>7}")
    print(f"  -> Mejor combo (train): EMA={mejor['ema']} slope={mejor['slope']}  ->  "
          f"OOS test WR {mejor['test_wr']*100:.1f}% EV {mejor['test_ev']*100:+.1f}%")

    print(f"\n[2] HORAS DEL DIA (UTC) — REAL agregado (EMA={EMA_BOT}, slope=0)")
    tz = CFG.get("filtro_hora", {}).get("timezone_offset", -4)
    print(f"{'UTC':>3} {'CL':>4} | {'TRAINtr':>8} {'wr':>7} | {'TESTtr':>8} {'wr':>7}  ")
    print("-" * W)
    for h in range(24):
        tr_wr = pooled_wr(hour_pool_real[h]["train"])
        te_wr = pooled_wr(hour_pool_real[h]["test"])
        marca = ""
        if hour_pool_real[h]["test"][0] >= 30:
            marca = "<< OOS+" if te_wr > BREAK_EVEN else ("  neg" if te_wr < 0.5 else "")
        print(f"{h:>3} {(h+tz)%24:>3}h | {hour_pool_real[h]['train'][0]:>8} {tr_wr*100:>6.1f}% | "
              f"{hour_pool_real[h]['test'][0]:>8} {te_wr*100:>6.1f}%  {marca}")
    horas_oos_pos = [h for h in range(24)
                     if hour_pool_real[h]["test"][0] >= 30 and pooled_wr(hour_pool_real[h]["test"]) > BREAK_EVEN]
    print(f"  -> Horas UTC con WR>BE OUT-OF-SAMPLE (pooled real, n>=30): {horas_oos_pos if horas_oos_pos else 'NINGUNA'}")

    # ── [3] MEJOR PENDIENTE POR PAR (top por EV OOS, solo REAL con muestra) ───
    reales = [(p, d) for p, d in por_par.items()
              if d["grupo"] == "real" and d["mejor_pendiente"] and d["mejor_pendiente"]["test_tr"] >= 20]
    reales.sort(key=lambda x: x[1]["mejor_pendiente"]["test_ev"], reverse=True)
    print(f"\n[3] MEJOR PENDIENTE POR PAR (REAL, top 15 por EV OUT-OF-SAMPLE)")
    print(f"{'PAR':>10} {'slope*':>9} | {'TRAINtr':>7} {'wr':>6} | {'TESTtr':>7} {'wr':>6} {'ev':>7}")
    print("-" * W)
    for p, d in reales[:15]:
        m = d["mejor_pendiente"]
        print(f"{p:>10} {m['slope']:>9.5f} | {m['train_tr']:>7} {m['train_wr']*100:>5.1f}% | "
              f"{m['test_tr']:>7} {m['test_wr']*100:>5.1f}% {m['test_ev']*100:>+6.1f}%")
    n_pos = sum(1 for _, d in reales if d["mejor_pendiente"]["test_ev"] > 0)
    print(f"  -> Pares REAL cuya mejor pendiente (elegida en train) es EV>0 en test: {n_pos}/{len(reales)}")

    # ── [4] HORARIOS OPTIMOS POR PAR validados OOS ───────────────────────────
    con_horas = {p: d["horas_robustas_oos"] for p, d in por_par.items() if d["horas_robustas_oos"]}
    reales_horas = {p: v for p, v in con_horas.items() if por_par[p]["grupo"] == "real"}
    print(f"\n[4] HORARIOS OPTIMOS POR PAR (buenos en TRAIN *y* validados en TEST)")
    print(f"  Pares con >=1 hora robusta OOS: {len(con_horas)} de {procesados}  (REAL: {len(reales_horas)})")
    ejemplos = sorted(reales_horas.items(), key=lambda x: -len(x[1]))[:12]
    for p, hrs in ejemplos:
        hrs_cl = [f"{h}({(h+tz)%24}CL)" for h in hrs]
        print(f"     {p:>10}: UTC {', '.join(str(h) for h in hrs)}   [{', '.join(hrs_cl)}]")
    print(f"  -> 'horas_por_par_sugeridas' en el JSON ({len(horas_por_par)} pares) usa SOLO horas validadas OOS.")

    print(f"\nJSON completo: {OUT_JSON}")
    print("=" * W)

    # ── Veredicto de sobreajuste ─────────────────────────────────────────────
    deg = (mejor["train_wr"] - mejor["test_wr"]) * 100
    print(f"\n[VEREDICTO]")
    print(f"  Mejor combo global: train {mejor['train_wr']*100:.1f}% -> test {mejor['test_wr']*100:.1f}% "
          f"(caida {deg:+.1f}pt) | EV OOS {mejor['test_ev']*100:+.1f}% | BE {BREAK_EVEN*100:.1f}%")
    combos_ev_pos = sum(1 for c in combos_rank if c["test_ev"] > 0 and c["test_tr"] >= 200)
    print(f"  Combos EMAxslope con EV_test>0 (muestra>=200): {combos_ev_pos}/{len(combos_rank)}")
    print(f"  Pares REAL con pendiente rentable OOS: {n_pos}/{len(reales)} | con hora robusta OOS: {len(reales_horas)}")
    if mejor["test_ev"] <= 0 and n_pos == 0:
        print("  >> Sin edge fuera de muestra ni global ni por par. Coherente con el historial del proyecto.")
    elif len(reales_horas) > 0 or n_pos > 0:
        print("  >> Hay pares que aguantan OOS, pero OJO: con 231 pares x 24h el azar produce falsos positivos.")
        print("     Trata cualquier seleccion por-par como hipotesis a re-validar con MAS datos, no como certeza.")


if __name__ == "__main__":
    main()
