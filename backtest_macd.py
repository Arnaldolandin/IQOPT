# backtest_macd.py - Backtest riguroso de MACD-crossover sobre TODOS los activos binarios de IQ.
#
# Estrategia: MACD line cruza signal line -> CALL (arriba) / PUT (abajo).
# 3 parametrizaciones MACD x 4 timeframes x 3 expiraciones = 36 combos POR ACTIVO.
# Des-solapado. Split 70/30 train/test. Walk-forward sobre top 5.
#
#   .venv314\Scripts\python.exe backtest_macd.py
import json, math, sys, time

import numpy as np
from iqoptionapi.stable_api import IQ_Option

# ── Config ──────────────────────────────────────────────────────────────────
N_VELAS = 16000
GRAN = 60
SPLIT = 0.70
MIN_TR = 30
MIN_TE = 15
MIN_VELAS = 500
WF_TRAIN = 1500
WF_TEST = 500
WF_MIN_TRADES = 20

MACD_PARAMS = [(8, 17, 9), (12, 26, 9), (19, 39, 9)]
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}
EXPIRIES = {
    "1m":  [5, 10, 15],
    "5m":  [10, 15, 30],
    "15m": [15, 30, 60],
    "30m": [30, 60, 90],
}


# ── IQ Option helpers ───────────────────────────────────────────────────────
def obtener_activos_binarios(api):
    """Devuelve lista (subyacente, payout_binary) para todos los activos binarios."""
    try:
        profits = api.get_all_profit()
    except Exception:
        profits = {}
    if not profits:
        return []

    activos = []
    seen = set()
    for key, info in profits.items():
        if not key.endswith("-op"):
            continue
        subyacente = key[:-3]  # "USDJPY-op" -> "USDJPY"
        if subyacente in seen:
            continue
        seen.add(subyacente)
        payout = None
        if isinstance(info, dict):
            payout = info.get("binary")
        if payout is None or payout <= 0:
            continue
        activos.append((subyacente, payout))

    activos.sort(key=lambda x: -x[1])
    return activos


def bajar_velas_1m(api, par, total):
    """Descarga velas 1m hacia atras en lotes de ~1000, con timeout por lote."""
    todas = {}
    endtime = time.time()
    intentos_vacios = 0
    while len(todas) < total:
        try:
            lote = api.get_candles(par, GRAN, 1000, endtime)
        except Exception:
            break
        if not lote:
            intentos_vacios += 1
            if intentos_vacios >= 2:
                break
            time.sleep(1)
            continue
        intentos_vacios = 0
        for v in lote:
            todas[v["from"]] = v
        endtime = min(v["from"] for v in lote) - 1
        if len(lote) < 2:
            break
    velas = [todas[k] for k in sorted(todas)]
    return velas


# ── Indicadores ─────────────────────────────────────────────────────────────
def rsi_series_inline(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = gain[:period].mean()
    al = loss[:period].mean()
    for i in range(period, n):
        if i > period:
            ag = (ag * (period - 1) + gain[i - 1]) / period
            al = (al * (period - 1) + loss[i - 1]) / period
        rs = ag / al if al > 0 else 999
        out[i] = 100 - 100 / (1 + rs)
    return out


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_series(closes, fast, slow, signal):
    ema_f = ema(closes, fast)
    ema_s = ema(closes, slow)
    macd_line = ema_f - ema_s
    sig_line = ema(macd_line, signal)
    return macd_line, sig_line


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


def resamplear_closes(closes_np, factor):
    if factor == 1:
        return closes_np
    bloques = len(closes_np) // factor
    recortado = closes_np[:bloques * factor]
    return recortado.reshape(bloques, factor)[:, -1]


# ── Evaluacion ──────────────────────────────────────────────────────────────
def evaluar(sig, closes, h, start, end):
    wins = tr = 0
    i = start
    while i < end - h:
        s = sig[i]
        if not s:
            i += 1
            continue
        gano = (closes[i + h] > closes[i]) if s == "call" else (closes[i + h] < closes[i])
        tr += 1
        wins += 1 if gano else 0
        i += h
    return tr, wins


def pval(w, n, p0):
    if n == 0:
        return 1.0
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - mu) / sd) / math.sqrt(2))


def ev_calc(wr, payout):
    return wr * payout - (1 - wr)


# ── Backtest ────────────────────────────────────────────────────────────────
def run_backtest(closes_all, par, payout):
    be = 1.0 / (1.0 + payout)
    n = len(closes_all)
    filas = []
    closes_np = np.asarray(closes_all, dtype=float)

    for tf_name, tf_factor in TIMEFRAMES.items():
        closes_tf = resamplear_closes(closes_np, tf_factor)
        n_tf = len(closes_tf)
        cut_tf = int(n_tf * SPLIT)

        for fast, slow, sig_p in MACD_PARAMS:
            if slow >= n_tf - 1:
                continue
            macd_l, sig_l = macd_series(closes_tf, fast, slow, sig_p)
            senales = cruzar_macd(macd_l, sig_l)

            for h in EXPIRIES[tf_name]:
                ttr, tw = evaluar(senales, closes_tf, h, 0, cut_tf)
                ter, te = evaluar(senales, closes_tf, h, cut_tf, n_tf)
                if ttr < MIN_TR or ter < MIN_TE:
                    continue

                tr_wr = tw / ttr
                te_wr = te / ter
                filas.append({
                    "par": par, "payout": payout, "be": be,
                    "tf": tf_name,
                    "macd": f"({fast},{slow},{sig_p})",
                    "h": h,
                    "tr_n": ttr, "tr_wr": tr_wr, "tr_ev": ev_calc(tr_wr, payout),
                    "te_n": ter, "te_wr": te_wr, "te_ev": ev_calc(te_wr, payout),
                    "te_p": pval(te, ter, be),
                })
    filas.sort(key=lambda x: x["tr_ev"], reverse=True)
    return filas


# ── Walk-forward ────────────────────────────────────────────────────────────
def run_walkforward(closes_all, payout, tf_name, tf_factor, fast, slow, sig_p, h):
    be = 1.0 / (1.0 + payout)
    closes_np = np.asarray(closes_all, dtype=float)
    closes_tf = resamplear_closes(closes_np, tf_factor)
    n = len(closes_tf)

    macd_l, sig_l = macd_series(closes_tf, fast, slow, sig_p)
    senales = cruzar_macd(macd_l, sig_l)

    oos_tr = oos_w = 0
    is_wr_sum = 0.0
    folds = 0
    rsi_tf = rsi_series_inline(closes_tf, 14)
    fix_h = min(10, h)
    fix_tr = fix_w = 0

    i0 = 50
    while i0 + WF_TRAIN + WF_TEST <= n:
        tr_s, tr_e = i0, i0 + WF_TRAIN
        te_s, te_e = tr_e, tr_e + WF_TEST

        t, w = evaluar(senales, closes_tf, h, te_s, te_e)
        oos_tr += t
        oos_w += w

        tis, wis = evaluar(senales, closes_tf, h, tr_s, tr_e)
        if tis >= WF_MIN_TRADES:
            is_wr_sum += wis / tis
            folds += 1

        ft, fw = 0, 0
        j = te_s
        while j < te_e - fix_h:
            rv = rsi_tf[j]
            if np.isnan(rv):
                j += 1
                continue
            side = "call" if rv < 30 else ("put" if rv > 70 else None)
            if side is None:
                j += 1
                continue
            gano = (closes_tf[j + fix_h] > closes_tf[j]) if side == "call" else (closes_tf[j + fix_h] < closes_tf[j])
            ft += 1
            fw += 1 if gano else 0
            j += fix_h
        fix_tr += ft
        fix_w += fw
        i0 += WF_TEST

    is_avg = (is_wr_sum / folds * 100) if folds else float("nan")
    oos_wr = (oos_w / oos_tr * 100) if oos_tr else float("nan")
    fix_wr = (fix_w / fix_tr * 100) if fix_tr else float("nan")

    return {
        "folds": folds, "is_avg": is_avg,
        "oos_n": oos_tr, "oos_wr": oos_wr,
        "oos_p": pval(oos_w, oos_tr, be),
        "oos_ev": ev_calc(oos_wr / 100, payout) * 100,
        "fix_n": fix_tr, "fix_wr": fix_wr,
        "fix_p": pval(fix_w, fix_tr, be),
        "fix_ev": ev_calc(fix_wr / 100, payout) * 100,
        "brecha": is_avg - oos_wr if not np.isnan(is_avg) else float("nan"),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    print("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        print(f"NO CONECTO: {reason}")
        sys.exit(1)
    api.change_balance("PRACTICE")

    activos = obtener_activos_binarios(api)
    if not activos:
        print("No se encontraron activos binarios.")
        sys.exit(1)

    print(f"\nActivos binarios ({len(activos)}):")
    for nombre, p in activos:
        print(f"  {nombre:12} payout {p*100:5.1f}%  BE {100/(1+p):.1f}%")
    print()

    todas_filas = []
    datos_cache = {}  # par -> closes_1m

    for idx, (par, payout) in enumerate(activos):
        print(f"[{idx+1}/{len(activos)}] {par} (payout {payout*100:.0f}%)...")
        try:
            velas = bajar_velas_1m(api, par, N_VELAS)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if len(velas) < MIN_VELAS:
            print(f"  solo {len(velas)} velas, salto")
            continue
        closes_1m = np.array([float(v["close"]) for v in velas])
        dias = (velas[-1]["from"] - velas[0]["from"]) / 86400
        print(f"  {len(closes_1m)} velas (~{dias:.1f}d)")

        filas = run_backtest(closes_1m, par, payout)
        todas_filas.extend(filas)
        datos_cache[par] = closes_1m
        print(f"  {len(filas)} combos evaluados")
        time.sleep(0.5)

    if not todas_filas:
        print("\nSin resultados.")
        return

    todas_filas.sort(key=lambda x: x["tr_ev"], reverse=True)

    # ── Output ──────────────────────────────────────────────────────────
    W = 120
    print("\n" + "=" * W)
    print("TODOS LOS COMBOS — TOP 30 (ranked by train EV)")
    print("=" * W)
    print(f"{'#':>2} {'ACTIVO':10} {'TF':>4} {'MACD':>14} {'exp':>5} | "
          f"{'TRAIN':>20} | {'TEST':>32}")
    print(f"{'':>2} {'':10} {'':>4} {'':>14} {'':>5} | "
          f"{'n':>5} {'WR':>7} {'EV':>8} | {'n':>5} {'WR':>7} {'EV':>8} {'p':>7}")
    print("-" * W)

    for k, f in enumerate(todas_filas[:30], 1):
        marca = " <==" if f["te_ev"] > 0 and f["te_p"] < 0.05 else ""
        print(f"{k:>2} {f['par']:10} {f['tf']:>4} {f['macd']:>14} {f['h']:>3}m | "
              f"{f['tr_n']:>5} {f['tr_wr']*100:5.1f}% {f['tr_ev']*100:+6.1f}% | "
              f"{f['te_n']:>5} {f['te_wr']*100:5.1f}% {f['te_ev']*100:+6.1f}% "
              f"{f['te_p']:6.3f}{marca}")

    # ── Top 5 OOS ──────────────────────────────────────────────────────
    validos = [f for f in todas_filas if f["te_n"] >= 30]
    validos.sort(key=lambda x: x["te_ev"], reverse=True)

    print("\n" + "=" * W)
    print("TOP 5 POR EV OUT-OF-SAMPLE (n_test >= 30)")
    print("=" * W)
    for k, f in enumerate(validos[:5], 1):
        print(f"  #{k} {f['par']:8} {f['tf']} MACD{f['macd']} exp {f['h']}m -> "
              f"TEST: n={f['te_n']} WR={f['te_wr']*100:.1f}% EV={f['te_ev']*100:+.1f}% "
              f"p={f['te_p']:.3f} | TRAIN WR={f['tr_wr']*100:.1f}% | payout {f['payout']*100:.0f}%")

    positivos = [f for f in validos if f["te_ev"] > 0 and f["te_p"] < 0.05]
    print(f"\nCombos EV_test > 0 y p < 0.05: {len(positivos)}")
    for f in positivos:
        print(f"  {f['par']:8} {f['tf']} MACD{f['macd']} exp {f['h']}m: "
              f"WR={f['te_wr']*100:.1f}% n={f['te_n']} p={f['te_p']:.3f}")

    # ── Walk-forward top 5 ─────────────────────────────────────────────
    if not validos:
        print("\nSin combos validos para walk-forward.")
        return

    print("\n" + "=" * W)
    print("WALK-FORWARD — TOP 5")
    print("=" * W)

    for k, f in enumerate(validos[:5], 1):
        closes_activo = datos_cache.get(f["par"])
        if closes_activo is None:
            print(f"\n  #{k} {f['par']} — sin datos")
            continue
        fast, slow, sig_p = [int(x) for x in f["macd"][1:-1].split(",")]
        tf_factor = TIMEFRAMES[f["tf"]]
        wf = run_walkforward(closes_activo, f["payout"], f["tf"], tf_factor, fast, slow, sig_p, f["h"])
        be = f["be"] * 100
        print(f"\n  #{k} {f['par']} {f['tf']} MACD{f['macd']} exp {f['h']}m (payout {f['payout']*100:.0f}%)")
        print(f"     Folds: {wf['folds']}")
        if wf["folds"] > 0:
            print(f"     IN-SAMPLE:  WR {wf['is_avg']:.1f}%")
        print(f"     OUT-SAMPLE: n={wf['oos_n']:5} WR {wf['oos_wr']:.1f}%  "
              f"vs BE {wf['oos_wr']-be:+.1f}pt  p={wf['oos_p']:.3f}  EV {wf['oos_ev']:+.1f}%")
        if wf["fix_n"] > 0:
            print(f"     BASELINE:   n={wf['fix_n']:5} WR {wf['fix_wr']:.1f}%  "
                  f"vs BE {wf['fix_wr']-be:+.1f}pt  p={wf['fix_p']:.3f}  EV {wf['fix_ev']:+.1f}%")
        print(f"     Brecha sobreajuste: {wf['brecha']:+.1f} puntos")

    print("\n" + "=" * W)
    print("DONE")


if __name__ == "__main__":
    main()
