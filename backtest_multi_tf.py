# backtest_multi_tf.py - Barrido multi-timeframe de la estrategia MACD+EMA+slope.
# Base: cache_ohlc/ (velas 5m). Se resamplea a 5/10/15/30m.
# Para cada TF barre: MACD x modo-EMA (ninguna/unica/apilada) x pendiente.
# Train/test OOS. Reporta mejor combo por TF, mejor global y par-hora del mejor.
#
# NOTA: la base 5m cubre ~1 mes, asi que el TEST OOS aqui es ~9 dias. Sirve para
# RANQUEAR combos/TF; el 1m (descarga aparte, ~2.5 meses) dara OOS de 1 mes.

import json, math, os, glob
import numpy as np

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
TRAIN_FRAC = 0.70
H = 2                                  # se mantiene 2 barras (expiry = 2xTF)
CACHE_DIR = "cache_ohlc"
WARMUP = 210
MIN_TR = 60                            # min trades por segmento para tener en cuenta un combo

# TF en factores sobre 5m
TFS = [(1, "5m"), (2, "10m"), (3, "15m"), (6, "30m")]
MACDS = [(6, 13, 5), (12, 26, 9), (8, 17, 9), (5, 10, 3)]
# modo EMA: ("none",) / ("single", p) / ("stack", fast, slow)
EMODES = [("none",), ("single", 100), ("single", 50), ("stack", 50, 100)]
SLOPES = [0.0, 0.0001, 0.0005, 0.001]
EMA_PERIODS = sorted({p for m in EMODES for p in m[1:]})


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_lines(c, fast, slow, sig):
    ml = ema(c, fast) - ema(c, slow)
    return ml, ema(ml, sig)


def resample_last(a, f):
    if f == 1:
        return np.asarray(a, float)
    n = len(a) // f
    return np.asarray(a[:n * f], float).reshape(n, f)[:, -1]


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def pval(w, n, p0=BREAK_EVEN):
    if n == 0:
        return 1.0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - n * p0) / sd) / math.sqrt(2))


def signals(c, ml, sl, emode, slope, emacache):
    n = len(c)
    diff = ml - sl
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
    cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
    kind = emode[0]
    if kind == "single":
        e = emacache[emode[1]]
        cu &= c > e; cd &= c < e
    elif kind == "stack":
        ef = emacache[emode[1]]; es = emacache[emode[2]]
        cu &= (c > ef) & (ef > es); cd &= (c < ef) & (ef < es)
    if slope > 0:
        pr = np.abs(c); pr[pr == 0] = 1.0
        sp = np.zeros(n); sp[1:] = (diff[1:] - diff[:-1]) / pr[1:]
        cu &= sp >= slope; cd &= sp <= -slope
    cu[:WARMUP] = False; cd[:WARMUP] = False
    return cu, cd


def evaluar(cu, cd, wc, wp, lo, hi, hour_arr=None, hour=None):
    rng = np.zeros(len(cu), bool); rng[lo:hi] = True
    cs = cu & rng; ps = cd & rng
    if hour is not None:
        hm = hour_arr == hour; cs &= hm; ps &= hm
    tr = int(cs.sum() + ps.sum())
    wins = int((wc & cs).sum() + (wp & ps).sum())
    return tr, wins, (wins / tr if tr else 0.0)


def emode_str(e):
    if e[0] == "none":
        return "sin-EMA"
    if e[0] == "single":
        return f"EMA{e[1]}"
    return f"EMA{e[1]}>{e[2]}"


def main():
    archivos = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    # pool[(tf,macd,emode,slope)][seg] = [tr, wins]  (solo REAL)
    pool = {}
    sig_count = {}
    # guardar por-activo del combo para el analisis par-hora posterior
    procesados = 0
    real_paths = []

    for path in archivos:
        par = os.path.basename(path)[:-5].replace("_", "/")
        if "-OTC" in par:
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
            closes5 = np.asarray(d["close"], float)
            times5 = np.asarray(d["times"], float)
        except Exception:
            continue
        if len(closes5) < 2000:
            continue
        real_paths.append(path)
        procesados += 1

        for factor, tfname in TFS:
            c = resample_last(closes5, factor)
            t = resample_last(times5, factor)
            n = len(c)
            if n < WARMUP + 200:
                continue
            split = int(n * TRAIN_FRAC)
            wc = np.zeros(n, bool); wp = np.zeros(n, bool)
            wc[:n - H] = c[H:] > c[:n - H]
            wp[:n - H] = c[H:] < c[:n - H]
            hour_arr = ((t % 86400) // 3600).astype(int)
            emacache = {p: ema(c, p) for p in EMA_PERIODS}
            tr_lo, tr_hi = WARMUP, max(WARMUP, split - H)
            te_lo, te_hi = split, n - H

            for mcfg in MACDS:
                ml, sl = macd_lines(c, *mcfg)
                for emode in EMODES:
                    for slope in SLOPES:
                        cu, cd = signals(c, ml, sl, emode, slope, emacache)
                        trn = evaluar(cu, cd, wc, wp, tr_lo, tr_hi)
                        tst = evaluar(cu, cd, wc, wp, te_lo, te_hi)
                        key = (tfname, mcfg, emode, slope)
                        p = pool.setdefault(key, {"tr": [0, 0], "te": [0, 0]})
                        p["tr"][0] += trn[0]; p["tr"][1] += trn[1]
                        p["te"][0] += tst[0]; p["te"][1] += tst[1]
                        if tst[0] >= MIN_TR and pval(tst[1], tst[0]) < 0.05 and ev_de_wr(tst[2]) > 0:
                            sig_count[key] = sig_count.get(key, 0) + 1

    def wr(seg):
        return seg[1] / seg[0] if seg[0] else 0.0

    rows = []
    for key, p in pool.items():
        tf, mcfg, emode, slope = key
        rows.append({
            "tf": tf, "macd": mcfg, "emode": emode_str(emode), "slope": slope,
            "train_tr": p["tr"][0], "train_wr": wr(p["tr"]), "train_ev": ev_de_wr(wr(p["tr"])),
            "test_tr": p["te"][0], "test_wr": wr(p["te"]), "test_ev": ev_de_wr(wr(p["te"])),
            "sig": sig_count.get(key, 0), "_key": key,
        })

    # Seleccion honesta: elegir por EV en TRAIN (muestra minima), reportar TEST
    cand = [r for r in rows if r["train_tr"] >= 400]
    cand.sort(key=lambda r: r["train_ev"], reverse=True)
    mejor = cand[0] if cand else max(rows, key=lambda r: r["train_ev"])

    W = 104
    print("=" * W)
    print(f"BARRIDO MULTI-TF  |  base 5m -> {[t[1] for t in TFS]}  |  {procesados} activos REALES  |  "
          f"payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})  |  H={H} barras  |  train/test {int(TRAIN_FRAC*100)}/{int((1-TRAIN_FRAC)*100)}")
    print("=" * W)

    # Efecto PURO del TF: parametros fijos (MACD 6/13/5, sin EMA, sin slope)
    print("\n[0] EFECTO DEL TF con parametros FIJOS (MACD(6,13,5), sin-EMA, slope 0) — sin minar params")
    print(f"{'TF':>5} | {'TRAINtr':>8} {'wr':>6} {'ev':>7} | {'TESTtr':>8} {'wr':>6} {'ev':>7}")
    print("-" * W)
    for _, tf in TFS:
        r = next((x for x in rows if x["tf"] == tf and x["macd"] == (6, 13, 5)
                  and x["emode"] == "sin-EMA" and x["slope"] == 0.0), None)
        if r:
            print(f"{tf:>5} | {r['train_tr']:>8} {r['train_wr']*100:>5.1f}% {r['train_ev']*100:>+6.1f}% | "
                  f"{r['test_tr']:>8} {r['test_wr']*100:>5.1f}% {r['test_ev']*100:>+6.1f}%")

    # Mejor combo por TF (elegido en train, reportado en test)
    print("\n[1] MEJOR COMBO POR TF (elegido por EV train, reportado OOS test)")
    print(f"{'TF':>4} {'MACD':>12} {'EMA':>10} {'slope':>8} | {'TRAINtr':>7} {'wr':>6} {'ev':>7} | {'TESTtr':>7} {'wr':>6} {'ev':>7} {'sig':>4}")
    print("-" * W)
    for _, tf in TFS:
        sub = [r for r in rows if r["tf"] == tf and r["train_tr"] >= 400]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["train_ev"])
        print(f"{tf:>4} {str(b['macd']):>12} {b['emode']:>10} {b['slope']:>8.4f} | "
              f"{b['train_tr']:>7} {b['train_wr']*100:>5.1f}% {b['train_ev']*100:>+6.1f}% | "
              f"{b['test_tr']:>7} {b['test_wr']*100:>5.1f}% {b['test_ev']*100:>+6.1f}% {b['sig']:>4}")

    # Top 15 combos globales por EV test (informativo)
    print("\n[2] TOP 15 COMBOS por EV OUT-OF-SAMPLE (muestra test >= 200)")
    print(f"{'TF':>4} {'MACD':>12} {'EMA':>10} {'slope':>8} | {'TESTtr':>7} {'wr':>6} {'ev':>7} {'sig':>4}")
    print("-" * W)
    top = [r for r in rows if r["test_tr"] >= 200]
    top.sort(key=lambda r: r["test_ev"], reverse=True)
    for r in top[:15]:
        print(f"{r['tf']:>4} {str(r['macd']):>12} {r['emode']:>10} {r['slope']:>8.4f} | "
              f"{r['test_tr']:>7} {r['test_wr']*100:>5.1f}% {r['test_ev']*100:>+6.1f}% {r['sig']:>4}")

    n_pos = sum(1 for r in rows if r["test_ev"] > 0 and r["test_tr"] >= 200)
    print(f"\n  Combos con EV_test>0 (muestra>=200): {n_pos} de {sum(1 for r in rows if r['test_tr']>=200)}")

    # ── Par-hora para el mejor combo global (elegido en train) ───────────────
    tf_b, macd_b, emode_b, slope_b = mejor["_key"]
    factor_b = dict((v, k) for k, v in TFS)[tf_b]
    print(f"\n[3] PAR-HORA con el mejor combo (train): TF={tf_b} MACD{macd_b} {emode_str(emode_b)} slope={slope_b}")
    print(f"    -> OOS global de ese combo: test WR {mejor['test_wr']*100:.1f}% EV {mejor['test_ev']*100:+.1f}% ({mejor['test_tr']} trades)")

    hour_pool = {h: {"tr": [0, 0], "te": [0, 0]} for h in range(24)}
    par_hora = {}   # par -> horas robustas (train Y test > BE)
    for path in real_paths:
        par = os.path.basename(path)[:-5].replace("_", "/")
        d = json.load(open(path, encoding="utf-8"))
        c = resample_last(np.asarray(d["close"], float), factor_b)
        t = resample_last(np.asarray(d["times"], float), factor_b)
        n = len(c)
        if n < WARMUP + 200:
            continue
        split = int(n * TRAIN_FRAC)
        wc = np.zeros(n, bool); wp = np.zeros(n, bool)
        wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
        hour_arr = ((t % 86400) // 3600).astype(int)
        emacache = {p: ema(c, p) for p in EMA_PERIODS}
        ml, sl = macd_lines(c, *macd_b)
        cu, cd = signals(c, ml, sl, emode_b, slope_b, emacache)
        robustas = []
        for h in range(24):
            ttr = evaluar(cu, cd, wc, wp, WARMUP, max(WARMUP, split - H), hour_arr, h)
            tte = evaluar(cu, cd, wc, wp, split, n - H, hour_arr, h)
            hp = hour_pool[h]
            hp["tr"][0] += ttr[0]; hp["tr"][1] += ttr[1]
            hp["te"][0] += tte[0]; hp["te"][1] += tte[1]
            if ttr[0] >= 10 and ttr[2] > BREAK_EVEN and tte[0] >= 5 and tte[2] > BREAK_EVEN:
                robustas.append(h)
        if robustas:
            par_hora[par] = robustas

    print(f"\n    Horas del dia (UTC) pooled REAL con ese combo:")
    print(f"    {'UTC':>3} | {'TRAINtr':>7} {'wr':>6} | {'TESTtr':>7} {'wr':>6}")
    horas_oos = []
    for h in range(24):
        twr = wr(hour_pool[h]["tr"]); ewr = wr(hour_pool[h]["te"])
        marca = ""
        if hour_pool[h]["te"][0] >= 20:
            marca = " <OOS+" if ewr > BREAK_EVEN else ""
            if ewr > BREAK_EVEN:
                horas_oos.append(h)
        print(f"    {h:>3} | {hour_pool[h]['tr'][0]:>7} {twr*100:>5.1f}% | {hour_pool[h]['te'][0]:>7} {ewr*100:>5.1f}%{marca}")
    print(f"    -> Horas UTC con WR>BE OOS (pooled, n>=20): {horas_oos if horas_oos else 'NINGUNA'}")
    print(f"    -> Pares con >=1 hora robusta (train Y test): {len(par_hora)} de {procesados}")

    # ── Veredicto ────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("[VEREDICTO]")
    print(f"  Mejor combo (train): TF={tf_b} MACD{macd_b} {emode_str(emode_b)} slope={slope_b}")
    print(f"    train WR {mejor['train_wr']*100:.1f}% -> test WR {mejor['test_wr']*100:.1f}% | EV OOS {mejor['test_ev']*100:+.1f}% | BE {BREAK_EVEN*100:.1f}%")
    print(f"  Combos EV_test>0 (muestra>=200): {n_pos} | Horas OOS+ pooled: {len(horas_oos)}")
    total_combos = len(rows)
    print(f"  Espacio explorado: {total_combos} combos (TF x MACD x EMA x slope). Con tantos, el azar")
    print(f"  produce ganadores aparentes; cualquier 'optimo' aqui es hipotesis a validar con mas datos.")

    out = {
        "config": {"payout": PAYOUT, "break_even": round(BREAK_EVEN, 4), "H": H,
                   "train_frac": TRAIN_FRAC, "tfs": [t[1] for t in TFS],
                   "macds": [list(m) for m in MACDS], "emodes": [emode_str(e) for e in EMODES],
                   "slopes": SLOPES, "activos_real": procesados, "nota_test": "OOS ~9 dias (base 5m ~1 mes)"},
        "mejor_combo": {"tf": tf_b, "macd": list(macd_b), "ema": emode_str(emode_b), "slope": slope_b,
                        "train_wr": round(mejor["train_wr"], 4), "test_wr": round(mejor["test_wr"], 4),
                        "test_ev": round(mejor["test_ev"], 4), "test_tr": mejor["test_tr"]},
        "ranking": sorted([{k: v for k, v in r.items() if k != "_key"} for r in rows],
                          key=lambda r: r["train_ev"], reverse=True)[:60],
        "par_hora_mejor_combo": par_hora,
    }
    json.dump(out, open("backtest_multi_tf.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nJSON: backtest_multi_tf.json")


if __name__ == "__main__":
    main()
