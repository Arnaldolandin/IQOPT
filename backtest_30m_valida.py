# backtest_30m_valida.py - Validacion RIGUROSA del timeframe 30m con ~6-7 meses.
# Lee cache_ohlc_30m/ (30m directo, pares reales). OOS = ultimos 60 dias (2 meses).
# Barre MACD x modo-EMA x pendiente; reporta si algun combo/hora/par AGUANTA fuera de muestra,
# con control de azar (cuantos ganadores esperaria el puro ruido).

import json, math, os, glob
import numpy as np

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
TEST_DAYS = 60                     # OOS de 2 meses
H = 2                              # 2 barras de 30m = 1h de expiracion
CACHE_DIR = "cache_ohlc_30m"
WARMUP = 210
MIN_TR = 60

MACDS = [(6, 13, 5), (12, 26, 9), (8, 17, 9), (5, 10, 3)]
EMODES = [("none",), ("single", 100), ("single", 50), ("stack", 50, 100)]
SLOPES = [0.0, 0.0001, 0.0005, 0.001]
EMA_PERIODS = sorted({p for m in EMODES for p in m[1:]})


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_lines(c, fast, slow, sig):
    ml = ema(c, fast) - ema(c, slow)
    return ml, ema(ml, sig)


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def pval(w, n, p0=BREAK_EVEN):
    if n == 0:
        return 1.0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - n * p0) / sd) / math.sqrt(2))


def prob_supera_be(n):
    # P(WR > BE) bajo H0: WR real = 0.5 (aprox normal)
    if n <= 0:
        return 0.0
    sd = math.sqrt(0.25 / n)
    z = (BREAK_EVEN - 0.5) / sd
    return 0.5 * math.erfc(z / math.sqrt(2))


def signals(c, ml, sl, emode, slope, emacache):
    n = len(c); diff = ml - sl
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
    cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
    k = emode[0]
    if k == "single":
        e = emacache[emode[1]]; cu &= c > e; cd &= c < e
    elif k == "stack":
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


def emstr(e):
    return "sin-EMA" if e[0] == "none" else (f"EMA{e[1]}" if e[0] == "single" else f"EMA{e[1]}>{e[2]}")


def wr(seg):
    return seg[1] / seg[0] if seg[0] else 0.0


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    if not files:
        print(f"No hay {CACHE_DIR}/ (baja primero con download_ohlc_30m.py)")
        return

    pool = {}; sig_count = {}
    per_asset = {}          # par -> {combo_key: {"tr":(..),"te":(..)}}  para par-hora luego
    spans = []
    procesados = 0
    for path in files:
        par = os.path.basename(path)[:-5].replace("_", "/")
        try:
            d = json.load(open(path, encoding="utf-8"))
            c = np.asarray(d["close"], float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < WARMUP + 300:
            continue
        spans.append((t[-1] - t[0]) / 86400)
        procesados += 1
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        wc = np.zeros(n, bool); wp = np.zeros(n, bool)
        wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
        hour_arr = ((t % 86400) // 3600).astype(int)
        emacache = {p: ema(c, p) for p in EMA_PERIODS}
        tr_lo, tr_hi = WARMUP, max(WARMUP, split - H)
        te_lo, te_hi = split, n - H

        pa = {}
        for mcfg in MACDS:
            ml, sl = macd_lines(c, *mcfg)
            for emode in EMODES:
                for slope in SLOPES:
                    cu, cd = signals(c, ml, sl, emode, slope, emacache)
                    trn = evaluar(cu, cd, wc, wp, tr_lo, tr_hi)
                    tst = evaluar(cu, cd, wc, wp, te_lo, te_hi)
                    key = (mcfg, emstr(emode), slope)
                    p = pool.setdefault(key, {"tr": [0, 0], "te": [0, 0]})
                    p["tr"][0] += trn[0]; p["tr"][1] += trn[1]
                    p["te"][0] += tst[0]; p["te"][1] += tst[1]
                    if tst[0] >= MIN_TR and pval(tst[1], tst[0]) < 0.05 and ev_de_wr(tst[2]) > 0:
                        sig_count[key] = sig_count.get(key, 0) + 1
                    pa[key] = {"tr": trn, "te": tst}
        per_asset[par] = pa

    rows = []
    for key, p in pool.items():
        mcfg, em, slope = key
        rows.append({"macd": mcfg, "ema": em, "slope": slope,
                     "train_tr": p["tr"][0], "train_wr": wr(p["tr"]), "train_ev": ev_de_wr(wr(p["tr"])),
                     "test_tr": p["te"][0], "test_wr": wr(p["te"]), "test_ev": ev_de_wr(wr(p["te"])),
                     "sig": sig_count.get(key, 0), "_key": key})

    W = 100
    print("=" * W)
    print(f"VALIDACION 30m  |  {procesados} pares REALES  |  span mediano {np.median(spans):.0f}d  |  "
          f"payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})  |  H={H} (1h exp)  |  OOS test = ultimos {TEST_DAYS}d")
    print("=" * W)

    base = next((r for r in rows if r["macd"] == (6, 13, 5) and r["ema"] == "sin-EMA" and r["slope"] == 0.0), None)
    if base:
        print(f"\n[BASE] MACD(6,13,5) sin-EMA slope0:  train {base['train_wr']*100:.1f}% ({base['train_tr']} tr)  ->  "
              f"TEST {base['test_wr']*100:.1f}% EV {base['test_ev']*100:+.1f}% ({base['test_tr']} tr)")

    print("\n[1] TOP 15 COMBOS por EV OUT-OF-SAMPLE (test 2 meses, muestra >= 150)")
    print(f"{'MACD':>12} {'EMA':>10} {'slope':>8} | {'TRAINtr':>7} {'wr':>6} | {'TESTtr':>7} {'wr':>6} {'ev':>7} {'sig':>4}")
    print("-" * W)
    top = [r for r in rows if r["test_tr"] >= 150]
    top.sort(key=lambda r: r["test_ev"], reverse=True)
    for r in top[:15]:
        print(f"{str(r['macd']):>12} {r['ema']:>10} {r['slope']:>8.4f} | {r['train_tr']:>7} {r['train_wr']*100:>5.1f}% | "
              f"{r['test_tr']:>7} {r['test_wr']*100:>5.1f}% {r['test_ev']*100:>+6.1f}% {r['sig']:>4}")

    cand = [r for r in rows if r["train_tr"] >= 300]
    cand.sort(key=lambda r: r["train_ev"], reverse=True)
    mejor = cand[0] if cand else max(rows, key=lambda r: r["train_ev"])
    print(f"\n  Mejor combo elegido en TRAIN: MACD{mejor['macd']} {mejor['ema']} slope={mejor['slope']}")
    print(f"    train {mejor['train_wr']*100:.1f}% -> TEST {mejor['test_wr']*100:.1f}% EV {mejor['test_ev']*100:+.1f}% ({mejor['test_tr']} tr)")

    # control de azar
    n_pos = sum(1 for r in rows if r["test_ev"] > 0 and r["test_tr"] >= 150)
    n_combos = sum(1 for r in rows if r["test_tr"] >= 150)
    esperado_azar = sum(prob_supera_be(r["test_tr"]) for r in rows if r["test_tr"] >= 150)
    print(f"\n[2] CONTROL DE AZAR")
    print(f"  Combos con EV_test>0: {n_pos} de {n_combos}  |  esperados por AZAR (WR=50%): ~{esperado_azar:.0f}")
    print(f"  Activos significativos OOS (suma sobre combos): {sum(sig_count.values())}")

    # ── Par-hora con el mejor combo ──────────────────────────────────────────
    key_b = mejor["_key"]
    hour_pool = {h: {"tr": [0, 0], "te": [0, 0]} for h in range(24)}
    par_hora = {}
    for path in files:
        par = os.path.basename(path)[:-5].replace("_", "/")
        d = json.load(open(path, encoding="utf-8"))
        c = np.asarray(d["close"], float); t = np.asarray(d["times"], float)
        n = len(c)
        if n < WARMUP + 300:
            continue
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        wc = np.zeros(n, bool); wp = np.zeros(n, bool)
        wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
        hour_arr = ((t % 86400) // 3600).astype(int)
        emacache = {p: ema(c, p) for p in EMA_PERIODS}
        ml, sl = macd_lines(c, *key_b[0])
        emode = ("none",) if key_b[1] == "sin-EMA" else (
            ("stack", 50, 100) if ">" in key_b[1] else ("single", int(key_b[1][3:])))
        cu, cd = signals(c, ml, sl, emode, key_b[2], emacache)
        robustas = []
        for h in range(24):
            ttr = evaluar(cu, cd, wc, wp, WARMUP, max(WARMUP, split - H), hour_arr, h)
            tte = evaluar(cu, cd, wc, wp, split, n - H, hour_arr, h)
            hp = hour_pool[h]
            hp["tr"][0] += ttr[0]; hp["tr"][1] += ttr[1]
            hp["te"][0] += tte[0]; hp["te"][1] += tte[1]
            if ttr[0] >= 15 and ttr[2] > BREAK_EVEN and tte[0] >= 10 and tte[2] > BREAK_EVEN:
                robustas.append(h)
        if robustas:
            par_hora[par] = robustas

    print(f"\n[3] HORAS (UTC) pooled con el mejor combo (MACD{key_b[0]} {key_b[1]} slope={key_b[2]})")
    horas_oos = []
    for h in range(24):
        twr = wr(hour_pool[h]["tr"]); ewr = wr(hour_pool[h]["te"])
        mark = ""
        if hour_pool[h]["te"][0] >= 20 and ewr > BREAK_EVEN:
            mark = " <OOS+"; horas_oos.append(h)
        if hour_pool[h]["te"][0] >= 20:
            print(f"  UTC {h:>2} | train {hour_pool[h]['tr'][0]:>5} {twr*100:>5.1f}% | test {hour_pool[h]['te'][0]:>5} {ewr*100:>5.1f}%{mark}")
    print(f"  -> Horas OOS+ (n>=20): {horas_oos if horas_oos else 'NINGUNA'}")
    print(f"  -> Pares con hora robusta (train Y test): {len(par_hora)}")

    print("\n" + "=" * W)
    print("[VEREDICTO 30m]")
    if base:
        print(f"  Baseline 30m (params fijos) OOS 2 meses: {base['test_wr']*100:.1f}% WR, EV {base['test_ev']*100:+.1f}% (BE {BREAK_EVEN*100:.1f}%)")
    print(f"  Mejor combo OOS: {mejor['test_wr']*100:.1f}% WR EV {mejor['test_ev']*100:+.1f}%")
    print(f"  Ganadores EV>0: {n_pos} vs ~{esperado_azar:.0f} por azar  ->  " +
          ("dentro del ruido" if n_pos <= esperado_azar * 1.3 else "POR ENCIMA del azar (revisar)"))
    veredicto = "RENTABLE" if (base and base["test_ev"] > 0) or mejor["test_ev"] > 0.02 else "NO rentable"
    print(f"  => 30m {veredicto} fuera de muestra a 2 meses.")

    json.dump({"config": {"payout": PAYOUT, "be": round(BREAK_EVEN, 4), "H": H, "test_days": TEST_DAYS,
                          "activos": procesados, "span_mediano_d": round(float(np.median(spans)), 0)},
               "baseline": base and {k: v for k, v in base.items() if k != "_key"},
               "mejor_combo": {k: v for k, v in mejor.items() if k != "_key"},
               "azar": {"ganadores": n_pos, "esperado": round(esperado_azar, 1), "combos": n_combos},
               "ranking": sorted([{k: v for k, v in r.items() if k != "_key"} for r in rows],
                                 key=lambda r: r["test_ev"], reverse=True)[:40],
               "par_hora": par_hora},
              open("backtest_30m_valida.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nJSON: backtest_30m_valida.json")


if __name__ == "__main__":
    main()
