import json, math, time, os
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

DATA_DIR = "cache_ohlc_1m"
EXPIRY_OPTIONS = [2, 5, 10]
PAYOUT = 0.85
MIN_TRADES = 30
BE = 1.0 / (1.0 + PAYOUT)

MACD_GRID = []
for fast in [3, 5, 6, 8, 10, 12, 15]:
    for slow in [10, 13, 15, 20, 26, 30, 40]:
        if fast < slow:
            for signal in [3, 5, 7, 9]:
                MACD_GRID.append((fast, slow, signal))

EMA_SLOW_OPTS = [0, 30, 50, 75, 100, 150, 200]
EMA_FAST_OPTS = [0, 10, 20, 30, 50]
SLOPE_OPTS = [0.0, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]

TOTAL_MACD = len(MACD_GRID)
TOTAL_FILTER = len(EMA_SLOW_OPTS) * len(EMA_FAST_OPTS) * len(SLOPE_OPTS)
TOTAL = TOTAL_MACD * TOTAL_FILTER * len(EXPIRY_OPTIONS)


def fast_ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def pval_binomial(w, n, p0=0.5):
    if n == 0:
        return 1.0
    mu = n * p0
    sd = math.sqrt(n * p0 * (1 - p0))
    if sd == 0:
        return 1.0
    return 0.5 * math.erfc(((w - 0.5 - mu) / sd) / math.sqrt(2))


def ev_val(wr, payout):
    return wr * payout - (1 - wr)


def load_assets():
    assets = {}
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        par = fname.replace(".json", "")
        try:
            d = json.load(open(os.path.join(DATA_DIR, fname), encoding="utf-8"))
            closes = np.array(d["close"], dtype=float)
            if len(closes) < 500:
                continue
            assets[par] = closes
        except Exception:
            pass
    return assets


def main():
    t0 = time.time()
    print(f"Cargando datos de {DATA_DIR}...")
    assets = load_assets()
    print(f"{len(assets)} activos cargados")

    all_spans = sorted(set(
        [s for f, sl, s in MACD_GRID] +
        [f for f, sl, s in MACD_GRID] +
        [sl for f, sl, s in MACD_GRID] +
        [e for e in EMA_SLOW_OPTS if e > 0] +
        [e for e in EMA_FAST_OPTS if e > 0]
    ))
    print(f"Spans EMA a precomputar: {all_spans}")
    print(f"\nGrid:")
    print(f"  MACD combos: {TOTAL_MACD}")
    print(f"  Filtros: {len(EMA_SLOW_OPTS)} ema_slow x {len(EMA_FAST_OPTS)} ema_fast x {len(SLOPE_OPTS)} slope = {TOTAL_FILTER}")
    print(f"  Expiraciones: {EXPIRY_OPTIONS}")
    print(f"  TOTAL: {TOTAL} configuraciones")
    print(f"  Payout: {PAYOUT:.0%} | Break-even WR: {BE*100:.1f}%")
    print()

    precomputed = {}
    t_pre = time.time()
    for par, closes in assets.items():
        emas = {}
        for sp in all_spans:
            if sp <= len(closes):
                emas[sp] = fast_ema(closes, sp)
        precomputed[par] = {"closes": closes, "emas": emas}
    print(f"EMAs precomputadas en {time.time()-t_pre:.1f}s")

    results = []
    combos = 0
    t_start = time.time()

    for macd_idx, (fast, slow, sig) in enumerate(MACD_GRID):
        macd_signals = {}
        for par, data in precomputed.items():
            closes = data["closes"]
            emas = data["emas"]
            n = len(closes)
            if slow not in emas or fast not in emas:
                continue
            macd_line = emas[fast] - emas[slow]
            sig_line = fast_ema(macd_line, sig)
            diff = macd_line - sig_line
            diff_prev = np.empty_like(diff)
            diff_prev[1:] = diff[:-1]
            diff_prev[0] = 0
            precio_ref = np.where(closes != 0, np.abs(closes), 1.0)
            slope_arr = (diff - diff_prev) / precio_ref
            cross_up = (diff_prev <= 0) & (diff > 0)
            cross_dn = (diff_prev >= 0) & (diff < 0)
            macd_signals[par] = (closes, cross_up, cross_dn, slope_arr)

        for ema_slow_p in EMA_SLOW_OPTS:
            for ema_fast_p in EMA_FAST_OPTS:
                if ema_slow_p == 0 and ema_fast_p > 0:
                    continue
                if ema_slow_p > 0 and ema_fast_p > 0 and ema_fast_p >= ema_slow_p:
                    continue

                ema_masks = {}
                for par, data in precomputed.items():
                    if par not in macd_signals:
                        continue
                    closes = data["closes"]
                    emas = data["emas"]
                    n = len(closes)
                    if ema_slow_p > 0 and ema_slow_p in emas:
                        ema_s = emas[ema_slow_p]
                        if ema_fast_p > 0 and ema_fast_p in emas:
                            ema_f = emas[ema_fast_p]
                            call_ok = (closes > ema_f) & (ema_f > ema_s)
                            put_ok = (closes < ema_f) & (ema_f < ema_s)
                        else:
                            call_ok = closes > ema_s
                            put_ok = closes < ema_s
                        ema_masks[par] = (call_ok, put_ok)
                    else:
                        ema_masks[par] = None

                for slope in SLOPE_OPTS:
                    for expiry in EXPIRY_OPTIONS:
                        h = expiry
                        total_all = 0
                        wins_all = 0

                        for par in macd_signals:
                            closes, cross_up, cross_dn, slope_arr = macd_signals[par]
                            n = len(closes)
                            if h >= n:
                                continue

                            cu = cross_up
                            cd = cross_dn

                            ema_m = ema_masks.get(par)
                            if ema_m is not None:
                                call_ok, put_ok = ema_m
                                cu = cu & call_ok
                                cd = cd & put_ok

                            if slope > 0:
                                cu = cu & (slope_arr >= slope)
                                cd = cd & (slope_arr <= -slope)

                            price_entry = closes[:n - h]
                            price_exit = closes[h:]

                            call_wins = cu[:n - h] & (price_exit > price_entry)
                            put_wins = cd[:n - h] & (price_exit < price_entry)

                            total_all += int(np.count_nonzero(cu[:n - h] | cd[:n - h]))
                            wins_all += int(np.count_nonzero(call_wins | put_wins))

                        combos += 1
                        if total_all < MIN_TRADES:
                            continue

                        wr = wins_all / total_all
                        ev = ev_val(wr, PAYOUT)
                        p = pval_binomial(wins_all, total_all, BE)

                        n_ema = f"EMA{ema_fast_p}/{ema_slow_p}" if ema_fast_p > 0 else (f"EMA{ema_slow_p}" if ema_slow_p > 0 else "sin")
                        results.append({
                            "macd": f"({fast},{slow},{sig})",
                            "fast": fast, "slow": slow, "sig": sig,
                            "ema_fast": ema_fast_p, "ema_slow": ema_slow_p,
                            "slope": slope, "expiry": expiry,
                            "trades": total_all, "wins": wins_all,
                            "wr": round(wr, 4), "ev": round(ev, 6),
                            "p": round(p, 4),
                            "ema_label": n_ema,
                        })

        elapsed = time.time() - t_start
        pct = combos / TOTAL * 100
        eta = (elapsed / max(combos, 1)) * (TOTAL - combos) / 60
        best_ev = max((r["ev"] for r in results), default=0) if results else 0
        best_wr = max((r["wr"] for r in results), default=0) if results else 0
        n_sig = len([r for r in results if r["p"] < 0.05 and r["ev"] > 0])
        print(f"  MACD({fast},{slow},{sig}) [{macd_idx+1}/{TOTAL_MACD}] "
              f"| {combos}/{TOTAL} ({pct:.1f}%) "
              f"| ETA {eta:.1f}min "
              f"| best EV {best_ev*100:+.3f}% WR {best_wr*100:.1f}% "
              f"| sig {n_sig} | total {len(results)}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"BUSQUEDA COMPLETADA en {elapsed/60:.1f} min")
    print(f"Configuraciones evaluadas: {combos}")
    print(f"Configuraciones con EV > -5%: {len(results)}")
    print(f"{'='*120}\n")

    results.sort(key=lambda x: x["ev"], reverse=True)

    print(f"\n{'='*120}")
    print(f"TOP 50 MEJORES CONFIGURACIONES (por EV)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'MACD':>12} {'EMA':>12} {'SLOPE':>8} {'EXP':>4} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
    print("-" * 120)
    for i, r in enumerate(results[:50], 1):
        marca = " ***" if r["p"] < 0.05 and r["ev"] > 0 else ""
        print(f"{i:3d} {r['macd']:>12} {r['ema_label']:>12} {r['slope']:>8.4f} {r['expiry']:>4} "
              f"{r['trades']:>7} {r['wr']*100:6.1f}% {r['ev']*100:+7.3f}% {r['p']:6.3f}{marca}")

    macd_stats = defaultdict(list)
    for r in results:
        macd_stats[r["macd"]].append(r["ev"])
    macd_avg = [(m, np.mean(evs), len(evs), max(evs)) for m, evs in macd_stats.items()]
    macd_avg.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*120}")
    print(f"TOP 20 MACD POR EV MAX")
    print(f"{'='*120}")
    print(f"{'MACD':>12} {'# CONFIGS':>10} {'EV PROM':>10} {'EV MAX':>10}")
    print("-" * 50)
    for m, avg, cnt, mx in macd_avg[:20]:
        print(f"{m:>12} {cnt:10d} {avg*100:+9.3f}% {mx*100:+9.3f}%")

    sig = [r for r in results if r["p"] < 0.05 and r["ev"] > 0]
    print(f"\n{'='*120}")
    print(f"ESTADISTICAMENTE SIGNIFICATIVAS (p<0.05, EV>0): {len(sig)}")
    print(f"{'='*120}")
    if sig:
        print(f"{'#':>3} {'MACD':>12} {'EMA':>12} {'SLOPE':>8} {'EXP':>4} {'TRADES':>7} {'WR':>7} {'EV':>8} {'P-VAL':>7}")
        print("-" * 120)
        for i, r in enumerate(sig[:30], 1):
            print(f"{i:3d} {r['macd']:>12} {r['ema_label']:>12} {r['slope']:>8.4f} {r['expiry']:>4} "
                  f"{r['trades']:>7} {r['wr']*100:6.1f}% {r['ev']*100:+7.3f}% {r['p']:6.3f}")
    else:
        print("Ninguna configuracion cumple p<0.05 con EV>0.")

    print(f"\n{'='*120}")
    print(f"TOP 20 POR # DE ACTIVOS CON EV POSITIVO")
    print(f"{'='*120}")
    from collections import Counter
    win_rate_by_config = Counter()
    for r in results:
        key = (r["macd"], r["ema_label"], r["slope"], r["expiry"])
        if r["ev"] > 0:
            win_rate_by_config[key] += 1
    top_by_assets = win_rate_by_config.most_common(20)
    print(f"{'MACD':>12} {'EMA':>12} {'SLOPE':>8} {'EXP':>4} {'# ASSETS EV+':>14}")
    print("-" * 60)
    for (m, e, s, ex), cnt in top_by_assets:
        print(f"{m:>12} {e:>12} {s:>8.4f} {ex:>4} {cnt:>14}")

    with open("exhaustive_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "fecha": datetime.now().isoformat(),
            "config": {
                "payout": PAYOUT, "min_trades": MIN_TRADES,
                "assets": len(assets), "macd_combos": TOTAL_MACD,
                "filter_combos": TOTAL_FILTER, "expiry_options": EXPIRY_OPTIONS,
                "total_evaluados": combos, "elapsed_min": round(elapsed/60, 1),
            },
            "top_50": results[:50],
            "significativos": sig[:50],
            "macd_ranking": [{"macd": m, "configs": cnt, "ev_avg": round(avg, 6), "ev_max": round(mx, 6)}
                             for m, avg, cnt, mx in macd_avg[:30]],
            "total_resultados": len(results),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en exhaustive_results.json")


if __name__ == "__main__":
    main()
