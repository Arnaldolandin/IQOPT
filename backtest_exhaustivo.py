# backtest_exhaustivo.py - Busqueda EXHAUSTIVA de cualquier config rentable.
# Barre estrategia x parametros x timeframe x hora, y para cada una hace WALK-FORWARD
# (WR pooled por ventana de 30 dias). Reporta solo lo que aguanta la MAYORIA de meses > break-even.
# Base: cache_ohlc_5m (6 meses, 50 reales), resampleada a 5/10/15/30m.
#   python backtest_exhaustivo.py

import json, math, os, glob
import numpy as np

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
H = 2                       # holdear 2 barras
WARMUP = 260
WIN_DAYS = 30
CACHE_DIR = "cache_ohlc_5m"
MIN_OPS_WIN = 15           # min ops por ventana para contarla
TFS = [(1, "5m"), (2, "10m"), (3, "15m"), (6, "30m")]

# Estrategias (nombre, params). tipo: bb / rsi / macd / fade
STRATS = [
    ("bb_2.0", "bb", {"p": 20, "k": 2.0}),
    ("bb_2.5", "bb", {"p": 20, "k": 2.5}),
    ("rsi_25_75", "rsi", {"p": 14, "os": 25, "ob": 75}),
    ("rsi_20_80", "rsi", {"p": 14, "os": 20, "ob": 80}),
    ("fade_macd", "fade", {"f": 6, "s": 13, "sig": 5}),
    ("macd_puro", "macd", {"f": 6, "s": 13, "sig": 5, "ema": 0}),
    ("macd_ema50", "macd", {"f": 6, "s": 13, "sig": 5, "ema": 50}),
    ("macd_12_26_9_ema50", "macd", {"f": 12, "s": 26, "sig": 9, "ema": 50}),
]


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def roll_sum(x, p):
    cs = np.cumsum(x); n = len(x); out = np.full(n, np.nan)
    out[p:] = cs[p:] - cs[:-p]; out[p - 1] = cs[p - 1]; return out


def rsi(c, period):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = roll_sum(up, period) / period; ad = roll_sum(dn, period) / period
    rs = au / np.where(ad == 0, np.nan, ad); r = 100 - 100 / (1 + rs)
    r[(ad == 0) & (au > 0)] = 100.0; r[(ad == 0) & (au == 0)] = 50.0; return r


def bollinger(c, period, k):
    n = len(c); cs = np.cumsum(c); cs2 = np.cumsum(c * c)
    mean = np.full(n, np.nan); std = np.full(n, np.nan)
    m = (cs[period:] - cs[:-period]) / period; m2 = (cs2[period:] - cs2[:-period]) / period
    std[period:] = np.sqrt(np.clip(m2 - m * m, 0, None)); mean[period:] = m
    return mean - k * std, mean + k * std


def resample_last(a, f):
    if f == 1:
        return np.asarray(a, float)
    n = len(a) // f
    return np.asarray(a[:n * f], float).reshape(n, f)[:, -1]


def cross_below(x, lv):
    n = len(x); m = np.zeros(n, bool); m[1:] = (x[:-1] >= lv[:-1]) & (x[1:] < lv[1:]); return m


def cross_above(x, lv):
    n = len(x); m = np.zeros(n, bool); m[1:] = (x[:-1] <= lv[:-1]) & (x[1:] > lv[1:]); return m


def signals(tipo, prm, c):
    n = len(c); cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    if tipo == "bb":
        lo, up = bollinger(c, prm["p"], prm["k"]); cu = cross_below(c, lo); cd = cross_above(c, up)
    elif tipo == "rsi":
        r = rsi(c, prm["p"])
        cu = np.zeros(n, bool); cd = np.zeros(n, bool)
        cu[1:] = (r[:-1] >= prm["os"]) & (r[1:] < prm["os"])
        cd[1:] = (r[:-1] <= prm["ob"]) & (r[1:] > prm["ob"])
    elif tipo in ("macd", "fade"):
        ml = ema(c, prm["f"]) - ema(c, prm["s"]); sl = ema(ml, prm["sig"]); diff = ml - sl
        up_ = np.zeros(n, bool); dn_ = np.zeros(n, bool)
        up_[1:] = (diff[:-1] <= 0) & (diff[1:] > 0); dn_[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
        if tipo == "macd":
            cu, cd = up_, dn_
            if prm.get("ema"):
                ev = ema(c, prm["ema"]); cu = cu & (c > ev); cd = cd & (c < ev)
        else:  # fade: al reves
            cu, cd = dn_, up_
    cu[:WARMUP] = False; cd[:WARMUP] = False
    return cu, cd


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    # cargar y determinar rango temporal global (para ventanas alineadas)
    raw = []
    t0 = t1 = None
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        if len(c) < 3000:
            continue
        raw.append((c, t)); t0 = t[0] if t0 is None else min(t0, t[0]); t1 = t[-1] if t1 is None else max(t1, t[-1])
    nwin = int((t1 - t0) // (WIN_DAYS * 86400))
    npar = len(raw)
    print(f"Exhaustivo: {npar} pares reales | {nwin} ventanas de {WIN_DAYS}d | "
          f"{len(STRATS)} estrategias x {len(TFS)} TF x 24h + all | BE {BE:.1%}")

    resultados = []   # (strat, tf, hora, windows_be, nwin_val, median_wr, total_ops, total_wr)
    for factor, tfname in TFS:
        # pre-resamplear por par
        rs = [(resample_last(c, factor), resample_last(t, factor)) for c, t in raw]
        winlen = WIN_DAYS * 86400
        for sname, tipo, prm in STRATS:
            # acumular por (hora, ventana): [ops, wins]; hora 24 = todas
            acc = np.zeros((25, nwin, 2), dtype=np.int64)
            for c, t in rs:
                n = len(c)
                if n < WARMUP + 200:
                    continue
                cu, cd = signals(tipo, prm, c)
                wc = np.zeros(n, bool); wp = np.zeros(n, bool)
                wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
                sig = cu | cd
                sig[n - H:] = False
                idx = np.where(sig)[0]
                if len(idx) == 0:
                    continue
                hora = ((t[idx] % 86400) // 3600).astype(int)
                widx = np.clip(((t[idx] - t0) // winlen).astype(int), 0, nwin - 1)
                gan = np.where(cu[idx], wc[idx], wp[idx])
                for h_, w_, g_ in zip(hora, widx, gan):
                    acc[h_, w_, 0] += 1; acc[h_, w_, 1] += g_
                    acc[24, w_, 0] += 1; acc[24, w_, 1] += g_
            # evaluar cada hora (y 'todas'=24)
            for h in range(25):
                wrs = []
                tot_ops = tot_w = 0
                for wv in range(nwin):
                    o, w = acc[h, wv]
                    tot_ops += o; tot_w += w
                    if o >= MIN_OPS_WIN:
                        wrs.append(w / o)
                if len(wrs) < max(3, nwin - 1):   # exigir cobertura en casi todas las ventanas
                    continue
                wrs = np.array(wrs)
                wbe = int((wrs > BE).sum())
                resultados.append((sname, tfname, h, wbe, len(wrs), float(np.median(wrs)),
                                   int(tot_ops), tot_w / tot_ops if tot_ops else 0.0))

    # ROBUSTAS: mayoria de ventanas > BE y mediana > BE
    robustas = [r for r in resultados if r[3] >= math.ceil(r[4] * 0.8) and r[5] > BE and r[6] >= 300]
    robustas.sort(key=lambda r: (r[3] / r[4], r[5]), reverse=True)

    W = 104
    print("\n" + "=" * W)
    print("CONFIGS ROBUSTAS (>=80% de las ventanas sobre break-even Y mediana > BE)")
    print(f"{'ESTRATEGIA':>20} {'TF':>4} {'HORA':>5} | {'vent>BE':>8} {'medWR':>7} {'globWR':>7} {'globEV':>7} {'ops':>7}")
    print("-" * W)
    if not robustas:
        print("  NINGUNA. Ninguna combinacion supera break-even de forma robusta (mayoria de meses).")
    for sname, tf, h, wbe, nv, med, ops, gwr in robustas[:40]:
        hstr = "todas" if h == 24 else f"UTC{h}"
        print(f"{sname:>20} {tf:>4} {hstr:>5} | {wbe:>3}/{nv:<3} {med*100:>6.1f}% {gwr*100:>6.1f}% "
              f"{ev_de_wr(gwr)*100:>+6.1f}% {ops:>7}")

    # Resumen 'all-hours' por estrategia/TF (para ver el panorama general, casi todo -EV)
    print("\n" + "=" * W)
    print("PANORAMA — TODAS las horas juntas (global WR OOS por estrategia x TF)")
    print(f"{'ESTRATEGIA':>20} | " + " ".join(f"{tf:>7}" for _, tf in TFS))
    print("-" * W)
    allh = {(s[0], tf): None for s in STRATS for _, tf in TFS}
    for sname, tf, h, wbe, nv, med, ops, gwr in resultados:
        if h == 24:
            allh[(sname, tf)] = gwr
    for sname, _, _ in STRATS:
        row = f"{sname:>20} | "
        for _, tf in TFS:
            g = allh.get((sname, tf))
            row += f"{(g*100 if g else 0):>6.1f}% "
        print(row)
    print(f"\n(break-even = {BE*100:.1f}%. Verde solo si > eso Y aguanta walk-forward arriba)")

    json.dump({"be": round(BE, 4), "pares": npar, "ventanas": nwin,
               "robustas": [{"strat": r[0], "tf": r[1], "hora": ("todas" if r[2] == 24 else r[2]),
                             "vent_be": r[3], "vent": r[4], "median_wr": round(r[5], 4),
                             "global_wr": round(r[7], 4), "ops": r[6]} for r in robustas],
               "todas_horas": [{"strat": s, "tf": tf, "global_wr": round(g, 4)}
                               for (s, tf), g in allh.items() if g]},
              open("backtest_exhaustivo.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nJSON: backtest_exhaustivo.json")


if __name__ == "__main__":
    main()
