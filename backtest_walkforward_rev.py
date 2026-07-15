# backtest_walkforward_rev.py - Walk-forward de las estrategias de reversion ganadoras.
# Mide el WR/EV en VENTANAS consecutivas de ~30 dias sobre los ~7 meses de 30m.
# Objetivo: ver si el edge (~53-54%) PERSISTE mes a mes o fue un mes afortunado.
#   python backtest_walkforward_rev.py [cache_dir] [dias_ventana]

import json, math, os, glob, sys
import numpy as np

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
H = 2
WARMUP = 60
CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_30m"
WIN_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# Candidatas: (nombre, ATR minimo)
CANDS = [
    ("rsi_20_80", 0.000),
    ("rsi_20_80", 0.001),
    ("rsi_20_80", 0.002),
    ("rsi_25_75", 0.001),
    ("bollinger", 0.000),
    ("bollinger", 0.005),
]


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def roll_sum(x, p):
    cs = np.cumsum(x); n = len(x); out = np.full(n, np.nan)
    out[p:] = cs[p:] - cs[:-p]; out[p - 1] = cs[p - 1]
    return out


def rsi(c, period=14):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = roll_sum(up, period) / period; ad = roll_sum(dn, period) / period
    rs = au / np.where(ad == 0, np.nan, ad)
    r = 100 - 100 / (1 + rs)
    r[(ad == 0) & (au > 0)] = 100.0; r[(ad == 0) & (au == 0)] = 50.0
    return r


def bollinger(c, period=20, k=2.0):
    n = len(c); cs = np.cumsum(c); cs2 = np.cumsum(c * c)
    mean = np.full(n, np.nan); std = np.full(n, np.nan)
    m = (cs[period:] - cs[:-period]) / period
    m2 = (cs2[period:] - cs2[:-period]) / period
    var = np.clip(m2 - m * m, 0, None)
    mean[period:] = m; std[period:] = np.sqrt(var)
    return mean - k * std, mean + k * std


def atr_pct_series(h, l, c, period=14):
    n = len(c); tr = np.empty(n); tr[0] = h[0] - l[0]; pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    cs = np.cumsum(tr); atr = np.full(n, np.nan); atr[period:] = (cs[period:] - cs[:-period]) / period
    pr = np.abs(c); pr[pr == 0] = 1.0
    return np.nan_to_num(atr / pr, nan=0.0)


def cross_below(x, level):
    n = len(x); m = np.zeros(n, bool)
    lv = level if np.ndim(level) else np.full(n, level, float)
    m[1:] = (x[:-1] >= lv[:-1]) & (x[1:] < lv[1:]); return m


def cross_above(x, level):
    n = len(x); m = np.zeros(n, bool)
    lv = level if np.ndim(level) else np.full(n, level, float)
    m[1:] = (x[:-1] <= lv[:-1]) & (x[1:] > lv[1:]); return m


def signals(name, c):
    n = len(c); cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    if name == "rsi_20_80":
        r = rsi(c, 14); cu = cross_below(r, 20); cd = cross_above(r, 80)
    elif name == "rsi_25_75":
        r = rsi(c, 14); cu = cross_below(r, 25); cd = cross_above(r, 75)
    elif name == "bollinger":
        lo, up = bollinger(c, 20, 2.0); cu = cross_below(c, lo); cd = cross_above(c, up)
    cu[:WARMUP] = False; cd[:WARMUP] = False
    return cu, cd


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def pval(w, n, p0=BREAK_EVEN):
    if n == 0:
        return 1.0
    sd = math.sqrt(n * p0 * (1 - p0))
    return 1.0 if sd == 0 else 0.5 * math.erfc(((w - 0.5 - n * p0) / sd) / math.sqrt(2))


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if not files:
        print(f"No hay datos REALES en {CACHE_DIR}/"); return

    # rango temporal global
    t0 = None; t1 = None
    datos = []
    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
            c = np.asarray(d["close"], float); h = np.asarray(d.get("high", d["close"]), float)
            l = np.asarray(d.get("low", d["close"]), float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        if len(c) < WARMUP + 300:
            continue
        datos.append((c, h, l, t))
        t0 = t[0] if t0 is None else min(t0, t[0])
        t1 = t[-1] if t1 is None else max(t1, t[-1])

    n_win = int((t1 - t0) // (WIN_DAYS * 86400))
    bordes = [t0 + i * WIN_DAYS * 86400 for i in range(n_win + 1)]
    W = 96
    print("=" * W)
    print(f"WALK-FORWARD REVERSION  |  {CACHE_DIR}  |  {len(datos)} pares  |  {n_win} ventanas de {WIN_DAYS}d  |  "
          f"payout {PAYOUT:.0%} (BE {BREAK_EVEN:.1%})")
    print("=" * W)

    from datetime import datetime, timezone
    etiquetas = [datetime.fromtimestamp(bordes[i], timezone.utc).strftime("%m-%d") for i in range(n_win)]

    # acumular por candidata x ventana
    for name, atr_min in CANDS:
        win = [[0, 0] for _ in range(n_win)]
        for c, h, l, t in datos:
            n = len(c)
            wc = np.zeros(n, bool); wp = np.zeros(n, bool)
            wc[:n - H] = c[H:] > c[:n - H]; wp[:n - H] = c[H:] < c[:n - H]
            cu, cd = signals(name, c)
            if atr_min > 0:
                ap = atr_pct_series(h, l, c, 14); vok = ap >= atr_min
                cu = cu & vok; cd = cd & vok
            widx = np.clip(((t - t0) // (WIN_DAYS * 86400)).astype(int), 0, n_win - 1)
            sig_idx = np.where(cu | cd)[0]
            sig_idx = sig_idx[sig_idx < n - H]
            for i in sig_idx:
                w = widx[i]
                gan = wc[i] if cu[i] else wp[i]
                win[w][0] += 1; win[w][1] += 1 if gan else 0

        wrs = [(win[i][1] / win[i][0]) if win[i][0] else 0.0 for i in range(n_win)]
        # resumen: cuantas ventanas > BE, WR global
        tot_tr = sum(win[i][0] for i in range(n_win)); tot_w = sum(win[i][1] for i in range(n_win))
        gwr = tot_w / tot_tr if tot_tr else 0.0
        n_gt_be = sum(1 for i in range(n_win) if win[i][0] >= 30 and wrs[i] > BREAK_EVEN)
        n_val = sum(1 for i in range(n_win) if win[i][0] >= 30)
        print(f"\n{name} ATR>={atr_min:.3f}  | global WR {gwr*100:.1f}% EV {ev_de_wr(gwr)*100:+.1f}% "
              f"({tot_tr} tr) | ventanas>BE: {n_gt_be}/{n_val}")
        cells = []
        for i in range(n_win):
            if win[i][0] >= 30:
                mk = "+" if wrs[i] > BREAK_EVEN else " "
                cells.append(f"{etiquetas[i]}:{wrs[i]*100:4.0f}%{mk}")
            else:
                cells.append(f"{etiquetas[i]}:  -- ")
        # imprimir en filas de 6
        for j in range(0, n_win, 6):
            print("   " + "  ".join(cells[j:j + 6]))

    print("\n" + "=" * W)
    print("LECTURA: una estrategia con edge REAL debe estar >BE en la MAYORIA de ventanas,")
    print("no solo en el global. Si salta arriba/abajo del 53.5%, el edge no es fiable.")


if __name__ == "__main__":
    main()
