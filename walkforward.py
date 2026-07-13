# walkforward.py - Prueba out-of-sample honesta del edge RSI-reversion sobre USDJPY (IQ).
#
# Metodo: ventana deslizante. En cada tramo TRAIN se elige el mejor (umbral RSI, horizonte)
# por WR in-sample; esa eleccion se aplica al tramo TEST siguiente (nunca visto). Se agregan
# todos los TEST -> WR out-of-sample real. Si el OOS colapsa a ~50%, el "edge" era sobreajuste.
# Baseline: la config fija del bot (30/70 @10m) aplicada a los mismos tramos TEST.
#
# Resultado del 2026-07-13 (25.4 dias, 19 folds): IS 65.0% -> OOS 57.5% (n=332, p=0.130),
# baseline 30/70@10m 53.7% (bajo break-even). Brecha de sobreajuste 7.5 pt.
#
#   .venv314\Scripts\python.exe walkforward.py
import json, math, time

import numpy as np
from iqoptionapi.stable_api import IQ_Option

BE = 0.543                      # break-even IQ (payout 84%)
PAYOUT = 0.84
WARMUP = 100
TRAIN = 5000
TEST = 1000
MIN_TRAIN_TRADES = 25
GRID = [(25, 75), (30, 70), (35, 65), (40, 60)]
HORIZONS = [5, 10, 15]


def rsi_series(c, p=14):
    c = np.asarray(c, float); n = len(c); out = np.full(n, np.nan)
    if n < p + 1: return out
    d = np.diff(c); g = np.where(d > 0, d, 0.); l = np.where(d < 0, -d, 0.)
    ag, al = g[:p].mean(), l[:p].mean()
    for i in range(p, n):
        if i > p: ag = (ag*(p-1)+g[i-1])/p; al = (al*(p-1)+l[i-1])/p
        out[i] = 100 - 100/(1 + (ag/al if al > 0 else 999))
    return out


def evaluar(rsi, closes, lo, hi, h, start, end):
    """RSI-reversion des-solapado sobre indices [start, end). Devuelve (trades, wins)."""
    wins = tr = 0; i = start
    while i < end - h:
        v = rsi[i]
        if np.isnan(v): i += 1; continue
        side = "call" if v < lo else ("put" if v > hi else None)
        if side is None: i += 1; continue
        gano = (closes[i+h] > closes[i]) if side == "call" else (closes[i+h] < closes[i])
        tr += 1; wins += 1 if gano else 0; i += h
    return tr, wins


def pval(w, n, p0=BE):
    if n == 0: return 1.0
    mu = n*p0; sd = math.sqrt(n*p0*(1-p0))
    if sd == 0: return 1.0
    return 0.5*math.erfc(((w-0.5-mu)/sd)/math.sqrt(2))


def bajar(api, par, total):
    """Baja `total` velas de 1-min hacia atras, en lotes (la API limita ~1000)."""
    todas = {}; et = time.time()
    while len(todas) < total:
        lote = api.get_candles(par, 60, 1000, et)
        if not lote: break
        for v in lote: todas[v["from"]] = v
        et = min(v["from"] for v in lote) - 1
        if len(lote) < 2: break
    return [todas[k] for k in sorted(todas)]


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"]); api.connect(); api.change_balance("PRACTICE")
    print("Descargando USDJPY (esto tarda)...")
    velas = bajar(api, "USDJPY", 25000)
    closes = [float(v["close"]) for v in velas]
    n = len(closes)
    dias = (velas[-1]["from"] - velas[0]["from"]) / 86400
    rsi = rsi_series(closes)
    print(f"USDJPY {n} velas 1m (~{dias:.1f} dias) | BE={BE:.1%}\n")

    oos_tr = oos_w = 0                       # agregado out-of-sample (config elegida en train)
    fix_tr = fix_w = 0                       # baseline fijo 30/70 @10m sobre los mismos test
    is_wr_sum = 0.0; folds = 0
    print(f"{'fold':>4} {'train_pick':>18} {'IS_WR':>7} | {'OOS_n':>6} {'OOS_WR':>7}")
    i0 = WARMUP
    while i0 + TRAIN + TEST <= n:
        tr_s, tr_e = i0, i0 + TRAIN
        te_s, te_e = tr_e, tr_e + TEST
        # elegir mejor config en TRAIN
        best = None
        for (lo, hi) in GRID:
            for h in HORIZONS:
                t, w = evaluar(rsi, closes, lo, hi, h, tr_s, tr_e)
                if t < MIN_TRAIN_TRADES: continue
                wr = w / t
                if best is None or wr > best[0]:
                    best = (wr, lo, hi, h, t)
        if best is None:
            i0 += TEST; continue
        _, lo, hi, h, tn = best
        # aplicar al TEST (out-of-sample) y comparar con la config fija del bot
        t, w = evaluar(rsi, closes, lo, hi, h, te_s, te_e)
        ft, fw = evaluar(rsi, closes, 30, 70, 10, te_s, te_e)
        oos_tr += t; oos_w += w
        fix_tr += ft; fix_w += fw
        is_wr_sum += best[0]; folds += 1
        oos_wr = (w/t*100) if t else float('nan')
        print(f"{folds:>4} {f'{lo}/{hi} @{h}m':>18} {best[0]*100:6.1f}% | {t:>6} {oos_wr:6.1f}%")
        i0 += TEST

    print("\n===== RESUMEN =====")
    is_avg = is_wr_sum/folds*100 if folds else float('nan')
    oos_wr = oos_w/oos_tr*100 if oos_tr else float('nan')
    fix_wr = fix_w/fix_tr*100 if fix_tr else float('nan')
    p_oos = pval(oos_w, oos_tr); p_fix = pval(fix_w, fix_tr)
    ev = lambda wr: (wr/100)*PAYOUT - (1-wr/100)
    print(f"Folds: {folds}")
    print(f"IN-SAMPLE  (mejor de cada train, promedio): WR {is_avg:5.1f}%   <- lo que 'promete' el tuning")
    print(f"OUT-SAMPLE (esa eleccion en test):  n={oos_tr:5} WR {oos_wr:5.1f}%  vs BE {oos_wr-BE*100:+.1f}pt  p={p_oos:.3f}  EV/trade {ev(oos_wr)*100:+.1f}%")
    print(f"BASELINE fijo 30/70@10m en test:    n={fix_tr:5} WR {fix_wr:5.1f}%  vs BE {fix_wr-BE*100:+.1f}pt  p={p_fix:.3f}  EV/trade {ev(fix_wr)*100:+.1f}%")
    print(f"\nBrecha sobreajuste (IS - OOS): {is_avg-oos_wr:+.1f} puntos")


if __name__ == "__main__":
    main()
