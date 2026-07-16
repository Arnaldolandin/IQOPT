# backtest_horas_macd.py - Mejores HORAS por par para la estrategia MACD + filtro EMA.
# Estrategia EXACTA del bot: cruce MACD(fast,slow,signal) + EMA(macd_ema) a favor de tendencia
# (+ ATR opcional). Params tomados de config.json. Rigor: train/test con OOS FIJO de 60 dias,
# hora robusta = buena en train Y en test, con control de azar.
#   python backtest_horas_macd.py [cache_dir] [test_days]

import json, math, os, glob, sys
import numpy as np

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
H = 2                         # holdear 2 barras (a 5m = 10min de expiracion)
CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
TEST_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MIN_TR_HORA = 20             # min senales por hora en train para postular
MIN_TE_HORA = 10             # min senales por hora en test para validar

CFG = json.load(open("config.json", encoding="utf-8"))["operacion"]
MFAST = CFG.get("macd_fast", 6)
MSLOW = CFG.get("macd_slow", 13)
MSIG = CFG.get("macd_signal", 5)
EMA_P = CFG.get("macd_ema", 50)
MIN_ATR = CFG.get("min_atr", 0.0)
ATR_P = CFG.get("atr_period", 14)
WARMUP = max(EMA_P * 5, MSLOW + MSIG + 2, 250)


def ema(c, span):
    c = np.asarray(c, float); a = 2.0 / (span + 1); out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def atr_pct(h, l, c, period):
    n = len(c); tr = np.empty(n); tr[0] = h[0] - l[0]; pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    cs = np.cumsum(tr); a = np.full(n, np.nan); a[period:] = (cs[period:] - cs[:-period]) / period
    pr = np.abs(c); pr[pr == 0] = 1.0
    return np.nan_to_num(a / pr, nan=0.0)


def signals(c, h, l):
    n = len(c)
    ml = ema(c, MFAST) - ema(c, MSLOW); sl = ema(ml, MSIG); diff = ml - sl
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[1:] = (diff[:-1] <= 0) & (diff[1:] > 0)
    cd[1:] = (diff[:-1] >= 0) & (diff[1:] < 0)
    if EMA_P:
        ev = ema(c, EMA_P)
        cu &= c > ev; cd &= c < ev
    if MIN_ATR > 0:
        vok = atr_pct(h, l, c, ATR_P) >= MIN_ATR
        cu &= vok; cd &= vok
    cu[:WARMUP] = False; cd[:WARMUP] = False
    return cu, cd


def ev_de_wr(wr):
    return wr * PAYOUT - (1 - wr)


def prob_be(n):
    if n <= 0:
        return 0.0
    z = (BE - 0.5) / math.sqrt(0.25 / n)
    return 0.5 * math.erfc(z / math.sqrt(2))


def wr(seg):
    return seg[1] / seg[0] if seg[0] else 0.0


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if not files:
        print(f"No hay datos en {CACHE_DIR}/ (baja con download_ohlc_5m.py)"); return

    hour_pool = {h: {"tr": [0, 0], "te": [0, 0]} for h in range(24)}
    por_par = {}
    spans = []
    procesados = 0
    for path in files:
        par = os.path.basename(path)[:-5].replace("_", "/")
        try:
            d = json.load(open(path, encoding="utf-8"))
            c = np.asarray(d["close"], float); hi = np.asarray(d.get("high", d["close"]), float)
            lo = np.asarray(d.get("low", d["close"]), float); t = np.asarray(d["times"], float)
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
        hora = ((t % 86400) // 3600).astype(int)
        cu, cd = signals(c, hi, lo)

        def eval_h(lo_i, hi_i, hh):
            m = np.zeros(n, bool); m[lo_i:hi_i] = True
            hm = hora == hh
            cs = cu & m & hm; ps = cd & m & hm
            tr = int(cs.sum() + ps.sum()); wins = int((wc & cs).sum() + (wp & ps).sum())
            return tr, wins

        robustas = []
        detalle = []
        for hh in range(24):
            ttr, wtr = eval_h(WARMUP, max(WARMUP, split - H), hh)
            tte, wte = eval_h(split, n - H, hh)
            wr_tr = (wtr / ttr) if ttr else 0.0
            wr_te = (wte / tte) if tte else 0.0
            detalle.append({"h": hh, "tr_tr": ttr, "wr_tr": round(wr_tr, 4),
                            "tr_te": tte, "wr_te": round(wr_te, 4)})
            hp = hour_pool[hh]
            hp["tr"][0] += ttr; hp["tr"][1] += wtr
            hp["te"][0] += tte; hp["te"][1] += wte
            if ttr >= MIN_TR_HORA and wr_tr > BE and tte >= MIN_TE_HORA and wr_te > BE:
                robustas.append(hh)
        por_par[par] = {"robustas": robustas, "detalle": detalle}

    horas_por_par = {p: d["robustas"] for p, d in por_par.items() if d["robustas"]}

    W = 96
    print("=" * W)
    print(f"MEJORES HORAS POR PAR  |  MACD({MFAST},{MSLOW},{MSIG}) + EMA{EMA_P}" +
          (f" + ATR>={MIN_ATR}" if MIN_ATR else "") + f"  |  {CACHE_DIR}")
    print(f"{procesados} pares REALES | span mediano {np.median(spans):.0f}d | H={H} ({H*5}min) | "
          f"payout {PAYOUT:.0%} (BE {BE:.1%}) | OOS test = ultimos {TEST_DAYS}d")
    print("=" * W)

    # Horas pooled (todos los pares)
    print("\n[1] HORAS DEL DIA (UTC) — pooled sobre todos los pares reales")
    tz = -4
    print(f"{'UTC':>3} {'CL':>4} | {'TRAINtr':>8} {'wr':>6} | {'TESTtr':>8} {'wr':>6} {'ev':>7}")
    print("-" * W)
    horas_oos = []
    for hh in range(24):
        wtr = wr(hour_pool[hh]["tr"]); wte = wr(hour_pool[hh]["te"])
        mark = ""
        if hour_pool[hh]["te"][0] >= 30:
            if wte > BE:
                mark = "  <OOS+"; horas_oos.append(hh)
        print(f"{hh:>3} {(hh+tz)%24:>3}h | {hour_pool[hh]['tr'][0]:>8} {wtr*100:>5.1f}% | "
              f"{hour_pool[hh]['te'][0]:>8} {wte*100:>5.1f}% {ev_de_wr(wte)*100:>+6.1f}%{mark}")
    print(f"  -> Horas UTC con WR>BE OUT-OF-SAMPLE (pooled, n>=30): {horas_oos if horas_oos else 'NINGUNA'}")

    # Por par: horas robustas (train Y test)
    print(f"\n[2] HORAS ROBUSTAS POR PAR (buenas en TRAIN *y* validadas en TEST)")
    con = {p: h for p, h in horas_por_par.items()}
    print(f"  Pares con >=1 hora robusta: {len(con)} de {procesados}")
    for p, hrs in sorted(con.items(), key=lambda x: -len(x[1]))[:25]:
        hrs_cl = ", ".join(f"{h}({(h+tz)%24}CL)" for h in hrs)
        print(f"     {p:>10}: {hrs_cl}")

    # Control de azar
    total_evaluadas = 0
    esperado = 0.0
    for p, d in por_par.items():
        for x in d["detalle"]:
            if x["tr_tr"] >= MIN_TR_HORA and x["wr_tr"] > BE and x["tr_te"] >= MIN_TE_HORA:
                total_evaluadas += 1
                esperado += prob_be(x["tr_te"])
    obtenidas = sum(len(h) for h in horas_por_par.values())
    print(f"\n[3] CONTROL DE AZAR")
    print(f"  Pares-hora robustos hallados: {obtenidas}")
    print(f"  Esperados por AZAR (de los que pasan train+muestra): ~{esperado:.0f} de {total_evaluadas} candidatos")
    veredicto = "dentro del ruido" if obtenidas <= esperado * 1.3 else "POR ENCIMA del azar"
    print(f"  -> {veredicto}")

    out = {
        "config": {"macd": [MFAST, MSLOW, MSIG], "ema": EMA_P, "min_atr": MIN_ATR,
                   "H": H, "payout": PAYOUT, "be": round(BE, 4), "test_days": TEST_DAYS,
                   "cache_dir": CACHE_DIR, "activos": procesados,
                   "span_mediano_d": round(float(np.median(spans)), 0)},
        "horas_pooled": {str(h): {"train_tr": hour_pool[h]["tr"][0], "train_wr": round(wr(hour_pool[h]["tr"]), 4),
                                  "test_tr": hour_pool[h]["te"][0], "test_wr": round(wr(hour_pool[h]["te"]), 4)}
                         for h in range(24)},
        "horas_por_par_robustas": horas_por_par,
        "azar": {"obtenidas": obtenidas, "esperado": round(esperado, 1), "candidatos": total_evaluadas},
        "por_par_detalle": {p: d["detalle"] for p, d in por_par.items()},
    }
    json.dump(out, open("backtest_horas_macd.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nJSON: backtest_horas_macd.json")


if __name__ == "__main__":
    main()
