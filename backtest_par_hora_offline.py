# backtest_par_hora_offline.py - Analisis par-hora POOLED (offline, desde cache_closes/).
# Agrega el WR de TEST por hora-del-dia UTC sobre TODOS los pares (real y OTC por separado),
# en vez de por-par-hora, para evitar el multiple-testing que infla falsos ganadores.
# Config: MACD del config.json, velas 5m, expiry = h velas (bot real). Split 70/30.
#
#   .venv314\Scripts\python.exe backtest_par_hora_offline.py
import json, os, glob
from datetime import datetime, timezone
import numpy as np

CACHE_DIR = "cache_closes"
SPLIT = 0.70
WARMUP = 60


def ema(c, span):
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_cruces(closes, fast, slow, sig_p):
    if len(closes) < slow + sig_p + 2:
        return np.zeros(len(closes), dtype=int)
    macd_l = ema(closes, fast) - ema(closes, slow)
    sig_l = ema(macd_l, sig_p)
    out = np.zeros(len(closes), dtype=int)
    for i in range(1, len(closes)):
        if macd_l[i-1] <= sig_l[i-1] and macd_l[i] > sig_l[i]:
            out[i] = 1
        elif macd_l[i-1] >= sig_l[i-1] and macd_l[i] < sig_l[i]:
            out[i] = -1
    return out


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    fa, sl, si = cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"]
    tf_seg = cfg["operacion"]["timeframe_seg"]
    expiry_min = cfg["operacion"]["expiry_min"]
    h = max(1, round(expiry_min * 60 / tf_seg))

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    print(f"Cache: {len(files)} pares | MACD({fa},{sl},{si}) TF {tf_seg//60}m exp {expiry_min}m (h={h}) | pooled por hora UTC\n")

    # hour -> [w, n] en test, por grupo
    horas = {g: {hh: [0, 0] for hh in range(24)} for g in ("real", "otc")}
    n_real = n_otc = 0

    for fp in files:
        name = os.path.basename(fp)[:-5]
        grp = "otc" if "-OTC" in name else "real"
        try:
            d = json.load(open(fp, encoding="utf-8"))
            closes = np.asarray(d["closes"], float)
            times = d.get("times", [])
        except Exception:
            continue
        if len(closes) < 400 or len(times) != len(closes):
            continue
        if grp == "real": n_real += 1
        else: n_otc += 1
        n = len(closes); cut = int(n * SPLIT)
        sigs = macd_cruces(closes, fa, sl, si)
        for i in range(WARMUP, n - h):
            s = sigs[i]
            if s == 0 or i < cut:
                continue
            hh = datetime.fromtimestamp(times[i], tz=timezone.utc).hour
            gano = (closes[i + h] > closes[i]) if s == 1 else (closes[i + h] < closes[i])
            horas[grp][hh][1] += 1; horas[grp][hh][0] += 1 if gano else 0

    print(f"Pares: {n_real} reales + {n_otc} OTC\n")
    for g in ("real", "otc"):
        be = 53.5 if g == "real" else 54.1
        print("=" * 50)
        print(f"GRUPO {g.upper()} | WR test por hora UTC (BE ~{be}%)")
        print("=" * 50)
        print(f"{'horaUTC':>7s} {'WR':>7s} {'n':>7s}   {'(Chile=-4h)':>12s}")
        mejores = []
        for hh in range(24):
            w, nn = horas[g][hh]
            wr = (100.0 * w / nn) if nn else float("nan")
            mark = " *" if nn and wr >= be else ""
            ch = (hh - 4) % 24
            print(f"{hh:7d} {wr:6.2f}% {nn:7d}   {ch:02d}:00 Chile{mark}")
            if nn:
                mejores.append((wr, nn, hh))
        tot_w = sum(horas[g][hh][0] for hh in range(24))
        tot_n = sum(horas[g][hh][1] for hh in range(24))
        print(f"  TOTAL WR: {100.0*tot_w/tot_n:.2f}% (n={tot_n})")
        mejores.sort(reverse=True)
        print(f"  Top 3 horas: " + ", ".join(f"{hh}h={wr:.1f}%(n={nn})" for wr, nn, hh in mejores[:3]))
        print()

    print("Guardado backtest_par_hora_offline.json")
    json.dump(horas, open("backtest_par_hora_offline.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
