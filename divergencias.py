# divergencias.py - Testea DIVERGENCIAS (RSI y MACD) y RUPTURAS (breakouts de rango).
# Divergencia bajista: precio hace maximo mas alto pero el indicador hace maximo mas bajo -> PUT.
# Divergencia alcista: precio minimo mas bajo pero indicador minimo mas alto -> CALL.
# Ruptura: cierre supera el maximo (o rompe el minimo) de las ultimas N barras.
# Entrada realista: en la barra de CONFIRMACION del swing (swing + w), no en el swing (que necesita futuro).
# Reporta WR/EV estandar y OPERABLE (entrada +1 barra, sin rebote), train/test OOS.
#   python divergencias.py [cache_dir]

import json, glob, os, sys
import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
H = 2
BE = 1.0 / 1.87
W = 3           # ventana de swing
NBRK = 20       # ventana de ruptura
TEST_DAYS = 60


def ema(c, s):
    a = 2.0 / (s + 1); o = np.copy(c)
    for i in range(1, len(c)):
        o[i] = a * c[i] + (1 - a) * o[i - 1]
    return o


def rsi(c, p=14):
    n = len(c); d = np.zeros(n); d[1:] = c[1:] - c[:-1]
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    cs = np.cumsum(up); cd = np.cumsum(dn); au = np.full(n, np.nan); ad = np.full(n, np.nan)
    au[p:] = (cs[p:] - cs[:-p]) / p; ad[p:] = (cd[p:] - cd[:-p]) / p
    rs = au / np.where(ad == 0, np.nan, ad); return np.nan_to_num(100 - 100 / (1 + rs), nan=50.0)


def swings(c, w):
    n = len(c); hi = np.zeros(n, bool); lo = np.zeros(n, bool)
    for i in range(w, n - w):
        seg = c[i - w:i + w + 1]
        if c[i] == seg.max() and c[i] > c[i - 1]:
            hi[i] = True
        if c[i] == seg.min() and c[i] < c[i - 1]:
            lo[i] = True
    return hi, lo


def señales_divergencia(c, ind):
    """Devuelve dict indice_confirmacion -> lado, para divergencias de `ind` vs precio."""
    hi, lo = swings(c, W)
    out = {}
    # bajista: dos swing highs consecutivos, precio sube e indicador baja
    idxh = np.where(hi)[0]
    for a, b in zip(idxh[:-1], idxh[1:]):
        if c[b] > c[a] and ind[b] < ind[a]:
            out[b + W] = "put"          # confirmacion W barras despues del swing
    idxl = np.where(lo)[0]
    for a, b in zip(idxl[:-1], idxl[1:]):
        if c[b] < c[a] and ind[b] > ind[a]:
            out[b + W] = "call"
    return out


def rupturas(c):
    """Breakout de rango: cierre > max(N) -> call (continuacion), < min(N) -> put."""
    n = len(c); out = {}
    for i in range(NBRK, n):
        hh = c[i - NBRK:i].max(); ll = c[i - NBRK:i].min()
        if c[i] > hh:
            out[i] = "call"
        elif c[i] < ll:
            out[i] = "put"
    return out


def evaluar(files, generador, nombre):
    tr = [0, 0, 0]  # ops, wins_std, wins_gap
    te = [0, 0, 0]
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        n = len(c)
        if n < 400:
            continue
        split = int(np.searchsorted(t, t[-1] - TEST_DAYS * 86400))
        sig = generador(c)
        for i, lado in sig.items():
            if i + 1 + H >= n:
                continue
            # estandar: entrada en i
            gs = (c[i + H] > c[i]) if lado == "call" else (c[i + H] < c[i])
            # operable: entrada en i+1
            gg = (c[i + 1 + H] > c[i + 1]) if lado == "call" else (c[i + 1 + H] < c[i + 1])
            box = tr if i < split else te
            box[0] += 1; box[1] += int(gs); box[2] += int(gg)
    def wr(x, k):
        return x[k] / x[0] * 100 if x[0] else 0
    ev = lambda w: (w / 100 * 0.87 - (1 - w / 100)) * 100
    print(f"\n{nombre}")
    print(f"  TRAIN: {tr[0]:>6} ops | std {wr(tr,1):.2f}% | operable {wr(tr,2):.2f}%")
    print(f"  TEST : {te[0]:>6} ops | std {wr(te,1):.2f}% (EV {ev(wr(te,1)):+.1f}%) | "
          f"operable {wr(te,2):.2f}% (EV {ev(wr(te,2)):+.1f}%)")
    return wr(te, 2)


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    print("=" * 80)
    print(f"DIVERGENCIAS Y RUPTURAS  |  {CACHE}  |  {len(files)} pares REALES  |  break-even {BE*100:.1f}%")
    print("  (std = entrada en la barra; operable = entrada +1, sin rebote bid-ask)")
    print("=" * 80)

    resultados = {}
    resultados["RSI div"] = evaluar(files, lambda c: señales_divergencia(c, rsi(c, 14)), "DIVERGENCIA RSI(14)")
    resultados["MACD div"] = evaluar(files, lambda c: señales_divergencia(c, ema(c, 6) - ema(c, 13)), "DIVERGENCIA MACD hist")
    resultados["Ruptura"] = evaluar(files, rupturas, f"RUPTURA de rango {NBRK} (breakout continuacion)")
    # ruptura invertida (fade del breakout = reversion)
    def rup_fade(c):
        return {i: ("put" if l == "call" else "call") for i, l in rupturas(c).items()}
    resultados["Ruptura-fade"] = evaluar(files, rup_fade, f"RUPTURA-FADE (reversion del breakout)")

    print("\n" + "=" * 80)
    print(f"RESUMEN (WR operable OOS, break-even {BE*100:.1f}%):")
    for k, v in resultados.items():
        print(f"  {k:>14}: {v:.2f}%  {'>BE' if v > BE*100 else '(pierde)'}")
    print("\nRecordatorio: el techo lo pone el modelo-libre (signo ~49.5% operable). Ningun patron")
    print("de precio puede superar sistematicamente lo que la serie no contiene.")


if __name__ == "__main__":
    main()
