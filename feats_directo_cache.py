# feats_directo_cache.py - Dataset del modelo DIRECTO: una fila por VELA (no solo por
# disparo del primario), con etiquetas para horizonte 1 y 2.
#
#   .venv314\Scripts\python.exe feats_directo_cache.py [--jobs 8] [--pares N]
#
# Diferencia clave con meta_feats_cache.py: alli solo se guardaban las velas donde bbrev
# o stoch disparaban (676k filas). Aqui se evalua CADA vela (~1.9M filas), porque el
# modelo directo no tiene primario que le acote el menu.
#
# Etiquetas: y1 = close[i+1] > close[i]; y2 = close[i+2] > close[i].
# Se guardan ademas los flags de continuidad (ok1/ok2): la opcion solo existe si la vela
# de liquidacion esta a h*300s exactos de la de decision. Sin eso se cuentan gaps de fin
# de semana como opciones de 5 o 10 minutos.
# Los empates (close igual) se marcan en fz1/fz2: feed parado, no son operables.
import argparse
import glob
import json
import os
import time

import numpy as np

from ml_features import mtf_hasta
from ml_features_v2 import extract_features_v2, FEATURE_NAMES_V2

CACHE = "cache_ohlc_5m"
OUT = "cache_feats_directo"
VENTANA = 100      # velas que ve extract_features
MTF_VENTANA = 180  # 3 x 60 barras MTF
INICIO = 300       # margen para que las EMA converjan


def velas_de(d):
    o, h, l, c, t = d["open"], d["high"], d["low"], d["close"], d["times"]
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])]
            for i in range(len(c))]


def procesar(par):
    dst = os.path.join(OUT, par + ".npz")
    if os.path.exists(dst):
        return par, -1        # ya hecho (reanudable)
    with open(os.path.join(CACHE, par + ".json"), encoding="utf-8") as f:
        V = velas_de(json.load(f))
    n = len(V)
    ts, fs, y1, y2, ok1, ok2, fz1, fz2 = [], [], [], [], [], [], [], []
    for i in range(INICIO, n - 2):
        cmtf = mtf_hasta(V[max(0, i - MTF_VENTANA + 1):i + 1], factor=3, max_barras=60)
        cmtf = cmtf if len(cmtf) >= 2 else None
        fv, _ = extract_features_v2(V[max(0, i - VENTANA + 1):i + 1], velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        c0, c1, c2 = V[i][4], V[i + 1][4], V[i + 2][4]
        t0, t1, t2 = V[i][0], V[i + 1][0], V[i + 2][0]
        ts.append(t0)
        fs.append(fv.astype(np.float32))
        y1.append(int(c1 > c0)); y2.append(int(c2 > c0))
        ok1.append(int(t1 - t0 == 300)); ok2.append(int(t2 - t0 == 600))
        fz1.append(int(c1 == c0)); fz2.append(int(c2 == c0))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        dst, t=np.array(ts, np.float64), X=np.array(fs, np.float32),
        y1=np.array(y1, np.int8), y2=np.array(y2, np.int8),
        ok1=np.array(ok1, np.int8), ok2=np.array(ok2, np.int8),
        fz1=np.array(fz1, np.int8), fz2=np.array(fz2, np.int8))
    return par, len(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pares", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pares = [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if a.pares:
        pares = pares[:a.pares]
    print(f"[DIRECTO] {len(pares)} pares | {len(FEATURE_NAMES_V2)} features | destino {OUT}/",
          flush=True)

    t0 = time.time()
    tot = 0
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for k, (par, n) in enumerate(ex.map(procesar, pares), 1):
                tot += max(n, 0)
                estado = "ya estaba" if n < 0 else f"{n} filas"
                print(f"[{k}/{len(pares)}] {par}: {estado} "
                      f"(acum {tot}, {time.time()-t0:.0f}s)", flush=True)
    else:
        for k, par in enumerate(pares, 1):
            par, n = procesar(par)
            tot += max(n, 0)
            print(f"[{k}/{len(pares)}] {par}: {n} filas ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nLISTO: {tot} filas en {time.time()-t0:.0f}s -> {OUT}/")


if __name__ == "__main__":
    main()
