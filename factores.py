# factores.py - Descompone el movimiento de cada activo en SISTEMICO e IDIOSINCRATICO.
#
# EL PROBLEMA QUE RESUELVE: hoy el modelo ve que ETHUSD cayo 2 ATR y no puede saber si
#   (a) todo el mercado cripto cayo  -> movimiento sistemico, tiende a CONTINUAR
#   (b) solo ETH se movio            -> idiosincratico, tiende a REVERTIR
# Son situaciones opuestas y para el son identicas. Toda la informacion que tiene hoy
# sale de la misma serie del mismo activo; esto es lo unico que agrega informacion NUEVA.
#
# COMO: por cada vela se calcula el retorno medio de la FAMILIA del activo (excluyendo
# al activo mismo, o el factor estaria contaminado por lo que queremos explicar) y se
# obtiene el residuo = retorno propio - beta * retorno de la familia.
#
# El resultado se cachea en factores_<cache>.npz: {times, <activo>_sis, <activo>_res, ...}
import json
import os

import numpy as np

FAMILIAS = {
    "fx": ("EURUSD USDJPY GBPUSD AUDUSD USDCHF USDCAD NZDUSD EURGBP EURJPY GBPJPY "
           "EURCHF AUDJPY CHFJPY EURAUD EURCAD EURNZD GBPAUD GBPCAD GBPCHF GBPNZD "
           "AUDCAD AUDCHF AUDNZD CADCHF NZDCAD NZDJPY USDBRL").split(),
    "cripto": "BTCUSD ETHUSD XRPUSD".split(),
    "metales": "XAUUSD".split(),
    "acciones": ("APPLE MSFT GOOGLE AMAZON FACEBOOK TESLA INTEL CISCO NIKE MCDON COKE "
                 "CITI JPM GS MORSTAN ALIBABA BAIDU SNAP AIG").split(),
    "energia": "USOUSD".split(),
}
FAMILIA_DE = {a: f for f, act in FAMILIAS.items() for a in act}
VENTANA_BETA = 60      # velas para estimar la beta contra la familia


def familia_de(par):
    return FAMILIA_DE.get(par)


def _cargar(cache, par):
    p = os.path.join(cache, par + ".json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return np.array(d["times"], np.int64), np.array(d["close"], np.float64)


def construir(cache, salida=None):
    """Devuelve dict {par: (times, sistemico, residuo)}.

    sistemico = retorno medio de la familia SIN el propio activo, en la misma vela.
    residuo   = retorno propio - beta_movil * sistemico.
    """
    salida = salida or f"factores_{os.path.basename(cache)}.npz"
    series = {}
    for par in FAMILIA_DE:
        r = _cargar(cache, par)
        if r is not None and len(r[0]) > VENTANA_BETA + 10:
            t, c = r
            ret = np.zeros(len(c))
            ret[1:] = np.diff(c) / c[:-1]          # retorno relativo, comparable entre activos
            series[par] = (t, ret)
    print(f"[FACTORES] {len(series)} activos con datos en {cache}")

    out = {}
    for par, (t, ret) in series.items():
        fam = FAMILIA_DE[par]
        pares_fam = [q for q in series if FAMILIA_DE[q] == fam and q != par]
        if not pares_fam:
            print(f"  {par}: familia '{fam}' sin otros miembros -> factor nulo")
            out[par] = (t, np.zeros(len(t)), ret.copy())
            continue

        # alinear por timestamp: el indice de cada vela en cada par de la familia
        idx = {q: {int(tt): k for k, tt in enumerate(series[q][0])} for q in pares_fam}
        sis = np.zeros(len(t))
        for k, tt in enumerate(t):
            vals = [series[q][1][idx[q][int(tt)]] for q in pares_fam if int(tt) in idx[q]]
            if vals:
                sis[k] = float(np.mean(vals))

        # beta movil: cuanto del movimiento propio explica la familia. Se calcula solo
        # con datos PASADOS (ventana que termina en k-1) para no filtrar futuro.
        beta = np.zeros(len(t))
        for k in range(VENTANA_BETA, len(t)):
            a = sis[k - VENTANA_BETA:k]
            b = ret[k - VENTANA_BETA:k]
            va = float(np.dot(a, a))
            beta[k] = float(np.dot(a, b) / va) if va > 1e-18 else 0.0
        res = ret - beta * sis
        out[par] = (t, sis, res)
        print(f"  {par:9s} fam={fam:9s} miembros={len(pares_fam):2d} "
              f"beta_med={np.median(beta[VENTANA_BETA:]):+.2f}")

    datos = {}
    for par, (t, sis, res) in out.items():
        datos[f"{par}__t"] = t
        datos[f"{par}__sis"] = sis.astype(np.float32)
        datos[f"{par}__res"] = res.astype(np.float32)
    np.savez_compressed(salida, **datos)
    print(f"[SAVE] {salida}")
    return out


def cargar_factores(path):
    """{par: (times, sistemico, residuo)} desde el npz."""
    z = np.load(path)
    pares = sorted({k.split("__")[0] for k in z.files})
    return {p: (z[f"{p}__t"], z[f"{p}__sis"], z[f"{p}__res"]) for p in pares}


if __name__ == "__main__":
    import sys
    construir(sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m")
