# meta_feats_cache.py - Precomputa las features del meta-labeling por activo y las cachea
# en cache_feats/<cache>__<activo>.npz. REANUDABLE: salta lo ya hecho.
#
# Por que: construir las señales cuesta ~50s por activo (~45 min para 50 pares) y los runs
# largos mueren en este entorno. Cacheando por activo, cada ejecucion avanza lo que pueda y
# la siguiente continua. Despues, el walk-forward sobre los .npz tarda segundos.
#   python meta_feats_cache.py [cache_dir] [--fx]
import os, sys, json, glob, bisect, time, warnings
import numpy as np
from ml_features import extract_features, mtf_hasta
warnings.filterwarnings("ignore")
CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
SOLO_FX = "--fx" in sys.argv
MTF_ESTRICTO = "--fuga" not in sys.argv        # --fuga = reproduce el bug de look-ahead
# --combo: 2do primario stoch %K si bbrev no dispara (como main.py con usar_combo).
# --gap0: etiqueta desde la vela de decision (convencion de meta_train_iq.py) en vez de
# la siguiente. Para ENTRENAR el modelo de produccion hay que usar ambos, o el modelo
# no ve los setups de stoch que el bot si opera.
COMBO = "--combo" in sys.argv
GAP0 = "--gap0" in sys.argv
OUT = "cache_feats"
SUF = ("" if MTF_ESTRICTO else "_FUGA") + ("_COMBO" if COMBO else "") + ("_G0" if GAP0 else "")
NCON = 2; K = 2.0; PERIOD = 20; GAP = 1
FX = set("EURUSD USDJPY GBPUSD AUDUSD USDCHF USDCAD NZDUSD EURGBP EURJPY GBPJPY EURCHF "
         "AUDJPY CADJPY CHFJPY EURAUD EURCAD EURNZD GBPAUD GBPCAD GBPCHF GBPNZD AUDCAD "
         "AUDCHF AUDNZD CADCHF NZDCAD NZDCHF NZDJPY USDNOK USDSEK USDTRY USDZAR".split())


def velas_de(d):
    o, h, l, c, t = d["open"], d["high"], d["low"], d["close"], d["times"]
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    """Devuelve (barras, t_cierre). t_cierre = timestamp de la ULTIMA vela del grupo.

    Ojo: la barra lleva como t[0] el inicio del grupo, pero agrega high/low/close de
    las f velas. Seleccionar por el INICIO (como se hacia antes) mete barras que
    contienen velas POSTERIORES a la de decision -> look-ahead en rsi_mtf/adx_mtf.
    Por eso devolvemos aparte el cierre y filtramos por el.
    """
    barras, cierres = [], []
    for i in range(0, len(velas) - f + 1, f):
        g = velas[i:i + f]
        barras.append([g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]])
        cierres.append(g[-1][0])
    return barras, cierres


def build(V, Vmtf, mep):
    """mep = tiempos de CIERRE de las barras MTF (ver mtf()). Con MTF_ESTRICTO solo se
    usan barras ya cerradas en la vela de decision; sin el, se reproduce el bug para
    poder medir cuanto valia la fuga."""
    closes = [v[4] for v in V]; N = len(V)
    highs = [v[2] for v in V]; lows = [v[3] for v in V]
    gap = 0 if GAP0 else GAP
    if not MTF_ESTRICTO:
        mep = [b[0] for b in Vmtf]             # comportamiento antiguo (con look-ahead)
    ts, fs, ys, fz = [], [], [], []
    for i in range(max(PERIOD, 60), N - NCON - GAP - 1):
        w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
        if sd <= 0:
            continue
        z = (closes[i] - sma) / sd
        side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None and COMBO:             # 2do primario: stoch %K extremo
            pk = 14; hh = max(highs[i - pk + 1:i + 1]); ll = min(lows[i - pk + 1:i + 1])
            kk = 100 * (closes[i] - ll) / (hh - ll) if hh > ll else 50.0
            side = "CALL" if kk < 20 else "PUT" if kk > 80 else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = win[-1][0]
        if MTF_ESTRICTO:                       # anclado a la vela i (igual que el bot)
            cmtf = mtf_hasta(V[max(0, i - 179):i + 1], factor=3, max_barras=60)
            cmtf = cmtf if len(cmtf) >= 2 else None
        else:                                  # bug historico: barras por su INICIO
            k = bisect.bisect_right(mep, ep); cmtf = Vmtf[max(0, k - 60):k] if k >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        base = closes[i + gap]; fut = closes[i + gap + NCON]
        ts.append(ep); fs.append(fv)
        ys.append(int(fut > base) if side == "CALL" else int(fut < base))
        fz.append(int(fut == base))            # feed parado (mercado cerrado)
    return ts, fs, ys, fz


def main():
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if SOLO_FX:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in FX]
    print(f"[CACHE FEATS] {CACHE} | {len(files)} activos | destino {OUT}/", flush=True)
    hechos = 0
    for f in files:
        tag = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(OUT, f"{os.path.basename(CACHE)}{SUF}__{tag}.npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        try:
            d = json.load(open(f, encoding="utf-8"))
            if len(d.get("close", [])) < 400 or "times" not in d:
                np.savez_compressed(dst, t=np.zeros(0), X=np.zeros((0, 1), np.float32),
                                    y=np.zeros(0, np.int8), fz=np.zeros(0, np.int8))
                print(f"  {tag}: descartado (pocas velas o sin times)", flush=True); continue
            V = velas_de(d); barras, cierres = mtf(V); ts, fs, ys, fz = build(V, barras, cierres)
        except Exception as e:
            print(f"  {tag}: ERROR {e}", flush=True); continue
        np.savez_compressed(dst, t=np.array(ts, np.float64),
                            X=np.array(fs, np.float32), y=np.array(ys, np.int8),
                            fz=np.array(fz, np.int8))
        hechos += 1
        print(f"  {tag}: {len(ys)} senales en {time.time()-t0:.0f}s -> {os.path.basename(dst)}", flush=True)
    pend = sum(1 for f in files
               if not os.path.exists(os.path.join(
                   OUT, f"{os.path.basename(CACHE)}{SUF}__{os.path.splitext(os.path.basename(f))[0]}.npz")))
    print(f"[FIN] cacheados ahora: {hechos} | pendientes: {pend}", flush=True)


if __name__ == "__main__":
    main()
