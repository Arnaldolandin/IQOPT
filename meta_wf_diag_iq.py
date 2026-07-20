# meta_wf_diag_iq.py - Diagnostico del walk-forward de umbral fijo. Uso:
#   python meta_wf_diag_iq.py [cache_dir]        (default cache_ohlc_5m)
#
# Por que existe: meta_wf_umbrales_iq.py dio 57.5%@0.56 en cache_ohlc_5m, pero el sweep
# out-of-fold en cache_ohlc daba 54.3% con el mismo umbral. El WF no puede salir MEJOR que
# el OOF; una de las dos medidas esta contaminada. Este script anade:
#   1) desglose POR ACTIVO (el edge se concentra en acciones/iliquidos? el bot opera FX)
#   2) % de velas CONGELADAS (fut==base): mercado cerrado -> precio repetido -> etiquetas basura
#   3) el mismo WF, para poder correrlo sobre cualquier caché y comparar manzanas con manzanas
import os, sys, json, glob, bisect, warnings
import numpy as np
from ml_features import extract_features
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
CACHE = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
NCON = 2; K = 2.0; PERIOD = 20; BE = 0.532; GAP = 1
UMBRAL = 0.56; NFOLDS = 5
SOLO_FX = os.environ.get("SOLO_FX") == "1"     # restringe a pares FX (lo que opera el bot)
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)
# Pares FX (lo que realmente opera el bot en regulares). El resto = acciones/indices/cripto.
FX = set("EURUSD USDJPY GBPUSD AUDUSD USDCHF USDCAD NZDUSD EURGBP EURJPY GBPJPY EURCHF "
         "AUDJPY CADJPY CHFJPY EURAUD EURCAD EURNZD GBPAUD GBPCAD GBPCHF GBPNZD AUDCAD "
         "AUDCHF AUDNZD CADCHF NZDCAD NZDCHF NZDJPY USDNOK USDSEK USDTRY USDZAR".split())


def velas_de(d):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    if "times" not in d:                       # sin timestamps no hay cronologia fiable
        raise ValueError("sin 'times'")
    t = d["times"]
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def mtf(velas, f=3):
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (velas[i:i + f] for i in range(0, len(velas) - f + 1, f))]


def build(V, Vmtf, tag):
    closes = [v[4] for v in V]; N = len(V); out = []; mep = [v[0] for v in Vmtf]
    for i in range(max(PERIOD, 60), N - NCON - GAP - 1):
        w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
        if sd <= 0:
            continue
        z = (closes[i] - sma) / sd
        side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = win[-1][0]
        k = bisect.bisect_right(mep, ep); cmtf = Vmtf[max(0, k - 60):k] if k >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        base = closes[i + GAP]; fut = closes[i + GAP + NCON]
        won = int(fut > base) if side == "CALL" else int(fut < base)
        congelada = int(fut == base)           # precio identico 2 velas despues = feed parado
        out.append((ep, fv, won, tag, congelada))
    return out


def resumen(nombre, wr_ops):
    if not wr_ops:
        return
    ws = np.array([w for w, _ in wr_ops]); ns = np.array([n for _, n in wr_ops])
    wp = float((ws * ns).sum() / ns.sum()); N = int(ns.sum())
    se = (wp * (1 - wp) / N) ** 0.5; z = (wp - BE) / se
    print(f"{nombre}: WR {wp*100:.2f}% ({N} ops) EV {ev(wp)*100:+.1f}% z={z:.2f} "
          f"{'SIGNIFICATIVO' if z > 1.64 else 'no signif.'}")


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    if SOLO_FX:                                # menos datos: aisla la pregunta y baja memoria
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in FX]
    print(f"[FASE] cargando {len(files)} ficheros de {CACHE}", flush=True)
    rows = []
    for idx, f in enumerate(files):
        if idx % 5 == 0:
            print(f"  [{idx}/{len(files)}] senales acumuladas: {len(rows)}", flush=True)
        tag = os.path.splitext(os.path.basename(f))[0]
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if len(d.get("close", [])) < 400:
            continue
        try:
            V = velas_de(d)
        except ValueError:
            print(f"[SKIP] {tag}: sin 'times'"); continue
        rows.extend(build(V, mtf(V), tag))
    print(f"[FASE] ordenando y armando matrices ({len(rows)} senales)", flush=True)
    rows.sort(key=lambda r: r[0])
    X = np.array([r[1] for r in rows], dtype=np.float32); y = np.array([r[2] for r in rows])
    tags = np.array([r[3] for r in rows]); frz = np.array([r[4] for r in rows]); n = len(X)
    rows.clear()                               # libera la lista de tuplas antes de entrenar
    print(f"[FASE] X = {X.shape} {X.dtype} ({X.nbytes/1e6:.0f} MB)", flush=True)
    es_fx = np.array([t in FX for t in tags])

    print(f"DIAGNOSTICO WF | {CACHE} | REGULARES ({len(files)} ficheros) | umbral {UMBRAL}")
    print(f"{n} senales | BE {BE*100:.1f}% | FX {es_fx.sum()} / no-FX {(~es_fx).sum()}")
    print(f"velas CONGELADAS (fut==base): {frz.mean()*100:.2f}% global | "
          f"FX {frz[es_fx].mean()*100 if es_fx.any() else 0:.2f}% | "
          f"no-FX {frz[~es_fx].mean()*100 if (~es_fx).any() else 0:.2f}%\n")

    # Walk-forward: entrena con el pasado, predice el 10% siguiente (ciego).
    P = np.full(n, np.nan)
    for k in range(NFOLDS):
        tr1 = int(n * (0.40 + 0.10 * k)); te1 = int(n * (0.50 + 0.10 * k))
        print(f"[FASE] fold {k+1}/{NFOLDS}: entrenando con {tr1} filas", flush=True)
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[:tr1], y[:tr1])
        P[tr1:te1] = m.predict_proba(X[tr1:te1])[:, 1]
        del m
    ev_m = ~np.isnan(P); sel = ev_m & (P >= UMBRAL)
    print(f"[GLOBAL]  base {y[ev_m].mean()*100:.2f}% -> seleccion {y[sel].mean()*100:.2f}% "
          f"({int(sel.sum())} ops)")
    resumen("  todo      ", [(float(y[sel].mean()), int(sel.sum()))])
    for etq, msk in (("solo FX   ", es_fx), ("solo no-FX", ~es_fx)):
        s = sel & msk
        if s.sum():
            resumen(f"  {etq}", [(float(y[s].mean()), int(s.sum()))])

    print(f"\n[POR ACTIVO] umbral {UMBRAL}, minimo 100 ops, ordenado por WR")
    filas = []
    for t in sorted(set(tags)):
        s = sel & (tags == t)
        if s.sum() >= 100:
            filas.append((float(y[s].mean()), int(s.sum()), t, float(frz[tags == t].mean())))
    for wr, nn, t, fz in sorted(filas, reverse=True):
        marca = "FX " if t in FX else "   "
        print(f"  {marca}{t:<12} {wr*100:6.2f}% {nn:6d} ops  EV {ev(wr)*100:+6.1f}%  congeladas {fz*100:5.2f}%")
    if not filas:
        print("  (ningun activo con >=100 ops)")


if __name__ == "__main__":
    main()
