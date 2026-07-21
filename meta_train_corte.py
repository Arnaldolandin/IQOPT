# meta_train_corte.py - Reentrena el meta-labeling con CORTE TEMPORAL ESTRICTO y
# EMBARGO CROSS-SECCIONAL, y evalua el filtro horario SOLO sobre el tramo held-out.
#
#   .venv314\Scripts\python.exe meta_train_corte.py [--test-frac 0.35] [--embargo-h 24]
#
# POR QUE EXISTE ESTE SCRIPT (leer antes de tocar nada):
#
# meta_train_iq.py valida con TimeSeriesSplit sobre las filas de los 50 pares ordenadas
# por timestamp. En cada instante hay ~50 señales simultaneas de pares muy correlacionados
# (EURUSD/EURGBP/EURAUD..., o APPLE/GOOGLE/AMAZON moviendose juntos). El corte train/test
# cae DENTRO de esos bloques simultaneos: el modelo entrena con EURUSD de las 14:35 y
# testea con EURGBP de las 14:35, que es casi la misma operacion. Eso infla el OOF
# (reporta 62-70% WR) mientras el walk-forward honesto da ~53-55% y el bot en vivo produce
# P centradas en 0.50. No es look-ahead temporal: es leakage cross-seccional.
#
# Aqui: un unico corte temporal, embargo de +-N horas alrededor del corte para que ningun
# bloque simultaneo cruce la frontera, y el modelo NUNCA ve el tramo de test.
#
# CONTINUIDAD: se exige que la vela de liquidacion este a exactamente NCON*300s de la de
# decision. Sin esto se cuentan gaps de fin de semana como opciones de 10 minutos.
import argparse
import glob
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CACHE_JSON = "cache_ohlc_5m"
FEATS = "cache_feats"
NCON = 2
PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def continuos(par):
    """Timestamps de decision cuya liquidacion esta a NCON*300s exactos."""
    p = os.path.join(CACHE_JSON, par + ".json")
    if not os.path.isfile(p):
        return None
    t = json.load(open(p, encoding="utf-8"))["times"]
    return {t[i] for i in range(len(t) - NCON) if t[i + NCON] - t[i] == NCON * 300}


def cargar(excluir):
    T, X, Y, P = [], [], [], []
    for f in sorted(glob.glob(os.path.join(FEATS, "*.npz"))):
        par = os.path.basename(f)[:-4].split("__")[-1]
        for suf in ("_COMBO_G0", "_COMBO", "_G0", "_FUGA"):
            par = par.replace(suf, "")
        if par in excluir:
            continue
        d = np.load(f)
        t, x, y = d["t"], d["X"], d["y"]
        fz = d["fz"] if "fz" in d else np.zeros(len(t), np.int8)
        if len(t) == 0:
            continue
        ok = continuos(par)
        m = (fz == 0)
        if ok is not None:
            m &= np.array([ts in ok for ts in t])
        if not m.any():
            continue
        T.append(t[m]); X.append(x[m]); Y.append(y[m])
        P.extend([par] * int(m.sum()))
        print(f"  {par:10s} {int(m.sum()):7d} filas (de {len(t)})", flush=True)
    return (np.concatenate(T), np.concatenate(X), np.concatenate(Y), np.array(P))


def wr_ev(y):
    if len(y) == 0:
        return None, 0.0
    w = float(y.mean())
    return w, w * PAYOUT - (1 - w)


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--embargo-h", type=float, default=24.0)
    ap.add_argument("--excluir", default="BTCUSD")
    ap.add_argument("--umbral", type=float, default=0.54)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--salida", default="models/meta_bbrev_corte.pkl")
    a = ap.parse_args()

    ex = {p.strip() for p in a.excluir.split(",") if p.strip()}
    print(f"[CARGA] {FEATS}/  excluyendo {sorted(ex)}")
    T, X, Y, P = cargar(ex)
    orden = np.argsort(T)
    T, X, Y, P = T[orden], X[orden], Y[orden], P[orden]
    f = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    print(f"\ntotal {len(T)} filas | {f(T[0])} -> {f(T[-1])} | WR base {100*Y.mean():.2f}%")

    # ---- corte temporal + embargo ----
    t_corte = float(np.quantile(T, 1 - a.test_frac))
    emb = a.embargo_h * 3600
    m_tr = T < (t_corte - emb)
    m_te = T > (t_corte + emb)
    n_emb = len(T) - int(m_tr.sum()) - int(m_te.sum())
    print(f"\n[CORTE] {f(t_corte)}  embargo +-{a.embargo_h:.0f}h")
    print(f"  train {int(m_tr.sum())} filas ({f(T[m_tr][0])} -> {f(T[m_tr][-1])})")
    print(f"  test  {int(m_te.sum())} filas ({f(T[m_te][0])} -> {f(T[m_te][-1])})")
    print(f"  embargadas {n_emb} filas (no se usan en ningun lado)")

    # ---- entrenamiento SOLO con train ----
    print("\n[FIT] entrenando solo con el tramo de train...")
    mdl = HistGradientBoostingClassifier(**PARAMS).fit(X[m_tr], Y[m_tr])
    p_te = mdl.predict_proba(X[m_te])[:, 1]
    y_te, t_te, par_te = Y[m_te], T[m_te], P[m_te]

    print(f"\n=== HELD-OUT (el modelo nunca vio estos datos) ===")
    w, e = wr_ev(y_te)
    print(f"  baseline (todas): WR {100*w:.2f}%  n={len(y_te)}  EV/op {e:+.4f}")
    print(f"  break-even = {100*BREAK_EVEN:.2f}%")
    print(f"{'umbral':>7} {'n':>8} {'%sel':>6} {'WR%':>7} {'EV/op':>8}")
    for thr in (0.52, 0.54, 0.56, 0.58, 0.60, 0.65):
        m = p_te >= thr
        if m.sum() == 0:
            print(f"{thr:>7} {0:>8}      -       -        -")
            continue
        w2, e2 = wr_ev(y_te[m])
        print(f"{thr:>7} {int(m.sum()):>8} {100*m.mean():>5.1f}% {100*w2:>6.2f} {e2:>+8.4f}")

    # ---- filtro horario SOLO sobre held-out ----
    m = p_te >= a.umbral
    filas = [{"ts": float(t_te[i]), "res": "win" if y_te[i] else "loss",
              "hora_utc": datetime.fromtimestamp(t_te[i], timezone.utc).hour,
              "par": par_te[i]}
             for i in range(len(y_te)) if m[i]]
    print(f"\n=== FILTRO HORARIO sobre held-out (P>={a.umbral}, n={len(filas)}) ===")
    if len(filas) < 500:
        print("  muestra insuficiente para evaluar 24 celdas horarias. No concluyo.")
        _guardar(mdl, a.salida)
        return

    por_hora = defaultdict(list)
    for r in filas:
        por_hora[r["hora_utc"]].append(r)
    print(f"{'h':>3} {'n':>6} {'WR%':>7}")
    for h in range(24):
        sub = por_hora.get(h, [])
        if sub:
            wr = sum(1 for r in sub if r["res"] == "win") / len(sub)
            print(f"{h:>3} {len(sub):>6} {100*wr:>6.2f}{' *' if wr >= BREAK_EVEN else ''}")

    import bt_horas_analizar as A
    w_f, n_f, w_b, n_b = A.walk_forward(filas, a.folds, a.min_n, 0.0)
    if w_f is None:
        print("\n  el filtro no selecciono ninguna hora -> sin operaciones OOS")
        _guardar(mdl, a.salida)
        return
    delta = (w_f - w_b) * 100
    print(f"\n  sin filtro : WR {100*w_b:.2f}%  n={n_b}")
    print(f"  con filtro : WR {100*w_f:.2f}%  n={n_f}")
    print(f"  mejora     : {delta:+.2f} pt")

    rnd = random.Random(12345)
    horas = [r["hora_utc"] for r in filas]
    nulos = []
    for _ in range(a.perm):
        rnd.shuffle(horas)
        for r, h in zip(filas, horas):
            r["_h"] = h
        w_p, _, w_bp, _ = A.walk_forward(filas, a.folds, a.min_n, 0.0, clave="_h")
        if w_p is not None:
            nulos.append((w_p - w_bp) * 100)
    if nulos:
        nulos.sort()
        pval = (sum(1 for x in nulos if x >= delta) + 1) / (len(nulos) + 1)
        print(f"\n  permutacion: nulo mediana {nulos[len(nulos)//2]:+.2f} pt, "
              f"p95 {nulos[int(0.95*(len(nulos)-1))]:+.2f} pt")
        print(f"  real {delta:+.2f} pt -> p = {pval:.4f}")
        print("  VEREDICTO:", "supera al azar" if (pval < 0.05 and delta > 0)
              else "NO se distingue del azar")
    _guardar(mdl, a.salida)


def _guardar(mdl, salida):
    import pickle
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    with open(salida, "wb") as fh:
        pickle.dump(mdl, fh)
    print(f"\n[SAVE] {salida}  (entrenado SOLO con el tramo de train)")


if __name__ == "__main__":
    main_()
