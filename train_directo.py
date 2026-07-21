# train_directo.py - Entrena y evalua el modelo DIRECTO (sin primario) con el mismo
# arnes limpio que uso meta_train_corte.py.
#
#   .venv314\Scripts\python.exe train_directo.py [--horizonte 2] [--max-train 0]
#
# El modelo predice P(sube) sobre CADA vela. Regla de trading simetrica:
#   CALL si p >= thr ; PUT si p <= 1-thr ; nada en el medio.
#
# PAYOUT SEGUN HORIZONTE (esto no es un detalle, cambia el liston):
#   h=1 -> expiry 5m  -> instrumento turbo  (~83%) -> break-even 54.64%
#   h=2 -> expiry 10m -> instrumento binary (~87%) -> break-even 53.48%
#
# Controles heredados de la sesion del 2026-07-21:
#   - corte temporal estricto + embargo +-24h (mata la fuga cross-seccional)
#   - solo filas con continuidad (ok_h==1): la vela de liquidacion a h*300s exactos
#   - sin empates (fz_h==0): feed parado, no operable
#   - BTCUSD excluido: 76 señales historicas, 0 entradas logradas
#   - resultados separados dentro/fuera de 20-22 UTC (rollover), porque ahi IQ
#     rechaza ordenes y el edge medido no es ejecutable
import argparse
import glob
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FEATS = "cache_feats_directo"
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)
PAYOUT = {1: 0.83, 2: 0.87}
ROLLOVER = (20, 21, 22)


def be(payout):
    return 1.0 / (1.0 + payout)


def cargar(h, excluir):
    T, X, Y, P = [], [], [], []
    for f in sorted(glob.glob(os.path.join(FEATS, "*.npz"))):
        par = os.path.basename(f)[:-4]
        if par in excluir:
            continue
        d = np.load(f)
        m = (d[f"ok{h}"] == 1) & (d[f"fz{h}"] == 0)
        if not m.any():
            continue
        T.append(d["t"][m]); X.append(d["X"][m]); Y.append(d[f"y{h}"][m])
        P.extend([par] * int(m.sum()))
    return (np.concatenate(T), np.concatenate(X), np.concatenate(Y).astype(np.int8),
            np.array(P))


def evaluar(p, y, hor, payout, etq):
    """Regla simetrica. Devuelve nada, imprime la tabla."""
    B = be(payout)
    roll = np.isin(hor, ROLLOVER)
    print(f"\n=== {etq} | payout {payout:.0%} | break-even {100*B:.2f}% ===")
    print(f"{'thr':>6} {'zona':>14} {'n':>9} {'%':>6} {'WR':>8} {'EV/op':>9}")
    for thr in (0.52, 0.54, 0.56, 0.58, 0.60, 0.65):
        sel = (p >= thr) | (p <= 1 - thr)
        lado_call = p >= thr
        gano = np.where(lado_call, y == 1, y == 0)
        for nombre, m in (("TODO", sel),
                          ("solo 20-22", sel & roll),
                          ("SIN rollover", sel & ~roll)):
            n = int(m.sum())
            if n == 0:
                print(f"{thr:>6} {nombre:>14} {0:>9}      -        -         -")
                continue
            w = gano[m].mean()
            e = w * payout - (1 - w)
            print(f"{thr:>6} {nombre:>14} {n:>9} {100*n/len(y):>5.1f}% "
                  f"{100*w:>7.2f}% {e:>+9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizonte", type=int, default=0,
                    help="1, 2 o 0 = ambos")
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--embargo-h", type=float, default=24.0)
    ap.add_argument("--excluir", default="BTCUSD")
    ap.add_argument("--max-train", type=int, default=0,
                    help="submuestrear train a N filas (0 = todas) si falta RAM")
    a = ap.parse_args()

    ex = {s.strip() for s in a.excluir.split(",") if s.strip()}
    horizontes = [1, 2] if a.horizonte == 0 else [a.horizonte]

    for h in horizontes:
        print(f"\n{'='*70}\nHORIZONTE {h} ({5*h} min)\n{'='*70}")
        T, X, Y, P = cargar(h, ex)
        o = np.argsort(T)
        T, X, Y, P = T[o], X[o], Y[o], P[o]
        f = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
        print(f"filas {len(T)} | {f(T[0])} -> {f(T[-1])} | "
              f"tasa base P(sube) {100*Y.mean():.2f}%")

        t_corte = float(np.quantile(T, 1 - a.test_frac))
        emb = a.embargo_h * 3600
        m_tr, m_te = T < (t_corte - emb), T > (t_corte + emb)
        print(f"corte {f(t_corte)} | train {int(m_tr.sum())} | test {int(m_te.sum())} | "
              f"embargadas {len(T)-int(m_tr.sum())-int(m_te.sum())}")

        Xtr, Ytr = X[m_tr], Y[m_tr]
        if a.max_train and len(Xtr) > a.max_train:
            idx = np.linspace(0, len(Xtr) - 1, a.max_train).astype(int)
            Xtr, Ytr = Xtr[idx], Ytr[idx]
            print(f"train submuestreado a {len(Xtr)} filas")

        print("entrenando...", flush=True)
        mdl = HistGradientBoostingClassifier(**PARAMS).fit(Xtr, Ytr)
        p = mdl.predict_proba(X[m_te])[:, 1]
        y_te, t_te = Y[m_te], T[m_te]
        hor = np.array([datetime.fromtimestamp(x, timezone.utc).hour for x in t_te])

        dias = (t_te.max() - t_te.min()) / 86400
        print(f"held-out: {dias:.0f} dias | P mediana {np.median(p):.3f} "
              f"| min {p.min():.3f} max {p.max():.3f}")
        evaluar(p, y_te, hor, PAYOUT[h], f"DIRECTO h={h}")

        import pickle
        os.makedirs("models", exist_ok=True)
        dst = f"models/directo_h{h}.pkl"
        with open(dst, "wb") as fh:
            pickle.dump(mdl, fh)
        print(f"\n[SAVE] {dst}")


if __name__ == "__main__":
    main()
