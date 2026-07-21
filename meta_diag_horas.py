# meta_diag_horas.py - Diagnostico: el edge del held-out, ¿esta concentrado en la
# ventana de rollover (20-22 UTC) y/o en pocos activos? Si al quitar esas horas el
# EV se cae, el "filtro horario" seria un artefacto del feed, no un edge operable.
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

import meta_train_corte as M

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)


def main():
    T, X, Y, P = M.cargar({"BTCUSD"})
    o = np.argsort(T)
    T, X, Y, P = T[o], X[o], Y[o], P[o]
    t_corte = float(np.quantile(T, 0.65))
    emb = 24 * 3600
    m_te = T > (t_corte + emb)
    mdl = pickle.load(open("models/meta_bbrev_corte.pkl", "rb"))
    p = mdl.predict_proba(X[m_te])[:, 1]
    y, t, par = Y[m_te], T[m_te], P[m_te]

    sel = p >= 0.54
    y, t, par, p = y[sel], t[sel], par[sel], p[sel]
    hor = np.array([datetime.fromtimestamp(x, timezone.utc).hour for x in t])
    ev = lambda yy: (yy.mean() * PAYOUT - (1 - yy.mean())) if len(yy) else 0.0

    print(f"n={len(y)}  WR {100*y.mean():.2f}%  EV/op {ev(y):+.4f}  (BE {100*BE:.2f}%)\n")

    roll = np.isin(hor, [20, 21, 22])
    print("=== quitando la ventana de rollover (20-22 UTC) ===")
    print(f"  dentro 20-22 : n={int(roll.sum()):6d}  WR {100*y[roll].mean():.2f}%  "
          f"EV {ev(y[roll]):+.4f}")
    print(f"  fuera        : n={int((~roll).sum()):6d}  WR {100*y[~roll].mean():.2f}%  "
          f"EV {ev(y[~roll]):+.4f}")
    solo21 = hor == 21
    print(f"  solo h21     : n={int(solo21.sum()):6d}  WR {100*y[solo21].mean():.2f}%  "
          f"EV {ev(y[solo21]):+.4f}")

    print("\n=== composicion por activo de h20-22 vs resto ===")
    c_in, c_out = defaultdict(int), defaultdict(int)
    for a, r in zip(par, roll):
        (c_in if r else c_out)[a] += 1
    print(f"{'activo':<10} {'en 20-22':>9} {'fuera':>7} {'WR 20-22':>9}")
    for a, n in sorted(c_in.items(), key=lambda kv: -kv[1])[:12]:
        m = roll & (par == a)
        print(f"{a:<10} {n:>9} {c_out.get(a,0):>7} {100*y[m].mean():>8.2f}%")

    print("\n=== WR por activo (todo el held-out, P>=0.54) ===")
    filas = []
    for a in sorted(set(par)):
        m = par == a
        if m.sum() >= 100:
            filas.append((a, int(m.sum()), 100 * y[m].mean(), ev(y[m])))
    filas.sort(key=lambda r: -r[2])
    print(f"{'activo':<10} {'n':>7} {'WR%':>7} {'EV/op':>8}")
    for a, n, w, e in filas[:10]:
        print(f"{a:<10} {n:>7} {w:>6.2f} {e:>+8.4f}")
    print("  ...")
    for a, n, w, e in filas[-5:]:
        print(f"{a:<10} {n:>7} {w:>6.2f} {e:>+8.4f}")

    print("\n=== distribucion de P (para comparar con el bot en vivo) ===")
    pall = mdl.predict_proba(X[m_te])[:, 1]
    for thr in (0.50, 0.52, 0.54, 0.56, 0.60):
        print(f"  P>={thr}: {100*(pall >= thr).mean():5.1f}% de las señales")
    print(f"  P mediana {np.median(pall):.3f}  max {pall.max():.3f}")


if __name__ == "__main__":
    main()
