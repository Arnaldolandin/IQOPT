# meta_par_hora.py - ¿Tiene cada par horas rentables propias?
#
#   .venv314\Scripts\python.exe meta_par_hora.py [--umbral 0.54] [--min-n 30]
#
# Metodo: se parte el HELD-OUT (que el modelo nunca vio) en dos mitades temporales.
#   SELECCION: en la 1a mitad se eligen, por cada par, las horas con WR >= break-even.
#   VALIDACION: se miden esas mismas celdas par-hora en la 2a mitad.
# Si el efecto es real, las celdas elegidas deben seguir rindiendo. Si es ruido,
# revierten a la media (~52%).
#
# Control: se repite eligiendo horas AL AZAR (misma cantidad por par). Si las celdas
# elegidas por WR no rinden mas que las elegidas al azar, no hay efecto par-hora.
import argparse
import pickle
import random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

import meta_train_corte as M

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)


def ev(y):
    return (y.mean() * PAYOUT - (1 - y.mean())) if len(y) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.54)
    ap.add_argument("--min-n", type=int, default=30, help="muestra minima por celda par-hora")
    ap.add_argument("--sin-rollover", action="store_true",
                    help="excluir 20-22 UTC (la ventana donde IQ rechaza ordenes)")
    ap.add_argument("--perm", type=int, default=200)
    a = ap.parse_args()

    T, X, Y, P = M.cargar({"BTCUSD"})
    o = np.argsort(T)
    T, X, Y, P = T[o], X[o], Y[o], P[o]
    m_te = T > (float(np.quantile(T, 0.65)) + 24 * 3600)
    mdl = pickle.load(open("models/meta_bbrev_corte.pkl", "rb"))
    p = mdl.predict_proba(X[m_te])[:, 1]
    y, t, par = Y[m_te], T[m_te], P[m_te]

    sel = p >= a.umbral
    y, t, par = y[sel], t[sel], par[sel]
    hor = np.array([datetime.fromtimestamp(x, timezone.utc).hour for x in t])
    if a.sin_rollover:
        k = ~np.isin(hor, [20, 21, 22])
        y, t, par, hor = y[k], t[k], par[k], hor[k]
        print("[MODO] excluida la ventana de rollover 20-22 UTC")

    corte = np.median(t)
    m1, m2 = t <= corte, t > corte
    f = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    print(f"n={len(y)}  WR global {100*y.mean():.2f}%  (BE {100*BE:.2f}%)")
    print(f"SELECCION {f(t[m1][0])}->{f(t[m1][-1])} n={int(m1.sum())} | "
          f"VALIDACION {f(t[m2][0])}->{f(t[m2][-1])} n={int(m2.sum())}\n")

    # --- celdas par-hora en la mitad de SELECCION ---
    cel1 = defaultdict(list)
    for i in np.where(m1)[0]:
        cel1[(par[i], hor[i])].append(y[i])
    cel2 = defaultdict(list)
    for i in np.where(m2)[0]:
        cel2[(par[i], hor[i])].append(y[i])

    elegidas = {k: np.array(v) for k, v in cel1.items()
                if len(v) >= a.min_n and np.mean(v) >= BE}
    con_datos = {k for k, v in cel1.items() if len(v) >= a.min_n}
    print(f"celdas par-hora con n>={a.min_n} en seleccion: {len(con_datos)}")
    print(f"  de ellas 'rentables' (WR>=BE): {len(elegidas)} "
          f"({100*len(elegidas)/max(len(con_datos),1):.0f}%)")
    print(f"  por azar puro se esperaria ~50%\n")

    pares_con = defaultdict(int)
    for (pa, h) in elegidas:
        pares_con[pa] += 1
    print(f"pares con al menos 1 hora 'rentable': {len(pares_con)} de {len(set(par))}")
    print(f"mediana de horas rentables por par: "
          f"{np.median([pares_con[x] for x in set(par)]) if pares_con else 0:.0f}\n")

    # --- validacion en la 2a mitad ---
    def medir(celdas):
        vv = [v for k in celdas for v in cel2.get(k, [])]
        return np.array(vv)

    val = medir(elegidas.keys())
    todo2 = y[m2]
    print("=== VALIDACION (2a mitad, celdas elegidas en la 1a) ===")
    print(f"  todas las señales : n={len(todo2):6d}  WR {100*todo2.mean():.2f}%  "
          f"EV {ev(todo2):+.4f}")
    if len(val):
        print(f"  celdas elegidas   : n={len(val):6d}  WR {100*val.mean():.2f}%  "
              f"EV {ev(val):+.4f}")
        print(f"  diferencia        : {100*(val.mean()-todo2.mean()):+.2f} pt")
    else:
        print("  celdas elegidas   : sin datos en la 2a mitad")
        return

    # --- control: elegir la MISMA cantidad de celdas al azar ---
    rnd = random.Random(7)
    pool = sorted(con_datos)
    difs = []
    for _ in range(a.perm):
        r = rnd.sample(pool, min(len(elegidas), len(pool)))
        vv = medir(r)
        if len(vv):
            difs.append(100 * (vv.mean() - todo2.mean()))
    if difs:
        difs.sort()
        real = 100 * (val.mean() - todo2.mean())
        pval = (sum(1 for x in difs if x >= real) + 1) / (len(difs) + 1)
        print(f"\n=== CONTROL: mismas N celdas elegidas AL AZAR ({a.perm} veces) ===")
        print(f"  azar: mediana {difs[len(difs)//2]:+.2f} pt, "
              f"p95 {difs[int(0.95*(len(difs)-1))]:+.2f} pt")
        print(f"  real: {real:+.2f} pt  ->  p = {pval:.4f}")
        print("  VEREDICTO:", "hay efecto par-hora real"
              if (pval < 0.05 and real > 0) else
              "NO se distingue de elegir horas al azar")

    # --- ¿las celdas que sobreviven son de rollover? ---
    if len(elegidas):
        rr = sum(1 for (_, h) in elegidas if h in (20, 21, 22))
        print(f"\n  de las {len(elegidas)} celdas elegidas, {rr} "
              f"({100*rr/len(elegidas):.0f}%) caen en 20-22 UTC")


if __name__ == "__main__":
    main()
