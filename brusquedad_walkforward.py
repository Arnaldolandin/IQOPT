# -*- coding: utf-8 -*-
"""VALIDACION de la brusquedad: walk-forward en periodos disjuntos + controles.

QUE SE VALIDA
-------------
`reversion_condicionada_test.py` encontro que la reversion funciona mejor cuando el
estiramiento ocurrio DE GOLPE (todo el movimiento de 5 velas concentrado en la ultima)
que cuando llego a goteo:

    Q5 brusquedad @0.56 -> 55.60%  n=6732  IC [54.41, 56.79]  mitades 56.04/55.16
    todas       @0.56 -> 54.31%  n=11368 IC [53.39, 55.23]

Encaja con la teoria: la reversion es una apuesta a SOBRERREACCION, y un shock de una
vela es sobrerreaccion; un desplazamiento repartido en 5 velas es deriva ordenada.

POR QUE NO BASTA LO ANTERIOR
----------------------------
Se miraron ~30 celdas. La historia del proyecto dice que a esa escala aparece un ganador
por azar casi seguro (forex, EURCHF fade, hora y contexto multi-escala fueron los cuatro
falsos positivos anteriores). Y "persiste en las dos mitades" es persistencia DENTRO del
mismo bloque de test: no es un periodo disjunto.

Aqui se exige lo que ninguno de esos cuatro aguanto:

  1) WALK-FORWARD en 4 folds consecutivos: entrenar en el pasado, medir en el tramo
     siguiente, avanzar. Cada fold es un periodo DISJUNTO de los demas. La pregunta no es
     "cual es el WR medio" sino EN CUANTOS FOLDS bate BE. Un edge real bate en casi todos;
     el ruido alterna.

  2) CONTROL DE SELECTIVIDAD (el que de verdad puede tumbar esto): Q5 podria ser
     "senal fuerte" con otro nombre -- una vela brusca estira el Bollinger, el modelo se
     pone confiado, y estariamos redescubriendo el umbral que ya subimos de 0.54 a 0.56.
     Se compara Q5 contra un umbral MAS ALTO ajustado para dar el MISMO numero de
     operaciones. Si el umbral igualado rinde lo mismo, la brusquedad no aporta nada.

  3) CONTROL DE ETIQUETAS BARAJADAS por fold: suelo de ruido.

  4) EV por operacion al payout real 87%, que es lo unico que decide si se toca el bot.

Resto de salvaguardas de la casa: embargo 66 velas, continuidad exacta a H*300 s, sin
BTCUSD, rollover 20-22 UTC fuera SIEMPRE.
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier

from horizonte_corto_test import feats, roll_mean, wr_ic, BE, ROLL, EMB_VELAS, CACHE

H = 2
SEED = 42
THR = 0.56           # el umbral que corre en produccion hoy
N_FOLDS = 4
PAYOUT = 0.87


def cargar():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    X_l, y_l, t_l, b_l = [], [], [], []
    print("Cargando pares...", flush=True)
    for par in pares:
        ruta = os.path.join(CACHE, par + ".json")
        if not os.path.isfile(ruta):
            continue
        d = json.load(open(ruta, encoding="utf-8"))
        o = np.asarray(d["open"], float); h = np.asarray(d["high"], float)
        l = np.asarray(d["low"], float);  c = np.asarray(d["close"], float)
        t = np.asarray(d["times"], float)
        vol = d.get("volume")
        vol = np.asarray(vol, float) if vol else None
        n = len(c)
        if n < 500:
            continue
        X = feats(o, h, l, c, vol, t)
        cont = np.zeros(n, bool)
        cont[:n - H] = (t[H:] - t[:-H] == H * 300) & (c[H:] != c[:-H])
        y = np.zeros(n, int)
        y[:n - H] = (c[H:] > c[:-H]).astype(int)

        r1 = np.zeros(n); r1[1:] = c[1:] - c[:-1]
        camino = np.zeros(n)
        camino[5:] = roll_mean(np.abs(r1), 5)[5:] * 5
        brus = np.where(camino > 0, np.abs(r1) / np.maximum(camino, 1e-12), np.nan)

        ok = np.isfinite(X).all(1) & cont & np.isfinite(brus)
        X_l.append(X[ok]); y_l.append(y[ok]); t_l.append(t[ok]); b_l.append(brus[ok])
        print(f"  {par}", end=" ", flush=True)
    print()
    X = np.vstack(X_l); y = np.concatenate(y_l)
    t = np.concatenate(t_l); b = np.concatenate(b_l)
    orden = np.argsort(t, kind="stable")          # walk-forward = orden temporal global
    return X[orden], y[orden], t[orden], b[orden]


def entrena_pred(Xtr, ytr, Xte, barajar=False):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                       l2_regularization=1.0, random_state=SEED)
    yy = np.random.RandomState(SEED).permutation(ytr) if barajar else ytr
    m.fit(Xtr, yy)
    return m.predict_proba(Xte)[:, 1]


def mide(p, y, thr, mask):
    sel = ((p >= thr) | (p <= 1 - thr)) & mask
    n = int(sel.sum())
    if n < 200:
        return None
    gano = np.where(p >= thr, y == 1, y == 0)
    wr, lo, hi = wr_ic(int(gano[sel].sum()), n)
    ev = wr * PAYOUT - (1 - wr)            # EV por unidad de stake al payout real
    return wr, lo, hi, n, ev


def thr_igualado(p, mask, objetivo):
    """Umbral que deja pasar ~'objetivo' operaciones dentro de mask. Es el control clave:
    compara la brusquedad contra PURA SELECTIVIDAD con la misma cantidad de operaciones."""
    conf = np.abs(p - 0.5)
    c = np.sort(conf[mask])[::-1]
    if objetivo >= len(c):
        return 0.5
    return 0.5 + c[objetivo - 1]


def main():
    X, y, t, b = cargar()
    n = len(y)
    EMB = EMB_VELAS * 300
    print(f"\n{n:,} velas utilizables | folds {N_FOLDS} | umbral {THR} | "
          f"break-even {100*BE:.2f}% | payout {PAYOUT:.0%}")

    # walk-forward: el fold k entrena con TODO lo anterior y mide en su tramo
    bordes = [int(n * (0.40 + 0.15 * k)) for k in range(N_FOLDS + 1)]
    q80_global = None
    res = {"todas": [], "q5": [], "igualado": [], "control": []}

    for k in range(N_FOLDS):
        ini, fin = bordes[k], bordes[k + 1]
        t_ini = t[ini]
        tr = t < t_ini - EMB                       # embargo antes del inicio del fold
        te = np.zeros(n, bool); te[ini:fin] = True
        hora = ((t + 300) // 3600 % 24).astype(int)
        te &= ~np.isin(hora, ROLL)                 # rollover fuera SIEMPRE
        if tr.sum() < 50000 or te.sum() < 20000:
            continue

        # el corte de brusquedad se fija SOLO con datos de entrenamiento: usar el
        # quantil del test seria mirar el futuro para definir el filtro
        q80 = float(np.nanquantile(b[tr], 0.8))
        p = entrena_pred(X[tr], y[tr], X[te])
        p_ctl = entrena_pred(X[tr], y[tr], X[te], barajar=True)
        yte, bte = y[te], b[te]
        todo = np.ones(int(te.sum()), bool)
        q5 = bte >= q80

        r_all = mide(p, yte, THR, todo)
        r_q5 = mide(p, yte, THR, q5)
        r_ctl = mide(p_ctl, yte, THR, q5)
        # control de selectividad: mismo numero de ops que Q5, pero elegidas por confianza
        r_ig = None
        if r_q5:
            thr_ig = thr_igualado(p, todo, r_q5[3])
            r_ig = mide(p, yte, thr_ig, todo)

        d0 = np.datetime64(int(t[ini]), "s"); d1 = np.datetime64(int(t[fin - 1]), "s")
        print(f"\n--- fold {k+1}: {str(d0)[:10]} a {str(d1)[:10]} "
              f"(train {int(tr.sum()):,} / test {int(te.sum()):,}, q80 brusq {q80:.3f}) ---")
        for nom, r in (("todas @0.56", r_all), ("Q5 brusquedad", r_q5),
                       ("umbral igualado (control)", r_ig), ("barajado (control)", r_ctl)):
            if r is None:
                print(f"    {nom:<28} n insuficiente")
                continue
            wr, lo, hi, nn, ev = r
            marca = " <-- bate BE" if lo > BE else ""
            print(f"    {nom:<28} n={nn:>6} | WR {100*wr:.2f}% [{100*lo:.2f},{100*hi:.2f}] "
                  f"| EV/op {ev:+.4f}{marca}")
        for key, r in (("todas", r_all), ("q5", r_q5), ("igualado", r_ig), ("control", r_ctl)):
            if r:
                res[key].append(r)
        q80_global = q80

    print(f"\n{'='*82}\nRESUMEN WALK-FORWARD ({len(res['q5'])} folds)\n{'='*82}")
    print(f"{'variante':<28}{'folds sobre BE':>16}{'WR medio':>11}{'EV medio':>11}{'n total':>10}")
    for key, nom in (("todas", "todas @0.56"), ("q5", "Q5 brusquedad"),
                     ("igualado", "umbral igualado"), ("control", "barajado")):
        rs = res[key]
        if not rs:
            continue
        sobre = sum(1 for r in rs if r[0] > BE)
        wr = float(np.mean([r[0] for r in rs]))
        ev = float(np.mean([r[4] for r in rs]))
        nt = sum(r[3] for r in rs)
        print(f"{nom:<28}{sobre:>10}/{len(rs):<5}{100*wr:>10.2f}%{ev:>11.4f}{nt:>10}")

    print(f"\nVEREDICTO: la brusquedad solo se aplica si Q5 bate BE en CASI TODOS los folds")
    print(f"Y le saca ventaja clara al 'umbral igualado'. Si empatan, es selectividad")
    print(f"disfrazada y la palanca sigue siendo el umbral, que ya esta puesto.")
    if q80_global:
        print(f"\nCorte operativo (q80 del ultimo fold): brusquedad >= {q80_global:.3f}")
        print("brusquedad = |ret_1vela| / suma|ret| de las ultimas 5 velas")


if __name__ == "__main__":
    main()
