# -*- coding: utf-8 -*-
"""Dos condicionantes de la reversion que no estan en la lista de cerrados.

Se prueban SOLO DOS, a proposito. Este proyecto ha cazado 4 falsos positivos por
multiplicidad (forex, EURCHF fade, hora, contexto multi-escala): cuantos mas subgrupos se
miran, mas seguro es que alguno "bata BE" por azar. Los dos de aqui tienen motivo previo,
no son barrido.

1) ASIMETRIA PUT/CALL
   Motivo: en las operaciones REALES del bot (rsi_iq.log, 527 cierres del 25 al 28 de
   julio de 2026) los dos lados no se parecen:
       PUT   n=267  WR 54.31%
       CALL  n=260  WR 45.77%
   8.5 puntos de diferencia, z~1.95 sobre esa n. Indicio, no prueba -- justo lo que hay
   que llevar a datos con n grande. Hipotesis economica: la reversion tras una CAIDA
   (comprar) y tras una SUBIDA (vender) no son simetricas; el efecto apalancamiento hace
   que la volatilidad suba tras las caidas, y una reversion en alta volatilidad es mas
   ruidosa. `simetrico_edge.py` NO cubre esto: mide payoff simetrico en perpetuos, no el
   lado de la senal.

2) BRUSQUEDAD DEL ESTIRAMIENTO (movimiento de golpe vs goteo)
   Motivo: la reversion es una apuesta a SOBRERREACCION. Un mismo desplazamiento de 2 ATR
   concentrado en UNA vela es un shock que suele corregirse; el mismo desplazamiento
   repartido en 5 velas es una tendencia ordenada, y ahi la reversion es justo la apuesta
   equivocada. Distinto del ADX (que mide tendencia en 14 velas y se cerro porque tiraba
   los pullbacks) y distinto del nivel de Bollinger (que mide CUANTO se estiro, no como).
   No aparece en los angulos cerrados.

Salvaguardas: corte temporal por par + embargo 66 velas, continuidad exacta a H*300 s, sin
BTCUSD, rollover 20-22 UTC separado, control de etiquetas barajadas, IC 95% y persistencia
en las dos mitades del test. Un resultado solo cuenta si el IC INFERIOR supera BE Y
persiste en ambas mitades.
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier

from horizonte_corto_test import feats, roll_mean, wr_ic, BE, ROLL, EMB_VELAS, CACHE

H = 2                      # el horizonte con el que vive el bot
SEED = 42
THRS = (0.54, 0.56, 0.58)

CRIPTO = {"ETHUSD", "XRPUSD"}
STOCKS = {"AIG", "ALIBABA", "AMAZON", "APPLE", "BAIDU", "CISCO", "CITI", "COKE", "FACEBOOK",
          "GOOGLE", "GS", "INTEL", "JPM", "MCDON", "MORSTAN", "MSFT", "NIKE", "SNAP", "TESLA"}


def bloque(nombre, p, y, t, hora, mask, thr):
    """WR fuera de rollover + IC + persistencia. Devuelve (wr, lo, n) o None."""
    sel = ((p >= thr) | (p <= 1 - thr)) & mask & ~np.isin(hora, ROLL)
    n = int(sel.sum())
    if n < 400:
        print(f"    {nombre:<30} n={n} insuficiente")
        return None
    gano = np.where(p >= thr, y == 1, y == 0)
    wr, lo, hi = wr_ic(int(gano[sel].sum()), n)
    tt, gg = t[sel], gano[sel]
    med = np.median(tt)
    a, b = gg[tt <= med], gg[tt > med]
    pers = ""
    if len(a) >= 150 and len(b) >= 150:
        ok = (a.mean() > BE and b.mean() > BE)
        pers = f" | mitades {100*a.mean():.2f}/{100*b.mean():.2f} {'PERSISTE' if ok else 'no'}"
    marca = "  <-- BATE BE" if lo > BE else ""
    print(f"    {nombre:<30} n={n:>7} | WR {100*wr:.2f}% [{100*lo:.2f},{100*hi:.2f}]{pers}{marca}")
    return wr, lo, n


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    EMB = EMB_VELAS * 300

    Xtr_l, ytr_l = [], []
    Xte_l, yte_l, tte_l, gru_l, brus_l = [], [], [], [], []

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
        finite = np.isfinite(X).all(1)

        cont = np.zeros(n, bool)
        cont[:n - H] = (t[H:] - t[:-H] == H * 300) & (c[H:] != c[:-H])
        y = np.zeros(n, int)
        y[:n - H] = (c[H:] > c[:-H]).astype(int)

        # BRUSQUEDAD: cuanto del movimiento de 5 velas ocurrio en la ULTIMA.
        # ~1 = todo de golpe (shock, candidato a sobrerreaccion)
        # ~0 = goteo repartido (deriva ordenada, la reversion es la apuesta equivocada)
        r1 = np.zeros(n); r1[1:] = c[1:] - c[:-1]
        r5 = np.zeros(n); r5[5:] = c[5:] - c[:-5]
        camino = np.zeros(n)                       # suma de |movimientos| de las 5 velas
        a1 = np.abs(r1)
        camino[5:] = roll_mean(a1, 5)[5:] * 5
        brus = np.where(camino > 0, np.abs(r1) / np.maximum(camino, 1e-12), np.nan)

        corte = float(np.quantile(t, 0.65))
        tr = finite & cont & (t < corte - EMB)
        te = finite & cont & (t > corte + EMB) & np.isfinite(brus)

        Xtr_l.append(X[tr]); ytr_l.append(y[tr])
        Xte_l.append(X[te]); yte_l.append(y[te]); tte_l.append(t[te]); brus_l.append(brus[te])
        g = 2 if par in CRIPTO else (1 if par in STOCKS else 0)
        gru_l.append(np.full(int(te.sum()), g, np.int8))
        print(f"  {par}", end=" ", flush=True)
    print()

    Xtr = np.vstack(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)
    tte = np.concatenate(tte_l); gru = np.concatenate(gru_l); brus = np.concatenate(brus_l)
    hte = ((tte + 300) // 3600 % 24).astype(int)
    print(f"\ntrain {len(Xtr):,} | test {len(Xte):,} | break-even {100*BE:.2f}%")

    def entrena(barajar=False):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                           l2_regularization=1.0, random_state=SEED)
        yy = np.random.RandomState(SEED).permutation(ytr) if barajar else ytr
        m.fit(Xtr, yy)
        return m.predict_proba(Xte)[:, 1]

    print("Entrenando (H=2)...", flush=True)
    p = entrena()
    print("Entrenando control barajado...", flush=True)
    p_ctl = entrena(barajar=True)

    todo = np.ones(len(yte), bool)

    print(f"\n{'='*80}\n1) ASIMETRIA PUT / CALL\n{'='*80}")
    print("   en vivo: PUT 54.31% (n=267) vs CALL 45.77% (n=260)")
    for thr in THRS:
        print(f"\n  umbral {thr}:")
        lado_call = (p >= thr)
        lado_put = (p <= 1 - thr)
        bloque("ambos lados", p, yte, tte, hte, todo, thr)
        bloque("solo CALL (P alto)", p, yte, tte, hte, lado_call, thr)
        bloque("solo PUT  (P bajo)", p, yte, tte, hte, lado_put, thr)
    print("\n  control barajado (suelo de ruido):")
    bloque("  ambos lados", p_ctl, yte, tte, hte, todo, 0.54)

    print(f"\n{'='*80}\n2) BRUSQUEDAD DEL ESTIRAMIENTO (quintiles)\n{'='*80}")
    print("   Q1 = movimiento repartido (goteo)  ...  Q5 = todo en la ultima vela (shock)")
    qs = np.nanquantile(brus, [0.2, 0.4, 0.6, 0.8])
    qidx = np.digitize(brus, qs)
    for thr in (0.54, 0.56):
        print(f"\n  umbral {thr}:")
        for q in range(5):
            bloque(f"Q{q+1} brusquedad", p, yte, tte, hte, qidx == q, thr)

    print(f"\n{'='*80}\n3) CRUCE de los dos (solo si alguno dio algo arriba)\n{'='*80}")
    for q in (0, 4):
        for nom, m in (("CALL", p >= 0.56), ("PUT", p <= 0.44)):
            bloque(f"Q{q+1} x {nom}", p, yte, tte, hte, (qidx == q) & m, 0.56)

    print(f"\n{'='*80}\nPor clase de activo (contexto, thr 0.56)\n{'='*80}")
    for g, nom in ((0, "forex"), (1, "stocks"), (2, "cripto")):
        bloque(nom, p, yte, tte, hte, gru == g, 0.56)
        bloque(f"  {nom} solo PUT", p, yte, tte, hte, (gru == g) & (p <= 0.44), 0.56)
        bloque(f"  {nom} solo CALL", p, yte, tte, hte, (gru == g) & (p >= 0.56), 0.56)

    print(f"\nSolo cuenta lo que tenga IC inferior > {100*BE:.2f}% Y persista en las dos "
          f"mitades.\nY aun asi: se han mirado ~30 celdas, asi que una sola que pase el "
          f"corte es\nsospechosa de multiplicidad y necesita validacion en periodo disjunto.")


if __name__ == "__main__":
    main()
