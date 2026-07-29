# -*- coding: utf-8 -*-
"""H=1 (5 min) al payout de BINARY: el horizonte que nunca se midio.

POR QUE ESTE TEST EXISTE
------------------------
El CLAUDE.md descarta el horizonte 1 asi: "expiry<=5 -> turbo (~83%, break-even 54.64%).
El horizonte 1 (5 min) esta descartado por esto: el payout se come la ventaja". Y
`horizonte_test.py` prueba HS = [2, 3, 4, 6]: **H=1 nunca se midio**, se descarto por el
supuesto del payout, no por un numero.

El supuesto es FALSO en la practica. Medido el 2026-07-28 sobre las operaciones reales del
log (rsi_iq.log, emparejando ENTRADA con CIERRE de forma inequivoca):

  - IQ liquida en marcas de reloj de 15 min: 434 de 527 cierres caen en minuto multiplo
    de 15, y 496 de 527 en los primeros 10 s del minuto.
  - El bot pide expiry_min=10 -> instrumento 'binary' -> payout 87%, PERO la opcion vence
    en la siguiente marca, que puede estar a 1 minuto. Duracion real medida: media 8.97
    min, minimo 0.1 min (una entrada a las 13:45:55 liquido a las 13:46).
  - En esas opciones cortas IQ PAGA EL 87% IGUAL: profit medio +0.870 sobre stake 1.00 en
    las ganadas de 0-2.5 min, +0.857 en las de 2.5-5 min. No las cobra como turbo.

=> Se puede tener exposicion de 5 minutos a break-even 53.48%, no 54.64%. Y la memoria
   `reversion-corto-plazo-real-2026-07-23` dice que el edge de reversion es MAXIMO
   inmediato y se PARTE A LA MITAD a los 5 min: delay0 56.3% -> delay2(10m) 53.4%.
   O sea: el horizonte corto es justo donde deberia vivir la senal.

QUE MIDE
--------
El bot no elige H: se lo impone el reloj. Decide al cerrar una vela de 5m y la opcion
vence en la siguiente marca de 15, asi que el minuto de cierre determina el horizonte:

    minuto de cierre mod 15 == 10  ->  vence en 5 min   -> H=1
    minuto de cierre mod 15 ==  5  ->  vence en 10 min  -> H=2
    minuto de cierre mod 15 ==  0  ->  vence en 15 min  -> H=3

Se comparan cuatro cosas, todas contra break-even 53.48%:

  A) H=2 puro           - lo que el bot CREE que hace (y contra lo que se entreno).
  B) H real segun reloj - lo que el bot REALMENTE obtiene hoy (mezcla de 1, 2 y 3).
  C) filtro :10         - operar SOLO las velas que cierran en minuto 10/25/40/55 (H=1),
                          usando el modelo entrenado a H=2. Accionable YA, sin reentrenar.
  D) modelo a H=1       - reentrenado para el horizonte corto. El techo de la idea.

SALVAGUARDAS (las de la casa, ninguna es opcional)
--------------------------------------------------
  - Corte temporal estricto por par + embargo de 66 velas a cada lado.
  - Continuidad: la vela de liquidacion a H*300 s EXACTOS (mata gaps de fin de semana).
  - Sin BTCUSD (trampa #5: 0 entradas historicas, no se puede operar).
  - Rollover 20-22 UTC separado SIEMPRE (trampa #3).
  - Control de etiquetas barajadas = suelo de ruido. Si el real no lo supera, no hay senal.
  - Persistencia en 2 mitades del test: el patron que ha matado 4 ideas de este proyecto es
    "1a mitad bate BE, 2a no". Sin persistencia no se aplica nada.

Baseline HGB, no el LSTM: esto es un test de VIABILIDAD. Si H=1 no despega aqui, no vale
la pena reentrenar 50 LSTM. Y usar los .pt ya entrenados seria fuga in-sample (trampa #1).
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = "cache_ohlc_5m_v2"
BE = 1.0 / 1.87                 # 53.48% - payout 87% VERIFICADO en opciones cortas
ROLL = (20, 21, 22)             # rollover UTC: IQ rechaza, los WR de ahi son artefacto
EMB_VELAS = 66                  # embargo = L(64) + H(2), en velas
THRS = (0.52, 0.53, 0.54, 0.55, 0.56, 0.58)
SEED = 42


def roll_mean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    r = np.full(len(x), np.nan)
    r[w - 1:] = (cs[w:] - cs[:-w]) / w
    return r


def roll_std(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    cs2 = np.cumsum(np.insert(x * x, 0, 0.0))
    r = np.full(len(x), np.nan)
    m = (cs[w:] - cs[:-w]) / w
    m2 = (cs2[w:] - cs2[:-w]) / w
    r[w - 1:] = np.sqrt(np.maximum(m2 - m * m, 0))
    return r


def feats(o, h, l, c, vol, t):
    """MISMAS features que horizonte_test.py, a proposito: para que la comparacion entre
    horizontes sea de horizontes y no de features."""
    n = len(c)
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = roll_mean(tr, 14)
    atr = np.where((atr > 0) & np.isfinite(atr), atr, np.nan)
    cols = {}
    for k in (1, 2, 3, 5, 10):
        r = np.full(n, np.nan)
        r[k:] = (c[k:] - c[:-k])
        cols[f"ret{k}"] = r / atr
    cols["cuerpo"] = (c - o) / atr
    cols["rango"] = (h - l) / atr
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0)
    ll = np.where(d < 0, -d, 0.0)
    ag = roll_mean(g, 14)
    al = roll_mean(ll, 14)
    cols["rsi"] = 100 - 100 / (1 + ag / (al + 1e-12))
    sma = roll_mean(c, 20)
    sd = roll_std(c, 20)
    cols["bb"] = (c - (sma - 2 * sd)) / (4 * sd + 1e-12)   # Bollinger %B: la feature dominante
    cols["atr_rel"] = atr / c
    if vol is not None:
        vm = roll_mean(vol, 20)
        cols["vol_rel"] = np.where(vm > 0, vol / np.maximum(vm, 1e-9) - 1, 0.0)
    else:
        cols["vol_rel"] = np.zeros(n)
    hora = (t // 3600) % 24
    cols["hsin"] = np.sin(2 * np.pi * hora / 24)
    cols["hcos"] = np.cos(2 * np.pi * hora / 24)
    return np.column_stack([cols[k] for k in cols])


def wr_ic(k, n):
    """WR e intervalo de confianza al 95% (normal). Sin IC, cualquier subgrupo 'bate BE'."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    e = 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n)
    return p, p - e, p + e


def evaluar(p, y, hora, mask, etiqueta, thr_fijo=None):
    """WR fuera de rollover al mejor umbral (o a uno fijo), con IC y n."""
    fuera = ~np.isin(hora, ROLL) & mask
    mejor = None
    lista = (thr_fijo,) if thr_fijo else THRS
    for thr in lista:
        sel = ((p >= thr) | (p <= 1 - thr)) & fuera
        n = int(sel.sum())
        if n < 300:
            continue
        gano = np.where(p >= thr, y == 1, y == 0)
        k = int(gano[sel].sum())
        wr, lo, hi = wr_ic(k, n)
        if mejor is None or wr > mejor[0]:
            mejor = (wr, lo, hi, n, thr, sel, gano)
    if mejor is None:
        print(f"  {etiqueta:<34} n insuficiente")
        return None
    wr, lo, hi, n, thr, sel, gano = mejor
    marca = "  <-- BATE BE" if lo > BE else ("  (IC incluye BE)" if hi > BE else "")
    print(f"  {etiqueta:<34} thr {thr:.2f} | n={n:>7} | WR {100*wr:.2f}% "
          f"[{100*lo:.2f}, {100*hi:.2f}]{marca}")
    return wr, n, thr, sel, gano


def persistencia(t, sel, gano, etiqueta):
    """El test que ha matado 4 ideas: 1a mitad bate BE, 2a no. Si no persiste, es ruido."""
    tt = t[sel]
    gg = gano[sel]
    if len(tt) < 600:
        return
    med = np.median(tt)
    a, b = gg[tt <= med], gg[tt > med]
    if len(a) < 200 or len(b) < 200:
        return
    wa, wb = a.mean(), b.mean()
    ok = "PERSISTE" if (wa > BE and wb > BE) else "NO persiste"
    print(f"  {'':<34} mitades: {100*wa:.2f}% (n={len(a)}) / {100*wb:.2f}% (n={len(b)})  -> {ok}")


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    EMB = EMB_VELAS * 300

    acc = {"Xtr": [], "Xte": [], "tte": [],
           "ytr1": [], "ytr2": [], "yte1": [], "yte2": [], "yte3": [],
           "cte1": [], "cte2": [], "cte3": [], "ctr1": [], "ctr2": [], "hte": []}

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

        # continuidad y etiqueta por horizonte: la vela de liquidacion a H*300 s EXACTOS
        cont, y = {}, {}
        for H in (1, 2, 3):
            cc = np.zeros(n, bool)
            cc[:n - H] = (t[H:] - t[:-H] == H * 300) & (c[H:] != c[:-H])
            cont[H] = cc
            yy = np.zeros(n, int)
            yy[:n - H] = (c[H:] > c[:-H]).astype(int)
            y[H] = yy

        corte = float(np.quantile(t, 0.65))
        tr = finite & (t < corte - EMB)
        te = finite & (t > corte + EMB)

        acc["Xtr"].append(X[tr]); acc["Xte"].append(X[te]); acc["tte"].append(t[te])
        acc["ytr1"].append(y[1][tr]); acc["ytr2"].append(y[2][tr])
        acc["ctr1"].append(cont[1][tr]); acc["ctr2"].append(cont[2][tr])
        for H in (1, 2, 3):
            acc[f"yte{H}"].append(y[H][te]); acc[f"cte{H}"].append(cont[H][te])
        # hora UTC del INSTANTE DE DECISION = cierre de la vela = t + 300
        acc["hte"].append(((t[te] + 300) // 3600 % 24).astype(int))
        print(f"  {par}", end=" ", flush=True)
    print()

    Xtr = np.vstack(acc["Xtr"]); Xte = np.vstack(acc["Xte"])
    tte = np.concatenate(acc["tte"]); hte = np.concatenate(acc["hte"])
    ytr = {H: np.concatenate(acc[f"ytr{H}"]) for H in (1, 2)}
    ctr = {H: np.concatenate(acc[f"ctr{H}"]) for H in (1, 2)}
    yte = {H: np.concatenate(acc[f"yte{H}"]) for H in (1, 2, 3)}
    cte = {H: np.concatenate(acc[f"cte{H}"]) for H in (1, 2, 3)}

    # minuto de reloj del INSTANTE DE DECISION (cierre de vela) -> horizonte que impone IQ
    minuto = ((tte + 300) // 60 % 15).astype(int)
    H_reloj = np.select([minuto == 10, minuto == 5, minuto == 0], [1, 2, 3], default=0)

    print(f"\ntrain {len(Xtr):,} velas | test {len(Xte):,} velas | break-even {100*BE:.2f}%")
    print(f"reparto del horizonte que impone el reloj: "
          f"H=1 {100*np.mean(H_reloj==1):.1f}%  H=2 {100*np.mean(H_reloj==2):.1f}%  "
          f"H=3 {100*np.mean(H_reloj==3):.1f}%  otros {100*np.mean(H_reloj==0):.1f}%")

    def entrena(H, barajar=False):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                           l2_regularization=1.0, random_state=SEED)
        msk = ctr[H]
        yy = ytr[H][msk]
        if barajar:
            yy = np.random.RandomState(SEED).permutation(yy)
        m.fit(Xtr[msk], yy)
        return m.predict_proba(Xte)[:, 1]

    print("\nEntrenando modelo H=2 (el horizonte con el que vive el bot)...", flush=True)
    p2 = entrena(2)
    print("Entrenando modelo H=1 (horizonte corto)...", flush=True)
    p1 = entrena(1)
    print("Entrenando controles barajados (suelo de ruido)...", flush=True)
    p2_ctl = entrena(2, barajar=True)
    p1_ctl = entrena(1, barajar=True)

    print(f"\nAUC OOS  |  H=2: {roc_auc_score(yte[2][cte[2]], p2[cte[2]]):.4f}"
          f"   H=1: {roc_auc_score(yte[1][cte[1]], p1[cte[1]]):.4f}"
          f"   (control H=1: {roc_auc_score(yte[1][cte[1]], p1_ctl[cte[1]]):.4f})")

    print(f"\n{'='*78}\nA) H=2 PURO - lo que el bot cree que hace\n{'='*78}")
    r = evaluar(p2, yte[2], hte, cte[2], "modelo H2 -> etiqueta H2")
    if r: persistencia(tte, r[3], r[4], "A")

    print(f"\n{'='*78}\nB) H REAL SEGUN RELOJ - lo que el bot obtiene de verdad hoy\n{'='*78}")
    # cada vela se evalua contra el horizonte que el reloj le impone
    y_real = np.where(H_reloj == 1, yte[1], np.where(H_reloj == 2, yte[2], yte[3]))
    c_real = np.where(H_reloj == 1, cte[1], np.where(H_reloj == 2, cte[2], cte[3])) & (H_reloj > 0)
    r = evaluar(p2, y_real, hte, c_real, "modelo H2 -> etiqueta H real")
    if r: persistencia(tte, r[3], r[4], "B")
    for H in (1, 2, 3):
        m = c_real & (H_reloj == H)
        evaluar(p2, y_real, hte, m, f"   desglose: reloj impone H={H}")

    print(f"\n{'='*78}\nC) FILTRO :10 - operar solo las velas de H=1, modelo actual\n{'='*78}")
    print("   (accionable YA: es un filtro de minuto, no requiere reentrenar nada)")
    m10 = cte[1] & (H_reloj == 1)
    r = evaluar(p2, yte[1], hte, m10, "modelo H2 -> etiqueta H1, min :10")
    if r: persistencia(tte, r[3], r[4], "C")
    r = evaluar(p2_ctl, yte[1], hte, m10, "  control barajado")

    print(f"\n{'='*78}\nD) MODELO ENTRENADO A H=1 - el techo de la idea\n{'='*78}")
    r = evaluar(p1, yte[1], hte, cte[1], "modelo H1 -> etiqueta H1, todas")
    if r: persistencia(tte, r[3], r[4], "D")
    r = evaluar(p1, yte[1], hte, m10, "modelo H1 -> etiqueta H1, min :10")
    if r: persistencia(tte, r[3], r[4], "D10")
    evaluar(p1_ctl, yte[1], hte, cte[1], "  control barajado")

    print(f"\n{'='*78}\nDENTRO de rollover (20-22 UTC) - IQ RECHAZA AHI, es solo control\n{'='*78}")
    dentro = np.isin(hte, ROLL)
    for nom, p, y, cm in (("H1 min :10", p2, yte[1], m10), ("H2", p2, yte[2], cte[2])):
        sel = ((p >= 0.54) | (p <= 0.46)) & cm & dentro
        if sel.sum() > 300:
            gano = np.where(p >= 0.54, y == 1, y == 0)
            wr, lo, hi = wr_ic(int(gano[sel].sum()), int(sel.sum()))
            print(f"  {nom:<34} n={int(sel.sum()):>7} | WR {100*wr:.2f}% "
                  f"[{100*lo:.2f}, {100*hi:.2f}]")

    print(f"\nLectura: solo cuenta si el IC INFERIOR supera {100*BE:.2f}% Y persiste en las "
          f"dos mitades\ny el control barajado se queda abajo. Cualquier otra cosa es ruido.")


if __name__ == "__main__":
    main()
