# screening.py - Evalua TODOS los activos con un tramo de validacion final intocado.
#
#   .venv314\Scripts\python.exe screening.py [--etapa 1|2] [--semillas 1]
#
# POR QUE ESTE DISEÑO: probar 50 activos y quedarse con los que se ven bien es
# seleccion multiple. Con 50 pruebas, varios superan cualquier filtro por azar (es lo
# que paso con los 210 combos de 2026-07-13 y con XAUUSD). La unica defensa es reservar
# un tramo que NO se mira durante la seleccion:
#
#   |---- entrenamiento ----|-- test de SELECCION --|-- VALIDACION FINAL --|
#            ~60%                     ~20%                    ~20%
#                              aqui se filtra          se mira UNA sola vez,
#                                                      solo con los finalistas
#
# ETAPA 1: entrena cada activo con 1 semilla y reporta metricas del test de seleccion.
#          NO toca la validacion final.
# ETAPA 2: reentrena los finalistas con N semillas y los evalua en la validacion final.
#
# El filtro de admision (las 5 condiciones que ETHUSD cumple y XAUUSD no) esta en
# cumple_criterio(). Se fija ANTES de mirar resultados y no se negocia despues.
import argparse
import glob
import json
import os

import numpy as np

import seq_model as S
import train_seq_save as T

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)
UMBRALES = (0.52, 0.53, 0.54, 0.56, 0.58)

# Conversion señal->operacion medida en rsi_iq.log (2026-07). Sin esto se entrenan
# modelos para activos que IQ nunca deja operar: BTCUSD tuvo 76 señales y 0 entradas.
CONVERSION_MINIMA = 0.30
NO_EJECUTAN = {"BTCUSD"}          # 0/76 historico
POCO_EJECUTAN = {"XRPUSD", "GBPCAD", "EURCAD", "CADCHF", "AUDCHF", "EURGBP"}


def particion(t, frac_train=0.60, frac_sel=0.20, embargo_velas=66):
    """train / test de seleccion / validacion final, con embargo entre bloques."""
    emb = embargo_velas * 300
    q1 = float(np.quantile(t, frac_train))
    q2 = float(np.quantile(t, frac_train + frac_sel))
    return (t < q1 - emb,
            (t > q1 + emb) & (t < q2 - emb),
            t > q2 + emb)


def perfil(p, y, t):
    """WR por umbral, solo fuera del rollover. Devuelve lista de (thr, n, wr)."""
    hor = ((t // 3600) % 24).astype(int)
    fuera = ~np.isin(hor, ROLLOVER)
    out = []
    for thr in UMBRALES:
        sel = ((p >= thr) | (p <= 1 - thr)) & fuera
        n = int(sel.sum())
        if n < 20:
            out.append((thr, n, None))
            continue
        gano = np.where(p >= thr, y == 1, y == 0)
        out.append((thr, n, float(gano[sel].mean())))
    return out


def lados(p, y, thr):
    """WR de CALL y PUT por separado: si el lado dominante esta bajo break-even, el
    activo pierde plata en la mayoria de sus operaciones (le paso a XAUUSD)."""
    call, put = p >= thr, p <= 1 - thr
    wc = float((y[call] == 1).mean()) if call.sum() >= 20 else None
    wp = float((y[put] == 0).mean()) if put.sum() >= 20 else None
    return wc, wp, int(call.sum()), int(put.sum())


def cumple_criterio(perf, wc, wp, n_min=1000):
    """Las 5 condiciones, fijadas ANTES de ver ningun resultado.

    1. WR sobre break-even en TODOS los umbrales con muestra suficiente
    2. perfil monotono (a mas confianza, mas acierto)
    3. CALL y PUT ambos sobre break-even
    4. algun umbral con n > n_min
    5. el activo ejecuta en IQ (se filtra antes de entrenar)
    """
    validos = [(thr, n, wr) for thr, n, wr in perf if wr is not None]
    if len(validos) < 3:
        return False, "pocos umbrales con muestra"
    if any(wr <= BE for _, _, wr in validos):
        return False, "algun umbral bajo break-even"
    wrs = [wr for _, _, wr in validos]
    if any(b < a - 0.005 for a, b in zip(wrs, wrs[1:])):
        return False, "perfil no monotono"
    if not any(n > n_min for _, n, _ in validos):
        return False, f"sin umbral con n>{n_min}"
    if wc is None or wp is None:
        return False, "un lado sin muestra"
    if wc <= BE or wp <= BE:
        return False, f"un lado bajo break-even (CALL {100*wc:.1f} PUT {100*wp:.1f})"
    return True, "PASA"


def candidatos(cache):
    pares = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(cache, "*.json")))
    return [p for p in pares if p not in NO_EJECUTAN and p not in POCO_EJECUTAN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, default=1, choices=[1, 2])
    ap.add_argument("--semillas", type=int, default=1)
    ap.add_argument("--L", type=int, default=S.L_DEFECTO)
    ap.add_argument("--finalistas", default="", help="etapa 2: coma-separados")
    a = ap.parse_args()

    E = T.cfg_entrenamiento()
    E["semillas"] = a.semillas
    cache = E.get("cache", "cache_ohlc_5m_v2")
    pares = ([p.strip() for p in a.finalistas.split(",") if p.strip()]
             if a.finalistas else candidatos(cache))

    print(f"[SCREENING etapa {a.etapa}] {len(pares)} activos | cache={cache} | "
          f"semillas={a.semillas} | break-even {100*BE:.2f}%")
    print(f"excluidos por no ejecutar: {sorted(NO_EJECUTAN | POCO_EJECUTAN)}\n")

    class Args:
        pass
    aa = Args(); aa.arq = E["arquitectura"]; aa.L = a.L; aa.epocas = int(E["epocas"])
    aa.salida = ""

    resultados = []
    for i, par in enumerate(pares, 1):
        try:
            X, y, t = T.dataset(par, a.L, cache, None)
        except Exception as e:
            print(f"[{i}/{len(pares)}] {par}: error leyendo ({e})")
            continue
        if len(X) < 5000:
            print(f"[{i}/{len(pares)}] {par}: solo {len(X)} muestras, salteado")
            continue
        m_tr, m_sel, m_fin = particion(t)
        # ETAPA 1: entrena con 'train' y mide en 'seleccion'. La validacion final ni se
        #          toca -- si se mirara aqui dejaria de ser independiente.
        # ETAPA 2: los finalistas se reentrenan con train+seleccion y se miden UNA sola
        #          vez en la validacion final, que hasta ahora nadie vio.
        if a.etapa == 1:
            m_fit, m_eval = m_tr, m_sel
        else:
            m_fit, m_eval = (m_tr | m_sel), m_fin
        Xtr, ytr = X[m_fit], y[m_fit]
        k = int(0.85 * len(Xtr))
        ps = []
        for s in range(max(1, a.semillas)):
            _, p_s, vl = T.entrenar_una(Xtr[:k], ytr[:k], Xtr[k:], ytr[k:],
                                        X[m_eval], aa, E, int(E["seed"]) + s)
            ps.append(p_s)
        p_sel = np.mean(ps, axis=0)
        perf = perfil(p_sel, y[m_eval], t[m_eval])
        m_sel = m_eval          # el resto del bucle usa m_sel para lados/etiquetas
        mejor = max((x for x in perf if x[2] is not None),
                    key=lambda x: x[2], default=(None, 0, None))
        wc, wp, nc, npu = lados(p_sel, y[m_sel], mejor[0]) if mejor[0] else (None, None, 0, 0)
        ok, motivo = cumple_criterio(perf, wc, wp)
        resultados.append((par, ok, motivo, perf, mejor, vl))
        txt = " ".join(f"{thr}:{100*wr:.1f}%(n{n})" if wr else f"{thr}:-"
                       for thr, n, wr in perf)
        print(f"[{i}/{len(pares)}] {par:10s} {'PASA ' if ok else 'no   '} {motivo:38s} {txt}",
              flush=True)

    print("\n" + "=" * 70)
    pasan = [r[0] for r in resultados if r[1]]
    print(f"PASAN LA SELECCION: {len(pasan)} de {len(resultados)}")
    for p in pasan:
        print(f"   {p}")
    if a.etapa == 1 and pasan:
        print(f"\nEtapa 2 (validacion final, tramo intocado):")
        print(f"  screening.py --etapa 2 --semillas 5 --finalistas {','.join(pasan)}")
    json.dump({"etapa": a.etapa, "pasan": pasan,
               "todos": [(r[0], r[1], r[2]) for r in resultados]},
              open(f"screening_etapa{a.etapa}.json", "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
