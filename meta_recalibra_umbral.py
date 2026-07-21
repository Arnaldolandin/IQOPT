# meta_recalibra_umbral.py - Recalibra bb_ml_threshold del modelo de PRODUCCION
# (models/meta_bbrev_iq.pkl) por PERCENTIL, sin reentrenar.
#
#   .venv314\Scripts\python.exe meta_recalibra_umbral.py
#
# POR QUE POR PERCENTIL: el umbral 0.60 validado sobre models/meta_bbrev_corte.pkl no
# se traslada por valor, porque los dos modelos no comparten escala de probabilidad.
# Lo que sí se traslada es la FRACCION de señales seleccionada.
#
# DE DONDE SALE LA MUESTRA LIMPIA: meta_bbrev_iq.pkl se entreno con cache_ohlc_5m, que
# termina el 2026-07-16. Todo lo que rsi_iq.log registro DESPUES de esa fecha son
# predicciones sobre datos que el modelo nunca vio -> percentiles no contaminados.
#
# OJO CON LAS HORAS: rsi_iq.log escribe hora LOCAL de Chile (UTC-4); el filtro de
# main.py compara contra UTC. Aqui se convierte sumando 4 h.
import json
import re
from collections import Counter
from datetime import datetime, timedelta

LOG = "rsi_iq.log"
# NO es la fecha de fin del entrenamiento: es la MTIME de models/meta_bbrev_iq.pkl.
# El log arranca el 17-jul y contiene predicciones de DOS modelos distintos (el anterior
# y el actual, reentrenado el 20-jul 15:06). Mezclarlos rompe la escala de probabilidad:
# el modelo viejo llegaba a P=0.944 y el actual no pasa de ~0.59. Hay que quedarse solo
# con lo posterior al reentrenamiento.
CORTE_ENTRENAMIENTO = datetime(2026, 7, 20, 15, 6, 0)
ROLLOVER = {20, 21, 22}
# Fraccion objetivo. OJO CON LA BASE: tiene que ser sobre TODAS las señales fuera del
# rollover, no sobre las que ya pasaron un umbral previo. En meta_bbrev_corte.pkl,
# fuera del rollover, P>=0.60 selecciona 496 de 195499 señales -> 0.254%.
# (Usar 496/25728 = 1.93% seria comparar contra la base "P>=0.54", que no es la misma.)
OBJETIVO = 496 / 195499

pat = re.compile(r'^\[([0-9T:\-\.]+)\]\s+(\S+)\s+\|.*meta P (\d\.\d+)')


def main():
    ps_todo, ps_sin_roll = [], []
    n_pre = 0
    for ln in open(LOG, encoding="utf-8", errors="replace"):
        m = pat.match(ln)
        if not m:
            continue
        ts = datetime.fromisoformat(m.group(1))
        if ts <= CORTE_ENTRENAMIENTO:
            n_pre += 1
            continue
        # Los -OTC no se operan (excluir_otc: true) pero el bot los escaneaba antes del
        # reinicio del 21-jul. Si entran, inflan la base y bajan el umbral de mas.
        if "-OTC" in m.group(2):
            continue
        p = float(m.group(3))
        h_utc = (ts + timedelta(hours=4)).hour     # log = hora local Chile (UTC-4)
        ps_todo.append(p)
        if h_utc not in ROLLOVER:
            ps_sin_roll.append(p)

    print(f"señales del modelo ANTERIOR (descartadas): {n_pre}")
    print(f"señales POSTERIORES (limpias): {len(ps_todo)}  "
          f"| fuera de rollover: {len(ps_sin_roll)}")
    if len(ps_sin_roll) < 1000:
        print("muestra insuficiente para recalibrar con confianza")
        return

    ps_sin_roll.sort()
    n = len(ps_sin_roll)
    idx = int((1 - OBJETIVO) * n)
    thr = ps_sin_roll[min(idx, n - 1)]
    print(f"\nfraccion objetivo (del modelo con corte, P>=0.60 sin rollover): "
          f"{100*OBJETIVO:.2f}%")
    print(f"umbral equivalente en el modelo de PRODUCCION: {thr:.3f}")

    print(f"\n{'umbral':>7} {'%sel':>7} {'n':>7}   (fuera de rollover)")
    for t in (0.54, 0.56, 0.58, 0.60, thr):
        k = sum(1 for x in ps_sin_roll if x >= t)
        print(f"{t:>7.3f} {100*k/n:>6.2f}% {k:>7}")

    print("\ndistribucion de P del modelo de produccion (datos no vistos):")
    for q in (50, 75, 90, 95, 98, 99, 99.5):
        print(f"  p{q:<5} {ps_sin_roll[int(q/100*(n-1))]:.3f}")
    print(f"  max    {ps_sin_roll[-1]:.3f}")

    print("\n--- cuantas operaciones/dia implicaria ---")
    dias = 0
    try:
        prim = min(datetime.fromisoformat(l.split(']')[0][1:])
                   for l in [] ) if False else None
    except Exception:
        pass
    print(f"  con {len(ps_sin_roll)} señales fuera de rollover en el log posterior al corte,")
    print(f"  un umbral {thr:.3f} deja {sum(1 for x in ps_sin_roll if x >= thr)} operaciones.")


if __name__ == "__main__":
    main()
