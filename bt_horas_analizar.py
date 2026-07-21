# bt_horas_analizar.py - Diseño A: filtro por HORA UTC agregada (24 celdas).
#
#   .venv314\Scripts\python.exe bt_horas_analizar.py [--folds 6] [--perm 1000]
#
# Tres capas, en orden de exigencia:
#   1. In-sample: WR por hora sobre todo el periodo (solo descriptivo, NO decide nada).
#   2. Walk-forward anclado: para cada fold, se eligen las horas usando SOLO el pasado
#      y se evaluan en el futuro inmediato. Es el unico numero que vale.
#   3. Control de permutacion: se repite el walk-forward completo con las etiquetas de
#      hora barajadas. Si el resultado real no supera al percentil 95 del nulo, la
#      "mejora" es indistinguible de elegir horas al azar.
import argparse
import json
import random
from collections import defaultdict

PAYOUT = 0.87
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)      # 0.5348


def wr(filas):
    dec = [f for f in filas if f["res"] != "empate"]
    if not dec:
        return None, 0
    return sum(1 for f in dec if f["res"] == "win") / len(dec), len(dec)


def ev(w, n):
    """EV por operacion en unidades de stake."""
    if w is None:
        return 0.0
    return w * PAYOUT - (1 - w)


def elegir_horas(train, min_n, margen, clave="hora_utc"):
    """Horas cuyo WR en TRAIN supera break-even + margen, con muestra suficiente.
    OJO: 'clave' debe ser la MISMA que se usa al filtrar el test. Si el train
    seleccionara por la hora real y el test filtrara por la barajada, la
    permutacion no seria un nulo valido y el control no probaria nada."""
    por_hora = defaultdict(list)
    for f in train:
        por_hora[f[clave]].append(f)
    buenas = set()
    for h, fs in por_hora.items():
        w, n = wr(fs)
        if w is not None and n >= min_n and w >= BREAK_EVEN + margen:
            buenas.add(h)
    return buenas


def walk_forward(filas, folds, min_n, margen, clave="hora_utc"):
    """Anclado: train = todo lo anterior al fold, test = el fold. Devuelve
    (wr_filtrado, n_filtrado, wr_sin_filtro, n_sin_filtro)."""
    filas = sorted(filas, key=lambda f: f["ts"])
    N = len(filas)
    corte = N // (folds + 1)          # el primer bloque es solo entrenamiento
    sel, base = [], []
    for k in range(1, folds + 1):
        train = filas[:corte * k]
        test = filas[corte * k: corte * (k + 1)] if k < folds else filas[corte * k:]
        if not train or not test:
            continue
        buenas = elegir_horas(train, min_n, margen, clave)
        sel.extend(f for f in test if f[clave] in buenas)
        base.extend(test)
    w_f, n_f = wr(sel)
    w_b, n_b = wr(base)
    return w_f, n_f, w_b, n_b


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="bt_horas_senales.json")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--min-n", type=int, default=200, help="muestra minima por hora en train")
    ap.add_argument("--margen", type=float, default=0.0, help="margen sobre break-even")
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--excluir", default="BTCUSD",
                    help="pares a excluir, separados por coma. BTCUSD por defecto: "
                         "aporta ~35%% de las señales y NUNCA ejecuta (siempre suspended), "
                         "asi que calibrar el filtro con el equivale a calibrarlo con "
                         "operaciones imposibles.")
    ap.add_argument("--solo", default="", help="analizar solo estos pares (coma)")
    a = ap.parse_args()

    filas = json.load(open(a.entrada, encoding="utf-8"))
    ex = {p.strip() for p in a.excluir.split(",") if p.strip()}
    if ex:
        antes = len(filas)
        filas = [f for f in filas if f["par"] not in ex]
        print(f"excluidos {sorted(ex)}: {antes - len(filas)} señales fuera "
              f"({100*(antes-len(filas))/max(antes,1):.1f}%)")
    if a.solo:
        keep = {p.strip() for p in a.solo.split(",") if p.strip()}
        filas = [f for f in filas if f["par"] in keep]
    filas.sort(key=lambda f: f["ts"])
    w, n = wr(filas)
    print(f"señales={len(filas)}  decididas={n}  WR global={100*w:.2f}%  "
          f"break-even={100*BREAK_EVEN:.2f}%  EV/op={ev(w,n):+.4f}")
    print(f"periodo: {filas[0]['ts']} -> {filas[-1]['ts']}")

    # ---- 1. In-sample por hora (descriptivo) ----
    print("\n=== IN-SAMPLE por hora UTC (NO es evidencia, solo descripcion) ===")
    print(f"{'h':>3} {'n':>6} {'WR%':>7} {'EV/op':>8}")
    por_hora = defaultdict(list)
    for f in filas:
        por_hora[f["hora_utc"]].append(f)
    for h in range(24):
        w_h, n_h = wr(por_hora.get(h, []))
        if n_h:
            marca = " *" if w_h >= BREAK_EVEN else ""
            print(f"{h:>3} {n_h:>6} {100*w_h:>6.2f} {ev(w_h,n_h):>+8.4f}{marca}")

    # ---- 2. Walk-forward ----
    print(f"\n=== WALK-FORWARD anclado ({a.folds} folds, min_n={a.min_n}, "
          f"margen={a.margen}) ===")
    w_f, n_f, w_b, n_b = walk_forward(filas, a.folds, a.min_n, a.margen)
    if w_f is None:
        print("el filtro no selecciono ninguna hora -> sin operaciones OOS")
        return
    print(f"  sin filtro : WR {100*w_b:.2f}%  n={n_b}  EV/op {ev(w_b,n_b):+.4f}")
    print(f"  con filtro : WR {100*w_f:.2f}%  n={n_f}  EV/op {ev(w_f,n_f):+.4f}")
    delta = (w_f - w_b) * 100
    print(f"  mejora     : {delta:+.2f} pt de WR  "
          f"(descarta {100*(1-n_f/max(n_b,1)):.0f}% de las operaciones)")

    # ---- 3. Control de permutacion ----
    print(f"\n=== CONTROL DE PERMUTACION ({a.perm} barajadas) ===")
    print("baraja la etiqueta de hora y repite el walk-forward entero.")
    rnd = random.Random(12345)
    horas = [f["hora_utc"] for f in filas]
    nulos = []
    for _ in range(a.perm):
        rnd.shuffle(horas)
        for f, h in zip(filas, horas):
            f["_h"] = h
        w_p, n_p, w_bp, _ = walk_forward(filas, a.folds, a.min_n, a.margen, clave="_h")
        if w_p is not None:
            nulos.append((w_p - w_bp) * 100)
    if not nulos:
        print("  el nulo nunca selecciono horas; control no concluyente")
        return
    nulos.sort()
    p95 = nulos[int(0.95 * (len(nulos) - 1))]
    mejores = sum(1 for x in nulos if x >= delta)
    pval = (mejores + 1) / (len(nulos) + 1)
    print(f"  nulo: mediana {nulos[len(nulos)//2]:+.2f} pt, p95 {p95:+.2f} pt "
          f"(n={len(nulos)})")
    print(f"  real: {delta:+.2f} pt")
    print(f"  p-valor empirico = {pval:.4f}")
    if pval < 0.05 and delta > 0:
        print("  VEREDICTO: la mejora supera al azar (p<0.05).")
    else:
        print("  VEREDICTO: NO se distingue del azar. No activar el filtro.")


if __name__ == "__main__":
    main_()
