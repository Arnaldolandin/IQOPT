# meta_umbrales_sin_roll.py - ¿Sobrevive el edge a umbrales altos al quitar la
# ventana de rollover (20-22 UTC), que es donde IQ rechaza las ordenes?
#
#   .venv314\Scripts\python.exe meta_umbrales_sin_roll.py
#
# Usa el modelo entrenado con corte estricto (models/meta_bbrev_corte.pkl) sobre el
# held-out que ese modelo nunca vio.
#
# El SE que se imprime es binomial y por tanto OPTIMISTA: ignora que las señales de
# pares correlacionados en el mismo instante no son independientes. El intervalo real
# es mas ancho que el mostrado.
import pickle
from datetime import datetime, timezone

import numpy as np

import meta_train_corte as M

PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)


def fila(etq, yy, n_dias):
    if len(yy) == 0:
        return f"{etq:>24} {0:>7}        -         -        -"
    w = yy.mean()
    e = w * PAYOUT - (1 - w)
    se = 100 * np.sqrt(w * (1 - w) / len(yy))
    return (f"{etq:>24} {len(yy):>7} {100*w:>7.2f}% {e:>+9.4f} "
            f"{len(yy)/n_dias:>7.1f}  +-{se:.2f}pt")


def main():
    T, X, Y, P = M.cargar({"BTCUSD"})
    o = np.argsort(T)
    T, X, Y = T[o], X[o], Y[o]
    m_te = T > (float(np.quantile(T, 0.65)) + 24 * 3600)
    mdl = pickle.load(open("models/meta_bbrev_corte.pkl", "rb"))
    p = mdl.predict_proba(X[m_te])[:, 1]
    y, t = Y[m_te], T[m_te]
    hor = np.array([datetime.fromtimestamp(x, timezone.utc).hour for x in t])
    roll = np.isin(hor, [20, 21, 22])
    dias = (t.max() - t.min()) / 86400.0

    print(f"\nheld-out: {dias:.0f} dias | break-even {100*BE:.2f}% | "
          f"n total {len(y)}")
    print(f"{'':>24} {'n':>7} {'WR':>8} {'EV/op':>9} {'ops/dia':>8}  SE(optimista)")
    for thr in (0.54, 0.56, 0.58, 0.60, 0.62, 0.65):
        m = p >= thr
        print()
        print(fila(f"P>={thr} TODO", y[m], dias))
        print(fila(f"P>={thr} solo 20-22", y[m & roll], dias))
        print(fila(f"P>={thr} SIN rollover", y[m & ~roll], dias))


if __name__ == "__main__":
    main()
