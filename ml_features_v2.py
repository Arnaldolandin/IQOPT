# ml_features_v2.py - Features extendidas para el modelo DIRECTO (sin primario).
#
# NO modifica ml_features.py a proposito: el bot en produccion y models/meta_bbrev_iq.pkl
# dependen de ese vector exacto. Tocarlo romperia el bot en el proximo reinicio.
#
# v2 = las 35 features actuales + la forma CRUDA de las ultimas K velas, normalizada
# por ATR. La hipotesis es que extract_features comprime 100 velas a 35 numeros y en esa
# compresion se pierde la estructura de mechas/cuerpos, que para reversion es central:
# un cierre en el extremo y un rechazo con mecha larga hoy son indistinguibles si los
# indicadores coinciden.
import numpy as np

from ml_features import extract_features, FEATURE_NAMES

K_VELAS = 15          # cuantas velas crudas se agregan


def _atr(velas, periodo=14):
    trs = []
    for i in range(1, len(velas)):
        h, l, pc = velas[i][2], velas[i][3], velas[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return 0.0
    return float(np.mean(trs[-periodo:]))


def _nombres_v2():
    n = list(FEATURE_NAMES)
    for j in range(K_VELAS):
        n += [f"c{j}_body", f"c{j}_upper", f"c{j}_lower", f"c{j}_range"]
    return n


FEATURE_NAMES_V2 = _nombres_v2()


def extract_features_v2(velas, velas_mtf=None):
    """velas = lista [epoch, o, h, l, c]. Devuelve (vector, precio)."""
    base, px = extract_features(velas, velas_mtf=velas_mtf)
    if len(base) == 0:
        return np.array([]), 0
    if len(velas) < K_VELAS + 1:
        return np.array([]), 0

    atr = _atr(velas)
    if not atr or atr <= 0:
        # sin volatilidad medible las razones no tienen escala -> descartar la fila
        return np.array([]), 0

    extra = np.empty(K_VELAS * 4, dtype=np.float64)
    ult = velas[-K_VELAS:]
    for j, v in enumerate(ult):
        o, h, l, c = v[1], v[2], v[3], v[4]
        cuerpo_alto = max(o, c)
        cuerpo_bajo = min(o, c)
        extra[j * 4 + 0] = (c - o) / atr          # cuerpo con signo
        extra[j * 4 + 1] = (h - cuerpo_alto) / atr  # mecha superior
        extra[j * 4 + 2] = (cuerpo_bajo - l) / atr  # mecha inferior
        extra[j * 4 + 3] = (h - l) / atr            # rango
    return np.concatenate([base, extra]), px
