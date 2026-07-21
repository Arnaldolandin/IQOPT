# seq_model.py - Modelo secuencial (LSTM / Transformer) para decidir CALL/PUT.
#
# LA REGLA DE ORO DE ESTE ARCHIVO: la ventana de features se construye en UNA sola
# funcion (`ventana_features`) que usan TANTO el entrenamiento COMO el bot en vivo.
# Si esas dos rutas divergen aunque sea en la normalizacion, el bot alimenta al modelo
# con algo distinto de lo que vio entrenando y falla en silencio: sigue devolviendo
# probabilidades con pinta razonable, pero sin ningun significado.
import json
import os

import numpy as np

L_DEFECTO = 64        # velas de contexto
N_FEATS = 6
ATR_P = 14


def ventana_features(V, L=L_DEFECTO):
    """V = lista de [t, o, h, l, c] en orden cronologico; la ULTIMA es la vela de
    decision (ya cerrada). Devuelve array (L, N_FEATS) float32, o None si no alcanza.

    Normaliza por ATR local: sin eso la escala depende del nivel del precio y el
    modelo termina aprendiendo el nivel en vez de la forma.
    """
    if len(V) < L + ATR_P + 1:
        return None
    t = np.array([r[0] for r in V], np.float64)
    o = np.array([r[1] for r in V], np.float64)
    h = np.array([r[2] for r in V], np.float64)
    l = np.array([r[3] for r in V], np.float64)
    c = np.array([r[4] for r in V], np.float64)
    i = len(V) - 1                      # indice de la vela de decision

    # ATR de las ultimas ATR_P velas, incluida la de decision
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = float(tr[i - ATR_P:i].mean())
    if not np.isfinite(a) or a <= 0:
        return None

    # la ventana no puede tener huecos: un salto de mercado adentro cambia el
    # significado de todas las features normalizadas
    if t[i] - t[i - L] != L * 300:
        return None

    sl = slice(i - L + 1, i + 1)
    oo, hh, ll, cc = o[sl], h[sl], l[sl], c[sl]
    ca = np.maximum(oo, cc)
    cb = np.minimum(oo, cc)
    ret = np.diff(np.concatenate([[c[i - L]], cc])) / a
    hora = ((t[sl] // 3600) % 24) / 24.0
    f = np.stack([ret, (cc - oo) / a, (hh - ca) / a, (cb - ll) / a, (hh - ll) / a,
                  np.sin(2 * np.pi * hora)], axis=1)
    if not np.isfinite(f).all():
        return None
    return f.astype(np.float32)


def velas_iq_a_filas(velas):
    """Convierte la lista de dicts de get_candles al formato [t,o,h,l,c]."""
    return [[float(v.get("from", i)), float(v["open"]),
             float(v.get("max", v.get("high", v["close"]))),
             float(v.get("min", v.get("low", v["close"]))), float(v["close"])]
            for i, v in enumerate(velas)]


# ---------------------------------------------------------------- arquitecturas
def construir_red(arq, n_feats=N_FEATS, L=L_DEFECTO):
    import torch.nn as nn
    import torch

    class LSTM(nn.Module):
        def __init__(self, f, hid=48):
            super().__init__()
            self.rnn = nn.LSTM(f, hid, batch_first=True)
            self.do = nn.Dropout(0.3)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(self.do(o[:, -1])).squeeze(-1)

    class Trafo(nn.Module):
        def __init__(self, f, d=48, heads=4, capas=2):
            super().__init__()
            self.inp = nn.Linear(f, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            cap = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.3,
                                             batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(cap, capas)
            self.fc = nn.Linear(d, 1)

        def forward(self, x):
            z = self.enc(self.inp(x) + self.pos)
            return self.fc(z.mean(1)).squeeze(-1)

    return LSTM(n_feats) if arq == "lstm" else Trafo(n_feats)


# ---------------------------------------------------------------- persistencia
_CACHE = {}


def guardar(net, arq, L, path, meta=None):
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"arq": arq, "L": L, "n_feats": N_FEATS, "meta": meta or {}}, f)


def cargar(path):
    """Devuelve (red_en_eval, L). Cachea por path: el bot llama a esto cada vela."""
    if path in _CACHE:
        return _CACHE[path]
    import torch
    with open(path + ".json", encoding="utf-8") as f:
        cfg = json.load(f)
    net = construir_red(cfg["arq"], cfg.get("n_feats", N_FEATS), cfg.get("L", L_DEFECTO))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    _CACHE[path] = (net, cfg.get("L", L_DEFECTO))
    return _CACHE[path]


def predecir_p(velas_iq, path):
    """velas_iq = lista de dicts de get_candles, INCLUIDA la vela en formacion.
    Devuelve P(sube) sobre la ultima vela CERRADA, o None."""
    import torch
    V = velas_iq_a_filas(velas_iq)[:-1]      # descarta la vela en formacion
    net, L = cargar(path)
    f = ventana_features(V, L)
    if f is None:
        return None
    with torch.no_grad():
        x = torch.tensor(f).unsqueeze(0)
        return float(torch.sigmoid(net(x)).item())
