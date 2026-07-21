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
HP_DEFECTO = {"hidden": 48, "dropout": 0.3, "capas": 2, "heads": 4}


def construir_red(arq, n_feats=N_FEATS, L=L_DEFECTO, hp=None):
    """hp = hiperparametros de ARQUITECTURA (no de entrenamiento). Se guardan junto
    al modelo: al cargarlo hay que reconstruir exactamente la misma red o los pesos
    no encajan."""
    import torch.nn as nn
    import torch
    h = dict(HP_DEFECTO)
    h.update(hp or {})

    class LSTM(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.rnn = nn.LSTM(f, h["hidden"], batch_first=True,
                               num_layers=max(1, int(h["capas"])) if h.get("lstm_capas") else 1)
            self.do = nn.Dropout(h["dropout"])
            self.fc = nn.Linear(h["hidden"], 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(self.do(o[:, -1])).squeeze(-1)

    class Trafo(nn.Module):
        def __init__(self, f):
            super().__init__()
            d = h["hidden"]
            self.inp = nn.Linear(f, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            cap = nn.TransformerEncoderLayer(d, h["heads"], d * 2, dropout=h["dropout"],
                                             batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(cap, int(h["capas"]))
            self.fc = nn.Linear(d, 1)

        def forward(self, x):
            z = self.enc(self.inp(x) + self.pos)
            return self.fc(z.mean(1)).squeeze(-1)

    return LSTM(n_feats) if arq == "lstm" else Trafo(n_feats)


# ---------------------------------------------------------------- persistencia
_CACHE = {}


def exportar_npz(net, arq, path):
    """Exporta los pesos a .npz para inferir SIN torch.

    Por que existe: en Windows Server torch suele fallar al cargar c10.dll
    (WinError 1114) por falta del runtime de Visual C++ o por un CPU sin AVX2, y son
    122 MB de dependencia para hacer unas pocas multiplicaciones de matrices. La LSTM
    de produccion es de 1 capa y 48 unidades: en numpy son 30 lineas y el servidor no
    necesita torch para nada. torch queda solo en la maquina de entrenamiento.

    Solo se soporta 'lstm'. El transformer es experimental y no se despliega.
    """
    if arq != "lstm":
        return False
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    try:
        datos = {
            "W_ih": sd["rnn.weight_ih_l0"], "W_hh": sd["rnn.weight_hh_l0"],
            "b_ih": sd["rnn.bias_ih_l0"], "b_hh": sd["rnn.bias_hh_l0"],
            "fc_w": sd["fc.weight"], "fc_b": sd["fc.bias"],
        }
    except KeyError:
        return False
    np.savez(path.replace(".pt", "") + ".npz", **datos)
    return True


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predecir_npz(f, path_npz):
    """Forward de la LSTM en numpy puro. f = (L, N_FEATS). Devuelve P(sube).

    Orden de compuertas de PyTorch en weight_ih/weight_hh: [i, f, g, o].
    En eval() el dropout es identidad, por eso no aparece.
    """
    d = _CACHE_NPZ.get(path_npz)
    if d is None:
        z = np.load(path_npz)
        d = {k: z[k].astype(np.float64) for k in z.files}
        _CACHE_NPZ[path_npz] = d
    W_ih, W_hh, b = d["W_ih"], d["W_hh"], d["b_ih"] + d["b_hh"]
    hid = W_hh.shape[1]
    h = np.zeros(hid)
    c = np.zeros(hid)
    for x in f:
        g = W_ih @ x + W_hh @ h + b
        i_, f_, g_, o_ = g[:hid], g[hid:2 * hid], g[2 * hid:3 * hid], g[3 * hid:]
        i_ = _sigmoid(i_); f_ = _sigmoid(f_); g_ = np.tanh(g_); o_ = _sigmoid(o_)
        c = f_ * c + i_ * g_
        h = o_ * np.tanh(c)
    return float(_sigmoid(d["fc_w"] @ h + d["fc_b"])[0])


_CACHE_NPZ = {}


def guardar(net, arq, L, path, hp=None, meta=None):
    """Guarda pesos + la receta para reconstruir la red. Sin 'hp' aqui, un cambio de
    hiperparametros en config.json dejaria modelos viejos imposibles de cargar."""
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)
    npz = exportar_npz(net, arq, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"arq": arq, "L": L, "n_feats": N_FEATS,
                   "hp": {**HP_DEFECTO, **(hp or {})}, "npz": npz,
                   "meta": meta or {}}, f)


def cargar(path):
    """Devuelve (red_en_eval, L). Cachea por path: el bot llama a esto cada vela."""
    if path in _CACHE:
        return _CACHE[path]
    import torch
    with open(path + ".json", encoding="utf-8") as f:
        cfg = json.load(f)
    net = construir_red(cfg["arq"], cfg.get("n_feats", N_FEATS),
                        cfg.get("L", L_DEFECTO), cfg.get("hp"))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    _CACHE[path] = (net, cfg.get("L", L_DEFECTO))
    return _CACHE[path]


def predecir_p(velas_iq, path):
    """velas_iq = lista de dicts de get_candles, INCLUIDA la vela en formacion.
    Devuelve P(sube) sobre la ultima vela CERRADA, o None.

    Prefiere el .npz (numpy puro): asi el servidor no necesita torch, que en Windows
    Server falla a menudo al cargar c10.dll. Cae a torch solo si no hay .npz.
    """
    V = velas_iq_a_filas(velas_iq)[:-1]      # descarta la vela en formacion
    npz = path.replace(".pt", "") + ".npz"
    if os.path.isfile(npz):
        with open(path + ".json", encoding="utf-8") as fh:
            L = json.load(fh).get("L", L_DEFECTO)
        f = ventana_features(V, L)
        if f is None:
            return None
        return predecir_npz(f, npz)
    import torch
    net, L = cargar(path)
    f = ventana_features(V, L)
    if f is None:
        return None
    with torch.no_grad():
        x = torch.tensor(f).unsqueeze(0)
        return float(torch.sigmoid(net(x)).item())
