# seq_model.py - Modelo secuencial (LSTM / Transformer) para decidir CALL/PUT.
#
# LA REGLA DE ORO DE ESTE ARCHIVO: la ventana de features se construye en UNA sola
# funcion (`ventana_features`) que usan TANTO el entrenamiento COMO el bot en vivo.
# Si esas dos rutas divergen aunque sea en la normalizacion, el bot alimenta al modelo
# con algo distinto de lo que vio entrenando y falla en silencio: sigue devolviendo
# probabilidades con pinta razonable, pero sin ningun significado.
import json
import os
import pickle

import numpy as np

L_DEFECTO = 64        # velas de contexto
# 7 base + 2 de volumen + RSI + Bollinger %B. Los factores cross-asset (sistemico/residuo)
# se probaron y NO mejoraron nada, y en vivo main.py no podia alimentarlos.
N_FEATS = 11           # 9 base + RSI + Bollinger %B
ATR_P = 14
VOL_P = 20            # ventana para normalizar el volumen
RSI_P = 14            # periodo RSI
BB_P = 20             # periodo Bollinger Bands
BB_K = 2.0            # desviaciones estandar de Bollinger
_MIN_DATA = L_DEFECTO + max(ATR_P, RSI_P, BB_P) + 1


def ventana_features(V, L=L_DEFECTO, vol=None, sis=None, res=None):
    """V = lista de [t, o, h, l, c] en orden cronologico; la ULTIMA es la vela de
    decision (ya cerrada). Devuelve array (L, N_FEATS) float32, o None si no alcanza.

    Normaliza por ATR local: sin eso la escala depende del nivel del precio y el
    modelo termina aprendiendo el nivel en vez de la forma.

    Extras opcionales, alineados con V (misma longitud):
      vol : volumen por vela. Aporta la conviccion detras del movimiento, que el precio
            solo no muestra. Se normaliza contra su propia media movil. Si falta se
            rellena con ceros: el modelo entrenado con volumen rendira peor, pero no
            fallara. (IQ dejo de enviarlo en 18 activos desde 2026-03.)

      sis, res : SE ACEPTAN Y SE IGNORAN. No entran en el vector.
            Vienen de factores.py (retorno sistemico de la familia e idiosincratico) y
            la idea era distinguir "cayo todo el mercado" -- que tiende a continuar -- de
            "cayo solo este activo" -- que tiende a revertir. Hasta el 2026-08-04 el
            codigo los calculaba, normalizaba y recortaba... y NO los metia en el stack,
            mientras este docstring afirmaba que res era "la feature clave". Se midio
            antes de conectarlos: HGB sobre la ventana aplanada, mismo split 65/35 con
            embargo, 29 pares, unico cambio 11 columnas vs 13:

                AUC medio 11f 0.5026 | 13f 0.5043 | delta +0.0017 | t = +0.78

            Sin señal (los deltas van de -0.026 a +0.024, dispersion 10x la media). Asi
            que se borro el calculo muerto y se deja la firma por compatibilidad con
            train_seq_save.py, que aun los pasa.

            OJO si algun dia se conectan: hay que implementar ANTES el calculo en
            main.py. Produccion llama predecir_p(velas, path, extras={"vol": vol}) con
            las velas de UN par, y factores.py necesita cruzar timestamps de toda la
            familia. Reentrenar con 13 features sin eso deja al bot alimentando ceros en
            las dos columnas nuevas: falla EN SILENCIO.
    """
    # El minimo depende de la L QUE SE PIDE, no de L_DEFECTO. Con la constante _MIN_DATA
    # (fijada a L_DEFECTO=64) una ventana de L=32 se rechazaba siempre aunque tuviera datos
    # de sobra, y una de L=96 pasaba la guarda sin tener suficientes filas. Para L=64 da
    # exactamente _MIN_DATA, asi que el comportamiento de produccion no cambia.
    if len(V) < L + max(ATR_P, RSI_P, BB_P) + 1:
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
    # Hora del dia en codificacion circular: hacen falta seno Y coseno. Con solo el
    # seno, las 3 y las 9 dan el mismo valor (sin es simetrico respecto de las 6) y el
    # modelo no puede distinguir la manana de la tarde.
    hora = ((t[sl] // 3600) % 24) / 24.0

    n = len(V)
    if vol is not None and len(vol) == n:
        vv = np.asarray(vol, np.float64)
        # media movil de VOL_P velas terminando en cada vela de la ventana
        med = np.empty(L)
        for j, k in enumerate(range(n - L, n)):
            w = vv[max(0, k - VOL_P + 1):k + 1]
            med[j] = w.mean() if len(w) else 0.0
        vol_rel = np.where(med > 0, vv[n - L:] / np.maximum(med, 1e-9) - 1.0, 0.0)
        vol_log = np.log1p(np.maximum(vv[n - L:], 0)) / 10.0
    else:
        vol_rel = np.zeros(L)
        vol_log = np.zeros(L)

    # --- RSI(14) rolling: cada posicion de la ventana tiene su propio RSI ---
    cc_ext = c[i - L - RSI_P:i + 1]     # L + RSI_P + 1 cierres
    diff_ext = np.diff(cc_ext)           # L + RSI_P retornos
    gains = np.where(diff_ext > 0, diff_ext, 0.0)
    losses = np.where(diff_ext < 0, -diff_ext, 0.0)
    rsi_arr = np.zeros(L, np.float32)
    if len(gains) >= RSI_P:
        ag = float(gains[:RSI_P].mean())
        al = float(losses[:RSI_P].mean())
        for k in range(RSI_P, len(gains)):
            ag = (ag * (RSI_P - 1) + float(gains[k])) / RSI_P
            al = (al * (RSI_P - 1) + float(losses[k])) / RSI_P
            pos = k - RSI_P              # 0..L-1
            if 0 <= pos < L:
                rs = ag / max(al, 1e-12)
                rsi_arr[pos] = 2.0 * (1.0 - 1.0 / (1.0 + rs)) - 1.0  # -1..1

    # --- Bollinger %B(20, 2) rolling: posicion dentro de las bandas ---
    cc_bb = c[i - L - BB_P + 1:i + 1]   # L + BB_P - 1 cierres
    bb_pct = np.zeros(L, np.float32)
    for j in range(L):
        chunk = cc_bb[j:j + BB_P]
        sma = float(chunk.mean())
        std = float(chunk.std())
        bw = 2.0 * max(std, 1e-12)
        bb_pct[j] = (float(cc_bb[j + BB_P - 1]) - (sma - max(std, 1e-12))) / bw
    bb_pct = np.clip(bb_pct * 2.0 - 1.0, -1.0, 1.0)

    f = np.stack([ret, (cc - oo) / a, (hh - ca) / a, (cb - ll) / a, (hh - ll) / a,
                  np.sin(2 * np.pi * hora), np.cos(2 * np.pi * hora),
                  vol_rel, vol_log, rsi_arr, bb_pct], axis=1)
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

    class BiLSTM(nn.Module):
        """LSTM bidireccional de 'capas' capas: lee la ventana (64 velas PASADAS) hacia
        delante y hacia atras. No filtra futuro -- la ventana entera es historia cerrada.
        Con >1 capa, PyTorch aplica dropout ENTRE capas. Usa el hidden FINAL de la ULTIMA
        capa en ambas direcciones (contexto completo), no o[:,-1]."""
        def __init__(self, f):
            super().__init__()
            nl = max(1, int(h["capas"]))
            self.rnn = nn.LSTM(f, h["hidden"], num_layers=nl, batch_first=True,
                               bidirectional=True, dropout=(h["dropout"] if nl > 1 else 0.0))
            self.do = nn.Dropout(h["dropout"])
            self.fc = nn.Linear(2 * h["hidden"], 1)

        def forward(self, x):
            _, (hn, _cn) = self.rnn(x)          # hn: (nl*2, batch, hidden)
            hcat = torch.cat([hn[-2], hn[-1]], dim=1)   # ultima capa: [fwd_final, bwd_final]
            return self.fc(self.do(hcat)).squeeze(-1)

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

    if arq == "lstm":
        return LSTM(n_feats)
    if arq == "bilstm":
        return BiLSTM(n_feats)
    return Trafo(n_feats)


# ---------------------------------------------------------------- persistencia
_CACHE = {}


def exportar_npz(net, arq, path):
    """Exporta los pesos a .npz para inferir SIN torch.

    Por que existe: en Windows Server torch suele fallar al cargar c10.dll
    (WinError 1114) por falta del runtime de Visual C++ o por un CPU sin AVX2, y son
    122 MB de dependencia para hacer unas pocas multiplicaciones de matrices. La LSTM
    de produccion es de 1 capa y 48 unidades: en numpy son 30 lineas y el servidor no
    necesita torch para nada. torch queda solo en la maquina de entrenamiento.

    Se soportan 'lstm' y 'bilstm'. El transformer es experimental y no se despliega.
    """
    if arq not in ("lstm", "bilstm"):
        return False
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    try:
        datos = {"fc_w": sd["fc.weight"], "fc_b": sd["fc.bias"]}
        # numero de capas y direcciones desde el propio state_dict
        nl = 0
        while f"rnn.weight_ih_l{nl}" in sd:
            nl += 1
        bidir = "rnn.weight_ih_l0_reverse" in sd
        for n in range(nl):
            suf = "" if n == 0 else f"_l{n}"     # capa 0 con nombres LEGACY (retrocompat)
            datos[f"W_ih{suf}"] = sd[f"rnn.weight_ih_l{n}"]
            datos[f"W_hh{suf}"] = sd[f"rnn.weight_hh_l{n}"]
            datos[f"b_ih{suf}"] = sd[f"rnn.bias_ih_l{n}"]
            datos[f"b_hh{suf}"] = sd[f"rnn.bias_hh_l{n}"]
            if bidir:
                datos[f"W_ih{suf}_r"] = sd[f"rnn.weight_ih_l{n}_reverse"]
                datos[f"W_hh{suf}_r"] = sd[f"rnn.weight_hh_l{n}_reverse"]
                datos[f"b_ih{suf}_r"] = sd[f"rnn.bias_ih_l{n}_reverse"]
                datos[f"b_hh{suf}_r"] = sd[f"rnn.bias_hh_l{n}_reverse"]
    except KeyError:
        return False
    np.savez(path.replace(".pt", "") + ".npz", **datos)
    return True


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_run(seq, W_ih, W_hh, b):
    """Corre una LSTM de 1 capa sobre 'seq' (T, feats). Devuelve (outputs (T, hid),
    hidden final). Orden de compuertas de PyTorch en weight_ih/weight_hh: [i, f, g, o]."""
    hid = W_hh.shape[1]
    h = np.zeros(hid); c = np.zeros(hid)
    outs = np.empty((len(seq), hid))
    for t in range(len(seq)):
        g = W_ih @ seq[t] + W_hh @ h + b
        i_, f_, g_, o_ = g[:hid], g[hid:2 * hid], g[2 * hid:3 * hid], g[3 * hid:]
        i_ = _sigmoid(i_); f_ = _sigmoid(f_); g_ = np.tanh(g_); o_ = _sigmoid(o_)
        c = f_ * c + i_ * g_
        h = o_ * np.tanh(c)
        outs[t] = h
    return outs, h


def predecir_npz(f, path_npz):
    """Forward en numpy puro de la LSTM / BiLSTM exportada (N capas). f = (L, N_FEATS).
    Devuelve P(sube). En eval() el dropout es identidad, por eso no aparece.

    General y retrocompatible: capa 0 con nombres legacy (W_ih...), capas >0 indexadas
    (W_ih_l1...). Bidireccional si hay pesos '_r'. Entre capas, la entrada de la siguiente
    es la salida concatenada [forward, backward] de la actual (como en PyTorch)."""
    d = _CACHE_NPZ.get(path_npz)
    if d is None:
        z = np.load(path_npz)
        d = {k: z[k].astype(np.float64) for k in z.files}
        _CACHE_NPZ[path_npz] = d
    nl = 1
    while f"W_ih_l{nl}" in d:
        nl += 1
    seq = np.asarray(f, np.float64)
    last_hf = last_hb = None
    for n in range(nl):
        suf = "" if n == 0 else f"_l{n}"
        outs_f, hf = _lstm_run(seq, d[f"W_ih{suf}"], d[f"W_hh{suf}"],
                               d[f"b_ih{suf}"] + d[f"b_hh{suf}"])
        if f"W_ih{suf}_r" in d:
            outs_br, hb = _lstm_run(seq[::-1], d[f"W_ih{suf}_r"], d[f"W_hh{suf}_r"],
                                    d[f"b_ih{suf}_r"] + d[f"b_hh{suf}_r"])
            outs_b = outs_br[::-1]                       # volver al orden temporal
            seq = np.concatenate([outs_f, outs_b], axis=1)
            last_hf, last_hb = hf, hb
        else:
            seq = outs_f; last_hf, last_hb = hf, None
    h = np.concatenate([last_hf, last_hb]) if last_hb is not None else last_hf
    return float(_sigmoid(d["fc_w"] @ h + d["fc_b"])[0])


_CACHE_NPZ = {}


def predecir_npz_ensemble(f, paths):
    """Promedia las probabilidades de varios modelos (una semilla cada uno).

    No agrega informacion: reduce la VARIANZA del ajuste. Con esta relacion señal-ruido,
    parte de lo que predice un modelo suelto es el azar de su inicializacion concreta.
    Devuelve (P promedio, desviacion entre modelos). La desviacion sirve de filtro: si
    los modelos discrepan mucho en una vela, es que ahi no hay nada que predecir.
    """
    ps = [predecir_npz(f, p) for p in paths]
    return float(np.mean(ps)), float(np.std(ps))


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


_ULTIMA_DISPERSION = [0.0]


def predecir_p(velas_iq, path, extras=None):
    """velas_iq = lista de dicts de get_candles, INCLUIDA la vela en formacion.
    Devuelve P(sube) sobre la ultima vela CERRADA, o None.

    extras = dict opcional {vol, sis, res} alineado con las velas (sin la ultima).

    Orden de preferencia:
      1) ENSEMBLE de .npz por semilla (base_s1.npz, base_s2.npz, ...) si existen.
      2) .npz unico: numpy puro, sin torch. Asi el servidor no necesita torch, que en
         Windows Server falla al cargar c10.dll.
      3) torch, solo si no hay ningun .npz.
    """
    V = velas_iq_a_filas(velas_iq)[:-1]      # descarta la vela en formacion
    base = path.replace(".pt", "")
    with open(path + ".json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    L = cfg.get("L", L_DEFECTO)

    # Guarda contra el fallo silencioso: si el modelo se entreno con otro numero de
    # features que el que produce ventana_features(), las matrices no cuadran. Sin este
    # chequeo el error caeria en el except de predecir_seq como un "err seq" generico.
    n_esperado = cfg.get("n_feats", N_FEATS)
    if n_esperado != N_FEATS:
        raise RuntimeError(
            f"modelo entrenado con {n_esperado} features y ventana_features() "
            f"produce {N_FEATS}. Reentrena con train_seq_save.py")

    f = ventana_features(V, L, **(extras or {}))
    if f is None:
        return None

    miembros = [f"{base}_s{k}.npz" for k in range(1, 10)
                if os.path.isfile(f"{base}_s{k}.npz")]
    if miembros:
        pr, disp = predecir_npz_ensemble(f, miembros)
        _ULTIMA_DISPERSION[0] = disp
        return pr

    _ULTIMA_DISPERSION[0] = 0.0
    npz = base + ".npz"
    if os.path.isfile(npz):
        return predecir_npz(f, npz)

    import torch
    net, _ = cargar(path)
    with torch.no_grad():
        return float(torch.sigmoid(net(torch.tensor(f).unsqueeze(0))).item())


# ---------------------------------------------------------------- HGB combinado
# Un HistGradientBoosting entrenado sobre la MISMA ventana aplanada (L*N_FEATS) que el
# LSTM. Vive en un .pkl al lado del .pt (misma L, mismas features): el bot promedia
# P(LSTM) y P(HGB). Si el .pkl falta o falla, el bot queda con el LSTM solo.


def predecir_hgb(velas_iq, path_pkl, extras=None):
    """P(sube) del HistGradientBoosting, con las MISMA features que el LSTM.

    velas_iq = lista de dicts de get_candles (INCLUIDA la vela en formacion).
    Devuelve P(sube) sobre la ultima vela CERRADA, o None.
    """
    V = velas_iq_a_filas(velas_iq)[:-1]
    with open(path_pkl, "rb") as fh:
        blob = pickle.load(fh)
    L = blob.get("L", L_DEFECTO)
    f = ventana_features(V, L, **(extras or {}))
    if f is None:
        return None
    return float(blob["modelo"].predict_proba(f.reshape(1, -1))[:, 1][0])


# ---------------------------------------------------------------- cerrojo de instancia
# Un unico proceso por rol (un watchdog, un bot). Dos bots operando a la vez = ordenes
# DUPLICADAS sobre la misma senal con el stake al doble, invisible porque el _lock de
# main.py no cruza procesos. El intento con open(path,"w") fallaba: "w" trunca a 0 bytes
# y msvcrt.locking del byte 0 sobre archivo vacio tiene EXITO para dos procesos. Aqui se
# garantiza 1 byte real de ancla y NO se trunca tras bloquear. Lock del SO: se libera solo
# si el proceso muere o el PC se apaga.
_CERROJOS = []

def tomar_cerrojo(path):
    """Descriptor si se obtuvo el cerrojo exclusivo, o None si ya lo tiene otro proceso."""
    try:
        import msvcrt
        f = open(path, "a+")
        f.seek(0, 2)
        if f.tell() == 0:
            f.write("x"); f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            f.close()
            return None
    except ImportError:
        try:
            import fcntl
            f = open(path, "a+")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                return None
        except ImportError:
            return open(path, "a+")
    try:
        f.seek(1); f.write(f" pid={os.getpid()}\n"); f.flush()
    except Exception:
        pass
    _CERROJOS.append(f)
    return f
