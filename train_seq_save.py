# train_seq_save.py - Entrena el modelo secuencial y lo GUARDA para que lo use el bot.
#
#   .venv314\Scripts\python.exe train_seq_save.py [--par EURUSD] [--arq lstm]
#
# Sin --par entrena todos los de config.json -> entrenamiento.pares.
#
# Usa seq_model.ventana_features(), la MISMA funcion que llama main.py en vivo. Si se
# cambia la construccion de features hay que reentrenar: el modelo queda atado a esa
# version del vector, y si divergen el bot falla EN SILENCIO (devuelve probabilidades
# de aspecto razonable pero sin significado).
#
# Corte temporal estricto + embargo de (L+H) velas, para que las ventanas de train y
# test no compartan barras a traves de la frontera.
#
# ENSEMBLE: con entrenamiento.semillas > 1 se entrena esa cantidad de modelos con
# semillas distintas y se guarda cada uno como *_sK. No agrega informacion: reduce la
# VARIANZA del ajuste, porque parte de lo que predice un modelo suelto es el azar de su
# inicializacion. El bot los promedia solo (seq_model.predecir_p los detecta).
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import seq_model as S

H = 2
PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)


def cfg_entrenamiento():
    """Lee config.json -> 'entrenamiento'. Los flags de CLI, si se pasan, mandan."""
    try:
        c = json.load(open("config.json", encoding="utf-8"))
    except Exception:
        c = {}
    d = {"pares": ["EURUSD"], "arquitectura": "lstm", "ventana_L": S.L_DEFECTO,
         "horizonte": 2, "epocas": 60, "batch_size": 256, "learning_rate": 1e-3,
         "weight_decay": 1e-2, "paciencia": 8, "test_frac": 0.35, "val_frac": 0.15,
         "seed": 42, "semillas": 1, "cache": "cache_ohlc_5m", "factores": "",
         "hp": dict(S.HP_DEFECTO)}
    d.update(c.get("entrenamiento", {}))
    d["hp"] = {**S.HP_DEFECTO, **(c.get("entrenamiento", {}).get("hp", {}))}
    return d


def dataset(par, L, cache="cache_ohlc_5m", fac=None):
    """Devuelve X (n,L,N_FEATS), y, t.

    'fac' = (times, sistemico, residuo) de factores.py, o None. El volumen se toma del
    propio json si esta (cache v2); si no, se rellena con ceros.
    """
    with open(os.path.join(cache, par + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    n = len(d["close"])
    V = [[d["times"][i], d["open"][i], d["high"][i], d["low"][i], d["close"][i]]
         for i in range(n)]
    vol = d.get("volume")
    vol = np.asarray(vol, np.float64) if vol else None

    sis = res = None
    if fac is not None:
        ft, fs, fr = fac
        pos = {int(x): k for k, x in enumerate(ft)}
        sis = np.zeros(n); res = np.zeros(n)
        for i in range(n):
            k = pos.get(int(d["times"][i]))
            if k is not None:
                sis[i] = fs[k]; res[i] = fr[k]

    cola = L + S.ATR_P
    X, y, t = [], [], []
    for i in range(cola + 1, n - H):
        if V[i + H][0] - V[i][0] != H * 300:      # continuidad de la opcion
            continue
        if V[i + H][4] == V[i][4]:                # empate: feed parado
            continue
        # Pasar solo la cola necesaria. Con V[:i+1] esto seria CUADRATICO:
        # ventana_features convierte toda la historia a numpy en cada iteracion
        # (~775M operaciones inutiles sobre 39k velas: 15 min contra 1).
        lo = max(0, i - cola)
        f = S.ventana_features(
            V[lo:i + 1], L,
            vol=None if vol is None else vol[lo:i + 1],
            sis=None if sis is None else sis[lo:i + 1],
            res=None if res is None else res[lo:i + 1])
        if f is None:
            continue
        X.append(f); y.append(int(V[i + H][4] > V[i][4])); t.append(V[i][0])
    return np.array(X, np.float32), np.array(y, np.int64), np.array(t, np.float64)


def metricas(p, y, t, etq):
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)
    print(f"\n=== {etq} | break-even {100*BE:.2f}% ===")
    print(f"  P mediana {np.median(p):.4f}  min {p.min():.4f}  max {p.max():.4f}")
    print(f"{'thr':>6} {'zona':>14} {'n':>7} {'WR':>8} {'EV/op':>9}")
    for thr in (0.52, 0.53, 0.54, 0.56, 0.58):
        sel = (p >= thr) | (p <= 1 - thr)
        gano = np.where(p >= thr, y == 1, y == 0)
        for nombre, m in (("TODO", sel), ("SIN rollover", sel & ~roll)):
            nn = int(m.sum())
            if nn < 20:
                print(f"{thr:>6} {nombre:>14} {nn:>7}       -         -")
                continue
            w = gano[m].mean()
            print(f"{thr:>6} {nombre:>14} {nn:>7} {100*w:>7.2f}% "
                  f"{w*PAYOUT-(1-w):>+9.4f}")


def entrenar_una(Xt, yt, Xv, yv, Xte, a, E, seed):
    """Entrena un modelo con una semilla. Devuelve (red, p_test, val_loss)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(int(seed))

    net = S.construir_red(a.arq, S.N_FEATS, a.L, E["hp"])
    opt = torch.optim.AdamW(net.parameters(), lr=float(E["learning_rate"]),
                            weight_decay=float(E["weight_decay"]))
    lossf = nn.BCEWithLogitsLoss()
    bs = int(E["batch_size"])
    Xt_t, yt_t = torch.tensor(Xt), torch.tensor(yt, dtype=torch.float32)
    Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv, dtype=torch.float32)
    mejor, mejor_est, pac = 1e9, None, 0
    for ep in range(a.epocas):
        net.train()
        perm = torch.randperm(len(Xt_t))
        for j in range(0, len(perm), bs):
            idx = perm[j:j + bs]
            opt.zero_grad()
            lossf(net(Xt_t[idx]), yt_t[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xv_t), yv_t).item()
        if vl < mejor - 1e-5:
            mejor, pac = vl, 0
            mejor_est = {kk: vv.clone() for kk, vv in net.state_dict().items()}
        else:
            pac += 1
            if pac >= int(E["paciencia"]):
                break
    if mejor_est:
        net.load_state_dict(mejor_est)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(Xte))).numpy()
    return net, p, mejor


def entrenar_par(par, a, E, fac=None):
    global H
    H = int(E["horizonte"])
    cache = E.get("cache", "cache_ohlc_5m")

    print(f"[{par}] arq={a.arq} L={a.L} H={H} cache={cache} "
          f"semillas={E.get('semillas', 1)} hp={E['hp']}", flush=True)
    X, y, t = dataset(par, a.L, cache, fac)
    if len(X) == 0:
        print(f"  {par}: sin muestras, salteado")
        return
    print(f"muestras {len(X)} | features {X.shape[2]} | "
          f"tasa base P(sube) {100*y.mean():.2f}%")

    emb = (a.L + H) * 300
    corte = float(np.quantile(t, 1 - a.test_frac))
    m_tr, m_te = t < (corte - emb), t > (corte + emb)
    f_ = lambda z: datetime.fromtimestamp(z, timezone.utc).strftime("%Y-%m-%d")
    print(f"corte {f_(corte)} | train {int(m_tr.sum())} | test {int(m_te.sum())}")

    Xtr, ytr = X[m_tr], y[m_tr]
    k = int((1.0 - float(E["val_frac"])) * len(Xtr))  # validacion TEMPORAL, nunca
    Xv, yv = Xtr[k:], ytr[k:]                         # aleatoria: con ventanas
    Xt, yt = Xtr[:k], ytr[:k]                         # solapadas filtraria el futuro
    Xte, yte, tte = X[m_te], y[m_te], t[m_te]

    salida = a.salida or f"models/seq_{a.arq}_{par}.pt"
    n_sem = max(1, int(E.get("semillas", 1)))
    base_seed = int(E.get("seed", 42))
    ps = []
    for s in range(n_sem):
        seed = base_seed + s
        net, p, vl = entrenar_una(Xt, yt, Xv, yv, Xte, a, E, seed)
        ps.append(p)
        print(f"  semilla {seed}: val_loss {vl:.5f}", flush=True)
        dst = salida if n_sem == 1 else salida.replace(".pt", f"_s{s+1}.pt")
        S.guardar(net, a.arq, a.L, dst, hp=E["hp"],
                  meta={"par": par, "H": H, "val_loss": vl, "seed": seed,
                        "corte": f_(corte), "n_train": int(len(Xt)),
                        "cache": cache, "semillas": n_sem})
        if n_sem > 1 and s == 0:
            # el .json que consulta el bot cuelga del nombre base
            import shutil
            shutil.copyfile(dst + ".json", salida + ".json")

    p_ens = np.mean(ps, axis=0)
    if n_sem > 1:
        for i, p in enumerate(ps, 1):
            metricas(p, yte, tte, f"{par} semilla {i} (individual)")
        disp = float(np.mean(np.std(ps, axis=0)))
        print(f"\n>>> dispersion media entre semillas: {disp:.4f}")
    metricas(p_ens, yte, tte,
             f"{a.arq.upper()} {par} ENSEMBLE x{n_sem}" if n_sem > 1
             else f"{a.arq.upper()} {par}")
    print(f"\n[SAVE] {salida} ({n_sem} semilla(s))")


def main():
    E = cfg_entrenamiento()
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", default="", help="uno solo; vacio = todos los de config")
    ap.add_argument("--arq", default=E["arquitectura"], choices=["lstm", "transformer"])
    ap.add_argument("--L", type=int, default=int(E["ventana_L"]))
    ap.add_argument("--test-frac", type=float, default=float(E["test_frac"]))
    ap.add_argument("--epocas", type=int, default=int(E["epocas"]))
    ap.add_argument("--semillas", type=int, default=int(E.get("semillas", 1)))
    ap.add_argument("--salida", default="")
    a = ap.parse_args()
    E["semillas"] = a.semillas

    facs = {}
    fpath = E.get("factores", "")
    if fpath and os.path.isfile(fpath):
        import factores
        facs = factores.cargar_factores(fpath)
        print(f"[FACTORES] {fpath}: {len(facs)} activos")
    elif fpath:
        print(f"[FACTORES] {fpath} no existe -> se entrena SIN factores cross-asset")

    pares = [a.par] if a.par else list(E.get("pares", ["EURUSD"]))
    for i, par in enumerate(pares, 1):
        print("\n" + "=" * 60)
        print(f"[{i}/{len(pares)}] {par}")
        print("=" * 60, flush=True)
        entrenar_par(par, a, E, facs.get(par))


if __name__ == "__main__":
    main()
