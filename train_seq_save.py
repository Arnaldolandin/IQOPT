# train_seq_save.py - Entrena el modelo secuencial y lo GUARDA para que lo use el bot.
#
#   .venv314\Scripts\python.exe train_seq_save.py --par EURUSD --arq lstm
#
# Usa seq_model.ventana_features(), la MISMA funcion que llama main.py en vivo. Si se
# cambia la construccion de features hay que reentrenar: el modelo guardado queda atado
# a esa version del vector.
#
# Corte temporal estricto + embargo de (L+H) velas, para que las ventanas de train y
# test no compartan barras a traves de la frontera.
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import seq_model as S

CACHE = "cache_ohlc_5m"
H = 2
PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)


def dataset(par, L):
    with open(os.path.join(CACHE, par + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    n = len(d["close"])
    V = [[d["times"][i], d["open"][i], d["high"][i], d["low"][i], d["close"][i]]
         for i in range(n)]
    X, y, t = [], [], []
    for i in range(L + S.ATR_P + 1, n - H):
        if V[i + H][0] - V[i][0] != H * 300:      # continuidad de la opcion
            continue
        if V[i + H][4] == V[i][4]:                # empate: feed parado
            continue
        f = S.ventana_features(V[:i + 1], L)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", default="EURUSD")
    ap.add_argument("--arq", default="lstm", choices=["lstm", "transformer"])
    ap.add_argument("--L", type=int, default=S.L_DEFECTO)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--salida", default="")
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    print(f"[{a.par}] construyendo ventanas L={a.L} H={H}...", flush=True)
    X, y, t = dataset(a.par, a.L)
    print(f"muestras {len(X)} | tasa base P(sube) {100*y.mean():.2f}%")

    emb = (a.L + H) * 300
    corte = float(np.quantile(t, 1 - a.test_frac))
    m_tr, m_te = t < (corte - emb), t > (corte + emb)
    f_ = lambda z: datetime.fromtimestamp(z, timezone.utc).strftime("%Y-%m-%d")
    print(f"corte {f_(corte)} | train {int(m_tr.sum())} | test {int(m_te.sum())}")

    Xtr, ytr = X[m_tr], y[m_tr]
    k = int(0.85 * len(Xtr))                 # validacion temporal, nunca aleatoria
    Xv, yv = Xtr[k:], ytr[k:]
    Xt, yt = Xtr[:k], ytr[:k]

    net = S.construir_red(a.arq, S.N_FEATS, a.L)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    Xt_t, yt_t = torch.tensor(Xt), torch.tensor(yt, dtype=torch.float32)
    Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv, dtype=torch.float32)
    mejor, mejor_est, pac = 1e9, None, 0
    for ep in range(a.epocas):
        net.train()
        perm = torch.randperm(len(Xt_t))
        for j in range(0, len(perm), 256):
            idx = perm[j:j + 256]
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
            if pac >= 8:
                break
        if ep % 10 == 0:
            print(f"  ep {ep:3d} val_loss {vl:.5f}", flush=True)
    if mejor_est:
        net.load_state_dict(mejor_est)
    net.eval()

    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(X[m_te]))).numpy()
    metricas(p, y[m_te], t[m_te], f"{a.arq.upper()} {a.par} (val_loss {mejor:.5f})")

    salida = a.salida or f"models/seq_{a.arq}_{a.par}.pt"
    S.guardar(net, a.arq, a.L, salida,
              meta={"par": a.par, "H": H, "val_loss": mejor,
                    "corte": f_(corte), "n_train": int(len(Xt))})
    print(f"\n[SAVE] {salida}  (+ .json con arq/L/meta)")


if __name__ == "__main__":
    main()
