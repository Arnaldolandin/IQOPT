# hp_search.py - Busqueda de hiperparametros para el seq LSTM, honesta.
#
#   .venv314\Scripts\python.exe hp_search.py ETHUSD
#
# Reusa train_seq_save.dataset() (misma ventana_features que el bot). Corte temporal +
# embargo. Elige por val_loss en VALIDACION (ultimo tramo del train) y REPORTA el WR en
# el bloque de TEST intocado, separando rollover. Aviso: sobre un objetivo casi-aleatorio
# esto sobreajusta facil; el WR de test manda, no el val_loss.
import sys, os, itertools
import numpy as np

import seq_model as S
import train_seq_save as T

PAR = sys.argv[1] if len(sys.argv) > 1 else "ETHUSD"
CACHE = "cache_ohlc_5m_v2"
L = 64; H = 2
PAYOUT = 0.87; BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)

GRID = {
    "hidden": [32, 48, 64],
    "dropout": [0.2, 0.3, 0.4],
    "lr": [1e-3, 5e-4],
}


def entrenar(Xt, yt, Xv, yv, Xte, hidden, dropout, lr, epocas=60, seed=42):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    hp = {"hidden": hidden, "dropout": dropout, "capas": 1, "heads": 4}
    net = S.construir_red("lstm", S.N_FEATS, L, hp)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    Xt_t, yt_t = torch.tensor(Xt), torch.tensor(yt, dtype=torch.float32)
    Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv, dtype=torch.float32)
    mejor, best_state, pac = 1e9, None, 0
    for ep in range(epocas):
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
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            pac += 1
            if pac >= 8:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(Xte))).numpy()
    return p, mejor


def wr_test(p, y, t, thr=0.54):
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)
    sel = ((p >= thr) | (p <= 1 - thr)) & ~roll
    if int(sel.sum()) < 50:
        return float("nan"), 0
    gano = np.where(p >= thr, y == 1, y == 0)
    return float(gano[sel].mean()), int(sel.sum())


def main():
    print(f"[{PAR}] cargando dataset...", flush=True)
    X, y, t = T.dataset(PAR, L, CACHE)
    emb = (L + H) * 300
    corte = float(np.quantile(t, 0.65))
    m_tr, m_te = t < (corte - emb), t > (corte + emb)
    Xtr, ytr = X[m_tr], y[m_tr]
    k = int(0.85 * len(Xtr))
    Xt, yt, Xv, yv = Xtr[:k], ytr[:k], Xtr[k:], ytr[k:]
    Xte, yte, tte = X[m_te], y[m_te], t[m_te]
    print(f"train {len(Xt)} | val {len(Xv)} | test {len(Xte)} | break-even {100*BE:.2f}%\n", flush=True)

    combos = list(itertools.product(GRID["hidden"], GRID["dropout"], GRID["lr"]))
    filas = []
    for i, (hid, do, lr) in enumerate(combos):
        p, vl = entrenar(Xt, yt, Xv, yv, Xte, hid, do, lr)
        wr54, n54 = wr_test(p, yte, tte, 0.54)
        wr56, n56 = wr_test(p, yte, tte, 0.56)
        filas.append((hid, do, lr, vl, wr54, n54, wr56, n56))
        print(f"[{i+1}/{len(combos)}] hidden={hid} dropout={do} lr={lr:.0e} | "
              f"val_loss {vl:.5f} | test WR@0.54 {100*wr54:.2f}% (n={n54}) "
              f"@0.56 {100*wr56:.2f}% (n={n56})", flush=True)

    print("\n=== RANKING por val_loss (el criterio de seleccion honesto) ===")
    filas.sort(key=lambda r: r[3])
    print(f"{'hidden':>7} {'drop':>5} {'lr':>6} {'val_loss':>9} {'WR@0.54':>9} {'WR@0.56':>9}")
    for hid, do, lr, vl, wr54, n54, wr56, n56 in filas:
        print(f"{hid:>7} {do:>5} {lr:>6.0e} {vl:>9.5f} "
              f"{100*wr54:>8.2f}% {100*wr56:>8.2f}%")
    b = filas[0]
    print(f"\nMEJOR por val_loss: hidden={b[0]} dropout={b[1]} lr={b[2]:.0e} "
          f"(val_loss {b[3]:.5f}, test WR@0.54 {100*b[4]:.2f}%)")
    print(f"ln(2)={np.log(2):.5f} -> si el mejor val_loss no baja de ahi con holgura, no hay senal que optimizar.")


if __name__ == "__main__":
    main()
