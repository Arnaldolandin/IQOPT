# eurusd_seq.py - LSTM y Transformer sobre EURUSD, horizonte 2 (10 min / binary).
#
#   .venv314\Scripts\python.exe eurusd_seq.py [--modelo lstm|transformer|ambos]
#
# POR QUE UN SOLO PAR: elimina de raiz la fuga cross-seccional que arruino las
# mediciones de esta sesion (~50 señales simultaneas de pares correlacionados cayendo
# a ambos lados del corte). Con un par solo queda el solapamiento entre ventanas
# contiguas, que se maneja con embargo.
#
# EL RIESGO AQUI ES EL OPUESTO: ~39k velas es MUY poco para un modelo secuencial y la
# relacion señal-ruido es minima. Por eso:
#   - modelos deliberadamente chicos (1 capa, hidden 32-64) y con dropout
#   - early stopping sobre un tramo de validacion tomado del FINAL del train (temporal,
#     nunca aleatorio: un split aleatorio filtraria el futuro por el solapamiento)
#   - baseline obligatoria (HistGradientBoosting) y control con etiquetas barajadas,
#     para saber cual es el suelo de ruido antes de creerle nada al modelo
#
# Horizonte 2 -> expiry 10m -> binary ~87% -> break-even 53.48%.
import argparse
import json
import os

import numpy as np

PAR = "EURUSD"
CACHE = "cache_ohlc_5m"
L = 64            # velas de contexto que ve el modelo secuencial
H = 2             # horizonte en velas (10 min)
PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)


def cargar():
    with open(os.path.join(CACHE, PAR + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    return (np.array(d["times"], np.float64), np.array(d["open"], np.float64),
            np.array(d["high"], np.float64), np.array(d["low"], np.float64),
            np.array(d["close"], np.float64))


def construir():
    """Devuelve X (n, L, 6), y (n,), t (n,). Normaliza por ATR local: sin eso la
    escala cambia con el nivel del precio y el modelo aprende el nivel, no la forma."""
    t, o, h, l, c = cargar()
    n = len(c)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.full(n, np.nan)
    k = 14
    for i in range(k, n):
        atr[i] = tr[i - k:i].mean()

    Xs, ys, ts = [], [], []
    for i in range(L, n - H):
        # continuidad: la opcion de 10 min solo existe si no hay hueco de mercado
        if t[i + H] - t[i] != H * 300:
            continue
        if t[i] - t[i - L] != L * 300:      # ventana sin huecos internos
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if c[i + H] == c[i]:                # empate: feed parado, no operable
            continue
        sl = slice(i - L + 1, i + 1)
        oo, hh, ll, cc = o[sl], h[sl], l[sl], c[sl]
        cuerpo_alto = np.maximum(oo, cc)
        cuerpo_bajo = np.minimum(oo, cc)
        ret = np.diff(np.concatenate([[c[i - L]], cc])) / a
        hora = ((t[sl] // 3600) % 24) / 24.0
        f = np.stack([
            ret,                          # retorno normalizado
            (cc - oo) / a,                # cuerpo
            (hh - cuerpo_alto) / a,       # mecha superior
            (cuerpo_bajo - ll) / a,       # mecha inferior
            (hh - ll) / a,                # rango
            np.sin(2 * np.pi * hora),     # hora del dia
        ], axis=1)
        Xs.append(f.astype(np.float32))
        ys.append(int(c[i + H] > c[i]))
        ts.append(t[i])
    return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(ts, np.float64)


def particion(t, test_frac=0.35, embargo=None):
    """Corte temporal + embargo. El embargo debe cubrir L+H velas: si no, las ventanas
    de train y test se solapan y el 'out-of-sample' comparte barras con el train."""
    if embargo is None:
        embargo = (L + H) * 300
    corte = np.quantile(t, 1 - test_frac)
    return t < (corte - embargo), t > (corte + embargo), corte


def metricas(p, y, t, etq):
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)
    print(f"\n=== {etq} | break-even {100*BE:.2f}% ===")
    print(f"  P mediana {np.median(p):.4f}  min {p.min():.4f}  max {p.max():.4f}")
    print(f"{'thr':>6} {'zona':>14} {'n':>7} {'WR':>8} {'EV/op':>9}")
    for thr in (0.52, 0.54, 0.56, 0.58, 0.60):
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


def baseline(Xtr, ytr, Xte, yte, tte):
    """HistGradientBoosting sobre la ventana aplanada. Si un LSTM no le gana a esto,
    la arquitectura secuencial no esta aportando nada."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.03, max_depth=4,
                                       l2_regularization=2.0, min_samples_leaf=40,
                                       random_state=42)
    m.fit(Xtr.reshape(len(Xtr), -1), ytr)
    p = m.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]
    metricas(p, yte, tte, "BASELINE HistGradientBoosting (ventana aplanada)")
    return p


def entrenar_torch(Xtr, ytr, Xva, yva, Xte, arq, epocas=60, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cpu"

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

    f = Xtr.shape[2]
    net = (LSTM(f) if arq == "lstm" else Trafo(f)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva, dtype=torch.float32)
    bs = 256
    mejor, mejor_est, paciencia = 1e9, None, 0
    for ep in range(epocas):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for j in range(0, len(perm), bs):
            idx = perm[j:j + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xva_t), yva_t).item()
        if vl < mejor - 1e-5:
            mejor, paciencia = vl, 0
            mejor_est = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            paciencia += 1
            if paciencia >= 8:
                break
        if ep % 10 == 0:
            print(f"    ep {ep:3d} val_loss {vl:.5f}", flush=True)
    if mejor_est:
        net.load_state_dict(mejor_est)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte))).numpy(), mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="ambos", choices=["lstm", "transformer", "ambos"])
    ap.add_argument("--control", action="store_true",
                    help="ademas, entrena con etiquetas barajadas para ver el suelo de ruido")
    a = ap.parse_args()

    print(f"[{PAR}] construyendo ventanas L={L} H={H}...", flush=True)
    X, y, t = construir()
    print(f"muestras {len(X)} | forma {X.shape} | tasa base P(sube) {100*y.mean():.2f}%")

    m_tr, m_te, corte = particion(t)
    from datetime import datetime, timezone
    f = lambda z: datetime.fromtimestamp(z, timezone.utc).strftime("%Y-%m-%d")
    print(f"corte {f(corte)} | train {int(m_tr.sum())} | test {int(m_te.sum())} | "
          f"embargadas {len(t)-int(m_tr.sum())-int(m_te.sum())}")

    Xtr, ytr = X[m_tr], y[m_tr]
    Xte, yte, tte = X[m_te], y[m_te], t[m_te]
    # validacion = ultimo 15% del train, temporal (nunca aleatoria: el solapamiento
    # entre ventanas contiguas filtraria el futuro y el early stopping mentiria)
    k = int(0.85 * len(Xtr))
    Xva, yva = Xtr[k:], ytr[k:]
    Xtr2, ytr2 = Xtr[:k], ytr[:k]
    print(f"train {len(Xtr2)} | val {len(Xva)} | test {len(Xte)}")

    baseline(Xtr2, ytr2, Xte, yte, tte)

    arqs = ["lstm", "transformer"] if a.modelo == "ambos" else [a.modelo]
    for arq in arqs:
        print(f"\n[{arq.upper()}] entrenando...", flush=True)
        p, vl = entrenar_torch(Xtr2, ytr2, Xva, yva, Xte, arq)
        metricas(p, yte, tte, f"{arq.upper()} (mejor val_loss {vl:.5f})")
        if a.control:
            rng = np.random.default_rng(0)
            yb = rng.permutation(ytr2)
            pc, _ = entrenar_torch(Xtr2, yb, Xva, yva, Xte, arq, epocas=25, seed=1)
            metricas(pc, yte, tte, f"{arq.upper()} CONTROL (etiquetas barajadas)")


if __name__ == "__main__":
    main()
