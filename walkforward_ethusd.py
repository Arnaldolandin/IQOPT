# walkforward_ethusd.py - Walk-forward expanding window sobre ETHUSD.
#
#   .venv314\Scripts\python.exe walkforward_ethusd.py
#
# Entrena en meses 1..k, testea en mes k+1. Cada muestra aparece UNA sola vez
# en test (sin data leakage). Agrega predicciones OOF por hora para estabilizar
# las estimaciones. Embargo de 1 dia entre train y test.
import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

PAR = "ETHUSD"
CACHE = "cache_ohlc_5m_v2"
L = 64
H = 2
PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)
ATR_P = 14
VOL_P = 20
EMBARGO_DIAS = 1        # 1 dia = 288 velas
TRAIN_MIN_MESES = 3     # meses de train (FIJO, no expandente)
TEST_MESES = 1          # ventana de test
EPOCAS = 15             # reducido para rapidez
TRAIN_MAX_MESES = 3     # ventana de train fija (no expandente)


def cargar():
    with open(os.path.join(CACHE, PAR + ".json"), encoding="utf-8") as f:
        d = json.load(f)
    return {
        "times": np.array(d["times"], np.float64),
        "open": np.array(d["open"], np.float64),
        "high": np.array(d["high"], np.float64),
        "low": np.array(d["low"], np.float64),
        "close": np.array(d["close"], np.float64),
        "volume": np.array(d.get("volume", []), np.float64) if "volume" in d else None,
    }


def construir(datos):
    t = datos["times"]
    o, h, l, c = datos["open"], datos["high"], datos["low"], datos["close"]
    vol = datos["volume"]
    n = len(c)

    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.full(n, np.nan)
    for i in range(ATR_P, n):
        atr[i] = tr[i - ATR_P:i].mean()

    Xs, ys, ts = [], [], []
    for i in range(L, n - H):
        if t[i + H] - t[i] != H * 300:
            continue
        if t[i] - t[i - L] != L * 300:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if c[i + H] == c[i]:
            continue

        sl = slice(i - L + 1, i + 1)
        oo, hh, ll, cc = o[sl], h[sl], l[sl], c[sl]
        ca = np.maximum(oo, cc)
        cb = np.minimum(oo, cc)
        ret = np.diff(np.concatenate([[c[i - L]], cc])) / a
        hora = ((t[sl] // 3600) % 24) / 24.0

        if vol is not None and len(vol) == n:
            vv = vol
            med = np.empty(L)
            for j, k_ in enumerate(range(n - L, n)):
                w = vv[max(0, k_ - VOL_P + 1):k_ + 1]
                med[j] = w.mean() if len(w) else 0.0
            vol_rel = np.where(med > 0, vv[n - L:] / np.maximum(med, 1e-9) - 1.0, 0.0)
            vol_log = np.log1p(np.maximum(vv[n - L:], 0)) / 10.0
        else:
            vol_rel = np.zeros(L)
            vol_log = np.zeros(L)

        esc = max(float(np.std(np.diff(c[-(L + 1):]) / np.maximum(c[-(L + 1):-1], 1e-12))), 1e-9)
        f = np.stack([ret, (cc - oo) / a, (hh - ca) / a, (cb - ll) / a, (hh - ll) / a,
                      np.sin(2 * np.pi * hora), np.cos(2 * np.pi * hora),
                      vol_rel, vol_log], axis=1)
        if not np.isfinite(f).all():
            continue
        Xs.append(f.astype(np.float32))
        ys.append(int(c[i + H] > c[i]))
        ts.append(t[i])

    return np.array(Xs, np.float32), np.array(ys, np.int64), np.array(ts, np.float64)


def entrenar_lstm(Xtr, ytr, Xva, yva, epocas=30, seed=42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class LSTM(nn.Module):
        def __init__(self, f, hid=48):
            super().__init__()
            self.rnn = nn.LSTM(f, hid, batch_first=True, num_layers=2, dropout=0.3)
            self.do = nn.Dropout(0.3)
            self.fc = nn.Linear(hid, 1)
        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(self.do(o[:, -1])).squeeze(-1)

    f_dim = Xtr.shape[2]
    net = LSTM(f_dim)
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
            if paciencia >= 5:
                break
    if mejor_est:
        net.load_state_dict(mejor_est)
    net.eval()
    print(f"    val_loss={mejor:.5f}  epocas={ep+1}", flush=True)
    return net, mejor


def predecir_batch(net, Xte):
    import torch
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte))).numpy()


def metricas(p, y, t, etq, thr=0.54):
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)
    sel = (p >= thr) | (p <= 1 - thr)
    gano = np.where(p >= thr, y == 1, y == 0)
    m_todo = sel
    m_sin = sel & ~roll
    nn = int(m_todo.sum())
    nn_sin = int(m_sin.sum())
    if nn > 0:
        w = gano[m_todo].mean()
        print(f"  {etq} | thr={thr} | TODO: n={nn} WR={100*w:.2f}% EV={w*PAYOUT-(1-w):+.4f}")
    if nn_sin > 0:
        w = gano[m_sin].mean()
        print(f"  {etq} | thr={thr} | SIN rollover: n={nn_sin} WR={100*w:.2f}% EV={w*PAYOUT-(1-w):+.4f}")


def analisis_por_hora(p_oof, y_oof, t_oof):
    hor = ((t_oof // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)

    print(f"\n{'='*70}")
    print(" WALK-FORWARD: ANALISIS POR HORA UTC (OOF aggregated)")
    print(f"{'='*70}")

    # --- Raw market ---
    print("\n--- Comportamiento RAW del mercado ---")
    print(f"{'hora':>4} {'n':>6} {'P(sube)':>8} {'bias':>7}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 100:
            continue
        print(f"{h:>4} {nn:>6} {100*y_oof[m].mean():>7.2f}% {y_oof[m].mean()-0.5:>+6.3f}")

    # --- Modelo por hora ---
    print("\n--- Desempeno LSTM por hora (thr=0.54) ---")
    print(f"{'hora':>4} {'n':>6} {'n_op':>6} {'WR':>8} {'EV/op':>9} {'P_med':>8}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 100:
            continue
        thr = 0.54
        sel = (p_oof[m] >= thr) | (p_oof[m] <= 1 - thr)
        gano = np.where(p_oof[m] >= thr, y_oof[m] == 1, y_oof[m] == 0)
        n_op = int(sel.sum())
        if n_op >= 20:
            w = gano[sel].mean()
            ev = w * PAYOUT - (1 - w)
            print(f"{h:>4} {nn:>6} {n_op:>6} {100*w:>7.2f}% {ev:>+9.4f} {np.median(p_oof[m]):>8.4f}")
        else:
            print(f"{h:>4} {nn:>6} {n_op:>6}       -         - {np.median(p_oof[m]):>8.4f}")

    # --- Sesiones ---
    print("\n--- Por sesion ---")
    sesiones = {
        "Asia (00-08)": range(0, 8),
        "Europa (08-14)": range(8, 14),
        "US (14-20)": range(14, 20),
        "Rollover (20-22)": range(20, 23),
        "Noche (22-00)": [22, 23],
    }
    for nombre, horas in sesiones.items():
        m = np.isin(hor, list(horas))
        nn = int(m.sum())
        if nn < 100:
            continue
        thr = 0.54
        sel = (p_oof[m] >= thr) | (p_oof[m] <= 1 - thr)
        gano = np.where(p_oof[m] >= thr, y_oof[m] == 1, y_oof[m] == 0)
        n_op = int(sel.sum())
        if n_op >= 20:
            w = gano[sel].mean()
            ev = w * PAYOUT - (1 - w)
            print(f"  {nombre:>20}: n={nn:>6}  ops={n_op:>5}  "
                  f"WR={100*w:.2f}%  EV={ev:+.4f}  base={100*y_oof[m].mean():.2f}%")
        else:
            print(f"  {nombre:>20}: n={nn:>6}  ops={n_op:>5}")

    # --- CALL vs PUT ---
    print("\n--- Precision CALL vs PUT por hora (thr=0.54) ---")
    print(f"{'hora':>4} {'n_call':>7} {'WR_call':>8} {'n_put':>7} {'WR_put':>8}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 100:
            continue
        thr = 0.54
        call_m = p_oof[m] >= thr
        put_m = p_oof[m] <= 1 - thr
        nc = int(call_m.sum())
        np_ = int(put_m.sum())
        wr_c = y_oof[m][call_m].mean() if nc >= 10 else float('nan')
        wr_p = (1 - y_oof[m])[put_m].mean() if np_ >= 10 else float('nan')
        print(f"{h:>4} {nc:>7} {100*wr_c:>7.2f}% {np_:>7} {100*wr_p:>7.2f}%")


def main():
    from datetime import timedelta

    print(f"[{PAR}] cargando datos de {CACHE}/...", flush=True)
    datos = cargar()
    t0 = datos["times"][0]
    t1 = datos["times"][-1]
    print(f"  velas: {len(datos['times'])}  "
          f"desde {datetime.fromtimestamp(t0, timezone.utc).strftime('%Y-%m-%d')} "
          f"hasta {datetime.fromtimestamp(t1, timezone.utc).strftime('%Y-%m-%d')}")

    print(f"[{PAR}] construyendo ventanas L={L} H={H}...", flush=True)
    X, y, t = construir(datos)
    print(f"  muestras: {len(X)} | tasa base P(sube): {100*y.mean():.2f}%")

    # --- Walk-forward splits (ventana FIJA) ---
    emb = EMBARGO_DIAS * 86400
    test_seg = TEST_MESES * 30 * 86400
    train_seg = TRAIN_MAX_MESES * 30 * 86400

    fold = 0
    p_oof = []
    y_oof = []
    t_oof = []
    val_losses = []

    test_ini_abs = t[0] + train_seg + emb

    while True:
        test_ini = test_ini_abs + fold * test_seg
        test_fin = test_ini + test_seg
        train_ini = test_ini - emb - train_seg
        train_fin = test_ini - emb

        if test_fin > t[-1] or train_ini < t[0]:
            break

        m_tr = (t >= train_ini) & (t < train_fin)
        m_te = (t >= test_ini) & (t < test_fin)

        n_tr = int(m_tr.sum())
        n_te = int(m_te.sum())
        if n_tr < 500 or n_te < 100:
            fold += 1
            continue

        Xtr, ytr = X[m_tr], y[m_tr]
        Xte, yte, tte = X[m_te], y[m_te], t[m_te]

        kv = int(0.85 * len(Xtr))
        Xva, yva = Xtr[kv:], ytr[kv:]
        Xtr2, ytr2 = Xtr[:kv], ytr[:kv]

        fd = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
        print(f"\n[fold {fold}] train: {fd(train_ini)}-{fd(train_fin)} ({n_tr}) | "
              f"test: {fd(test_ini)}-{fd(test_fin)} ({n_te})", flush=True)

        net, vl = entrenar_lstm(Xtr2, ytr2, Xva, yva, epocas=EPOCAS)
        val_losses.append(vl)

        p = predecir_batch(net, Xte)
        p_oof.append(p)
        y_oof.append(yte)
        t_oof.append(tte)

        metricas(p, yte, tte, f"fold {fold}")

        fold += 1

    # --- Agregar todo ---
    p_all = np.concatenate(p_oof)
    y_all = np.concatenate(y_oof)
    t_all = np.concatenate(t_oof)
    print(f"\n{'='*70}")
    print(f" WALK-FORWARD RESUMEN: {fold} folds, {len(p_all)} muestras OOF")
    print(f" val_loss promedio: {np.mean(val_losses):.5f}  "
          f"(ln2={np.log(2):.5f}, margen {np.log(2)-np.mean(val_losses):.5f})")
    print(f"{'='*70}")

    # Metricas globales OOF
    for thr in (0.52, 0.53, 0.54, 0.56):
        sel = (p_all >= thr) | (p_all <= 1 - thr)
        gano = np.where(p_all >= thr, y_all == 1, y_all == 0)
        nn = int(sel.sum())
        if nn > 0:
            w = gano[sel].mean()
            hor = ((t_all // 3600) % 24).astype(int)
            roll = np.isin(hor, ROLLOVER)
            sel_sin = sel & ~roll
            nn_sin = int(sel_sin.sum())
            w_sin = gano[sel_sin].mean() if nn_sin > 0 else 0
            print(f"  thr={thr}  TODO: n={nn:>6}  WR={100*w:.2f}%  EV={w*PAYOUT-(1-w):+.4f}"
                  f"  |  SIN rollover: n={nn_sin:>6}  WR={100*w_sin:.2f}%  EV={w_sin*PAYOUT-(1-w_sin):+.4f}")

    analisis_por_hora(p_all, y_all, t_all)


if __name__ == "__main__":
    main()
