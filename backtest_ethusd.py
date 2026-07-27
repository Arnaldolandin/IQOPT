# backtest_ethuusd.py - Backtest ETHUSD: LSTM vs baseline + analisis de patrones por hora.
#
#   .venv314\Scripts\python.exe backtest_ethusd.py
#
# Carga cache_ohlc_5m_v2/ETHUSD.json, entrena LSTM (y baseline + control barajado),
# y desglosa todo por hora UTC para encontrar patrones explotables.
import json
import os

import numpy as np

PAR = "ETHUSD"
CACHE = "cache_ohlc_5m_v2"
L = 64
H = 2
PAYOUT = 0.87
BE = 1.0 / (1.0 + PAYOUT)
ROLLOVER = (20, 21, 22)
ATR_P = 14
VOL_P = 20


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
    """X (n, L, 9), y (n,), t (n,) usando las mismas 9 features que el bot en vivo."""
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
            for j, k in enumerate(range(n - L, n)):
                w = vv[max(0, k - VOL_P + 1):k + 1]
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


def particion(t, test_frac=0.35, embargo=None):
    if embargo is None:
        embargo = (L + H) * 300
    corte = np.quantile(t, 1 - test_frac)
    return t < (corte - embargo), t > (corte + embargo), corte


def metricas_generales(p, y, t, etq):
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)
    print(f"\n{'='*60}")
    print(f" {etq}")
    print(f"{'='*60}")
    print(f"  P mediana {np.median(p):.4f}  min {p.min():.4f}  max {p.max():.4f}")
    print(f"  break-even: {100*BE:.2f}%")
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


def analisis_por_hora_completo(p, y, t, X):
    """Analisis detallado por hora UTC: raw behavior + desempeno del modelo."""
    hor = ((t // 3600) % 24).astype(int)
    roll = np.isin(hor, ROLLOVER)

    print(f"\n{'='*60}")
    print(" ANALISIS POR HORA UTC (test set)")
    print(f"{'='*60}")

    # --- 1. Comportamiento RAW del mercado por hora ---
    print("\n--- Comportamiento RAW del mercado ---")
    print(f"{'hora':>4} {'n':>6} {'P(sube)':>8} {'bias':>7} {'rango_med':>10} {'vol_med':>10}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 10:
            continue
        base_rate = y[m].mean()
        bias = base_rate - 0.5
        # rango y volumen promedio de esta hora
        rr = X[m, :, 4].mean()  # feature range = (high-low)/atr, promedio en la ventana
        vv = X[m, :, 7].mean()  # feature vol_rel
        print(f"{h:>4} {nn:>6} {100*base_rate:>7.2f}% {bias:>+6.3f} "
              f"{rr:>10.4f} {vv:>+10.4f}")

    # --- 2. Desempeno del modelo por hora ---
    print("\n--- Desempeno del modelo por hora (thr=0.54, CALL si P>=0.54, PUT si P<=0.46) ---")
    print(f"{'hora':>4} {'n':>6} {'n_op':>6} {'WR':>8} {'EV/op':>9} {'P_med':>8} "
          f"{'calls':>6} {'puts':>6}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 10:
            continue
        thr = 0.54
        sel = (p[m] >= thr) | (p[m] <= 1 - thr)
        gano = np.where(p[m] >= thr, y[m] == 1, y[m] == 0)
        n_op = int(sel.sum())
        calls = int((p[m] >= thr).sum())
        puts = int((p[m] <= 1 - thr).sum())
        if n_op > 0:
            w = gano[sel].mean()
            ev = w * PAYOUT - (1 - w)
            print(f"{h:>4} {nn:>6} {n_op:>6} {100*w:>7.2f}% {ev:>+9.4f} "
                  f"{np.median(p[m]):>8.4f} {calls:>6} {puts:>6}")
        else:
            print(f"{h:>4} {nn:>6} {n_op:>6}       -         - {np.median(p[m]):>8.4f} "
                  f"{calls:>6} {puts:>6}")

    # --- 3. Analisis S/ rollover ---
    print("\n--- Desempeno SIN rollover (horas 20-22 excluidas) ---")
    sin_roll = ~roll
    print(f"{'hora':>4} {'n':>6} {'n_op':>6} {'WR':>8} {'EV/op':>9}")
    for h in range(24):
        if h in ROLLOVER:
            continue
        m = (hor == h) & sin_roll
        nn = int(m.sum())
        if nn < 10:
            continue
        thr = 0.54
        sel = (p[m] >= thr) | (p[m] <= 1 - thr)
        gano = np.where(p[m] >= thr, y[m] == 1, y[m] == 0)
        n_op = int(sel.sum())
        if n_op > 0:
            w = gano[sel].mean()
            ev = w * PAYOUT - (1 - w)
            print(f"{h:>4} {nn:>6} {n_op:>6} {100*w:>7.2f}% {ev:>+9.4f}")
        else:
            print(f"{h:>4} {nn:>6} {n_op:>6}       -         -")

    # --- 4. Horas con mayor dispersion del modelo ---
    print("\n--- Horas donde el modelo esta mas seguro (|P-0.5| alto) ---")
    conf = np.abs(p - 0.5)
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 10:
            continue
        print(f"  hora {h:02d}: conf media {conf[m].mean():.4f}  "
              f"n_alto_conf (|P-0.5|>0.04) {int((conf[m] > 0.04).sum())}")

    # --- 5. Sesion Asia/Europa/US ---
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
        if nn < 10:
            continue
        thr = 0.54
        sel = (p[m] >= thr) | (p[m] <= 1 - thr)
        gano = np.where(p[m] >= thr, y[m] == 1, y[m] == 0)
        n_op = int(sel.sum())
        if n_op > 0:
            w = gano[sel].mean()
            ev = w * PAYOUT - (1 - w)
            print(f"  {nombre:>20}: n={nn:>5}  operaciones={n_op:>5}  "
                  f"WR={100*w:.2f}%  EV/op={ev:+.4f}  "
                  f"base_rate={100*y[m].mean():.2f}%")
        else:
            print(f"  {nombre:>20}: n={nn:>5}  operaciones=0")

    # --- 6. Matriz confusion por hora: CALL vs PUT accuracy ---
    print("\n--- Precision CALL vs PUT por hora (thr=0.54) ---")
    print(f"{'hora':>4} {'n_call':>7} {'WR_call':>8} {'n_put':>7} {'WR_put':>8}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 10:
            continue
        thr = 0.54
        call_mask = p[m] >= thr
        put_mask = p[m] <= 1 - thr
        nc = int(call_mask.sum())
        np_ = int(put_mask.sum())
        wr_c = y[m][call_mask].mean() if nc > 5 else float('nan')
        wr_p = (1 - y[m])[put_mask].mean() if np_ > 5 else float('nan')
        print(f"{h:>4} {nc:>7} {100*wr_c:>7.2f}% {np_:>7} {100*wr_p:>7.2f}%")


def baseline(Xtr, ytr, Xte, yte, tte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.03, max_depth=4,
                                       l2_regularization=2.0, min_samples_leaf=40,
                                       random_state=42)
    m.fit(Xtr.reshape(len(Xtr), -1), ytr)
    p = m.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]
    metricas_generales(p, yte, tte, "BASELINE HistGradientBoosting")
    analisis_por_hora_completo(p, yte, tte, Xte)
    return p


def entrenar_torch(Xtr, ytr, Xva, yva, Xte, arq="lstm", epocas=30, seed=42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cpu"

    class LSTM(nn.Module):
        def __init__(self, f, hid=48):
            super().__init__()
            self.rnn = nn.LSTM(f, hid, batch_first=True, num_layers=2,
                               dropout=0.3)
            self.do = nn.Dropout(0.3)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(self.do(o[:, -1])).squeeze(-1)

    f = Xtr.shape[2]
    net = LSTM(f).to(dev)
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
    print(f"  mejor val_loss: {mejor:.5f}  (ln2={np.log(2):.4f}, margen {np.log(2)-mejor:.5f})")
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte))).numpy(), mejor


def main():
    from datetime import datetime, timezone

    print(f"[{PAR}] cargando datos de {CACHE}/...", flush=True)
    datos = cargar()
    print(f"  velas: {len(datos['times'])}  "
          f"desde {datetime.fromtimestamp(datos['times'][0], timezone.utc).strftime('%Y-%m-%d')} "
          f"hasta {datetime.fromtimestamp(datos['times'][-1], timezone.utc).strftime('%Y-%m-%d')}")

    print(f"[{PAR}] construyendo ventanas L={L} H={H}...", flush=True)
    X, y, t = construir(datos)
    print(f"  muestras: {len(X)} | forma: {X.shape} | tasa base P(sube): {100*y.mean():.2f}%")

    # Split 20/80: entrenamos poco (2 meses) para testear en ~8 meses
    # y tener muestras por hora grandes (~1000+ ops/hora).
    m_tr, m_te, corte = particion(t, test_frac=0.80)
    f = lambda z: datetime.fromtimestamp(z, timezone.utc).strftime("%Y-%m-%d")
    print(f"  corte: {f(corte)} | train: {int(m_tr.sum())} | test: {int(m_te.sum())} | "
          f"embargadas: {len(t)-int(m_tr.sum())-int(m_te.sum())}")

    Xtr, ytr = X[m_tr], y[m_tr]
    Xte, yte, tte = X[m_te], y[m_te], t[m_te]
    k = int(0.85 * len(Xtr))
    Xva, yva = Xtr[k:], ytr[k:]
    Xtr2, ytr2 = Xtr[:k], ytr[:k]
    print(f"  train: {len(Xtr2)} | val: {len(Xva)} | test: {len(Xte)}")

    print("\n--- BASELINE ---")
    baseline(Xtr2, ytr2, Xte, yte, tte)

    print("\n--- LSTM ---")
    p, vl = entrenar_torch(Xtr2, ytr2, Xva, yva, Xte)
    metricas_generales(p, yte, tte, f"LSTM (val_loss={vl:.5f})")
    analisis_por_hora_completo(p, yte, tte, Xte)

    print("\n--- LSTM CONTROL (etiquetas barajadas, suelo de ruido) ---")
    rng = np.random.default_rng(0)
    yb = rng.permutation(ytr2)
    pc, _ = entrenar_torch(Xtr2, yb, Xva, yva, Xte, epocas=15, seed=1)
    metricas_generales(pc, yte, tte, "LSTM CONTROL (labels shuffled)")

    # --- Bonus: analisis crudo de la serie sin modelo ---
    print(f"\n{'='*60}")
    print(" ANALISIS CRUDO DE LA SERIE ETHUSD (sin modelo)")
    print(f"{'='*60}")
    hor = ((tte // 3600) % 24).astype(int)
    print("\nDireccionalidad cruda por hora:")
    print(f"{'hora':>4} {'n':>6} {'P(sube)':>8} {'bias':>7}")
    for h in range(24):
        m = hor == h
        nn = int(m.sum())
        if nn < 10:
            continue
        br = yte[m].mean()
        print(f"{h:>4} {nn:>6} {100*br:>7.2f}% {br-0.5:>+6.3f}")

    # Promedio movil de 7 dias del base rate por hora
    print("\nTendencia temporal: base rate promedio por hora en PRIMER vs ULTIMO tercio del test")
    tercio = len(tte) // 3
    for etiqueta, ini, fin in [("primer tercio", 0, tercio), ("ultimo tercio", 2*tercio, len(tte))]:
        sub_t = tte[ini:fin]
        sub_y = yte[ini:fin]
        sub_h = ((sub_t // 3600) % 24).astype(int)
        print(f"  {etiqueta}:")
        for h in range(24):
            m = sub_h == h
            nn = int(m.sum())
            if nn < 5:
                continue
            print(f"    hora {h:02d}: n={nn:>4}  P(sube)={100*sub_y[m].mean():.2f}%")


if __name__ == "__main__":
    main()
