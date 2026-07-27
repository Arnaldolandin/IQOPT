# -*- coding: utf-8 -*-
"""Calibracion del modelo seq: la P que devuelve, PREDICE el acierto?

Es el prerequisito de 'umbrales por par' y 'Kelly': ambos asumen que P mas alta = mas
aciertos. Si la curva de calibracion sale plana, no hay base para ninguno de los dos.

Honestidad (reglas del proyecto):
- OOS estricto: solo el TEST intocado de cada par (t > corte_entrenamiento + embargo).
- Continuidad de la opcion y sin empates (lo da dataset()).
- SIN BTCUSD (no ejecuta).
- Rollover 20-22 UTC SEPARADO.
- Mismo ensemble de 11 feats que el bot (predecir_npz_ensemble), no torch.
- IC de Wilson por bin: sin intervalo, un bin de n=30 'confirma' cualquier cosa.
"""
import json, os, glob, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import train_seq_save as T

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
BE = 1.0 / 1.87            # 0.5348
ROLL = (20, 21, 22)
MAX_POR_PAR = 600          # submuestreo del test por par (calibracion no necesita todo)

def semillas(par):
    s = sorted(glob.glob(f"models/seq_lstm_{par}_s*.npz"))
    return [x for x in s if "W_ih" in np.load(x).files and int(np.load(x)["W_ih"].shape[1]) == 11]

def _sig(x): return 1.0 / (1.0 + np.exp(-x))

def fwd_batch(X, d):
    """Forward de una LSTM de 1 capa sobre TODO el batch X (N,L,F) a la vez. Mismo calculo
    que predecir_npz (compuertas PyTorch [i,f,g,o]) pero vectorizado -> ~1000x mas rapido.
    Verificado que coincide con predecir_npz_ensemble punto a punto."""
    W_ih, W_hh, b = d["W_ih"], d["W_hh"], d["b_ih"] + d["b_hh"]
    hid = W_hh.shape[1]; N = X.shape[0]
    h = np.zeros((N, hid)); c = np.zeros((N, hid))
    Wih_T, Whh_T = W_ih.T, W_hh.T
    for tt in range(X.shape[1]):
        g = X[:, tt, :] @ Wih_T + h @ Whh_T + b
        i_ = _sig(g[:, :hid]); f_ = _sig(g[:, hid:2*hid])
        g_ = np.tanh(g[:, 2*hid:3*hid]); o_ = _sig(g[:, 3*hid:])
        c = f_ * c + i_ * g_
        h = o_ * np.tanh(c)
    return _sig(h @ d["fc_w"].T + d["fc_b"])[:, 0]

def ensemble_batch(X, sem_paths):
    """Media del ensemble sobre todas las semillas, en batch. X=(N,L,F) -> P=(N,)."""
    ps = []
    for sp in sem_paths:
        z = np.load(sp); d = {k: z[k].astype(np.float64) for k in z.files}
        ps.append(fwd_batch(X, d))
    return np.mean(ps, axis=0)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    import datetime
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1     # velas de contexto que exige ventana_features
    P_all, Y_all, T_all = [], [], []
    usados = 0
    for par in pares:
        sem = semillas(par)
        if len(sem) < 1:
            continue
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o = np.asarray(d["open"], float); h = np.asarray(d["high"], float)
        l = np.asarray(d["low"], float); c = np.asarray(d["close"], float)
        tt = np.asarray(d["times"], float)
        vol = d.get("volume"); vol = np.asarray(vol, float) if vol else None
        n = len(c)
        V = [[tt[i], o[i], h[i], l[i], c[i]] for i in range(n)]
        j = json.load(open(f"models/seq_lstm_{par}.pt.json"))
        ci = j["meta"].get("corte")
        corte = (datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
                 if ci else float(np.quantile(tt, 0.65)))
        # indices del TEST intocado (con embargo), continuidad de la opcion y sin empate
        cand = [i for i in range(COLA, n - H)
                if tt[i] > corte + EMB and tt[i + H] - tt[i] == H * 300 and c[i + H] != c[i]]
        if not cand:
            continue
        if len(cand) > MAX_POR_PAR:
            cand = sorted(rng.choice(cand, MAX_POR_PAR, replace=False))
        fs, ys, ts_ = [], [], []
        for i in cand:
            lo = i - COLA
            f = S.ventana_features(V[lo:i + 1], L, vol=(None if vol is None else vol[lo:i + 1]))
            if f is None:
                continue
            fs.append(f); ys.append(int(c[i + H] > c[i])); ts_.append(tt[i])
        if not fs:
            continue
        Pbatch = ensemble_batch(np.asarray(fs, np.float64), sem)
        P_all.extend(Pbatch.tolist()); Y_all.extend(ys); T_all.extend(ts_)
        usados += 1
        print(f"  {par:8s} test OOS n={len(fs)}", flush=True)

    P = np.array(P_all); Y = np.array(Y_all); TS = np.array(T_all)
    hor = ((TS // 3600) % 24).astype(int)
    fuera = ~np.isin(hor, ROLL)
    print(f"\n=== {usados} pares | {len(P)} predicciones OOS ({fuera.sum()} fuera de rollover) ===")
    print(f"tasa base de subida (OOS, fuera roll): {Y[fuera].mean():.4f}")
    print(f"rango de P: [{P.min():.3f}, {P.max():.3f}]  media {P.mean():.4f}  sd {P.std():.4f}")
    brier = np.mean((P[fuera] - Y[fuera])**2)
    base = Y[fuera].mean()
    brier_base = np.mean((base - Y[fuera])**2)
    print(f"Brier: {brier:.5f}  vs  predecir-siempre-la-base {brier_base:.5f}  "
          f"({'mejor' if brier < brier_base else 'PEOR/IGUAL'})")

    # --- curva de calibracion (bins de P), SOLO fuera de rollover ---
    bins = [0.0, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 1.0]
    print(f"\n=== CALIBRACION fuera de rollover (calibrado = 'sube real' sigue a 'P media') ===")
    print(f"{'bin P':>13} {'n':>6} {'P media':>8} {'sube real':>10} {'IC95 real':>16}")
    Pf, Yf = P[fuera], Y[fuera]
    for a, b in zip(bins[:-1], bins[1:]):
        m = (Pf >= a) & (Pf < b)
        n = int(m.sum())
        if n == 0: continue
        k = int(Yf[m].sum())
        lo, hi = wilson(k, n)
        flag = ""
        if a >= 0.50 and lo > 0.5: flag = " <- sube signif."
        if b <= 0.50 and hi < 0.5: flag = " <- baja signif."
        print(f"[{a:.2f},{b:.2f}) {n:6d} {Pf[m].mean():8.3f} {k/n:10.3f} "
              f"[{lo:.3f},{hi:.3f}]{flag}")

    # --- lo que importa para trading: WR de la apuesta simetrica por umbral ---
    print(f"\n=== WR de la APUESTA por umbral (CALL si P>=thr, PUT si P<=1-thr), fuera roll ===")
    print(f"break-even {100*BE:.2f}%")
    print(f"{'thr':>6} {'n':>7} {'WR':>8} {'IC95 WR':>16} {'EV/op':>8}")
    for thr in [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]:
        sel = ((Pf >= thr) | (Pf <= 1 - thr))
        n = int(sel.sum())
        if n < 30: continue
        gano = np.where(Pf[sel] >= thr, Yf[sel] == 1, Yf[sel] == 0)
        k = int(gano.sum()); wr = k / n
        lo, hi = wilson(k, n)
        ev = wr * 0.87 - (1 - wr)   # payout 0.87
        print(f"{thr:6.2f} {n:7d} {100*wr:7.2f}% [{100*lo:.1f}%,{100*hi:.1f}%] {ev:+8.4f}")

if __name__ == "__main__":
    main()
