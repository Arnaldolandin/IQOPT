# -*- coding: utf-8 -*-
"""Filtro ATR: filtrar velas de baja volatilidad mejora el WR?

Mismo marco OOS honesto que calibracion.py (test intocado, embargo, continuidad, sin
BTCUSD, rollover separado, ensemble de 11 feats = lo que hace el bot). Para cada punto
guarda ademas el ATR local (el MISMO que usa ventana_features / el bot: media del True
Range de las ultimas ATR_P velas).

El ATR absoluto NO es comparable entre pares (EURUSD ~1e-4, XAUUSD ~1, cripto ~miles),
asi que el analisis principal es por ATR RELATIVO (ATR/close) en quintiles. Se reporta
tambien el ATR absoluto en el bloque forex, que es donde aplican los umbrales propuestos.
"""
import json, os, glob, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C   # reusa semillas(), ensemble_batch(), wilson()

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
BE = 1.0 / 1.87
ROLL = (20, 21, 22)
MAX_POR_PAR = 600
THR = 0.54                 # umbral de apuesta actual del bot
FOREX = set("EURUSD EURGBP EURJPY GBPUSD AUDCAD GBPCHF EURCHF AUDJPY GBPJPY AUDUSD "
            "AUDCHF EURNZD GBPCAD NZDCAD EURAUD GBPAUD AUDNZD CADCHF EURCAD CHFJPY "
            "NZDJPY GBPNZD NZDUSD USDJPY USDCAD".split())

def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    import datetime
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    P_all, Y_all, T_all, AR_all, AA_all, FX_all = [], [], [], [], [], []
    for par in pares:
        sem = C.semillas(par)
        if len(sem) < 1:
            continue
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o = np.asarray(d["open"], float); h = np.asarray(d["high"], float)
        l = np.asarray(d["low"], float); c = np.asarray(d["close"], float)
        tt = np.asarray(d["times"], float)
        vol = d.get("volume"); vol = np.asarray(vol, float) if vol else None
        n = len(c)
        # True Range causal
        tr = np.zeros(n)
        tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        V = [[tt[i], o[i], h[i], l[i], c[i]] for i in range(n)]
        j = json.load(open(f"models/seq_lstm_{par}.pt.json"))
        ci = j["meta"].get("corte")
        corte = (datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
                 if ci else float(np.quantile(tt, 0.65)))
        cand = [i for i in range(COLA, n - H)
                if tt[i] > corte + EMB and tt[i + H] - tt[i] == H * 300 and c[i + H] != c[i]]
        if not cand:
            continue
        if len(cand) > MAX_POR_PAR:
            cand = sorted(rng.choice(cand, MAX_POR_PAR, replace=False))
        fs, ys, ts_, ars, aas = [], [], [], [], []
        for i in cand:
            lo = i - COLA
            f = S.ventana_features(V[lo:i + 1], L, vol=(None if vol is None else vol[lo:i + 1]))
            if f is None:
                continue
            atr = float(tr[i - S.ATR_P:i].mean())    # mismo ATR que ventana_features
            if not np.isfinite(atr) or atr <= 0:
                continue
            fs.append(f); ys.append(int(c[i + H] > c[i])); ts_.append(tt[i])
            ars.append(atr / c[i]); aas.append(atr)
        if not fs:
            continue
        P = C.ensemble_batch(np.asarray(fs, np.float64), sem)
        P_all.extend(P.tolist()); Y_all.extend(ys); T_all.extend(ts_)
        AR_all.extend(ars); AA_all.extend(aas); FX_all.extend([par in FOREX] * len(fs))
        print(f"  {par:8s} n={len(fs)}", flush=True)

    P = np.array(P_all); Y = np.array(Y_all); TS = np.array(T_all)
    AR = np.array(AR_all); AA = np.array(AA_all); FX = np.array(FX_all)
    hor = ((TS // 3600) % 24).astype(int); fuera = ~np.isin(hor, ROLL)
    # apuesta a THR, fuera de rollover
    apuesta = ((P >= THR) | (P <= 1 - THR)) & fuera
    gano = np.where(P >= THR, Y == 1, Y == 0)

    def wr(mask):
        n = int(mask.sum())
        if n == 0: return (float('nan'), 0, (0, 0))
        k = int(gano[mask].sum())
        return (k / n, n, C.wilson(k, n))

    print(f"\n=== {len(P)} puntos OOS | apuesta a thr {THR}, fuera de rollover ===")
    w, nn, ic = wr(apuesta)
    print(f"BASELINE (sin filtro ATR): WR {100*w:.2f}% n={nn} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}]  break-even {100*BE:.2f}%")

    print(f"\n=== por QUINTIL de ATR relativo (ATR/close), dentro de las apuestas ===")
    a_ar = AR[apuesta]
    qs = np.quantile(a_ar, [0.2, 0.4, 0.6, 0.8])
    print(f"cortes de ATR_rel: {['%.2e'%q for q in qs]}")
    print(f"{'quintil ATR_rel':>18} {'WR':>8} {'n':>6} {'IC95':>16} {'EV/op':>8}")
    edges = [-np.inf] + list(qs) + [np.inf]
    for qi in range(5):
        m = apuesta.copy()
        sub = (AR[apuesta] >= edges[qi]) & (AR[apuesta] < edges[qi + 1])
        idxs = np.where(apuesta)[0][sub]
        mm = np.zeros_like(apuesta); mm[idxs] = True
        w, nn, ic = wr(mm)
        ev = w * 0.87 - (1 - w)
        etq = f"Q{qi+1} {'(mas baja)' if qi==0 else '(mas alta)' if qi==4 else ''}"
        print(f"{etq:>18} {100*w:7.2f}% {nn:6d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

    print(f"\n=== filtro: quedarse SOLO con ATR_rel por encima de cada corte ===")
    print(f"{'filtro':>22} {'WR':>8} {'n':>7} {'IC95':>16} {'EV/op':>8}")
    for pct in [0, 25, 50, 75]:
        thr_ar = np.quantile(AR[apuesta], pct/100) if pct else -np.inf
        idxs = np.where(apuesta)[0][AR[apuesta] >= thr_ar]
        mm = np.zeros_like(apuesta); mm[idxs] = True
        w, nn, ic = wr(mm)
        ev = w * 0.87 - (1 - w)
        etq = "sin filtro" if pct == 0 else f"ATR_rel >= p{pct}"
        print(f"{etq:>22} {100*w:7.2f}% {nn:7d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

    print(f"\n=== FOREX solo: umbrales de ATR ABSOLUTO propuestos ===")
    fxa = apuesta & FX
    print(f"apuestas forex: {int(fxa.sum())}")
    print(f"{'filtro ATR abs':>18} {'WR':>8} {'n':>7} {'IC95':>16} {'EV/op':>8}")
    for thr_a in [0.0, 0.0001, 0.0002, 0.0005, 0.001]:
        idxs = np.where(fxa)[0][AA[fxa] >= thr_a]
        mm = np.zeros_like(apuesta); mm[idxs] = True
        w, nn, ic = wr(mm)
        if nn < 30:
            print(f"{('>= %.4f'%thr_a):>18} n={nn} (muy pocos)"); continue
        ev = w * 0.87 - (1 - w)
        print(f"{('>= %.4f'%thr_a):>18} {100*w:7.2f}% {nn:7d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

if __name__ == "__main__":
    main()
