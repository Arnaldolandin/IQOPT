# -*- coding: utf-8 -*-
"""Viabilidad de SEGUIMIENTO DE TENDENCIA con MACD + ADX (premisa: trend is your friend).

Antes de reentrenar 50 pares (~5h) con features nuevas, se mide si la tendencia se puede
seguir con edge a 5m/H=2:
  1. Estrategia trend-following pura: MACD alcista + ADX fuerte -> CALL (seguir), y al reves.
  2. Baseline HGB con features de TENDENCIA (MACD, ADX, DI) -> AUC OOS vs el actual (~0.53).
  3. Contraste: la MISMA senal fadeada (reversion) -> para ver que lado gana en los datos.
OOS estricto, sin BTCUSD, rollover separado, break-even 53.48%.
"""
import json, os, sys, math
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = "cache_ohlc_5m_v2"
H = 2
BE = 1/1.87
ROLL = (20, 21, 22)
ADX_MIN = 25          # umbral clasico de "tendencia fuerte"

def ema(x, span):
    a = 2/(span+1); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)): y[i] = a*x[i] + (1-a)*y[i-1]
    return y

def wilder(x, n):
    y = np.empty_like(x); y[:n] = np.nan; y[n-1] = np.nanmean(x[:n])
    for i in range(n, len(x)): y[i] = (y[i-1]*(n-1) + x[i]) / n
    return y

def macd_adx(o, h, l, c):
    n = len(c)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    # ADX / DI (Wilder 14)
    up = h[1:]-h[:-1]; dn = l[:-1]-l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    pdm = np.concatenate([[0], pdm]); mdm = np.concatenate([[0], mdm]); tr = np.concatenate([[0], tr])
    atr = wilder(tr, 14)
    pdi = 100 * wilder(pdm, 14) / np.where(atr > 0, atr, np.nan)
    mdi = 100 * wilder(mdm, 14) / np.where(atr > 0, atr, np.nan)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi+mdi) > 0, pdi+mdi, np.nan)
    adx = wilder(np.nan_to_num(dx), 14)
    return macd, sig, hist, adx, pdi, mdi, atr

def wil_ic(k, n, z=1.96):
    if n == 0: return (0, 0)
    p=k/n; d=1+z*z/n; cc=p+z*z/(2*n); hh=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((cc-hh)/d,(cc+hh)/d)

def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    tf_k = tf_n = 0
    Xtr, ytr, Xte, yte = [], [], [], []
    for par in pares:
        d = json.load(open(f"{CACHE}/{par}.json", encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);t=np.asarray(d["times"],float)
        n=len(c)
        macd, sig, hist, adx, pdi, mdi, atr = macd_adx(o,h,l,c)
        corte = float(np.quantile(t, 0.65)); emb = 66*300
        i = np.arange(40, n-H)
        cont = (t[i+H]-t[i]==H*300) & (c[i+H]!=c[i]) & np.isfinite(adx[i]) & np.isfinite(hist[i]) & (atr[i]>0)
        i = i[cont]
        subio = (c[i+H] > c[i]).astype(int)
        hor = (t[i]//3600) % 24
        a = np.where(atr[i]>0, atr[i], np.nan)
        F = np.column_stack([macd[i]/a, sig[i]/a, hist[i]/a, adx[i], pdi[i], mdi[i],
                             (c[i]-c[i-6])/a, (c[i]-c[i-12])/a,
                             np.sin(2*np.pi*hor/24), np.cos(2*np.pi*hor/24)])
        fin = np.isfinite(F).all(1)
        i, subio, hor, F = i[fin], subio[fin], hor[fin], F[fin]
        tr = t[i] < corte-emb; te = t[i] > corte+emb
        Xtr.append(F[tr]); ytr.append(subio[tr]); Xte.append(F[te]); yte.append(subio[te])
        # trend-following en test: ADX fuerte, fuera rollover, seguir el signo de MACD hist
        sel = te & (adx[i] >= ADX_MIN) & (~np.isin(hor, ROLL))
        call = hist[i][sel] > 0
        gano = np.where(call, subio[sel]==1, subio[sel]==0)
        tf_k += int(gano.sum()); tf_n += int(sel.sum())
    Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr);Xte=np.vstack(Xte);yte=np.concatenate(yte)

    print(f"=== SEGUIMIENTO DE TENDENCIA con MACD+ADX | break-even {100*BE:.2f}% ===\n")
    print(f"1) Estrategia trend-following pura (ADX>={ADX_MIN}, seguir MACD, fuera rollover):")
    wr = tf_k/tf_n if tf_n else float('nan'); lo,hi = wil_ic(tf_k, tf_n)
    ev = wr*0.87-(1-wr)
    print(f"   WR {100*wr:.2f}%  n={tf_n}  IC[{100*lo:.1f},{100*hi:.1f}]  EV/op {ev:+.4f}  "
          f"{'BATE BE' if lo>BE else 'no bate'}")
    print(f"   (fade = reversion daria {100*(1-wr):.2f}%)")

    print(f"\n2) Baseline HGB con features de TENDENCIA (MACD/ADX/DI), AUC OOS:")
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                       l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:,1]
    print(f"   AUC {roc_auc_score(yte,p):.4f}  (modelo ACTUAL con BB da ~0.53)")
    # importancia relativa: correlacion de cada feature con el acierto
    print(f"\n   veredicto: si el trend-following no bate BE y el AUC no supera ~0.53,")
    print(f"   sustituir BB por MACD+ADX y reentrenar (5h) no cambiaria el muro.")

if __name__ == "__main__":
    main()
