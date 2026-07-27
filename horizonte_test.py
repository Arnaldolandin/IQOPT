# -*- coding: utf-8 -*-
"""Test de VIABILIDAD de horizontes: hay mas senal a 3/4/6 velas que a 2?

Baseline rapido (HistGradientBoosting) sobre features tabulares vectorizadas, NO el LSTM:
solo queremos saber si algun horizonte tiene mas senal OOS antes de gastar horas
reentrenando. Corte temporal por par + embargo, sin BTCUSD, rollover separado para el WR.
Si el AUC OOS no sube con el horizonte, reentrenar a otro H no vale la pena.
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CACHE = "cache_ohlc_5m_v2"
HS = [2, 3, 4, 6]
BE = 1.0 / 1.87
ROLL = (20, 21, 22)

def roll_mean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    r = np.full(len(x), np.nan)
    r[w-1:] = (cs[w:] - cs[:-w]) / w
    return r

def roll_std(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    cs2 = np.cumsum(np.insert(x*x, 0, 0.0))
    r = np.full(len(x), np.nan)
    m = (cs[w:] - cs[:-w]) / w
    m2 = (cs2[w:] - cs2[:-w]) / w
    r[w-1:] = np.sqrt(np.maximum(m2 - m*m, 0))
    return r

def feats(o, h, l, c, vol, t):
    n = len(c)
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = roll_mean(tr, 14)
    atr = np.where((atr > 0) & np.isfinite(atr), atr, np.nan)
    cols = {}
    for k in (1, 2, 3, 5, 10):
        r = np.full(n, np.nan); r[k:] = (c[k:] - c[:-k])
        cols[f"ret{k}"] = r / atr
    cols["cuerpo"] = (c - o) / atr
    cols["rango"] = (h - l) / atr
    # RSI(14) aprox rolling-mean
    d = np.diff(c, prepend=c[0])
    g = np.where(d > 0, d, 0.0); ll = np.where(d < 0, -d, 0.0)
    ag = roll_mean(g, 14); al = roll_mean(ll, 14)
    cols["rsi"] = 100 - 100/(1 + ag/(al + 1e-12))
    # Bollinger %B(20,2)
    sma = roll_mean(c, 20); sd = roll_std(c, 20)
    cols["bb"] = (c - (sma - 2*sd)) / (4*sd + 1e-12)
    cols["atr_rel"] = atr / c
    if vol is not None:
        vm = roll_mean(vol, 20)
        cols["vol_rel"] = np.where(vm > 0, vol/np.maximum(vm, 1e-9) - 1, 0.0)
    else:
        cols["vol_rel"] = np.zeros(n)
    hora = (t // 3600) % 24
    cols["hsin"] = np.sin(2*np.pi*hora/24); cols["hcos"] = np.cos(2*np.pi*hora/24)
    names = list(cols.keys())
    X = np.column_stack([cols[k] for k in names])
    return X, names

def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    EMB = 66 * 300
    # acumular por horizonte
    data = {H: {"Xtr": [], "ytr": [], "Xte": [], "yte": [], "tte": []} for H in HS}
    for par in pares:
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);t=np.asarray(d["times"],float)
        vol=d.get("volume"); vol=np.asarray(vol,float) if vol else None
        n=len(c)
        X, _ = feats(o,h,l,c,vol,t)
        finite = np.isfinite(X).all(1)
        corte = float(np.quantile(t, 0.65))
        for H in HS:
            cont = np.zeros(n, bool)
            cont[:n-H] = (t[H:]-t[:-H] == H*300) & (c[H:] != c[:-H])
            y = (np.roll(c, -H) > c).astype(int)
            valid = finite & cont
            tr = valid & (t < corte - EMB)
            te = valid & (t > corte + EMB)
            data[H]["Xtr"].append(X[tr]); data[H]["ytr"].append(y[tr])
            data[H]["Xte"].append(X[te]); data[H]["yte"].append(y[te]); data[H]["tte"].append(t[te])
        print(f"  {par}", end=" ", flush=True)
    print()

    print(f"\n=== VIABILIDAD por horizonte (baseline HGB, AUC OOS) | break-even {100*BE:.2f}% ===")
    print(f"{'H':>3} {'velas/min':>10} {'n_test':>8} {'AUC OOS':>9} {'mejor WR fuera roll (n@thr)':>32}")
    for H in HS:
        Xtr=np.vstack(data[H]["Xtr"]); ytr=np.concatenate(data[H]["ytr"])
        Xte=np.vstack(data[H]["Xte"]); yte=np.concatenate(data[H]["yte"]); tte=np.concatenate(data[H]["tte"])
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,
                                         l2_regularization=1.0,random_state=42)
        m.fit(Xtr,ytr)
        p=m.predict_proba(Xte)[:,1]
        auc=roc_auc_score(yte,p)
        hor=((tte//3600)%24).astype(int); fuera=~np.isin(hor,ROLL)
        best=(0,0,0.0)
        for thr in (0.52,0.53,0.54,0.55,0.56,0.58):
            sel=((p>=thr)|(p<=1-thr))&fuera
            if sel.sum()<200: continue
            gano=np.where(p>=thr,yte==1,yte==0)
            wr=gano[sel].mean()
            if wr>best[0]: best=(wr,int(sel.sum()),thr)
        print(f"{H:>3} {H*5:>8}m {len(yte):>8} {auc:>9.4f}   {100*best[0]:>6.2f}% (n={best[1]} @{best[2]})")
    print(f"\nAUC 0.5 = azar. Si ningun H sube claramente sobre H=2, el horizonte no es palanca.")

if __name__ == "__main__":
    main()
