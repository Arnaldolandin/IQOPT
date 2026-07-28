# -*- coding: utf-8 -*-
"""Evalua otros timeframes: hay mas senal direccional en 15m/30m/1h que en 5m?

Resamplea el cache 5m de IQ a TF mas largos (respetando continuidad: solo agrupa velas
consecutivas a 300s, sin cruzar gaps de fin de semana). Baseline HGB sobre las features
tabulares, AUC OOS por TF. 1m solo cripto (Binance). Si ningun TF sube claramente sobre
5m, el timeframe no es palanca; si 15m/1h destaca, valdria entrenar ahi.
"""
import json, os, sys, urllib.request
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import horizonte_test as HT

CACHE = "cache_ohlc_5m_v2"
ROLL = (20, 21, 22)
BE = 1/1.87
H = 2   # horizonte en velas del propio TF

def resample(o,h,l,c,v,t,k):
    """Agrupa de k en k velas 5m consecutivas (paso 300s). Devuelve OHLCV+t del TF."""
    n=len(c); O=[];Hi=[];Lo=[];C=[];V=[];T=[]
    i=0
    while i+k<=n:
        # continuidad: las k velas deben ser consecutivas a 300s
        if t[i+k-1]-t[i]==(k-1)*300:
            O.append(o[i]);Hi.append(h[i:i+k].max());Lo.append(l[i:i+k].min())
            C.append(c[i+k-1]);V.append(v[i:i+k].sum() if v is not None else 0);T.append(t[i])
            i+=k
        else:
            i+=1
    return (np.array(O),np.array(Hi),np.array(Lo),np.array(C),
            (np.array(V) if v is not None else None),np.array(T))

def auc_tf(pares, k, tf_seg):
    Xtr=[];ytr=[];Xte=[];yte=[]
    for par in pares:
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);t=np.asarray(d["times"],float)
        v=d.get("volume"); v=np.asarray(v,float) if v else None
        if k>1:
            o,h,l,c,v,t=resample(o,h,l,c,v,t,k)
        if len(c)<500: continue
        X,_=HT.feats(o,h,l,c,v,t)
        fin=np.isfinite(X).all(1)
        cont=np.zeros(len(c),bool); cont[:len(c)-H]=(t[H:]-t[:-H]==H*tf_seg)&(c[H:]!=c[:-H])
        y=(np.roll(c,-H)>c).astype(int)
        valid=fin&cont
        corte=float(np.quantile(t,0.65)); emb=(64+H)*tf_seg
        tr=valid&(t<corte-emb); te=valid&(t>corte+emb)
        Xtr.append(X[tr]);ytr.append(y[tr]);Xte.append(X[te]);yte.append(y[te])
    Xtr=np.vstack(Xtr);ytr=np.concatenate(ytr);Xte=np.vstack(Xte);yte=np.concatenate(yte)
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,
                                     l2_regularization=1.0,random_state=42)
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,p), len(yte)

def main():
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    print(f"=== predecibilidad direccional por TF (baseline HGB, AUC OOS, H=2 velas del TF) ===")
    print(f"{'TF':>6} {'k(x5m)':>7} {'AUC OOS':>9} {'n_test':>9}")
    for tf,k in [("5m",1),("15m",3),("30m",6),("1h",12),("2h",24)]:
        try:
            auc,n=auc_tf(pares,k,300*k)
            print(f"{tf:>6} {k:>7} {auc:>9.4f} {n:>9}", flush=True)
        except Exception as e:
            print(f"{tf:>6}: fallo {type(e).__name__}: {str(e)[:60]}")
    print(f"\nAUC 0.5 = azar. 5m base ~0.526. Si ningun TF sube claro, el TF no es palanca.")

if __name__ == "__main__":
    main()
