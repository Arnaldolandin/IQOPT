# -*- coding: utf-8 -*-
"""Factibilidad + WR de timeframes MENORES (2m,1m,30s,15s) para cripto.

El cache de IQ es 5m; para sub-5m se usa Binance (= feed de IQ, corr 0.998). Baja 1m y 1s,
reconstruye 30s/15s desde 1s, y mide el AUC direccional OOS (baseline HGB) por TF vs 5m.
Solo cripto (Binance); forex sub-5m no tiene fuente aqui.

Recordatorio de factibilidad (no medible aqui, pero manda): expiry<=5m = turbo (BE 54.64%),
y el bot no puede escanear 49 pares en <1min. Este script responde solo a: hay MAS senal
direccional en TF cortos, que justifique el peor payout y una reingenieria del bot?
"""
import json, os, sys, time, urllib.request
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import horizonte_test as HT

H = 2

def binance(symbol, interval, n_target):
    """Ultimas ~n_target klines de 'interval'. Devuelve OHLCV+t (segundos)."""
    out = []
    end = None
    while len(out) < n_target:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000"
        if end: url += f"&endTime={end}"
        d = json.load(urllib.request.urlopen(url, timeout=20))
        if not d: break
        out = d + out
        end = d[0][0] - 1
        if len(d) < 1000: break
    o=np.array([float(k[1]) for k in out]); h=np.array([float(k[2]) for k in out])
    l=np.array([float(k[3]) for k in out]); c=np.array([float(k[4]) for k in out])
    v=np.array([float(k[5]) for k in out]); t=np.array([k[0]//1000 for k in out],float)
    return o,h,l,c,v,t

def agg(o,h,l,c,v,t,k,paso):
    """Agrupa de k en k velas consecutivas (paso seg)."""
    n=len(c);O=[];Hi=[];Lo=[];C=[];V=[];T=[];i=0
    while i+k<=n:
        if t[i+k-1]-t[i]==(k-1)*paso:
            O.append(o[i]);Hi.append(h[i:i+k].max());Lo.append(l[i:i+k].min())
            C.append(c[i+k-1]);V.append(v[i:i+k].sum());T.append(t[i]);i+=k
        else: i+=1
    return (np.array(O),np.array(Hi),np.array(Lo),np.array(C),np.array(V),np.array(T))

def auc(o,h,l,c,v,t,tf_seg,etq):
    X,_=HT.feats(o,h,l,c,v,t)
    fin=np.isfinite(X).all(1)
    cont=np.zeros(len(c),bool); cont[:len(c)-H]=(t[H:]-t[:-H]==H*tf_seg)&(c[H:]!=c[:-H])
    y=(np.roll(c,-H)>c).astype(int); valid=fin&cont
    corte=float(np.quantile(t,0.65)); emb=(64+H)*tf_seg
    tr=valid&(t<corte-emb); te=valid&(t>corte+emb)
    if te.sum()<500 or tr.sum()<500: return None
    m=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.05,max_depth=4,
                                     l2_regularization=1.0,random_state=42)
    m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]
    a=roc_auc_score(y[te],p)
    print(f"  {etq:>6} tf={tf_seg:>4}s  AUC {a:.4f}  n_test={int(te.sum())}", flush=True)
    return a

for sym in ["ETHUSDT","XRPUSDT"]:
    print(f"\n=== {sym} ===", flush=True)
    # 1m (para 1m,2m) y 1s (para 15s,30s)
    print("bajando 1m...", flush=True); o1,h1,l1,c1,v1,t1=binance(sym,"1m",40000)
    auc(o1,h1,l1,c1,v1,t1,60,"1m")
    a=agg(o1,h1,l1,c1,v1,t1,2,60); auc(*a,120,"2m")
    print("bajando 1s (tramo corto)...", flush=True); os_,hs,ls,cs,vs,ts=binance(sym,"1s",50000)
    a=agg(os_,hs,ls,cs,vs,ts,15,1); auc(*a,15,"15s")
    a=agg(os_,hs,ls,cs,vs,ts,30,1); auc(*a,30,"30s")
    auc(os_,hs,ls,cs,vs,ts,1,"1s")
print("\nreferencia 5m (medido antes): AUC ~0.526. AUC 0.5 = azar.")
