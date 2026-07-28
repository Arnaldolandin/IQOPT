# -*- coding: utf-8 -*-
"""Anadir features de CONTEXTO MULTI-ESCALA mejora el modelo?

El modelo ve 64 velas de 5m (~5.3h). Es ciego al contexto de escala mayor. Se prueba si
anadir features que aportan info NUEVA (no redundante con RSI/Bollinger, que ya derivan del
mismo precio) mejora el AUC OOS:
  - ret_4h, ret_dia : tendencia a 4h y 1 dia (fuera de su ventana).
  - pos_dia         : posicion del precio en el rango del dia (0=minimo, 1=maximo).
  - vol_ratio       : volatilidad reciente (14) / volatilidad del dia (288) -> regimen de vol.
Baseline HGB (features base) vs +contexto, mismo test. Si sube el AUC, reentrenar el LSTM
con esas features anadidas a cada vela valdria la pena.
"""
import json, os, sys, warnings
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import horizonte_test as HT

CACHE="cache_ohlc_5m_v2"; H=2
PARES=["ETHUSD","XRPUSD","EURUSD","GBPUSD","USDJPY","EURJPY","AUDUSD","XAUUSD","GBPJPY","EURGBP","AUDCAD","NZDUSD"]
D=288   # velas de 5m en un dia
MAX=1200

def main():
    Xb_tr,Xc_tr,ytr,Xb_te,Xc_te,yte=[],[],[],[],[],[]
    rng=np.random.default_rng(0)
    for par in PARES:
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);t=np.asarray(d["times"],float)
        v=d.get("volume");v=np.asarray(v,float) if v else None
        n=len(c)
        X,names=HT.feats(o,h,l,c,v,t)
        tr=np.zeros(n);tr[1:]=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
        atr=np.convolve(tr,np.ones(14)/14,mode="full")[:n]
        ret=np.zeros(n);ret[1:]=np.diff(c)/np.where(c[:-1]!=0,c[:-1],1)
        corte=float(np.quantile(t,0.65));emb=(64+H)*300
        finX=np.isfinite(X).all(1)
        cand=[i for i in range(D,n-H) if t[i+H]-t[i]==H*300 and c[i+H]!=c[i] and finX[i] and atr[i]>0]
        cand=[i for i in cand if t[i]<corte-emb or t[i]>corte+emb]
        if len(cand)>MAX: cand=sorted(rng.choice(cand,MAX,replace=False))
        for i in cand:
            a=atr[i]
            ret_4h=(c[i]-c[i-48])/a; ret_dia=(c[i]-c[i-D])/a
            hi=h[i-D:i+1].max(); lo=l[i-D:i+1].min()
            pos=(c[i]-lo)/(hi-lo) if hi>lo else 0.5
            vshort=ret[i-13:i+1].std(); vlong=ret[i-D+1:i+1].std()
            vratio=vshort/vlong if vlong>0 else 1.0
            ctx=[ret_4h,ret_dia,pos,vratio]
            if not all(np.isfinite(ctx)): continue
            y=int(c[i+H]>c[i])
            if t[i]<corte-emb: Xb_tr.append(X[i]);Xc_tr.append(ctx);ytr.append(y)
            else: Xb_te.append(X[i]);Xc_te.append(ctx);yte.append(y)
        print(f"  {par} listo",flush=True)
    Xb_tr=np.array(Xb_tr);Xc_tr=np.array(Xc_tr);ytr=np.array(ytr)
    Xb_te=np.array(Xb_te);Xc_te=np.array(Xc_te);yte=np.array(yte)
    print(f"\ntrain {len(ytr)} test {len(yte)}\n")
    def auc(Xtr,Xte):
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42).fit(Xtr,ytr)
        return roc_auc_score(yte,m.predict_proba(Xte)[:,1])
    a0=auc(Xb_tr,Xb_te)
    a1=auc(np.hstack([Xb_tr,Xc_tr]),np.hstack([Xb_te,Xc_te]))
    print(f"AUC base (features actuales):         {a0:.4f}")
    print(f"AUC base + contexto multi-escala:     {a1:.4f}")
    print(f"ganancia: {a1-a0:+.4f}  -> {'APORTA, valdria reentrenar' if a1-a0>0.003 else 'no aporta (dentro del ruido)'}")
    # cual de las 4 aporta mas (una a una sobre la base)
    print("\ncontribucion individual (base + 1 feature de contexto):")
    for j,nm in enumerate(["ret_4h","ret_dia","pos_dia","vol_ratio"]):
        aj=auc(np.hstack([Xb_tr,Xc_tr[:,[j]]]),np.hstack([Xb_te,Xc_te[:,[j]]]))
        print(f"  +{nm:10s} AUC {aj:.4f} ({aj-a0:+.4f})")

if __name__ == "__main__":
    main()
