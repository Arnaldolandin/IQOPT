# -*- coding: utf-8 -*-
"""Verificacion rigurosa de vol_ratio (regimen de volatilidad = std(14) / std(288 = 1 dia)).

Dos pruebas:
  A) REGIMEN: WR de las senales del modelo REAL (ensemble) por quintil de vol_ratio, con
     persistencia 2 mitades. Si el modelo acierta mas en cierto regimen -> filtro util.
  B) FEATURE vs baseline FUERTE: HGB sobre la ventana APLANADA (64x11) con y sin vol_ratio.
     Baseline fuerte (no de 1 vela) para ver si aporta info que el modelo no tenga ya.
OOS estricto, sin BTCUSD, rollover separado.
"""
import json, os, sys, math, warnings
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import seq_model as S
import calibracion as C

CACHE="cache_ohlc_5m_v2"; L,H=64,2; EMB=(L+H)*300; BE=1/1.87; ROLL=(20,21,22); THR=0.54
Dv=288; MAX=1200
PARES_B=["ETHUSD","XRPUSD","EURUSD","GBPUSD","USDJPY","EURJPY","AUDUSD","XAUUSD"]

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

def main():
    import datetime
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    rng=np.random.default_rng(0);COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
    Ps,VR,Ys,Ts=[],[],[],[]
    Xflat,Vflat,Yflat,Tflat=[],[],[],[]     # para prueba B (solo PARES_B)
    for par in pares:
        sem=C.semillas(par)
        if len(sem)<5: continue
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        ret=np.zeros(n);ret[1:]=np.diff(c)/np.where(c[:-1]!=0,c[:-1],1)
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(max(COLA,Dv),n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
        if not cand: continue
        if len(cand)>MAX: cand=sorted(rng.choice(cand,MAX,replace=False))
        fs,ys,ts_,vr,idx=[],[],[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            vs=ret[i-13:i+1].std();vl=ret[i-Dv+1:i+1].std()
            if vl<=0: continue
            fs.append(f);ys.append(int(c[i+H]>c[i]));ts_.append(tt[i]);vr.append(vs/vl);idx.append(i)
        if not fs: continue
        P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
        Ps.extend(P.tolist());VR.extend(vr);Ys.extend(ys);Ts.extend(ts_)
        if par in PARES_B:
            Xflat.append(np.asarray(fs,np.float64).reshape(len(fs),-1))
            Vflat.extend(vr);Yflat.extend(ys);Tflat.extend(ts_)
        print(f"  {par} listo",flush=True)
    P=np.array(Ps);VR=np.array(VR);Y=np.array(Ys);T=np.array(Ts)
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0);idx=np.where(ap)[0];med=np.median(T[ap])
    def wr(m):
        n=int(m.sum())
        if n==0:return(float('nan'),0,(0,0))
        k=int(gano[m].sum());return(k/n,n,wil(k,n))

    print(f"\n=== A) REGIMEN: WR por quintil de vol_ratio | BE {100*BE:.2f}% | {int(ap.sum())} senales ===")
    vv=VR[ap];qs=np.quantile(vv,[0.2,0.4,0.6,0.8]);edges=[-np.inf]+list(qs)+[np.inf]
    print(f"cortes vol_ratio: {[round(q,2) for q in qs]}")
    print(f"{'quintil':>22} {'WR':>8} {'n':>6} {'IC95':>16} {'1a/2a mit':>16}")
    for qi in range(5):
        sub=(vv>=edges[qi])&(vv<edges[qi+1]);mm=np.zeros_like(ap);mm[idx[sub]]=True
        w,n,ic=wr(mm);w1=wr(mm&(T<=med))[0];w2=wr(mm&(T>med))[0]
        per="AMBAS>BE" if (w1>BE and w2>BE) else ""
        etq=f"Q{qi+1}"+(" (vol comprimida)" if qi==0 else " (vol expandida)" if qi==4 else "")
        print(f"{etq:>22} {100*w:7.2f}% {n:6d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {100*w1:.1f}%/{100*w2:.1f}% {per}")

    print(f"\n=== B) FEATURE vs baseline FUERTE (ventana aplanada 64x11) ===")
    Xf=np.vstack(Xflat);Vf=np.array(Vflat);Yf=np.array(Yflat);Tf=np.array(Tflat)
    cor=np.quantile(Tf,0.5)  # split simple train/test dentro del pool B ya-OOS
    tr=Tf<=cor;te=Tf>cor
    def auc(A,B):
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42).fit(A,Yf[tr])
        return roc_auc_score(Yf[te],m.predict_proba(B)[:,1])
    a0=auc(Xf[tr],Xf[te])
    a1=auc(np.hstack([Xf[tr],Vf[tr,None]]),np.hstack([Xf[te],Vf[te,None]]))
    print(f"  AUC ventana aplanada:            {a0:.4f}")
    print(f"  AUC ventana aplanada + vol_ratio: {a1:.4f}  ({a1-a0:+.4f})")
    print(f"  -> {'APORTA sobre baseline fuerte' if a1-a0>0.003 else 'NO aporta (el modelo ya tiene esa info)'}")

if __name__ == "__main__":
    main()
