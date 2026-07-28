# -*- coding: utf-8 -*-
"""TEST DECISIVO: las features de OTROS activos anaden WR SOBRE el modelo del propio par?

El lead-lag cross-asset supero al control, pero corr~0.10 y concentrado en sesion/riesgo. La
unica pregunta que importa: sumar los retornos de otros activos al modelo, sube el WR OOS por
encima de lo que ya da el OHLC del propio par? Se mide en USDJPY (24h, operable):

  baseline = umbral sobre la P del modelo (solo OHLC de USDJPY)
  combinado = regresion logistica sobre [logit(P), retornos de LIDERES a varios lags]
              coeficientes AJUSTADOS en la 1a mitad, EVALUADOS en la 2a (y viceversa).

Si el combinado no bate al baseline en AMBAS mitades y por encima del control (lideres
barajados), la info externa no aporta. OOS estricto, sin fuga (pasado de lideres, coef fuera
de muestra), rollover fuera, y separado por sesion.
"""
import json, sys, math, datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C
from sklearn.linear_model import LogisticRegression

CACHE="cache_ohlc_5m_v2"; L,H=64,2; EMB=(L+H)*300; BE=1/1.87; ROLL=(20,21,22)
COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
OBJ="USDJPY"
LIDERES=["AUDUSD","AUDJPY","EURUSD","GBPUSD","XAUUSD","EURJPY","GBPJPY","AUDNZD","NZDUSD","USDCHF"]
LAGS=[1,2,3,6]
THR=0.54

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;cc=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((cc-h)/d,(cc+h)/d)

def close_idx(par):
    d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
    tt=np.asarray(d["times"],float);c=np.asarray(d["close"],float)
    return {int(tt[i]):c[i] for i in range(len(tt))}

def main():
    # --- P y etiqueta del objetivo en OOS ---
    sem=C.semillas(OBJ)
    d=json.load(open(f"{CACHE}/{OBJ}.json",encoding="utf-8"))
    o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
    c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float);n=len(c)
    vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
    V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
    j=json.load(open(f"models/seq_lstm_{OBJ}.pt.json"));ci=j["meta"].get("corte")
    corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
           if ci else float(np.quantile(tt,0.65)))
    cand=[i for i in range(COLA,n-H)
          if tt[i]>corte+EMB and (int(tt[i])//3600)%24 not in ROLL
          and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
    cand=cand[-4000:]
    import os
    global LIDERES
    LIDERES=[p for p in LIDERES if os.path.exists(f"{CACHE}/{p}.json")]
    print(f"lideres disponibles: {LIDERES}",flush=True)
    lid={p:close_idx(p) for p in LIDERES}
    fs=[];rows=[];ts_=[];ys=[]
    for i in cand:
        f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
        if f is None: continue
        t=int(tt[i]);feat=[]
        ok=True
        for p in LIDERES:
            cp=lid[p];a0=cp.get(t)
            for lg in LAGS:
                ap=cp.get(t-lg*300)
                if a0 is None or ap is None or ap<=0: feat.append(np.nan);ok=False
                else: feat.append(a0/ap-1.0)
        fs.append(f);rows.append(feat);ts_.append(t);ys.append(int(c[i+H]>c[i]))
    P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
    X=np.array(rows);T=np.array(ts_);Y=np.array(ys)
    good=np.all(np.isfinite(X),axis=1)
    P,X,T,Y=P[good],X[good],T[good],Y[good]
    print(f"{OBJ}: {len(P)} puntos OOS con lideres completos | BE {100*BE:.2f}%",flush=True)
    logit=np.log(np.clip(P,1e-6,1-1e-6)/np.clip(1-P,1e-6,1-1e-6))
    med=np.median(T)

    def wr_umbral(p):
        sel=(p>=THR)|(p<=1-THR);ng=int(sel.sum())
        if ng==0: return None
        g=np.where(p[sel]>=THR,Y[sel]==1,Y[sel]==0)
        return g.mean(),ng,int(g.sum())

    def eval_half(tr,te,shuffle=False):
        # estandarizar lideres con stats de train
        mu=X[tr].mean(0);sd=X[tr].std(0)+1e-9
        Xtr=(X[tr]-mu)/sd;Xte=(X[te]-mu)/sd
        Ztr=np.column_stack([logit[tr],Xtr]);Zte=np.column_stack([logit[te],Xte])
        if shuffle:
            r=np.random.default_rng(0).permutation(len(te))
            Zte=np.column_stack([logit[te],Xte[r]])
        clf=LogisticRegression(C=0.5,max_iter=2000).fit(Ztr,Y[tr])
        pc=clf.predict_proba(Zte)[:,1]
        # baseline en test
        b=wr_umbral_sub(P[te],Y[te])
        # combinado: mismas n operaciones que baseline, las mas confiadas
        nb=b[1] if b else 0
        conf=np.abs(pc-0.5)
        order=np.argsort(-conf)
        selc=order[:nb]
        g=np.where(pc[selc]>=0.5,Y[te][selc]==1,Y[te][selc]==0)
        return b,(g.mean(),nb,int(g.sum()))

    def wr_umbral_sub(p,y):
        sel=(p>=THR)|(p<=1-THR);ng=int(sel.sum())
        if ng==0: return (0,0,0)
        g=np.where(p[sel]>=THR,y[sel]==1,y[sel]==0)
        return g.mean(),ng,int(g.sum())

    idx=np.arange(len(P));h1=idx[T<=med];h2=idx[T>med]
    print(f"\n=== baseline (solo modelo) vs combinado (modelo+lideres), umbral {THR} ===")
    for tr,te,lab in [(h1,h2,"fit 1a -> test 2a"),(h2,h1,"fit 2a -> test 1a")]:
        b,cmb=eval_half(tr,te)
        _,cmb_sh=eval_half(tr,te,shuffle=True)
        lo_b,hi_b=wil(b[2],b[1]);lo_c,hi_c=wil(cmb[2],cmb[1])
        print(f"\n{lab}:")
        print(f"  baseline   WR {100*b[0]:6.2f}%  n={b[1]:4d}  IC[{100*lo_b:.1f},{100*hi_b:.1f}]")
        print(f"  combinado  WR {100*cmb[0]:6.2f}%  n={cmb[1]:4d}  IC[{100*lo_c:.1f},{100*hi_c:.1f}]  "
              f"({'+' if cmb[0]>=b[0] else ''}{100*(cmb[0]-b[0]):.2f} pts)")
        print(f"  control(lideres barajados)  WR {100*cmb_sh[0]:6.2f}%  n={cmb_sh[1]:4d}")

if __name__ == "__main__":
    main()
