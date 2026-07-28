# -*- coding: utf-8 -*-
"""Pullback vs reversion-pura: las senales del modelo A FAVOR de la tendencia rinden mas?

El modelo es reversion (apuesta contra el movimiento). Se separan sus senales en:
  - PULLBACK: la direccion apostada COINCIDE con la tendencia dominante (comprar el dip en
    tendencia alcista, vender el rebote en bajista) -> apuesta la CONTINUACION.
  - REVERSION PURA: la direccion apostada va CONTRA la tendencia -> apuesta el giro.
Si el pullback bate BE y la reversion-pura no, valdria reorientar el LSTM a pullback
(anadir tendencia y reentrenar). OOS estricto, sin BTCUSD, rollover separado, persistencia.
"""
import json, os, sys, math
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C

CACHE="cache_ohlc_5m_v2"
L,H=64,2; EMB=(L+H)*300; BE=1/1.87; ROLL=(20,21,22); THR=0.54; MAX=1500

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

def main():
    import datetime
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    rng=np.random.default_rng(0);COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
    Ps,TR24,TR50,Ys,Ts=[],[],[],[],[]
    for par in pares:
        sem=C.semillas(par)
        if len(sem)<5: continue
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(COLA,n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
        if not cand: continue
        if len(cand)>MAX: cand=sorted(rng.choice(cand,MAX,replace=False))
        fs,ys,ts_,t24,t50=[],[],[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            fs.append(f);ys.append(int(c[i+H]>c[i]));ts_.append(tt[i])
            t24.append(np.sign(c[i]-c[i-24]));t50.append(np.sign(c[i]-c[i-50]))
        if not fs: continue
        P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
        Ps.extend(P.tolist());TR24.extend(t24);TR50.extend(t50);Ys.extend(ys);Ts.extend(ts_)
        print(f"  {par:8s} n={len(fs)}",flush=True)
    P=np.array(Ps);TR24=np.array(TR24);TR50=np.array(TR50);Y=np.array(Ys);T=np.array(Ts)
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0)
    dirm=np.where(P>=THR,1,-1)              # direccion apostada por el modelo
    med=np.median(T[ap])
    def wr(m):
        n=int(m.sum())
        if n==0:return(float('nan'),0,(0,0))
        k=int(gano[m].sum());return(k/n,n,wil(k,n))
    def fila(nom,m):
        w,n,ic=wr(m)
        if n<40:print(f"{nom:38s} n={n}");return
        m1=m&(T<=med);m2=m&(T>med);w1=wr(m1)[0];w2=wr(m2)[0]
        per="AMBAS>BE" if (w1>BE and w2>BE) else ""
        print(f"{nom:38s} {100*w:6.2f}% n={n:5d} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}] | {100*w1:.1f}%/{100*w2:.1f}% {per}")
    print(f"\nBE {100*BE:.2f}% | {int(ap.sum())} senales\n")
    fila("BASELINE (todas)",ap)
    for trd,lab in [(TR24,"tendencia 2h"),(TR50,"tendencia 4h")]:
        favor=ap&(dirm==trd)               # pullback: apuesta A FAVOR de la tendencia
        contra=ap&(dirm==-trd)             # reversion pura: apuesta CONTRA
        print(f"\n-- segun {lab} --")
        fila("PULLBACK (a favor de tendencia)",favor)
        fila("REVERSION PURA (contra tendencia)",contra)
        print(f"   (n pullback {int(favor.sum())} vs reversion {int(contra.sum())})")

if __name__ == "__main__":
    main()
