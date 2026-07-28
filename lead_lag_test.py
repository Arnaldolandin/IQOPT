# -*- coding: utf-8 -*-
"""LEAD-LAG cross-asset: el retorno reciente de un activo A ANTICIPA la direccion de B a 10m?

Es la unica palanca que NO es 'las mismas senales reagrupadas': mete informacion EXTERNA al
OHLC del par. A=lider, B=objetivo. Feature = retorno de A en las ultimas k velas (hasta t,
SIN futuro). Etiqueta = signo del movimiento de B de t a t+H (futuro). Si algun lider predice
de forma robusta, es informacion nueva explotable.

Guardas (con 49x49xk tests el multiple-testing es brutal):
  1. Se compara la DISTRIBUCION de |corr| real contra un CONTROL con la etiqueta desalineada
     (rotada). Si la cola real no supera a la del control, no hay nada -> se acabo.
  2. Solo si la cola real destaca, se listan los top pares con PERSISTENCIA en 2 mitades.
Alineacion por timestamp comun, OOS estricto, sin BTCUSD, rollover fuera. Sin fuga: pasado
de A vs futuro de B; no se entrena nada, no se mezcla train/test.
"""
import json, sys, math, datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

CACHE="cache_ohlc_5m_v2"; H=2; EMB=(64+H)*300; ROLL=(20,21,22)
LAGS=[1,2,3,6]           # velas de historia del lider (5,10,15,30 min)
BE=1/1.87

def main():
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    # cargar close por par indexado por epoch
    closes={}; cortes={}
    for par in pares:
        try:
            d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        except FileNotFoundError:
            continue
        tt=np.asarray(d["times"],float); c=np.asarray(d["close"],float)
        closes[par]={int(tt[i]):c[i] for i in range(len(tt))}
        try:
            j=json.load(open(f"models/seq_lstm_{par}.pt.json")); ci=j["meta"].get("corte")
            cortes[par]=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
                         if ci else float(np.quantile(tt,0.65)))
        except FileNotFoundError:
            cortes[par]=float(np.quantile(tt,0.65))
    P=[p for p in pares if p in closes]
    # rejilla de timestamps comunes (union de todos), OOS y fuera de rollover
    corte_glob=max(cortes[p] for p in P)
    allt=set()
    for p in P: allt|=set(closes[p].keys())
    grid=np.array(sorted(t for t in allt if t>corte_glob+EMB and (t//3600)%24 not in ROLL))
    # matriz close alineada (nan si falta)
    M=np.full((len(grid),len(P)),np.nan)
    pos={int(t):k for k,t in enumerate(grid)}
    for jp,p in enumerate(P):
        cp=closes[p]
        for t,c in cp.items():
            k=pos.get(int(t))
            if k is not None: M[k,jp]=c
    print(f"{len(P)} pares | {len(grid)} timestamps OOS comunes", flush=True)

    # etiqueta forward de B: signo de close_B[t+H]-close_B[t] con continuidad (t+H a H*300 exactos)
    step=300
    # indice de t+H en la rejilla
    tH=grid+H*step
    idxH=np.array([pos.get(int(x),-1) for x in tH])
    valid_fwd=idxH>=0
    real=[]; ctrl=[]; detalle=[]
    for jb,B in enumerate(P):
        cb=M[:,jb]
        fwd=np.full(len(grid),np.nan)
        m=valid_fwd & np.isfinite(cb)
        ii=np.where(m)[0]
        cbH=np.array([M[idxH[i],jb] for i in ii])
        fb=np.sign(cbH-cb[ii])
        okf=cbH!=cb[ii]
        fwd_idx=ii[okf]; fwd_val=fb[okf]
        if len(fwd_idx)<200: continue
        yb=(fwd_val>0).astype(float)
        for ja,A in enumerate(P):
            if ja==jb: continue
            ca=M[:,ja]
            for lg in LAGS:
                if fwd_idx.min()-lg<0:
                    sel=fwd_idx[fwd_idx-lg>=0]
                else:
                    sel=fwd_idx
                a_now=ca[sel]; a_prev=ca[sel-lg]
                good=np.isfinite(a_now)&np.isfinite(a_prev)&(a_prev>0)
                if good.sum()<200: continue
                ret=a_now[good]/a_prev[good]-1.0
                # etiqueta de B alineada a esos sel
                pos_map={int(s):yb[k] for k,s in enumerate(fwd_idx)}
                yy=np.array([pos_map[int(s)] for s in sel[good]])
                if yy.std()==0 or ret.std()==0: continue
                r=np.corrcoef(ret,yy)[0,1]
                real.append(abs(r))
                # control: rotar la etiqueta media vuelta (rompe alineacion temporal)
                yr=np.roll(yy,len(yy)//2 or 1)
                ctrl.append(abs(np.corrcoef(ret,yr)[0,1]))
                detalle.append((abs(r),r,A,B,lg,len(yy),ret,yy))
    real=np.array(real); ctrl=np.array(ctrl)
    print(f"\n=== {len(real)} tests lider->objetivo (|corr|) ===")
    for q in [50,90,95,99,99.9]:
        print(f"  percentil {q:5.1f}:  real {np.percentile(real,q):.4f}   control {np.percentile(ctrl,q):.4f}")
    print(f"  max:            real {real.max():.4f}   control {ctrl.max():.4f}")
    # umbral = percentil 99.9 del control: cuantos reales lo pasan por AZAR esperado vs observado
    thr=np.percentile(ctrl,99.9)
    n_esp=len(real)*0.001
    n_obs=(real>thr).sum()
    print(f"\n  sobre el p99.9 del control ({thr:.4f}): esperados ~{n_esp:.0f} por azar, observados {n_obs}")

    # top pares que superan el control, con persistencia en 2 mitades
    print(f"\n=== TOP lead-lag que superan p99.9 control, con persistencia (corr 1a/2a mitad) ===")
    detalle.sort(reverse=True,key=lambda z:z[0])
    mostrados=0
    for ab,r,A,B,lg,n,ret,yy in detalle:
        if ab<=thr: break
        h=len(yy)//2
        r1=np.corrcoef(ret[:h],yy[:h])[0,1] if yy[:h].std()>0 else 0
        r2=np.corrcoef(ret[h:],yy[h:])[0,1] if yy[h:].std()>0 else 0
        pers="MISMO SIGNO" if (r1*r2>0 and abs(r1)>0.02 and abs(r2)>0.02) else ""
        print(f"  {A:9s} -> {B:9s} lag{lg} | corr {r:+.4f} n={n} | 1a/2a {r1:+.4f}/{r2:+.4f} {pers}")
        mostrados+=1
        if mostrados>=25: break
    if mostrados==0:
        print("  (ninguno supera el control: no hay lead-lag explotable)")

if __name__ == "__main__":
    main()
