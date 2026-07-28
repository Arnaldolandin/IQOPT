# -*- coding: utf-8 -*-
"""Combinar RANGO (ADX bajo) + SEÑAL FUERTE (|P-0.5| alto) para la reversion.

La reversion es mas fiable cuando (a) el mercado esta en rango (ADX bajo, el precio revierte
en vez de seguir) y (b) la senal es fuerte (P lejos de 0.5, precio claramente estirado). Se
mide la interseccion. OOS estricto, sin BTCUSD, rollover separado, persistencia 2 mitades.
El OOS (P, ADX, y, t) se cachea en TEMP para reusarlo en refinamientos.
"""
import json, os, sys, math
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C
from trend_following_test import macd_adx

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
BE = 1.0/1.87
ROLL = (20,21,22)
THR = 0.54
MAX_POR_PAR = 1500
NPZ = os.path.join(os.environ.get("TEMP","."), "oos_reversion.npz")

def wil(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

def generar():
    import datetime
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    rng=np.random.default_rng(0);COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
    Ps,ADXs,Ys,Ts=[],[],[],[]
    for par in pares:
        sem=C.semillas(par)
        if len(sem)<5: continue
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        _,_,_,adx,_,_,_=macd_adx(o,h,l,c)
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(COLA,n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300
              and c[i+H]!=c[i] and np.isfinite(adx[i])]
        if not cand: continue
        if len(cand)>MAX_POR_PAR: cand=sorted(rng.choice(cand,MAX_POR_PAR,replace=False))
        fs,ys,ts_,ad=[],[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            fs.append(f);ys.append(int(c[i+H]>c[i]));ts_.append(tt[i]);ad.append(adx[i])
        if not fs: continue
        P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
        Ps.extend(P.tolist());ADXs.extend(ad);Ys.extend(ys);Ts.extend(ts_)
        print(f"  {par:8s} n={len(fs)}",flush=True)
    Ps=np.array(Ps);ADX=np.array(ADXs);Y=np.array(Ys);T=np.array(Ts)
    np.savez(NPZ,P=Ps,ADX=ADX,Y=Y,T=T)
    return Ps,ADX,Y,T

def main():
    if os.path.exists(NPZ) and "--regen" not in sys.argv:
        z=np.load(NPZ);P,ADX,Y,T=z["P"],z["ADX"],z["Y"],z["T"];print("[oos cacheado]")
    else:
        P,ADX,Y,T=generar()
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0)
    conf=np.abs(P-0.5)                       # fuerza de la senal
    idx=np.where(ap)[0]
    med=np.median(T[ap])
    def wr(mask):
        n=int(mask.sum())
        if n==0: return (float('nan'),0,(0,0))
        k=int(gano[mask].sum());return (k/n,n,wil(k,n))
    def fila(nombre,mask):
        w,n,ic=wr(mask)
        if n<40: print(f"{nombre:34s} n={n} (pocos)");return
        m1=mask&(T<=med);m2=mask&(T>med);w1=wr(m1)[0];w2=wr(m2)[0]
        ev=w*0.87-(1-w)
        per="AMBAS>BE" if (w1>BE and w2>BE) else ""
        print(f"{nombre:34s} {100*w:6.2f}% n={n:5d} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}] "
              f"EV{ev:+.3f} | {100*w1:.1f}%/{100*w2:.1f}% {per}")

    aa=ADX[ap];cc=conf[ap]
    adx_p40=np.quantile(aa,0.40); conf_med=np.median(cc)
    print(f"=== BE {100*BE:.2f}% | senales {int(ap.sum())} | cortes: ADX<={adx_p40:.1f}(rango), conf>={conf_med:.3f}(fuerte) ===\n")
    fila("BASELINE (todas)", ap)
    r=np.zeros_like(ap);r[idx[aa<=adx_p40]]=True
    fila("solo RANGO (ADX<=p40)", r)
    s=np.zeros_like(ap);s[idx[cc>=conf_med]]=True
    fila("solo SENAL FUERTE (conf>=mediana)", s)
    fila("RANGO + SENAL FUERTE", r & s)

    print(f"\n=== matriz WR: ADX (filas) x fuerza de senal (columnas) ===")
    aq=np.quantile(aa,[0.33,0.66]); cq=np.quantile(cc,[0.33,0.66])
    print(f"{'':>14}{'conf baja':>12}{'conf media':>12}{'conf alta':>12}")
    for ai,(alo,ahi,alab) in enumerate([(-np.inf,aq[0],'ADX bajo'),(aq[0],aq[1],'ADX medio'),(aq[1],np.inf,'ADX alto')]):
        row=f"{alab:>14}"
        for clo,chi in [(-np.inf,cq[0]),(cq[0],cq[1]),(cq[1],np.inf)]:
            sub=(aa>=alo)&(aa<ahi)&(cc>=clo)&(cc<chi)
            mm=np.zeros_like(ap);mm[idx[sub]]=True
            w,n,_=wr(mm)
            row+=f"  {100*w:5.1f}%(n{n})" if n>=30 else f"  {'-':>10}"
        print(row)
    print(f"\n(sweet spot teorico = ADX bajo + conf alta, esquina inf-der; debe batir BE y persistir)")

if __name__ == "__main__":
    main()
