# -*- coding: utf-8 -*-
"""Optimizar la REVERSION filtrando por REGIMEN (ADX).

El modelo es de reversion (apuesta contra la tendencia). La reversion funciona en RANGO
y falla en TENDENCIA fuerte. ADX mide la fuerza de tendencia: ADX bajo = rango (reversion
deberia acertar), ADX alto = tendencia (reversion deberia fallar).

Hipotesis: el WR de las senales del modelo sube cuando ADX es bajo. Si el regimen de ADX
bajo bate BE de forma PERSISTENTE (2 mitades del test), tenemos un filtro real que aplicar
sin tocar el modelo: operar solo cuando ADX < umbral.

OOS estricto, sin BTCUSD, rollover separado, persistencia 2 mitades, control.
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
BE = 1.0 / 1.87
ROLL = (20, 21, 22)
THR = 0.54
MAX_POR_PAR = 1500

def wil(k, n, z=1.96):
    if n == 0: return (0, 0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)

def main():
    import datetime
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    Ps, ADXs, Ys, Ts = [], [], [], []
    for par in pares:
        sem = C.semillas(par)
        if len(sem) < 5: continue
        d = json.load(open(f"{CACHE}/{par}.json", encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        _,_,_,adx,_,_,_ = macd_adx(o,h,l,c)
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

    P=np.array(Ps);ADX=np.array(ADXs);Y=np.array(Ys);T=np.array(Ts)
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0)
    idx_ap=np.where(ap)[0]
    def wr(mask):
        n=int(mask.sum())
        if n==0: return (float('nan'),0,(0,0))
        k=int(gano[mask].sum());return (k/n,n,wil(k,n))
    w,nn,ic=wr(ap)
    print(f"\n=== {int(ap.sum())} senales OOS (thr {THR}, fuera roll) | BE {100*BE:.2f}% ===")
    print(f"BASELINE (sin filtro): WR {100*w:.2f}% n={nn} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}]")
    print(f"ADX medio en las senales: {ADX[ap].mean():.1f}")

    print(f"\n=== WR por QUINTIL de ADX (Q1=rango/tendencia debil, Q5=tendencia fuerte) ===")
    aa=ADX[ap];qs=np.quantile(aa,[0.2,0.4,0.6,0.8]);edges=[-np.inf]+list(qs)+[np.inf]
    print(f"cortes ADX: {[round(q,1) for q in qs]}")
    print(f"{'quintil':>22} {'WR':>8} {'n':>6} {'IC95':>16} {'EV/op':>8}")
    for qi in range(5):
        sub=(aa>=edges[qi])&(aa<edges[qi+1])
        mm=np.zeros_like(ap);mm[idx_ap[sub]]=True
        w,nn,ic=wr(mm);ev=w*0.87-(1-w)
        etq=f"Q{qi+1}"+(" (rango)" if qi==0 else " (tendencia)" if qi==4 else "")
        print(f"{etq:>22} {100*w:7.2f}% {nn:6d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

    print(f"\n=== filtro ADX < umbral (operar solo en rango) + persistencia 2 mitades ===")
    med=np.median(T[ap])
    print(f"{'filtro':>16} {'WR':>8} {'n':>7} {'IC95':>16} {'1a/2a mit':>16}")
    for pct in [100,60,40,25]:
        thr_a=np.quantile(aa,pct/100) if pct<100 else np.inf
        mm=np.zeros_like(ap);mm[idx_ap[aa<=thr_a]]=True
        w,nn,ic=wr(mm)
        m1=mm&(T<=med);m2=mm&(T>med);w1=wr(m1)[0];w2=wr(m2)[0]
        per=f"{100*w1:.1f}%/{100*w2:.1f}% {'AMBAS>BE' if w1>BE and w2>BE else ''}"
        etq="sin filtro" if pct==100 else f"ADX<=p{pct}"
        print(f"{etq:>16} {100*w:7.2f}% {nn:7d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {per:>16}")
    print(f"\n(si el WR sube al bajar ADX y persiste en ambas mitades -> filtro de regimen real)")

if __name__ == "__main__":
    main()
