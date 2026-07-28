# -*- coding: utf-8 -*-
"""Confirmacion CROSS-ASSET: las senales que coinciden con muchos pares rinden mas?

Mecanismo: los pares comparten monedas. Si muchos apuntan igual EN EL MISMO INSTANTE, es
una senal macro (p.ej. USD debil en todos los USD-pairs) mas fiable que un par solo. Se
mide el WR de cada senal segun el CONSENSO direccional del resto de pares en esa vela.

Sin fuga: el consenso se calcula en el momento de decision (no usa el futuro); se agrupa el
WR real por nivel de consenso. OOS estricto, sin BTCUSD, rollover separado, persistencia.
"""
import json, os, sys, math, datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C

CACHE="cache_ohlc_5m_v2"; L,H=64,2; EMB=(L+H)*300; BE=1/1.87; ROLL=(20,21,22); THR=0.54
N_TS=700   # timestamps (velas) a evaluar

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

def main():
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    rng=np.random.default_rng(0);COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
    # cargar cada par: velas, sem, corte, y un indice epoch->i
    data={}
    tmin=tmax=None
    for par in pares:
        sem=C.semillas(par)
        if len(sem)<5: continue
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        idx={int(tt[i]):i for i in range(len(tt))}
        data[par]=dict(o=o,h=h,l=l,c=c,tt=tt,vol=vol,sem=sem,corte=corte,idx=idx)
        lo,hi=tt[0],tt[-1]
        tmin=lo if tmin is None else max(tmin,lo); tmax=hi if tmax is None else min(tmax,hi)
    # timestamps candidatos: rejilla de 5m en el periodo OOS comun (t > max corte + emb)
    corte_glob=max(v["corte"] for v in data.values())
    grid=np.arange(corte_glob+EMB+COLA*300, tmax-H*300, 300)
    grid=grid[np.isin((grid//3600)%24, ROLL, invert=True)]   # fuera de rollover
    if len(grid)>N_TS: grid=np.sort(rng.choice(grid,N_TS,replace=False))
    print(f"{len(data)} pares | {len(grid)} timestamps OOS evaluados", flush=True)

    # para cada timestamp: P y y de cada par que lo tenga con continuidad
    filas=[]   # (t, par, P, y)
    for k,t in enumerate(grid):
        t=int(t)
        Ppar={}
        for par,v in data.items():
            i=v["idx"].get(t)
            if i is None or i<COLA or i+H>=len(v["c"]): continue
            if v["tt"][i+H]-v["tt"][i]!=H*300 or v["c"][i+H]==v["c"][i]: continue
            Vv=[[v["tt"][x],v["o"][x],v["h"][x],v["l"][x],v["c"][x]] for x in range(i-COLA,i+1)]
            f=S.ventana_features(Vv,L,vol=(None if v["vol"] is None else v["vol"][i-COLA:i+1]))
            if f is None: continue
            P=C.ensemble_batch(np.asarray([f],float),v["sem"])[0]
            Ppar[par]=(P,int(v["c"][i+H]>v["c"][i]))
        # consenso: fraccion de pares con senal (|P-0.5|>=0.04) alcista vs bajista
        dirs=[1 if P>=THR else -1 for P,_ in Ppar.values() if P>=THR or P<=1-THR]
        n_sig=len(dirs)
        if n_sig<3: continue
        up=sum(1 for x in dirs if x>0)
        for par,(P,y) in Ppar.items():
            if not (P>=THR or P<=1-THR): continue
            d_par=1 if P>=THR else -1
            # consenso a favor de ESTA senal entre los OTROS pares
            otros=n_sig-1
            de_acuerdo=(up-(1 if d_par>0 else 0)) if d_par>0 else ((n_sig-up)-(1 if d_par<0 else 0))
            cons=de_acuerdo/otros if otros>0 else 0.0
            gano=(y==1) if d_par>0 else (y==0)
            filas.append((t,cons,gano))
        if k%100==0: print(f"  {k}/{len(grid)}",flush=True)

    T=np.array([f[0] for f in filas]);CO=np.array([f[1] for f in filas]);G=np.array([f[2] for f in filas],float)
    med=np.median(T)
    print(f"\n=== {len(filas)} senales | BE {100*BE:.2f}% ===")
    print(f"WR global: {100*G.mean():.2f}%")
    print(f"\nWR por CONSENSO (fraccion de OTROS pares que apuntan igual):")
    print(f"{'consenso':>16} {'WR':>8} {'n':>6} {'IC95':>16} {'1a/2a':>14}")
    for lo,hi,lab in [(0.0,0.34,'bajo (<34%)'),(0.34,0.67,'medio'),(0.67,1.01,'alto (>=67%)')]:
        m=(CO>=lo)&(CO<hi);n=int(m.sum())
        if n<30: continue
        k=int(G[m].sum());wr=k/n;l2,h2=wil(k,n)
        w1=G[m&(T<=med)].mean();w2=G[m&(T>med)].mean()
        per='AMBAS>BE' if (w1>BE and w2>BE) else ''
        print(f"{lab:>16} {100*wr:7.2f}% {n:6d} [{100*l2:.1f},{100*h2:.1f}] {100*w1:.1f}%/{100*w2:.1f}% {per}")
    # correlacion consenso-acierto
    print(f"\ncorr(consenso, acierto): {np.corrcoef(CO,G)[0,1]:+.4f} (>0 = mas acuerdo -> mas acierto)")

if __name__ == "__main__":
    main()
