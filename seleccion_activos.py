# -*- coding: utf-8 -*-
"""Seleccion de activos: hay familias/pares que baten el break-even de forma PERSISTENTE?

Reusa el marco OOS (test intocado, embargo, continuidad, sin BTCUSD, rollover separado,
ensemble de 11 feats = lo que hace el bot). Guarda el dataset OOS en un .npz para reusarlo.

Prueba dura contra overfitting: parte el TEST en 2 mitades temporales. Si un par bate BE
en la 1a mitad pero NO en la 2a, es ruido, no seleccion. Un edge de verdad persiste.
"""
import json, os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
BE = 1.0 / 1.87
ROLL = (20, 21, 22)
THR = 0.54
MAX_POR_PAR = 800
OOS_NPZ = os.path.join(os.environ.get("TEMP", "."), "oos_seq.npz")

FOREX = set("EURUSD EURGBP EURJPY GBPUSD AUDCAD GBPCHF EURCHF AUDJPY GBPJPY AUDUSD AUDCHF "
            "EURNZD GBPCAD NZDCAD EURAUD GBPAUD AUDNZD CADCHF EURCAD CHFJPY NZDJPY GBPNZD "
            "NZDUSD USDJPY USDCAD".split())
CRIPTO = set("ETHUSD XRPUSD BTCUSD".split())
COMMO = set("XAUUSD USOUSD".split())
def familia(p): return "forex" if p in FOREX else "cripto" if p in CRIPTO else "commodity" if p in COMMO else "stock"

def generar_oos():
    import datetime
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    Ps, Ys, Ts, Pares = [], [], [], []
    for par in pares:
        sem = C.semillas(par)
        if len(sem) < 1: continue
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume"); vol=np.asarray(vol,float) if vol else None
        n=len(c); V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        j=json.load(open(f"models/seq_lstm_{par}.pt.json")); ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(COLA,n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
        if not cand: continue
        if len(cand)>MAX_POR_PAR: cand=sorted(rng.choice(cand,MAX_POR_PAR,replace=False))
        fs,ys,ts_=[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            fs.append(f); ys.append(int(c[i+H]>c[i])); ts_.append(tt[i])
        if not fs: continue
        P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
        Ps.extend(P.tolist()); Ys.extend(ys); Ts.extend(ts_); Pares.extend([par]*len(fs))
        print(f"  {par:8s} n={len(fs)}", flush=True)
    Ps=np.array(Ps);Ys=np.array(Ys);Ts=np.array(Ts);Pares=np.array(Pares)
    np.savez(OOS_NPZ, P=Ps, Y=Ys, T=Ts, par=Pares)
    return Ps,Ys,Ts,Pares

def main():
    if os.path.exists(OOS_NPZ) and "--regen" not in sys.argv:
        z=np.load(OOS_NPZ, allow_pickle=True); P,Y,T,Par=z["P"],z["Y"],z["T"],z["par"]
        print(f"[oos cargado de cache: {len(P)} puntos]")
    else:
        P,Y,T,Par=generar_oos()

    hor=((T//3600)%24).astype(int); fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR)) & fuera
    gano=np.where(P>=THR,Y==1,Y==0)
    def wr(mask):
        n=int(mask.sum());
        if n==0: return (float('nan'),0,(0,0))
        k=int(gano[mask].sum()); return (k/n,n,C.wilson(k,n))

    w,nn,ic=wr(ap)
    print(f"\n=== {len(P)} OOS | apuesta thr {THR} fuera roll | break-even {100*BE:.2f}% ===")
    print(f"BASELINE (todos): WR {100*w:.2f}% n={nn} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}]")

    print(f"\n=== por FAMILIA ===")
    print(f"{'familia':>10} {'WR':>8} {'n':>6} {'IC95':>16} {'EV/op':>8}")
    fams=np.array([familia(p) for p in Par])
    for fam in ["forex","stock","cripto","commodity"]:
        m=ap & (fams==fam); w,nn,ic=wr(m)
        if nn<30: continue
        ev=w*0.87-(1-w)
        print(f"{fam:>10} {100*w:7.2f}% {nn:6d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

    # persistencia: partir el test en 2 mitades temporales POR PAR
    print(f"\n=== PERSISTENCIA por par (mediana temporal del test de cada par) ===")
    print(f"un par con edge REAL bate BE en AMBAS mitades; si solo en una, es ruido\n")
    print(f"{'par':>9} {'n':>5} {'WR 1a':>7} {'WR 2a':>7} {'ambas>BE':>9}")
    filas=[]
    for par in sorted(set(Par)):
        m = ap & (Par==par)
        if m.sum()<40: continue
        tp=T[m]; med=np.median(tp)
        g=gano[m]
        m1=tp<=med; m2=tp>med
        if m1.sum()<15 or m2.sum()<15: continue
        wr1=g[m1].mean(); wr2=g[m2].mean(); wtot=g.mean()
        ambas = wr1>BE and wr2>BE
        filas.append((par,int(m.sum()),wr1,wr2,wtot,ambas))
    filas.sort(key=lambda r:-r[4])
    n_ambas=sum(1 for f in filas if f[5])
    for par,n,wr1,wr2,wtot,ambas in filas:
        print(f"{par:>9} {n:5d} {100*wr1:6.1f}% {100*wr2:6.1f}% {'  SI' if ambas else '  no':>9}")
    print(f"\npares evaluados: {len(filas)} | baten BE en AMBAS mitades: {n_ambas} "
          f"({100*n_ambas/max(len(filas),1):.0f}%)")
    print(f"si ~la mitad salen 'SI' por azar (~{100*(0.5*0.5):.0f}% esperado si WR~50/50), no hay seleccion real")

if __name__ == "__main__":
    main()
