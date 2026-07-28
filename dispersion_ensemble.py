# -*- coding: utf-8 -*-
"""Filtro de CONSENSO del ensemble: apostar solo cuando las 5 semillas acuerdan sube el WR?

El modelo es un ensemble de 5 semillas; su promedio es la P que usa el bot. Pero la
DISPERSION entre las 5 (std) mide la incertidumbre y nunca se ha usado como filtro. Idea:
cuando las semillas coinciden (baja dispersion), la senal deberia ser mas fiable.

OOS estricto, sin BTCUSD, rollover separado, y -- clave -- persistencia en 2 mitades del
test (para no confundir un filtro real con ruido, como paso con 'forex' y 'hora').
"""
import json, os, sys, math
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
MAX_POR_PAR = 1200

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)

def ensemble_con_disp(X, sem_paths):
    """Devuelve (P_media, dispersion) por punto: dispersion = std entre las semillas."""
    ps = []
    for sp in sem_paths:
        z = np.load(sp); d = {k: z[k].astype(np.float64) for k in z.files}
        ps.append(C.fwd_batch(X, d))
    ps = np.array(ps)                     # (n_sem, n_puntos)
    return ps.mean(0), ps.std(0)

def main():
    import datetime
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    Ps, Ds, Ys, Ts = [], [], [], []
    for par in pares:
        sem = C.semillas(par)
        if len(sem) < 5: continue          # solo pares con ensemble completo
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(COLA,n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
        if not cand: continue
        if len(cand)>MAX_POR_PAR: cand=sorted(rng.choice(cand,MAX_POR_PAR,replace=False))
        fs,ys,ts_=[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            fs.append(f);ys.append(int(c[i+H]>c[i]));ts_.append(tt[i])
        if not fs: continue
        P,D=ensemble_con_disp(np.asarray(fs,np.float64),sem)
        Ps.extend(P.tolist());Ds.extend(D.tolist());Ys.extend(ys);Ts.extend(ts_)
        print(f"  {par:8s} n={len(fs)}",flush=True)

    P=np.array(Ps);D=np.array(Ds);Y=np.array(Ys);T=np.array(Ts)
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0)
    def wr(mask):
        n=int(mask.sum())
        if n==0: return (float('nan'),0,(0,0))
        k=int(gano[mask].sum());return (k/n,n,wilson(k,n))

    w,nn,ic=wr(ap)
    print(f"\n=== {int(ap.sum())} apuestas OOS (thr {THR}, fuera roll) | break-even {100*BE:.2f}% ===")
    print(f"BASELINE (sin filtro): WR {100*w:.2f}% n={nn} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}]")
    print(f"dispersion entre semillas: media {D[ap].mean():.4f}  (0=acuerdo total)")

    print(f"\n=== WR por QUINTIL de dispersion (Q1=mas acuerdo) ===")
    dd=D[ap]; qs=np.quantile(dd,[0.2,0.4,0.6,0.8]); edges=[-np.inf]+list(qs)+[np.inf]
    print(f"{'quintil':>16} {'WR':>8} {'n':>6} {'IC95':>16} {'EV/op':>8}")
    idx_ap=np.where(ap)[0]
    for qi in range(5):
        sub=(dd>=edges[qi])&(dd<edges[qi+1])
        mm=np.zeros_like(ap);mm[idx_ap[sub]]=True
        w,nn,ic=wr(mm);ev=w*0.87-(1-w)
        etq=f"Q{qi+1}"+(" acuerdo" if qi==0 else " desacuerdo" if qi==4 else "")
        print(f"{etq:>16} {100*w:7.2f}% {nn:6d} [{100*ic[0]:.1f},{100*ic[1]:.1f}] {ev:+8.4f}")

    print(f"\n=== filtro: solo apuestas con dispersion BAJA (acuerdo), + persistencia 2 mitades ===")
    med_t=np.median(T[ap])
    for pct in [100,40,20,10]:
        thr_d=np.quantile(dd,pct/100) if pct<100 else np.inf
        mm=np.zeros_like(ap);mm[idx_ap[dd<=thr_d]]=True
        w,nn,ic=wr(mm)
        # persistencia
        tt2=T[mm]
        if nn>=60:
            m1=mm&(T<=med_t);m2=mm&(T>med_t)
            w1=wr(m1)[0];w2=wr(m2)[0]
            per=f"| 1a {100*w1:.1f}% 2a {100*w2:.1f}% {'AMBAS>BE' if w1>BE and w2>BE else ''}"
        else: per=""
        etq="sin filtro" if pct==100 else f"disp <= p{pct}"
        print(f"  {etq:>14}: WR {100*w:.2f}% n={nn} IC[{100*ic[0]:.1f},{100*ic[1]:.1f}] {per}")

if __name__ == "__main__":
    main()
