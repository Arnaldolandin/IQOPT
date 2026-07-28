# -*- coding: utf-8 -*-
"""Dependencia SERIAL de resultados: el acierto/fallo previo predice el siguiente?

Si el modelo esta calibrado deberia ser ~cero. Se mide, por par y en orden temporal, el WR
de una senal condicionado al resultado de la senal ANTERIOR YA RESUELTA (decision previa +
H velas < decision actual: el bot conoceria ese resultado antes de apostar). Asi la relacion
seria explotable ("parar tras perder" / "insistir tras ganar").

Se reporta tambien la version cruda (senal adyacente, aunque solape) para contraste: las
ventanas de velas consecutivas solapan casi del todo, asi que una autocorrelacion cruda alta
seria artefacto, no edge. OOS estricto, sin BTCUSD, rollover fuera, persistencia 2 mitades.
"""
import json, sys, math, datetime
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C

CACHE="cache_ohlc_5m_v2"; L,H=64,2; EMB=(L+H)*300; BE=1/1.87; ROLL=(20,21,22); THR=0.54
COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

MAX_CONTIG=2000   # bloque contiguo OOS mas reciente por par (preserva la secuencia)

def senales_par(par):
    """Devuelve (t, gano) de cada senal OOS del par, en orden temporal."""
    sem=C.semillas(par)
    if len(sem)<5: return None
    d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
    o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
    c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
    vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
    n=len(c)
    V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]     # se construye UNA vez y se corta
    j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
    corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
           if ci else float(np.quantile(tt,0.65)))
    cand=[i for i in range(COLA,n-H)
          if tt[i]>corte+EMB and (int(tt[i])//3600)%24 not in ROLL
          and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
    if len(cand)<50: return None
    cand=cand[-MAX_CONTIG:]                                # bloque contiguo mas reciente
    feats=[];meta=[]
    for i in cand:
        lo=i-COLA
        f=S.ventana_features(V[lo:i+1],L,vol=(None if vol is None else vol[lo:i+1]))
        if f is None: continue
        feats.append(f);meta.append((tt[i],int(c[i+H]>c[i])))
    if len(feats)<50: return None
    P=C.ensemble_batch(np.asarray(feats,float),sem)
    out=[]
    for k in range(len(P)):
        p=P[k];t,y=meta[k]
        if p>=THR: out.append((t, y==1))
        elif p<=1-THR: out.append((t, y==0))
    out.sort(key=lambda z:z[0])
    return out

def main():
    cfg=json.load(open("config.json",encoding="utf-8"))
    pares=[p for p in cfg["entrenamiento"]["pares"] if p!="BTCUSD"]
    # pares[prev_resuelto_gano] -> lista de (t, gano_actual)
    tras_win=[];tras_loss=[]          # version resuelta (explotable)
    adj_win=[];adj_loss=[]            # version cruda adyacente (solapa)
    ac=[]                             # autocorrelacion lag-1 por par
    ns=0
    for par in pares:
        s=senales_par(par)
        if not s: continue
        ns+=1
        g=np.array([1 if x[1] else 0 for x in s]);ts=np.array([x[0] for x in s])
        if len(g)>2 and g.std()>0: ac.append(np.corrcoef(g[:-1],g[1:])[0,1])
        # cruda adyacente
        for k in range(1,len(s)):
            (tp,gp)=s[k-1];(t,gc)=s[k]
            (adj_win if gp else adj_loss).append((t,gc))
        # resuelta: para cada senal, el ultimo previo con decision+H*300 < decision actual.
        # dos punteros O(n): jp avanza mientras s[jp] ya este resuelto antes de t=s[k]
        jp=-1
        for k in range(len(s)):
            t=s[k][0]
            while jp+1<k and s[jp+1][0]+H*300 < t: jp+=1
            # jp es el mayor indice <k cuyo trade ya cerro antes de t (si existe)
            if jp<0 or not (s[jp][0]+H*300 < t): continue
            (tras_win if s[jp][1] else tras_loss).append((t,s[k][1]))
    def rep(nombre,W,Lo):
        allt=W+Lo
        if not allt: print(f"{nombre}: sin datos");return
        med=np.median([x[0] for x in allt])
        for lab,arr in [("tras GANAR",W),("tras PERDER",Lo)]:
            n=len(arr)
            if n<30: print(f"  {lab:12} n={n} (insuf.)");continue
            gg=np.array([1 if x[1] else 0 for x in arr]);tt=np.array([x[0] for x in arr])
            wr=gg.mean();k=int(gg.sum());l2,h2=wil(k,n)
            w1=gg[tt<=med].mean();w2=gg[tt>med].mean()
            print(f"  {lab:12} WR {100*wr:6.2f}%  n={n:5d}  IC95[{100*l2:.1f},{100*h2:.1f}]  1a/2a {100*w1:.1f}/{100*w2:.1f}")
        # diferencia
        gw=np.array([1 if x[1] else 0 for x in W]);gl=np.array([1 if x[1] else 0 for x in Lo])
        if len(gw)>=30 and len(gl)>=30:
            dif=gw.mean()-gl.mean()
            se=math.sqrt(gw.var()/len(gw)+gl.var()/len(gl))
            print(f"  -> dif(ganar-perder) = {100*dif:+.2f} pts  z={dif/se if se>0 else 0:+.2f}")
    print(f"=== {ns} pares | BE {100*BE:.2f}% | THR {THR} ===\n")
    print("VERSION EXPLOTABLE (previo YA resuelto antes de apostar):")
    rep("resuelta",tras_win,tras_loss)
    print("\nVERSION CRUDA (senal adyacente 5m, SOLAPA -> referencia, no explotable):")
    rep("cruda",adj_win,adj_loss)
    if ac:
        ac=np.array(ac)
        print(f"\nautocorr lag-1 por par: media {ac.mean():+.4f}  (n={len(ac)} pares, |media|<0.03 = sin memoria)")

if __name__ == "__main__":
    main()
