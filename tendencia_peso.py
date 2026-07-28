# -*- coding: utf-8 -*-
"""Cuanto pesa la tendencia en las decisiones del modelo?

  1. corr(P-0.5, tendencia): + = el modelo sigue la tendencia (momentum); - = contra.
  2. % de senales alineadas con la tendencia vs contra-tendencia.
  3. Importancia por PERMUTACION: se baraja la serie de una feature entre puntos y se mide
     cuanto cae el AUC del ensemble. La feature 0 (retorno vela a vela) ES la tendencia.
OOS estricto, mismo ensemble de 11 feats del bot.
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score
import seq_model as S
import calibracion as C

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
THR = 0.54
PARES = ["ETHUSD","XRPUSD","EURUSD","GBPUSD","USDJPY","EURJPY","AUDUSD","XAUUSD"]
GRUPOS = {"retorno(tendencia)":[0], "forma_vela":[1,2,3,4], "hora":[5,6],
          "volumen":[7,8], "RSI":[9], "Bollinger":[10]}

def main():
    import datetime
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    XX, YY, PP = [], [], []
    TREND = {6:[],12:[],24:[],64:[]}
    ref_sem = None
    for par in PARES:
        sem = C.semillas(par)
        if len(sem) < 1: continue
        if ref_sem is None: ref_sem = sem
        d=json.load(open(f"{CACHE}/{par}.json",encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
        vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
        n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
        tr=np.zeros(n);tr[1:]=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
        atr=np.convolve(tr,np.ones(14)/14,mode="full")[:n]
        j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
        corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
               if ci else float(np.quantile(tt,0.65)))
        cand=[i for i in range(COLA,n-H) if tt[i]>corte+EMB and tt[i+H]-tt[i]==H*300 and c[i+H]!=c[i]]
        if not cand: continue
        if len(cand)>1000: cand=sorted(rng.choice(cand,1000,replace=False))
        fs=[];idx=[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None or atr[i]<=0: continue
            fs.append(f);idx.append(i)
        if not fs: continue
        Xp=np.asarray(fs,np.float64)
        XX.append(Xp);PP.append(C.ensemble_batch(Xp,sem))
        YY.extend(int(c[k+H]>c[k]) for k in idx)
        for K in TREND: TREND[K].extend((c[k]-c[k-K])/atr[k] if k>=K else 0.0 for k in idx)
        print(f"  {par} listo",flush=True)
    X=np.vstack(XX);P=np.concatenate(PP);Y=np.array(YY)
    for K in TREND: TREND[K]=np.array(TREND[K])
    print(f"\npuntos OOS: {len(P)}")

    print(f"\n=== 1. corr(P-0.5, tendencia)   (+ = momentum, - = reversion) ===")
    for K in [6,12,24,64]:
        cc=np.corrcoef(P-0.5,TREND[K])[0,1]
        tipo="MOMENTUM (sigue)" if cc>0.03 else "REVERSION (contra)" if cc<-0.03 else "neutro"
        print(f"  tendencia ultimas {K:>2} velas ({K*5:>3}min): corr {cc:+.4f}  {tipo}")

    print(f"\n=== 2. senales: alineadas con la tendencia (momentum) vs contra ===")
    sig=(P>=THR)|(P<=1-THR)
    dirn=np.where(P>=THR,1,-1)[sig]
    print(f"  senales totales: {int(sig.sum())} de {len(P)}")
    for K in [12,24]:
        tr_sig=np.sign(TREND[K][sig])
        mom=(dirn==tr_sig).mean()
        print(f"  vs tendencia {K*5}min: {100*mom:.1f}% momentum / {100*(1-mom):.1f}% contra")

    print(f"\n=== 3. importancia por permutacion (caida de AUC al romper cada grupo) ===")
    base=roc_auc_score(Y,C.ensemble_batch(X,ref_sem))
    print(f"  AUC base (proxy, semillas {PARES[0]} sobre todo el pool): {base:.4f}")
    res=[]
    for gname,cols in GRUPOS.items():
        Xp=X.copy(); perm=rng.permutation(len(Xp))
        for col in cols: Xp[:,:,col]=X[perm][:,:,col]
        a=roc_auc_score(Y,C.ensemble_batch(Xp,ref_sem))
        res.append((gname,base-a))
    res.sort(key=lambda r:-r[1])
    for g,caida in res:
        bar="#"*int(max(0,caida)*400)
        print(f"  {g:20s} caida AUC {caida:+.4f}  {bar}")
    print(f"\n  (mayor caida = feature mas influyente en la decision del modelo)")

if __name__ == "__main__":
    main()
