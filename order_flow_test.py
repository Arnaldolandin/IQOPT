# -*- coding: utf-8 -*-
"""Test de viabilidad de ORDER FLOW (cripto, Binance = feed IQ corr 0.998).

El OHLC ve el volumen TOTAL pero no su DIRECCION. Binance da en cada vela el 'taker buy
volume' (compras agresivas que cruzan el spread). El desequilibrio:
    OFI = (2*taker_buy - volumen) / volumen   in [-1,1]   (+ = presion compradora)
es informacion NUEVA que el modelo no tiene. Se mide si predice el movimiento a H=2 (10m):
  1. correlacion OFI vs retorno futuro.
  2. como SEÑAL: OFI extremo -> CALL/PUT, WR a H=2, persistencia 2 mitades.
  3. como FEATURE: baseline HGB (OHLC) vs + OFI, AUC OOS.
OOS estricto (corte temporal), rollover separado, break-even 53.48%.
"""
import json, os, sys, math, urllib.request, warnings
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import horizonte_test as HT

H=2; BE=1/1.87; ROLL=(20,21,22)
SYMS=["ETHUSDT","XRPUSDT","BTCUSDT"]

def binance_full(symbol, interval, n_target):
    out=[]; end=None
    while len(out)<n_target:
        url=f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000"
        if end: url+=f"&endTime={end}"
        d=json.load(urllib.request.urlopen(url,timeout=25))
        if not d: break
        out=d+out; end=d[0][0]-1
        if len(d)<1000: break
    return out

def wil(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return((c-h)/d,(c+h)/d)

def main():
    OFI_all=[];RETF=[];Y=[];T=[];Xb=[];Xo=[]
    for sym in SYMS:
        print(f"bajando {sym} 5m con order flow...",flush=True)
        d=binance_full(sym,"5m",40000)
        o=np.array([float(k[1]) for k in d]);h=np.array([float(k[2]) for k in d])
        l=np.array([float(k[3]) for k in d]);c=np.array([float(k[4]) for k in d])
        vol=np.array([float(k[5]) for k in d]);tb=np.array([float(k[9]) for k in d])  # taker buy base
        t=np.array([k[0]//1000 for k in d],float)
        ofi=np.where(vol>0,(2*tb-vol)/vol,0.0)                # desequilibrio [-1,1]
        ofi_ma=np.convolve(ofi,np.ones(5)/5,mode="full")[:len(ofi)]  # flujo suavizado 5 velas
        X,names=HT.feats(o,h,l,c,vol,t)
        finX=np.isfinite(X).all(1)
        n=len(c);corte=float(np.quantile(t,0.65));emb=(64+H)*300
        for i in range(20,n-H):
            if t[i+H]-t[i]!=H*300 or c[i+H]==c[i] or not finX[i] or not np.isfinite(ofi[i]): continue
            if not (t[i]<corte-emb or t[i]>corte+emb): continue
            OFI_all.append(ofi[i]);RETF.append((c[i+H]-c[i])/c[i]);Y.append(int(c[i+H]>c[i]));T.append(t[i])
            Xb.append(X[i]);Xo.append([ofi[i],ofi_ma[i]])
    OFI=np.array(OFI_all);RF=np.array(RETF);Y=np.array(Y);T=np.array(T)
    Xb=np.array(Xb);Xo=np.array(Xo)
    corte=float(np.quantile(T,0.65));tr=T<corte;te=T>=corte
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    print(f"\n=== {len(OFI)} puntos | BE {100*BE:.2f}% ===")
    print(f"1) corr(OFI, retorno futuro a {H*5}min): {np.corrcoef(OFI,RF)[0,1]:+.4f}  "
          f"(+ = mas compra agresiva -> sube)")

    # 2) senal: OFI extremo
    med=np.median(T[te&fuera])
    q=np.quantile(OFI[tr],[0.25,0.75])
    print(f"\n2) SENAL de order flow (OFI>={q[1]:.2f}->CALL, OFI<={q[0]:.2f}->PUT), TEST fuera roll:")
    m=te&fuera&((OFI>=q[1])|(OFI<=q[0]))
    call=OFI[m]>=q[1]; subio=Y[m]==1
    gano=np.where(call,subio,~subio)
    k=int(gano.sum());nn=len(gano);wr=k/nn;lo,hi=wil(k,nn)
    m1=T[m]<=med;m2=T[m]>med
    g1=np.where(OFI[m][m1]>=q[1],Y[m][m1]==1,Y[m][m1]==0);g2=np.where(OFI[m][m2]>=q[1],Y[m][m2]==1,Y[m][m2]==0)
    print(f"   WR {100*wr:.2f}% n={nn} IC[{100*lo:.1f},{100*hi:.1f}] | "
          f"1a {100*g1.mean():.1f}%/2a {100*g2.mean():.1f}% {'AMBAS>BE' if g1.mean()>BE and g2.mean()>BE else ''}")

    # 3) feature contra baseline OHLC
    def auc(A,B):
        mdl=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42).fit(A,Y[tr])
        return roc_auc_score(Y[te],mdl.predict_proba(B)[:,1])
    a0=auc(Xb[tr],Xb[te]); a1=auc(np.hstack([Xb[tr],Xo[tr]]),np.hstack([Xb[te],Xo[te]]))
    print(f"\n3) FEATURE: AUC baseline OHLC {a0:.4f} | + order flow {a1:.4f} ({a1-a0:+.4f}) "
          f"-> {'APORTA' if a1-a0>0.003 else 'no aporta'}")

if __name__ == "__main__":
    main()
