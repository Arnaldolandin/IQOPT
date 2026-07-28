# -*- coding: utf-8 -*-
"""Otros modelos son mas eficientes (mejor prediccion) que el LSTM?

Comparacion JUSTA: mismas features (ventana 64x11), mismo corte temporal, mismo test
intocado. El LSTM ve la secuencia; a los tabulares se les da la ventana APLANADA (704
features), o sea la misma informacion. AUC + logloss OOS. Control barajado = suelo.

Si todos empatan cerca de 0.52-0.53, el modelo NO es el cuello de botella (lo es la senal
en los datos). Si alguno bate al LSTM con holgura, valdria cambiarlo.
"""
import json, os, sys, warnings
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss
import seq_model as S
import calibracion as C

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
PARES = ["ETHUSD", "XRPUSD", "EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD"]
MAX_TR, MAX_TE = 20000, 12000

def build(par):
    import datetime
    d = json.load(open(f"{CACHE}/{par}.json", encoding="utf-8"))
    o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
    c=np.asarray(d["close"],float);tt=np.asarray(d["times"],float)
    vol=d.get("volume");vol=np.asarray(vol,float) if vol else None
    n=len(c);V=[[tt[i],o[i],h[i],l[i],c[i]] for i in range(n)]
    j=json.load(open(f"models/seq_lstm_{par}.pt.json"));ci=j["meta"].get("corte")
    corte=(datetime.datetime.fromisoformat(ci).replace(tzinfo=datetime.timezone.utc).timestamp()
           if ci else float(np.quantile(tt,0.65)))
    COLA=L+max(S.ATR_P,S.RSI_P,S.BB_P)+1
    Xtr,ytr,Xte,yte,Fte=[],[],[],[],[]
    sem=C.semillas(par)
    for i in range(COLA,n-H):
        if tt[i+H]-tt[i]!=H*300 or c[i+H]==c[i]: continue
        f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
        if f is None: continue
        y=int(c[i+H]>c[i])
        if tt[i]<corte-EMB: Xtr.append(f);ytr.append(y)
        elif tt[i]>corte+EMB: Xte.append(f);yte.append(y)
    return (np.array(Xtr),np.array(ytr),np.array(Xte),np.array(yte),sem)

def main():
    rng=np.random.default_rng(0)
    # acumular todos los pares (pool)
    XTR,YTR,XTE,YTE=[],[],[],[]
    P_lstm=[]   # prediccion del LSTM ensemble en test
    for par in PARES:
        try: Xtr,ytr,Xte,yte,sem=build(par)
        except Exception as e: print(f"  {par}: {type(e).__name__}"); continue
        if len(Xte)<200 or len(sem)<1: continue
        # LSTM ensemble en test
        p=C.ensemble_batch(Xte.astype(np.float64),sem)
        XTR.append(Xtr);YTR.append(ytr);XTE.append(Xte);YTE.append(yte);P_lstm.append(p)
        print(f"  {par:8s} train={len(Xtr)} test={len(Xte)}",flush=True)
    Xtr=np.vstack(XTR);ytr=np.concatenate(YTR);Xte=np.vstack(XTE);yte=np.concatenate(YTE)
    plstm=np.concatenate(P_lstm)
    # submuestreo para modelos lentos
    if len(Xtr)>MAX_TR:
        s=rng.choice(len(Xtr),MAX_TR,replace=False); Xtr,ytr=Xtr[s],ytr[s]
    # aplanar la ventana para tabulares
    Ftr=Xtr.reshape(len(Xtr),-1); Fte=Xte.reshape(len(Xte),-1)
    print(f"\npool: train {len(Xtr)} | test {len(Xte)} | features aplanadas {Ftr.shape[1]}\n")

    def rep(nombre, p):
        p=np.clip(p,1e-6,1-1e-6)
        print(f"  {nombre:22s} AUC {roc_auc_score(yte,p):.4f}  logloss {log_loss(yte,p):.5f}")

    print(f"=== capacidad predictiva OOS (AUC 0.5 = azar, ln2 logloss 0.6931) ===")
    rep("LSTM (actual, seq)", plstm)
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,
                                     l2_regularization=1.0,random_state=42).fit(Ftr,ytr)
    rep("GradientBoosting", m.predict_proba(Fte)[:,1])
    m=LogisticRegression(max_iter=1000,C=0.1).fit(Ftr,ytr)
    rep("LogisticRegression", m.predict_proba(Fte)[:,1])
    m=RandomForestClassifier(n_estimators=200,max_depth=8,n_jobs=-1,random_state=42).fit(Ftr,ytr)
    rep("RandomForest", m.predict_proba(Fte)[:,1])
    m=MLPClassifier(hidden_layer_sizes=(64,),max_iter=150,early_stopping=True,
                    random_state=42).fit(Ftr,ytr)
    rep("MLP (red densa)", m.predict_proba(Fte)[:,1])
    # control: barajar etiquetas de train, GB -> suelo
    m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,
                                     random_state=1).fit(Ftr,rng.permutation(ytr))
    rep("CONTROL barajado", m.predict_proba(Fte)[:,1])
    print(f"\nSi todos ~0.52-0.53, el modelo NO es el cuello: lo es la senal en los datos.")

if __name__ == "__main__":
    main()
