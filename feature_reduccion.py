# -*- coding: utf-8 -*-
"""Reducir features mejora el modelo? (menos overfitting con senal debil)

La importancia por permutacion (2026-07-27) mostro: Bollinger domina, retorno tiene
importancia NEGATIVA (mete ruido), forma_vela/RSI/volumen/hora ~0. Con senal tan debil,
un modelo con MENOS features puede generalizar mejor. Se compara el AUC OOS de distintos
subconjuntos de las 11 features (baseline HGB sobre la ventana aplanada, mismo test).
Si un subconjunto bate a las 11, valdria reentrenar el LSTM con menos features.

Columnas: 0=retorno 1-4=forma_vela 5-6=hora 7-8=volumen 9=RSI 10=Bollinger
"""
import json, os, sys, warnings
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from modelos_compara import build

PARES = ["ETHUSD","XRPUSD","EURUSD","GBPUSD","USDJPY","EURJPY","AUDUSD","XAUUSD","GBPJPY","EURGBP"]
SUBSETS = {
    "todas (11)":            list(range(11)),
    "sin retorno (10)":      [1,2,3,4,5,6,7,8,9,10],
    "sin retorno+forma (6)": [5,6,7,8,9,10],
    "Bollinger+RSI+vol (4)": [7,8,9,10],
    "Bollinger+RSI (2)":     [9,10],
    "solo Bollinger (1)":    [10],
}

def main():
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for par in PARES:
        try: Xtr,ytr,Xte,yte,_ = build(par)
        except Exception as e: print(f"  {par}: {type(e).__name__}"); continue
        if len(Xte) < 200: continue
        Xtr_l.append(Xtr);ytr_l.append(ytr);Xte_l.append(Xte);yte_l.append(yte)
        print(f"  {par} listo",flush=True)
    Xtr=np.vstack(Xtr_l);ytr=np.concatenate(ytr_l);Xte=np.vstack(Xte_l);yte=np.concatenate(yte_l)
    print(f"\npool: train {len(Xtr)} test {len(Xte)}\n")
    print(f"{'subconjunto':>24} {'AUC OOS':>9} {'vs 11':>8}")
    base=None
    for nom,cols in SUBSETS.items():
        Ftr=Xtr[:,:,cols].reshape(len(Xtr),-1)
        Fte=Xte[:,:,cols].reshape(len(Xte),-1)
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=4,
                                         l2_regularization=1.0,random_state=42).fit(Ftr,ytr)
        a=roc_auc_score(yte,m.predict_proba(Fte)[:,1])
        if base is None: base=a
        print(f"{nom:>24} {a:9.4f} {a-base:+8.4f}")
    print(f"\n(si un subconjunto reducido bate a 'todas', reentrenar el LSTM con menos features)")

if __name__ == "__main__":
    main()
