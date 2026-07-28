# -*- coding: utf-8 -*-
"""No-estacionariedad del edge de reversion: fluctua en el tiempo? es DETECTABLE?

No-estacionariedad sola no da edge: solo es explotable si el regimen es PERSISTENTE (una
epoca buena predice la siguiente -> puedes detectar "esto funciona" y operar). Se mide:
  1. varianza del WR entre bloques temporales vs la esperada por azar (hay regimen?).
  2. autocorrelacion lag-1 del WR (tiene MEMORIA -> detectable?).

Resultado 2026-07-28: varianza 1.52x (regimen leve) PERO autocorrelacion +0.016 (~0): el
edge oscila alrededor de 52.55% (< BE) de forma IMPREDECIBLE. NO detectable -> operar "solo
cuando funciona" es imposible. Usa el OOS cacheado por combinacion_reversion.py.
"""
import os, sys, numpy as np, datetime
sys.stdout.reconfigure(encoding="utf-8")

BE = 1/1.87; THR = 0.54; ROLL = (20,21,22); K = 20
NPZ = os.path.join(os.environ.get("TEMP","."), "oos_reversion.npz")

def main():
    if not os.path.exists(NPZ):
        print("Falta el OOS cacheado. Corre antes: python combinacion_reversion.py"); return
    z=np.load(NPZ);P,ADX,Y,T=z["P"],z["ADX"],z["Y"],z["T"]
    hor=((T//3600)%24).astype(int);fuera=~np.isin(hor,ROLL)
    ap=((P>=THR)|(P<=1-THR))&fuera
    gano=np.where(P>=THR,Y==1,Y==0).astype(float)[ap];t=T[ap]
    o=np.argsort(t);gano=gano[o];t=t[o]
    b=np.array_split(np.arange(len(gano)),K)
    wr=np.array([gano[ix].mean() for ix in b]);ns=np.array([len(ix) for ix in b])
    fechas=[datetime.datetime.utcfromtimestamp(t[ix[0]]).strftime('%m-%d') for ix in b]
    print(f"break-even {100*BE:.2f}% | {len(gano)} senales en {K} bloques de ~{ns.mean():.0f}\n")
    print("bloque  desde   WR")
    for i in range(K):
        mark="+" if wr[i]>BE else " "
        print(f"  {i+1:2d}  {fechas[i]}  {100*wr[i]:5.1f}% {mark}")
    p=gano.mean();var_obs=wr.var();var_bin=np.mean(p*(1-p)/ns)
    print(f"\nWR global {100*p:.2f}% | varianza WR: obs {var_obs:.5f} vs azar {var_bin:.5f} "
          f"= {var_obs/var_bin:.2f}x ({'NO-estacionario' if var_obs/var_bin>1.5 else 'compatible con azar'})")
    ac=np.corrcoef(wr[:-1],wr[1:])[0,1]
    print(f"autocorrelacion lag-1: {ac:+.3f} "
          f"({'PERSISTENTE->detectable' if ac>0.25 else 'SIN memoria->NO detectable'})")
    print(f"bloques que baten BE: {int((wr>BE).sum())}/{K}")

if __name__ == "__main__":
    main()
