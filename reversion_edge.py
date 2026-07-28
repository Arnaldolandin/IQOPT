# -*- coding: utf-8 -*-
"""Cambio de enfoque: explotar la REVERSION a corto plazo, no la direccion del LSTM.

El unico edge real que el proyecto documento (memoria reversion-corto-plazo-real): cuando
el precio se aleja demasiado (Bollinger %B / RSI extremos) tiende a volver. Aqui se mide
SIN el LSTM (senal pura de reversion), OOS honesto, contra control barajado, con la
verificacion de estabilidad (2 mitades) desde el inicio, y -- lo decisivo -- como decae
con el RETARDO de ejecucion (entrar 0, 1 o 2 velas despues de la senal).

Break-even correcto por horizonte: H=1 -> turbo 54.64% ; H>=2 -> binary 53.48%.
Si la reversion no bate su BE con control, no hay edge. Si lo bate pero muere al retardo 1,
el bot (que entra con retraso) no puede capturarlo.
"""
import json, os, sys, math
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import horizonte_test as HT   # reusa feats() vectorizado (rsi, bb, etc.)

CACHE = "cache_ohlc_5m_v2"
ROLL = (20, 21, 22)
BE = {1: 1/1.83, 2: 1/1.87, 3: 1/1.87}   # turbo vs binary
BB_HI, BB_LO = 0.95, 0.05                  # extremos de Bollinger %B (senal de reversion)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)

def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    rng = np.random.default_rng(0)
    # acumular senales de reversion OOS: (dir_apostada, c_en_senal, i, par, t, close_array_ref)
    # guardamos por par para poder mirar retardos con el close real
    registros = []   # (par, i, dir, t_i)  dir=+1 espera subir, -1 espera bajar
    closes = {}      # par -> (c, t)
    for par in pares:
        d = json.load(open(os.path.join(CACHE, par + ".json"), encoding="utf-8"))
        o=np.asarray(d["open"],float);h=np.asarray(d["high"],float);l=np.asarray(d["low"],float)
        c=np.asarray(d["close"],float);t=np.asarray(d["times"],float)
        vol=d.get("volume"); vol=np.asarray(vol,float) if vol else None
        n=len(c); closes[par]=(c,t)
        X,names=HT.feats(o,h,l,c,vol,t)
        bb=X[:,names.index("bb")]; rsi=X[:,names.index("rsi")]
        corte=float(np.quantile(t,0.65)); EMB=66*300
        for i in range(25,n-8):
            if t[i]<=corte+EMB: continue
            if not (np.isfinite(bb[i]) and np.isfinite(rsi[i])): continue
            # senal de reversion: banda superior -> espera BAJADA; inferior -> SUBIDA
            if bb[i]>=BB_HI or rsi[i]>=70: registros.append((par,i,-1,t[i]))
            elif bb[i]<=BB_LO or rsi[i]<=30: registros.append((par,i,+1,t[i]))

    print(f"=== senales de reversion OOS: {len(registros)} ===")
    fams_t=np.array([r[3] for r in registros])
    hor=((fams_t//3600)%24).astype(int); fuera=~np.isin(hor,ROLL)
    print(f"fuera de rollover: {int(fuera.sum())}\n")

    def evalua(H, retardo, mask_extra=None):
        """WR de apostar la reversion, entrando 'retardo' velas despues de la senal,
        resultado a H velas desde la entrada. Continuidad exacta."""
        k=nn=0; idxs=[]
        for j,(par,i,dr,ti) in enumerate(registros):
            if mask_extra is not None and not mask_extra[j]: continue
            c,t=closes[par]
            e=i+retardo                      # vela de entrada
            if e+H>=len(c): continue
            if t[e+H]-t[e]!=H*300: continue  # continuidad de la opcion
            if t[e]-t[i]!=retardo*300: continue  # el retardo tambien continuo
            if c[e+H]==c[e]: continue
            subio = c[e+H]>c[e]
            gano = (subio and dr>0) or ((not subio) and dr<0)
            k+=gano; nn+=1; idxs.append(j)
        return k,nn,idxs

    print(f"=== WR de reversion por HORIZONTE y RETARDO (fuera de rollover) ===")
    print(f"{'H':>3} {'retardo':>8} {'BE':>7} {'WR':>8} {'n':>6} {'IC95':>16} {'vs BE':>7}")
    for H in (1,2,3):
        be=BE[H]
        for ret in (0,1,2):
            # aplicar fuera-rollover como mask sobre registros
            k,nn,idxs=evalua(H,ret,fuera)
            if nn<50: continue
            wr=k/nn; lo,hi=wilson(k,nn)
            marca = "bate" if lo>be else ("no" )
            print(f"{H:>3} {ret:>8} {100*be:6.2f}% {100*wr:7.2f}% {nn:6d} [{100*lo:.1f},{100*hi:.1f}] {marca:>7}")
        print()

    # control barajado: barajar la direccion apostada -> suelo de ruido (H=1 ret=0)
    print(f"=== control barajado (H=1, retardo 0): suelo de ruido ===")
    dirs=np.array([r[2] for r in registros])
    dsh=rng.permutation(dirs)
    k=nn=0
    for j,(par,i,dr,ti) in enumerate(registros):
        if not fuera[j]: continue
        c,t=closes[par]; e=i
        if e+1>=len(c) or t[e+1]-t[e]!=300 or c[e+1]==c[e]: continue
        subio=c[e+1]>c[e]; g=(subio and dsh[j]>0) or ((not subio) and dsh[j]<0)
        k+=g; nn+=1
    if nn: print(f"control WR {100*k/nn:.2f}% n={nn} (debe rondar 50%)")

    # persistencia: 2 mitades temporales (H=1 ret=0)
    print(f"\n=== persistencia H=1 retardo 0 (2 mitades del test) ===")
    k,nn,idxs=evalua(1,0,fuera)
    ts_sel=np.array([registros[j][3] for j in idxs])
    med=np.median(ts_sel)
    for nom,mm in [("1a mitad",ts_sel<=med),("2a mitad",ts_sel>med)]:
        sub=[idxs[x] for x in range(len(idxs)) if mm[x]]
        kk=nn2=0
        for j in sub:
            par,i,dr,ti=registros[j]; c,t=closes[par]
            if i+1>=len(c) or t[i+1]-t[i]!=300 or c[i+1]==c[i]: continue
            subio=c[i+1]>c[i]; kk+=(subio and dr>0) or ((not subio) and dr<0); nn2+=1
        if nn2: print(f"  {nom}: WR {100*kk/nn2:.2f}% n={nn2} (BE turbo {100*BE[1]:.2f}%)")

if __name__ == "__main__":
    main()
