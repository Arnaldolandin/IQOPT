# -*- coding: utf-8 -*-
"""El edge del modelo es INUTIL en binarias (payout 87% -> BE 53.48%) pero puede ser
RENTABLE en payoff SIMETRICO (perpetuos/futuros, BE ~50% + comision).

Aqui se aplica la senal del modelo (ensemble 11 feats, la MISMA del bot) como long/short
en un perpetuo y se mide el PnL NETO de comisiones. OOS estricto, sin BTCUSD, rollover
separado. Foco cripto (Bybit) pero se reporta todo.

Clave: en simetrico no importa solo el signo (WR) sino el RETORNO capturado. Un modelo con
52% de acierto direccional PIERDE en binarias pero puede GANAR en perpetuo si el
movimiento medio supera la comision.
"""
import json, os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import seq_model as S
import calibracion as C

CACHE = "cache_ohlc_5m_v2"
L, H = 64, 2
EMB = (L + H) * 300
ROLL = (20, 21, 22)
THR = 0.54
MAX_POR_PAR = 1500
FEES = {"maker 0.02%": 0.0002, "taker 0.055%": 0.00055}   # por lado; ida+vuelta = x2
CRIPTO = {"ETHUSD", "XRPUSD"}

def main():
    import datetime
    cfg = json.load(open("config.json", encoding="utf-8"))
    pares = [p for p in cfg["entrenamiento"]["pares"] if p != "BTCUSD"]
    rng = np.random.default_rng(0)
    COLA = L + max(S.ATR_P, S.RSI_P, S.BB_P) + 1
    reg = {}   # par -> (P, dir_real_ret, t)
    for par in pares:
        sem = C.semillas(par)
        if len(sem) < 1: continue
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
        fs,ret,ts_=[],[],[]
        for i in cand:
            f=S.ventana_features(V[i-COLA:i+1],L,vol=(None if vol is None else vol[i-COLA:i+1]))
            if f is None: continue
            fs.append(f); ret.append((c[i+H]-c[i])/c[i]); ts_.append(tt[i])
        if not fs: continue
        P=C.ensemble_batch(np.asarray(fs,np.float64),sem)
        reg[par]=(P,np.array(ret),np.array(ts_))
        print(f"  {par:8s} n={len(fs)}",flush=True)

    def pnl_grupo(pares_g, etq):
        Ps=np.concatenate([reg[p][0] for p in pares_g if p in reg])
        R =np.concatenate([reg[p][1] for p in pares_g if p in reg])
        T =np.concatenate([reg[p][2] for p in pares_g if p in reg])
        hor=((T//3600)%24).astype(int); fuera=~np.isin(hor,ROLL)
        # senal: long si P>=THR, short si P<=1-THR
        lado=np.where(Ps>=THR,1.0,np.where(Ps<=1-THR,-1.0,0.0))
        m=(lado!=0)&fuera
        r=R[m]; d=lado[m]
        bruto=d*r                      # retorno capturado por operacion (sin fees)
        wr=(bruto>0).mean()
        print(f"\n=== {etq}: {int(m.sum())} operaciones (thr {THR}, fuera rollover) ===")
        print(f"  WR direccional {100*wr:.2f}% | retorno BRUTO medio/op {1e4*bruto.mean():+.2f} bps "
              f"| mov medio |ret| {1e4*np.abs(r).mean():.1f} bps")
        for fn,fee in FEES.items():
            neto=bruto-2*fee            # comision ida+vuelta
            tot=neto.sum(); med=neto.mean()
            # sharpe por operacion
            sh=med/ (neto.std()+1e-12)
            print(f"  fee {fn:14s}: neto medio/op {1e4*med:+.2f} bps | acumulado {100*tot:+.2f}% "
                  f"| Sharpe/op {sh:+.4f} | {'RENTABLE' if med>0 else 'pierde'}")

    pnl_grupo([p for p in reg if p in CRIPTO], "CRIPTO (ETH+XRP, estilo Bybit perp)")
    for p in CRIPTO:
        if p in reg: pnl_grupo([p], p)
    pnl_grupo([p for p in reg if p not in CRIPTO], "resto (forex+stocks)")

if __name__ == "__main__":
    main()
