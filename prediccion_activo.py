# prediccion_activo.py - Analiza 3 meses de un activo, mide la precision REAL del
# predictor de 15 indicadores (pesos optimizados) sobre ese activo, y emite la
# prediccion actual con velas frescas en vivo. Sin humo: la confianza que reporta
# es la tasa de acierto OPERABLE medida, no un numero inventado.
#   .venv314\Scripts\python.exe prediccion_activo.py EURUSD
import json, sys, time
from datetime import datetime, timezone
import numpy as np
import predictor_opt as P
import main as M

PAR = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
TF = 300
MESES = 3
BARS_3M = int(MESES * 30 * 24 * 60 / (TF / 60))   # ~25920 velas de 5m


def backtest_3m(o, h, l, c):
    """Precision REAL del predictor sobre este activo, 3 meses, entrada OPERABLE (+1 barra)."""
    V, _ = P.indicadores(o, h, l, c)
    V = np.nan_to_num(V)
    z = P.__dict__.get("_z")  # no existe; calculamos con los pesos de main
    W = np.asarray(M.PESOS_OPT, float); b = M.INTERCEPT_OPT
    zz = b + V.dot(W)
    p = 1.0 / (1.0 + np.exp(-np.clip(zz, -30, 30)))
    conf = np.abs(p - 0.5)
    lado = (p > 0.5).astype(float)                       # 1=call, 0=put
    n = len(c)
    # target OPERABLE: entrar en la apertura de la barra siguiente, resultado a +2 barras
    H = 2
    gan = np.zeros(n)
    gan[:n - 1 - H] = (c[1 + H:] > c[1:n - H]).astype(float)
    hi = n - 2 - H
    idx = np.arange(M.__dict__.get("WARM", 210) if False else 210, hi)
    mc = 0.02
    m = conf[idx] >= mc
    aciertos = (lado[idx] == gan[idx])
    wr_all = aciertos.mean() * 100
    wr_conf = aciertos[m].mean() * 100 if m.sum() else 0
    return wr_all, wr_conf, int(m.sum()), len(idx)


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    # ---- 1) PRECISION REAL sobre 3 meses (cache) ----
    d = json.load(open(f"cache_ohlc_5m/{PAR}.json", encoding="utf-8"))
    c = np.asarray(d["close"], float); o = np.asarray(d.get("open", d["close"]), float)
    h = np.asarray(d.get("high", d["close"]), float); l = np.asarray(d.get("low", d["close"]), float)
    s = max(0, len(c) - BARS_3M)
    o, h, l, c = o[s:], h[s:], l[s:], c[s:]
    wr_all, wr_conf, nops, ntot = backtest_3m(o, h, l, c)
    print("=" * 70)
    print(f"PRECISION REAL de la prediccion | {PAR} | 3 meses ({len(c)} velas 5m)")
    print(f"  entrada OPERABLE (+1 barra, sin rebote bid-ask), horizonte 10m")
    print(f"  acierto TODAS las senales:     {wr_all:.2f}%  ({ntot} predicciones)")
    print(f"  acierto con confianza>=0.02:   {wr_conf:.2f}%  ({nops} predicciones)")
    print(f"  break-even (payout 84%):       53.5%")
    print("=" * 70)

    # ---- 2) PREDICCION ACTUAL con velas frescas ----
    from iqoptionapi.stable_api import IQ_Option
    api = IQ_Option(cfg["email"], cfg["password"])
    ok, _ = api.connect()
    if not ok:
        print("no conecto para velas frescas"); return
    api.change_balance("PRACTICE"); time.sleep(2)
    v = api.get_candles(PAR, TF, 320, time.time())
    cc = np.array([x["close"] for x in v], float)
    hh = np.array([x["max"] for x in v], float)
    ll = np.array([x["min"] for x in v], float)
    # excluir la ultima vela en formacion
    cc, hh, ll = cc[:-1], hh[:-1], ll[:-1]
    precio = float(cc[-1])
    lado, conf, info = M.predecir(cc, hh, ll)
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"\nPREDICCION ACTUAL {PAR} @ {ahora} UTC")
    print(f"  precio de referencia: {precio:.5f}")
    print(f"  {info}")
    dirtxt = "SUBE (CALL)" if lado == "call" else "BAJA (PUT)"
    print(f"  >>> {dirtxt}  en los proximos 10 min")
    print(f"  VERIFICABLE: {PAR} {'>' if lado=='call' else '<'} {precio:.5f} a las "
          f"{datetime.now(timezone.utc).strftime('%H:%M')}+10min UTC")
    prob_real = wr_conf if conf >= 0.02 else wr_all
    print(f"  probabilidad honesta de acertar (medida en 3m): ~{prob_real:.1f}%")


if __name__ == "__main__":
    main()
