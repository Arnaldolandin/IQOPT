# predictor.py - Predictor direccional (el mismo analisis que hicimos a mano):
# baja datos frescos y vota con MACD+histograma, EMA50, ultimas velas, RSI y Bollinger.
# Emite CALL/PUT con desglose de votos, confianza y referencia verificable.
#   .venv314\Scripts\python.exe predictor.py                 # EURUSD, 5m, horizonte 15m, 1 vez
#   .venv314\Scripts\python.exe predictor.py EURUSD 300 3    # par, tf_seg, horizonte(velas)
#   .venv314\Scripts\python.exe predictor.py EURUSD 300 3 loop   # cada vela nueva
import json, sys, time
from datetime import datetime, timezone
import numpy as np
from iqoptionapi.stable_api import IQ_Option

PAR = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
TF = int(sys.argv[2]) if len(sys.argv) > 2 else 300
HORIZ = int(sys.argv[3]) if len(sys.argv) > 3 else 3      # velas hacia adelante
LOOP = len(sys.argv) > 4 and sys.argv[4] == "loop"


def ema(x, s):
    a = 2.0 / (s + 1); o = np.copy(x)
    for i in range(1, len(x)):
        o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o


def rsi(c, p=14):
    d = np.diff(c); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = np.convolve(up, np.ones(p) / p, "valid"); ad = np.convolve(dn, np.ones(p) / p, "valid")
    return 100 - 100 / (1 + au[-1] / ad[-1]) if ad[-1] > 0 else 100.0


def analizar(c):
    """Devuelve (lado, confianza, votos[list], detalle[dict])."""
    ml = ema(c, 6) - ema(c, 13); sig = ema(ml, 5)
    hist = ml - sig
    ema50 = ema(c, 50)[-1]
    r = rsi(c, 14)
    m20 = c[-20:].mean(); s20 = c[-20:].std(); bbz = (c[-1] - m20) / s20 if s20 > 0 else 0.0
    votos = []
    # 1) posicion MACD
    votos.append(("MACD>Sig" if hist[-1] > 0 else "MACD<Sig", 1 if hist[-1] > 0 else -1))
    # 2) momentum del histograma (crece/decrece)
    dmom = hist[-1] - hist[-2]
    votos.append(("hist " + ("creciendo" if dmom > 0 else "cayendo"), 1 if dmom > 0 else -1))
    # 3) tendencia EMA50
    votos.append(("precio " + ("sobre EMA50" if c[-1] > ema50 else "bajo EMA50"), 1 if c[-1] > ema50 else -1))
    # 4) ultimas 3 velas
    up3 = c[-1] > c[-2] > c[-3]; dn3 = c[-1] < c[-2] < c[-3]
    votos.append(("ult3 " + ("subiendo" if up3 else "bajando" if dn3 else "mixto"), 1 if up3 else -1 if dn3 else 0))
    # 5) RSI extremo (reversion)
    votos.append(("RSI " + (f"{r:.0f} sobrecompra" if r > 70 else f"{r:.0f} sobreventa" if r < 30 else f"{r:.0f} neutro"),
                  -1 if r > 70 else 1 if r < 30 else 0))
    # 6) Bollinger extremo (reversion)
    votos.append((f"BBz {bbz:+.2f}", -1 if bbz > 2 else 1 if bbz < -2 else 0))

    net = sum(v for _, v in votos)
    lado = "call" if net > 0 else "put" if net < 0 else ("call" if hist[-1] > 0 else "put")
    activos = sum(1 for _, v in votos if v != 0)
    conf = 50 + (abs(net) / max(activos, 1)) * 8   # 50%..~58% honesto
    detalle = dict(macd=ml[-1], sig=sig[-1], hist=hist[-1], ema50=ema50, rsi=r, bbz=bbz)
    return lado, conf, votos, detalle


def predecir(api):
    v = api.get_candles(PAR, TF, 300, time.time())
    c = np.array([x["close"] for x in v], float)
    # usar velas CERRADAS (excluir la ultima en formacion)
    cerr = c[:-1]
    precio = float(cerr[-1])
    lado, conf, votos, det = analizar(cerr)
    ahora = datetime.now(timezone.utc).strftime("%H:%M")
    print(f"\n=== PREDICCION {PAR} @ {ahora} UTC ===")
    print(f"precio ref: {precio:.5f}  | horizonte: {HORIZ} velas ({HORIZ*TF//60} min)")
    print(f"MACD {det['macd']:+.6f}/{det['sig']:+.6f}  hist {det['hist']:+.6f}  EMA50 {det['ema50']:.5f}  "
          f"RSI {det['rsi']:.0f}  BBz {det['bbz']:+.2f}")
    print("votos: " + " | ".join(f"{n}[{'+' if val>0 else '' }{val}]" for n, val in votos))
    net = sum(val for _, val in votos)
    print(f">>> {('CALL (SUBE)' if lado=='call' else 'PUT (BAJA)')}  | score {net:+d}  | confianza ~{conf:.0f}%")
    print(f"    verificable: {PAR} {'>' if lado=='call' else '<'} {precio:.5f} en {HORIZ*TF//60} min")
    return precio, lado


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    ok, _ = api.connect()
    if not ok:
        print("no conecto"); return
    api.change_balance("PRACTICE"); time.sleep(2)
    print(f"Predictor {PAR} tf={TF}s horizonte={HORIZ} velas {'(loop)' if LOOP else ''}")
    print("HONESTIDAD: es una lectura de confluencia. Acierta ~50%. Una prediccion no prueba nada.")
    if not LOOP:
        predecir(api)
        return
    ultima = None
    while True:
        try:
            v = api.get_candles(PAR, TF, 3, time.time())
            vc = int(v[-2]["from"])
            if vc != ultima:
                ultima = vc
                predecir(api)
            time.sleep(5)
        except Exception as e:
            print("[warn]", str(e)[:50]); time.sleep(5)


if __name__ == "__main__":
    main()
