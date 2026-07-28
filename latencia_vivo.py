# -*- coding: utf-8 -*-
"""Medidor de latencia EN VIVO: el feed de IQ va con retraso respecto a Binance?

Captura el precio de ETHUSD en IQ y de ETHUSDT en Binance a la vez, con el MISMO reloj
local, durante unos minutos, y cross-correlaciona los retornos para hallar el desfase.

  lag optimo > 0  -> el retorno de IQ(t) se parece al de Binance(t-lag): IQ va DETRAS
                     ese 'lag' de segundos -> ARBITRABLE (ves Binance, apuestas IQ).
  lag ~ 0         -> sincronizados: no hay arbitraje de latencia.

Mide el desfase OBSERVADO desde ESTA maquina (feed + red juntos), que es justo lo que
importa para operar desde aqui. Uso:  python latencia_vivo.py [segundos]  (def 300)
"""
import os, sys, time, json, threading
import numpy as np

DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 300
IQ_ACTIVE = "ETHUSD"
BIN_SYM = "ethusdt"

# --- .env ---
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env):
    for ln in open(_env, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, _, v = ln.partition("="); os.environ.setdefault(k.strip(), v.strip())

muestras_iq = []    # (ts_local, precio)
muestras_bin = []   # (ts_local, precio)
_stop = threading.Event()

# --- Binance: WebSocket de trades ---
def binance_ws():
    import websocket
    url = f"wss://stream.binance.com:9443/ws/{BIN_SYM}@trade"
    def on_msg(ws, msg):
        try:
            d = json.loads(msg)
            muestras_bin.append((time.time(), float(d["p"])))
        except Exception:
            pass
    def on_err(ws, e): pass
    while not _stop.is_set():
        try:
            ws = websocket.WebSocketApp(url, on_message=on_msg, on_error=on_err)
            ws.run_forever(ping_interval=20)
        except Exception:
            time.sleep(1)
        if _stop.is_set():
            break

# --- IQ: stream de velas en formacion ---
def iq_stream():
    from iqoptionapi.stable_api import IQ_Option
    api = IQ_Option(os.getenv("IQ_EMAIL"), os.getenv("IQ_PASSWORD"))
    ok, reason = api.connect()
    if not ok:
        print(f"[IQ] no conecto: {reason}"); _stop.set(); return
    api.change_balance("PRACTICE")
    api.start_candles_stream(IQ_ACTIVE, 1, 5)
    print("[IQ] stream iniciado", flush=True)
    last = None
    while not _stop.is_set():
        try:
            rc = api.get_realtime_candles(IQ_ACTIVE, 1)
            if rc:
                k = max(rc.keys())
                px = rc[k]["close"]
                if px != last:                     # solo cambios de precio
                    muestras_iq.append((time.time(), float(px)))
                    last = px
        except Exception:
            pass
        time.sleep(0.10)
    try: api.stop_candles_stream(IQ_ACTIVE, 1)
    except Exception: pass

def analizar():
    if len(muestras_iq) < 50 or len(muestras_bin) < 50:
        print(f"muestras insuficientes: IQ={len(muestras_iq)} Binance={len(muestras_bin)}")
        return
    iq = np.array(muestras_iq); bn = np.array(muestras_bin)
    t0 = max(iq[0,0], bn[0,0]); t1 = min(iq[-1,0], bn[-1,0])
    step = 0.5
    grid = np.arange(t0, t1, step)
    def ffill(m):
        idx = np.searchsorted(m[:,0], grid, side="right") - 1
        idx = np.clip(idx, 0, len(m)-1)
        return m[idx, 1]
    pi = ffill(iq); pb = ffill(bn)
    ri = np.diff(pi) / pi[:-1]; rb = np.diff(pb) / pb[:-1]
    print(f"\n=== latencia IQ vs Binance ({IQ_ACTIVE}) ===")
    print(f"muestras: IQ={len(iq)} (cambios de precio), Binance={len(bn)} trades")
    print(f"ventana alineada: {t1-t0:.0f}s a {step*1000:.0f}ms -> {len(grid)} puntos")
    print(f"corr contemporanea: {np.corrcoef(ri, rb)[0,1]:.4f}")
    print(f"\n{'lag(s)':>8} {'corr':>8}   (lag>0 = IQ va DETRAS de Binance)")
    mejor = (0.0, 0.0)
    for L in range(-4, 31):                     # -2s .. +15s en pasos de 500ms
        if L >= 0:
            a, b = ri[L:], rb[:len(rb)-L] if L > 0 else rb
        else:
            a, b = ri[:L], rb[-L:]
        m = min(len(a), len(b))
        if m < 50: continue
        c = np.corrcoef(a[:m], b[:m])[0,1]
        if c > mejor[1]: mejor = (L*step, c)
        bar = "#" * int(max(0, c) * 50)
        print(f"{L*step:>8.2f} {c:>8.4f}  {bar}")
    print(f"\n>>> lag optimo: {mejor[0]:+.2f}s (corr {mejor[1]:.4f})")
    if mejor[0] > 0.2:
        print(f">>> IQ va ~{mejor[0]:.2f}s DETRAS de Binance -> potencialmente ARBITRABLE")
    else:
        print(f">>> sincronizados (lag ~0): NO hay arbitraje de latencia observable desde aqui")

if __name__ == "__main__":
    print(f"midiendo {DUR}s... (ETHUSD IQ vs ETHUSDT Binance)", flush=True)
    tb = threading.Thread(target=binance_ws, daemon=True)
    ti = threading.Thread(target=iq_stream, daemon=True)
    tb.start(); ti.start()
    t_end = time.time() + DUR
    while time.time() < t_end and not _stop.is_set():
        time.sleep(2)
        if int(time.time()) % 30 < 2:
            print(f"  ...IQ={len(muestras_iq)} Binance={len(muestras_bin)}", flush=True)
    _stop.set(); time.sleep(1.5)
    analizar()
