# main.py - Bot REVERSION-Bollinger multi-activo MULTI-HILO en IQ Option.
#
# Estrategia: reversion con Bandas de Bollinger sobre velas cerradas.
#   CALL cuando el precio cruza por DEBAJO de la banda inferior (rebote).
#   PUT  cuando el precio cruza por ENCIMA de la banda superior.
#   Filtro ATR opcional (volatilidad minima). Multi-hilo, hasta max_trades a la vez.
#   Estrategia UNICA: predictor con 13 indicadores y pesos aprendidos (logistica). Config -> "operacion.min_conf".
#
#   .venv314\Scripts\python.exe main.py            # DEMO
#   .venv314\Scripts\python.exe main.py --dry      # solo loguea senales
#   .venv314\Scripts\python.exe main.py --real     # CUIDADO
import argparse
import json
import math
import os
import pickle
import time
import threading
from datetime import datetime, timezone

import numpy as np
from iqoptionapi.stable_api import IQ_Option

CFG = {}
POLL = 3
_balance_mode = "PRACTICE"

_lock = threading.Lock()
_trades_abiertos = 0
_sesion = {"trades": 0, "wins": 0, "pnl": 0.0, "balance_inicial": None}
_activos_ref = {"abiertos": 0}
_cruces_fallidos = set()
_ultima_ping = time.time()
_conectado = True

# Control de riesgo: PnL del dia (UTC) y timestamps de aperturas (ventana 1h).
_riesgo = {"fecha": None, "pnl_dia": 0.0, "ops": []}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open("rsi_iq.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def hay_capacidad():
    max_t = CFG.get("max_trades", 1)
    with _lock:
        return _trades_abiertos < max_t


def sumar_trade():
    global _trades_abiertos
    with _lock:
        _trades_abiertos += 1


def restar_trade():
    global _trades_abiertos
    with _lock:
        _trades_abiertos -= 1


def _reset_dia_si_cambia():
    # Debe llamarse con _lock tomado.
    hoy = datetime.now(timezone.utc).date().isoformat()
    if _riesgo["fecha"] != hoy:
        _riesgo["fecha"] = hoy
        _riesgo["pnl_dia"] = 0.0
        _riesgo["ops"] = []


def perdida_diaria_excedida():
    max_perd = CFG.get("riesgo", {}).get("max_perdida_diaria")
    if not max_perd:
        return False
    with _lock:
        _reset_dia_si_cambia()
        return -_riesgo["pnl_dia"] >= max_perd


def ops_hora_excedidas():
    max_ops = CFG.get("riesgo", {}).get("max_operaciones_hora")
    if not max_ops:
        return False
    ahora = time.time()
    with _lock:
        _reset_dia_si_cambia()
        _riesgo["ops"] = [t for t in _riesgo["ops"] if ahora - t < 3600]
        return len(_riesgo["ops"]) >= max_ops


def registrar_apertura():
    with _lock:
        _reset_dia_si_cambia()
        _riesgo["ops"].append(time.time())


def registrar_resultado(profit):
    with _lock:
        _reset_dia_si_cambia()
        _riesgo["pnl_dia"] += profit


def _profit_key(par):
    return par if "-OTC" in par else f"{par}-op"


def _instrumento(expiry):
    return "turbo" if expiry <= 5 else "binary"


def verificar_conexion(api):
    global _conectado, _ultima_ping
    ahora = time.time()
    if ahora - _ultima_ping < 30:
        return True
    _ultima_ping = ahora
    try:
        api.get_balance()
        _conectado = True
        return True
    except Exception:
        pass
    if _conectado:
        log("[RECONNECT] WebSocket caido, reconectando...")
        _conectado = False
    for intento in range(5):
        try:
            ok, reason = api.connect()
            if ok:
                api.change_balance(_balance_mode)
                log(f"[RECONNECT] Reconectado OK (intento {intento + 1})")
                try:
                    api.get_ALL_Binary_ACTIVES_OPCODE()
                except Exception:
                    pass
                _conectado = True
                return True
            log(f"[RECONNECT] Intento {intento + 1} fallo: {reason}")
        except Exception as e:
            log(f"[RECONNECT] Intento {intento + 1} excepcion: {e}")
        time.sleep(3 * (intento + 1))
    log("[RECONNECT] No se pudo reconectar tras 5 intentos")
    return False


def obtener_activos_binarios(api):
    configurados = CFG.get("pares_binarios", [])
    op = CFG.get("operacion", {})
    # solo_par: si esta definido, el bot opera UNICAMENTE ese activo (no destruye la lista)
    solo = (op.get("solo_par") or "").strip()
    if solo:
        configurados = [solo]
    # excluir_otc: opera solo los pares regulares (reales), descarta los -OTC
    if op.get("excluir_otc"):
        configurados = [p for p in configurados if "-OTC" not in p]
    if not configurados:
        return []
    try:
        profits = api.get_all_profit()
    except Exception:
        return []
    activos = []
    saltados = []
    from iqoptionapi.api import OP_code as _OP
    for par in configurados:
        if par not in _OP.ACTIVES:
            saltados.append(par)
            continue
        key = _profit_key(par)
        info = profits.get(key, {})
        payout = None
        if isinstance(info, dict):
            payout = info.get(_instrumento(CFG["operacion"]["expiry_min"])) or info.get("binary")
        if payout is None or payout <= 0:
            continue
        activos.append((par, payout))
    if saltados:
        log(f"Saltados: {', '.join(saltados)}")
    activos.sort(key=lambda x: -x[1])
    return activos


def atr_pct(highs, lows, closes, period):
    """ATR (Wilder) de las ultimas `period` velas, normalizado por precio (fraccion).
    Devuelve None si no hay suficientes velas."""
    n = len(closes)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr = sum(trs[-period:]) / period
    precio = abs(closes[-1]) or 1.0
    return atr / precio


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


# Pesos OPTIMIZADOS (regresion logistica sobre 15 indicadores, 6 meses de 5m, train/test OOS).
# Orden: macd_pos, macd_mom, ema50, ema200, ult3, rsi_ext, stoch_ext, stoch_kd, williamsR,
#        cci_ext, di_dir, roc, bb_ext, canal_pos, canal_slope
PESOS_OPT = [-0.02367, -0.01637, -0.02282, -0.00287, -0.01889, 0.01106, 0.0135,
             -0.02191, 0.0135, -0.02011, 0.00447, -0.00347, 0.04264, 0.02051, -0.01226]
INTERCEPT_OPT = -0.01608


def _rma_last(x, p):
    o = float(x[0])
    for i in range(1, len(x)):
        o = (o * (p - 1) + x[i]) / p
    return o


def _votos13(h, l, c):
    """15 votos (+1/-1/0) de la ultima barra, en el orden de PESOS_OPT."""
    c = np.asarray(c, float); h = np.asarray(h, float); l = np.asarray(l, float)
    n = len(c)
    def sg(x):
        return 1.0 if x > 0 else -1.0 if x < 0 else 0.0
    v = []
    ml = ema(c, 6) - ema(c, 13); sig = ema(ml, 5); hist = ml - sig
    v.append(sg(hist[-1]))                                        # macd_pos
    v.append(sg(hist[-1] - hist[-2]))                            # macd_mom
    v.append(1.0 if c[-1] > ema(c, 50)[-1] else -1.0)           # ema50
    v.append(1.0 if c[-1] > ema(c, 200)[-1] else -1.0)         # ema200
    v.append(1.0 if c[-1] > c[-2] > c[-3] else -1.0 if c[-1] < c[-2] < c[-3] else 0.0)  # ult3
    d = np.diff(c); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = up[-14:].mean(); ad = dn[-14:].mean(); rsi = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    v.append(1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.0)    # rsi_ext
    def stochk(i):
        hh = h[i - 13:i + 1].max(); ll = l[i - 13:i + 1].min()
        return 100 * (c[i] - ll) / (hh - ll) if hh > ll else 50.0
    kl = stochk(n - 1); dl = (stochk(n - 1) + stochk(n - 2) + stochk(n - 3)) / 3
    v.append(1.0 if kl < 20 else -1.0 if kl > 80 else 0.0)      # stoch_ext
    v.append(sg(kl - dl))                                        # stoch_kd
    hh = h[-14:].max(); ll = l[-14:].min()
    wr = -100 * (hh - c[-1]) / (hh - ll) if hh > ll else -50.0
    v.append(1.0 if wr < -80 else -1.0 if wr > -20 else 0.0)    # williamsR
    tp = (h + l + c) / 3; sma = tp[-20:].mean(); md = np.abs(tp[-20:] - sma).mean()
    cci = (tp[-1] - sma) / (0.015 * md) if md > 0 else 0.0
    v.append(1.0 if cci < -100 else -1.0 if cci > 100 else 0.0)  # cci_ext
    upm = np.zeros(n); dnm = np.zeros(n); upm[1:] = h[1:] - h[:-1]; dnm[1:] = l[:-1] - l[1:]
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0); ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    tr = np.zeros(n); tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = _rma_last(tr, 14); pdi = _rma_last(pdm, 14) / atr if atr > 0 else 0; ndi = _rma_last(ndm, 14) / atr if atr > 0 else 0
    v.append(sg(pdi - ndi))                                      # di_dir
    v.append(sg(c[-1] / c[-11] - 1) if n > 10 else 0.0)         # roc
    m20 = c[-20:].mean(); s20 = c[-20:].std(); bbz = (c[-1] - m20) / s20 if s20 > 0 else 0.0
    v.append(1.0 if bbz < -2 else -1.0 if bbz > 2 else 0.0)     # bb_ext
    # CANAL DE TENDENCIA DINAMICA (regresion lineal movil N=50)
    NC = 50; xc = np.arange(NC); xb = xc.mean(); den = ((xc - xb) ** 2).sum()
    win = c[-NC:]; slope = float(((xc - xb) / den * win).sum())    # pendiente OLS
    centro = win.mean() + slope * ((NC - 1) / 2.0)                 # linea de regresion al extremo
    sN = win.std(); czr = (c[-1] - centro) / sN if sN > 0 else 0.0
    v.append(1.0 if czr < -1 else -1.0 if czr > 1 else 0.0)       # canal_pos (extremo del canal)
    v.append(sg(slope))                                            # canal_slope (direccion del canal)
    return v, rsi, bbz


def predecir(closes, highs, lows):
    """Predictor con 13 indicadores y PESOS OPTIMIZADOS (logistica).
    Devuelve (lado, confianza, info_txt). confianza = |p-0.5|."""
    c = np.asarray(closes, dtype=float)
    if len(c) < 210:
        return None, 0.0, ""
    votos, rsi, bbz = _votos13(highs, lows, closes)
    z = INTERCEPT_OPT + sum(w * vt for w, vt in zip(PESOS_OPT, votos))
    p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
    conf = abs(p - 0.5)
    lado = "call" if p > 0.5 else "put"
    info = f"p {p:.3f} conf {conf:.3f} | RSI {rsi:.0f} BBz {bbz:+.2f}"
    return lado, conf, info


# ── Estrategia META-LABELING (portada de Deriv) ──────────────────────────────
# Primario bbrev (reversion Bollinger 2sigma) elige el lado; un modelo meta
# (HistGradientBoosting, 32 features) predice P(la senal gana) y filtra. El bot
# opera si P >= bb_ml_threshold. Modelo: models/meta_bbrev_iq.pkl (meta_train_iq.py).
_META_MODEL = None


def _cargar_meta(path):
    global _META_MODEL
    if _META_MODEL is None:
        with open(path, "rb") as f:
            _META_MODEL = pickle.load(f)
    return _META_MODEL


def _mtf_resample(V, f=3):
    """5m -> 15m para las features MTF (agrupa cada f velas)."""
    return [[g[0][0], g[0][1], max(x[2] for x in g), min(x[3] for x in g), g[-1][4]]
            for g in (V[i:i + f] for i in range(0, len(V) - f + 1, f))]


def predecir_meta(velas):
    """bbrev + meta-labeling (estrategia de Deriv). velas = lista de dicts de
    get_candles. Decide sobre la vela CERRADA. Devuelve (lado, P, info)."""
    import ml_features
    op = CFG["operacion"]
    bb_std = op.get("bb_std", 2.0); period = int(op.get("bb_period", 20))
    vc = velas[:-1]                                  # excluir la vela en formacion
    if len(vc) < 110:
        return None, 0.0, ""
    V = [[float(v.get("from", i)), float(v["open"]),
          float(v.get("max", v.get("high", v["close"]))),
          float(v.get("min", v.get("low", v["close"]))), float(v["close"])]
         for i, v in enumerate(vc)]
    closes = [x[4] for x in V]
    w = closes[-period:]; sma = sum(w) / len(w)
    sd = (sum((x - sma) ** 2 for x in w) / len(w)) ** 0.5
    if sd <= 0:
        return None, 0.0, ""
    z = (closes[-1] - sma) / sd
    if z <= -bb_std:
        lado = "call"
    elif z >= bb_std:
        lado = "put"
    else:
        return None, 0.0, f"z {z:+.2f} (sin extremo)"
    try:
        Vmtf = _mtf_resample(V); k = len(Vmtf)
        cmtf = Vmtf[max(0, k - 60):k] if k >= 2 else None
        fv, _ = ml_features.extract_features(V[-100:], velas_mtf=cmtf)
        if len(fv) == 0:
            return None, 0.0, f"z {z:+.2f} (features insuf)"
        model = _cargar_meta(op.get("bb_ml_model", "models/meta_bbrev_iq.pkl"))
        p = float(model.predict_proba(fv.reshape(1, -1))[0, 1])
    except Exception as e:
        return None, 0.0, f"z {z:+.2f} (err meta: {str(e)[:40]})"
    return lado, p, f"bbrev z {z:+.2f} {lado.upper()} | meta P {p:.3f}"


def _parse_result(res, stake, payout):
    win_flag, amount = None, None
    if isinstance(res, (list, tuple)):
        if len(res) >= 1:
            win_flag = res[0]
        if len(res) >= 2:
            amount = res[1]
    else:
        amount = res
    if amount is not None:
        try:
            amount = float(amount)
            return amount > 0, amount
        except (ValueError, TypeError):
            pass
    if win_flag is not None and (win_flag is True or str(win_flag).lower() in ("win", "true")):
        return True, stake * payout
    return False, -stake


def ejecutar_trade(api, par, lado, payout, stake, expiry, vela_id, info_txt=""):
    # El contador (sumar_trade) ya fue incrementado por el hilo principal
    # ANTES de lanzar este thread, para que hay_capacidad() no se pase de max_trades.
    try:
        ok, oid = api.buy(stake, f"{par}-op", lado, expiry)
        if not ok:
            motivo = str(oid)[:60] if oid else "desconocido"
            log(f"[SKIP] {par} {lado.upper()}: {motivo}")
            with _lock:
                _cruces_fallidos.add(f"{par}-{vela_id}")
            return
        log(f"[ENTRADA] {par} {lado.upper()} | payout {payout:.0%} | id={oid} | "
            f"exp {expiry}m | {info_txt}")

        res = api.check_win_v4(oid)
        gano, profit = _parse_result(res, stake, payout)

        registrar_resultado(profit)
        with _lock:
            _sesion["trades"] += 1
            _sesion["pnl"] += profit
            if gano:
                _sesion["wins"] += 1
            tr = _sesion["trades"]
            wr = _sesion["wins"] / tr * 100
            pnl = _sesion["pnl"]

        log(f"[CIERRE] {par} {lado.upper()} {'GANADA' if gano else 'PERDIDA'} | "
            f"profit ${profit:+.2f} | sesion: {tr} ops, WR {wr:.1f}%, PnL ${pnl:+.2f}")
    except Exception as e:
        log(f"[ERROR] {par}: {type(e).__name__}: {str(e)[:60]}")
    finally:
        restar_trade()
        with _lock:
            _activos_ref["abiertos"] = _trades_abiertos


def run(api, activos, dry=False):
    op = CFG["operacion"]
    log(f"=== Bot PREDICTOR-optimizado (min_conf {op.get('min_conf', 0.02)}) | {len(activos)} activos | "
        f"ATR min {op.get('min_atr', 0)} | {_instrumento(op['expiry_min'])} {op['expiry_min']}m | "
        f"stake ${op['stake']} | max {CFG.get('max_trades', 1)} trades | {'DRY-RUN' if dry else 'OPERANDO'} ===")

    filtro = CFG.get("filtro_hora", {})
    if filtro.get("habilitado"):
        horas_cfg = filtro.get("horas_por_par", {})
        offset = filtro.get("timezone_offset", 0)
        log(f"Filtro de hora ACTIVADO (UTC, Chile={offset}h) - {len(horas_cfg)} pares con horarios")
    else:
        log("Filtro de hora DESACTIVADO - opera 24/7 en todos los pares")

    ultimas_velas = {}
    _ultima_limpieza = time.time()
    _ultimo_reload = time.time()

    while True:
        try:
            if CFG.get("riesgo", {}).get("pausado"):
                log("[PAUSADO] Bot pausado via Telegram. Esperando...")
                time.sleep(10)
                continue

            if not dry and perdida_diaria_excedida():
                max_perd = CFG.get("riesgo", {}).get("max_perdida_diaria")
                log(f"[RIESGO] Perdida diaria >= ${max_perd}. Sin nuevas aperturas hoy (UTC). Durmiendo 60s...")
                time.sleep(60)
                continue

            if time.time() - _ultimo_reload > 30:
                try:
                    with open("config.json", encoding="utf-8") as f:
                        nuevo = json.load(f)
                    for k in ("operacion", "max_trades", "filtro_hora", "riesgo"):
                        if k in nuevo:
                            CFG[k] = nuevo[k]
                    _ultimo_reload = time.time()
                except Exception as e:
                    log(f"[RELOAD] Error: {e}")

            if time.time() - _ultima_limpieza > 3600:
                ahora = time.time()
                with _lock:
                    viejos = {k for k in _cruces_fallidos
                              if ahora - int(k.split("-")[-1]) > 3600}
                    _cruces_fallidos.difference_update(viejos)
                _ultima_limpieza = ahora

            if not verificar_conexion(api):
                log("[FATAL] Sin conexion, durmiendo 30s...")
                time.sleep(30)
                continue

            # Una sola consulta de payouts por ciclo (evita 231 llamadas/ciclo).
            try:
                profits_ciclo = api.get_all_profit()
            except Exception:
                profits_ciclo = {}

            for par, payout in activos:
                stake = CFG["operacion"]["stake"]
                expiry = CFG["operacion"]["expiry_min"]
                max_trades = CFG.get("max_trades", 1)

                if not dry and not hay_capacidad():
                    log(f"[LLENO] {max_trades} trades abiertos, esperando...")
                    time.sleep(10)
                    break

                p = profits_ciclo.get(_profit_key(par), {}).get(_instrumento(expiry))
                payout_ok = p is not None and p >= CFG["operacion"]["min_payout"]
                if not payout_ok:
                    continue

                filtro = CFG.get("filtro_hora", {})
                if filtro.get("habilitado") and "-OTC" not in par:
                    hora_utc = datetime.now(timezone.utc).hour
                    horas_par = filtro.get("horas_por_par", {}).get(par)
                    if horas_par is None or hora_utc not in horas_par:
                        continue

                try:
                    op_ = CFG["operacion"]
                    n_velas = 300   # ~5x EMA50 para que converja + margen
                    velas = api.get_candles(par, op_["timeframe_seg"], n_velas, time.time())
                except Exception:
                    continue
                if not velas or len(velas) < 60:
                    continue

                vela_cerrada = int(velas[-2]["from"])
                if ultimas_velas.get(par) == vela_cerrada:
                    continue
                ultimas_velas[par] = vela_cerrada

                closes = [float(v["close"]) for v in velas[:-1]]
                highs = [float(v.get("max", v.get("high", v["close"]))) for v in velas[:-1]]
                lows = [float(v.get("min", v.get("low", v["close"]))) for v in velas[:-1]]
                # ── Estrategia: 'meta' (bbrev + meta-labeling, portada de Deriv)
                #    o 'predictor' (13/15 indicadores logisticos). Config -> operacion.estrategia.
                estrategia = CFG["operacion"].get("estrategia", "meta")
                if estrategia == "meta":
                    lado, score, info_txt = predecir_meta(velas)
                    thr = CFG["operacion"].get("bb_ml_threshold", 0.60)
                    cumple = lado is not None and score >= thr
                    log(f"  {par:8s} | {info_txt} | " + (
                        ("SENAL " + lado.upper()) if cumple else
                        ("descartada meta (P<" + str(thr) + ")" if lado else "sin senal bbrev")))
                else:
                    lado, conf, info_txt = predecir(closes, highs, lows)
                    min_conf = CFG["operacion"].get("min_conf", 0.02)
                    cumple = lado is not None and conf >= min_conf
                    log(f"  {par:8s} | {info_txt} | "
                        f"{('SENAL ' + lado.upper()) if cumple else 'sin senal (conf<' + str(min_conf) + ')'}")
                if not cumple:
                    continue

                # Filtro ATR (volatilidad): solo operar si ATR/precio >= min_atr.
                # Evita entrar en rangos muertos donde el precio no se mueve.
                min_atr = CFG.get("operacion", {}).get("min_atr", 0.0)
                atr_p = CFG.get("operacion", {}).get("atr_period", 14)
                if min_atr and len(closes) > atr_p:
                    a = atr_pct(highs, lows, closes, atr_p)
                    if a is not None:
                        if a < min_atr:
                            log(f"  [FILTRO-ATR] {par} {lado.upper()} descartado "
                                f"(ATR {a:.4%} < min {min_atr:.4%})")
                            continue
                        info_txt = info_txt + f" | ATR {a:.4%}"

                clave = f"{par}-{vela_cerrada}"
                with _lock:
                    if clave in _cruces_fallidos:
                        continue

                if dry:
                    log(f"[DRY] {par} {lado.upper()} | {info_txt} | payout {p:.0%}")
                    continue

                if ops_hora_excedidas():
                    max_ops = CFG.get("riesgo", {}).get("max_operaciones_hora")
                    log(f"[RIESGO] {max_ops} ops/hora alcanzadas. Esperando...")
                    break

                # Reservar cupo ANTES de lanzar el hilo para no pasarse de max_trades.
                sumar_trade()
                registrar_apertura()
                with _lock:
                    _activos_ref["abiertos"] = _trades_abiertos

                t = threading.Thread(
                    target=ejecutar_trade,
                    args=(api, par, lado, p, stake, expiry, vela_cerrada, info_txt),
                    daemon=True,
                )
                t.start()

            time.sleep(POLL)

        except Exception as e:
            log(f"[WARN] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(POLL)


def main():
    global CFG, _balance_mode
    ap = argparse.ArgumentParser(description="Bot REVERSION-Bollinger multi-activo MULTI-HILO.")
    ap.add_argument("--real", action="store_true", help="Cuenta REAL (default: demo)")
    ap.add_argument("--dry", action="store_true", help="No opera, solo loguea senales")
    args = ap.parse_args()

    _balance_mode = "REAL" if args.real else "PRACTICE"

    with open("config.json", encoding="utf-8") as f:
        CFG = json.load(f)

    # Credenciales: variables de entorno tienen prioridad sobre config.json.
    # Permite NO guardar secretos en el repo (config.json esta gitignored/untracked).
    CFG["email"] = os.getenv("IQ_EMAIL") or CFG.get("email")
    CFG["password"] = os.getenv("IQ_PASSWORD") or CFG.get("password")
    tg = CFG.setdefault("telegram", {})
    tg["token"] = os.getenv("TELEGRAM_TOKEN") or tg.get("token")
    tg["chat_id"] = os.getenv("TELEGRAM_CHAT_ID") or tg.get("chat_id")
    if not CFG.get("email") or not CFG.get("password"):
        log("FALTAN CREDENCIALES: define IQ_EMAIL/IQ_PASSWORD o config.json")
        return

    api = IQ_Option(CFG["email"], CFG["password"])
    log("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}")
        return
    log("Conectado. Cambiando balance...")
    api.change_balance(_balance_mode)

    log("Actualizando opcode de activos...")
    done = [False]
    def _update():
        try:
            api.get_ALL_Binary_ACTIVES_OPCODE()
        except Exception:
            pass
        done[0] = True
    t = threading.Thread(target=_update, daemon=True)
    t.start()
    t.join(timeout=45)

    from iqoptionapi.api import OP_code
    for par in CFG.get("pares_binarios", []):
        if par in OP_code.ACTIVES and f"{par}-op" not in OP_code.ACTIVES:
            OP_code.ACTIVES[f"{par}-op"] = OP_code.ACTIVES[par]

    if done[0]:
        log("Opcode actualizado.")
    else:
        log("Opcode timeout, usando lista estatica.")

    if args.real:
        log("MODO REAL - dinero real")

    activos = obtener_activos_binarios(api)
    if not activos:
        log("No se encontraron activos binarios reales.")
        return

    log(f"Activos ({len(activos)}): {', '.join(f'{n}({p*100:.0f}%)' for n, p in activos)}")
    try:
        _sesion["balance_inicial"] = float(api.get_balance())
    except Exception:
        _sesion["balance_inicial"] = None
    log(f"Balance: {_sesion['balance_inicial']}")

    tg_cfg = CFG.get("telegram", {})
    if tg_cfg.get("habilitado") and tg_cfg.get("token") and tg_cfg.get("chat_id"):
        from telegram_commands import TelegramCommanderSimple
        commander = TelegramCommanderSimple(api, CFG, _sesion, _activos_ref)
        t = threading.Thread(target=commander.run,
                             args=(tg_cfg["token"], tg_cfg["chat_id"]),
                             daemon=True)
        t.start()
        log("Telegram bot iniciado.")

    run(api, activos, dry=args.dry)


if __name__ == "__main__":
    main()
