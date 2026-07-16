# main.py - Bot REVERSION-Bollinger multi-activo MULTI-HILO en IQ Option.
#
# Estrategia: reversion con Bandas de Bollinger sobre velas cerradas.
#   CALL cuando el precio cruza por DEBAJO de la banda inferior (rebote).
#   PUT  cuando el precio cruza por ENCIMA de la banda superior.
#   Filtro ATR opcional (volatilidad minima). Multi-hilo, hasta max_trades a la vez.
#   Todo configurable en config.json -> "operacion" (bb_period, bb_k, min_atr...).
#
#   .venv314\Scripts\python.exe main.py            # DEMO
#   .venv314\Scripts\python.exe main.py --dry      # solo loguea senales
#   .venv314\Scripts\python.exe main.py --real     # CUIDADO
import argparse
import json
import os
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
_sesion = {"trades": 0, "wins": 0, "pnl": 0.0}
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


def bb_last(closes, period, k):
    """Bandas de Bollinger: devuelve (close, close_prev, low, low_prev, up, up_prev) o None."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if n < period + 1:
        return None

    def banda(idx):
        w = c[idx - period + 1:idx + 1]
        m = w.mean()
        s = w.std()
        return m - k * s, m + k * s

    lo1, up1 = banda(n - 1)
    lo0, up0 = banda(n - 2)
    return c[-1], c[-2], lo1, lo0, up1, up0


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_last(closes, fast, slow, sig_p):
    """MACD crossover puro: (macd_now, sig_now, macd_prev, sig_prev) o None."""
    c = np.asarray(closes, dtype=float)
    if len(c) < slow + sig_p + 2:
        return None
    macd_line = ema(c, fast) - ema(c, slow)
    sig_line = ema(macd_line, sig_p)
    return macd_line[-1], sig_line[-1], macd_line[-2], sig_line[-2]


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
    est = op.get("estrategia", "bb_rev")
    if est == "macd":
        senal = f"MACD-cruce({op.get('macd_fast',6)},{op.get('macd_slow',13)},{op.get('macd_signal',5)})"
    else:
        senal = f"BB-reversion({op.get('bb_period',20)},{op.get('bb_k',2.0)})"
    log(f"=== Bot {senal} | {len(activos)} activos | ATR min {op.get('min_atr',0)} | "
        f"{_instrumento(op['expiry_min'])} {op['expiry_min']}m | stake ${op['stake']} | "
        f"max {CFG.get('max_trades', 1)} trades | {'DRY-RUN' if dry else 'OPERANDO'} ===")

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
                    bb_p = op_.get("bb_period", 20)
                    atr_pp = op_.get("atr_period", 14)
                    # ~5x el periodo del indicador mas lento para que CONVERJA (EMA del MACD y filtro EMA).
                    macd_need = op_.get("macd_slow", 13) * 5 + op_.get("macd_signal", 5)
                    ema_need = op_.get("macd_ema", 0) * 5
                    n_velas = min(1000, max(bb_p + 5, atr_pp + 5, macd_need, ema_need, 200))
                    velas = api.get_candles(par, op_["timeframe_seg"], n_velas, time.time())
                except Exception:
                    continue
                if not velas or len(velas) < max(bb_p, atr_pp, op_.get("macd_slow", 13) + op_.get("macd_signal", 5)) + 2:
                    continue

                vela_cerrada = int(velas[-2]["from"])
                if ultimas_velas.get(par) == vela_cerrada:
                    continue
                ultimas_velas[par] = vela_cerrada

                closes = [float(v["close"]) for v in velas[:-1]]
                highs = [float(v.get("max", v.get("high", v["close"]))) for v in velas[:-1]]
                lows = [float(v.get("min", v.get("low", v["close"]))) for v in velas[:-1]]
                # ── Generacion de senal segun estrategia ──────────────────────
                # bb_rev = reversion Bollinger (CALL bajo banda inf, PUT sobre sup)
                # macd   = cruce MACD; con filtro EMA (macd_ema): solo a favor de la EMA
                estrategia = CFG["operacion"].get("estrategia", "bb_rev")
                lado = None
                info_txt = ""

                if estrategia == "macd":
                    r = macd_last(closes,
                                  CFG["operacion"].get("macd_fast", 6),
                                  CFG["operacion"].get("macd_slow", 13),
                                  CFG["operacion"].get("macd_signal", 5))
                    if r is None:
                        continue
                    m_now, s_now, m_prev, s_prev = r
                    if m_prev <= s_prev and m_now > s_now:
                        lado = "call"
                    elif m_prev >= s_prev and m_now < s_now:
                        lado = "put"
                    info_txt = f"MACD {m_now:+.6f}/{s_now:+.6f}"
                    log(f"  {par:8s} | MACD {m_now:+.6f} / Sig {s_now:+.6f} | "
                        f"{'CRUCE ' + lado.upper() if lado else 'sin cruce'}")

                    # Filtro EMA de tendencia: solo operar A FAVOR de la EMA.
                    ema_p = CFG["operacion"].get("macd_ema", 0)
                    if lado and ema_p and len(closes) >= ema_p:
                        ema_v = float(ema(np.asarray(closes, dtype=float), ema_p)[-1])
                        precio = closes[-1]
                        if (lado == "call" and precio <= ema_v) or (lado == "put" and precio >= ema_v):
                            log(f"  [FILTRO-EMA{ema_p}] {par} {lado.upper()} descartado "
                                f"(contra tendencia: precio {precio:.5f} vs EMA {ema_v:.5f})")
                            continue
                        info_txt += f" | EMA{ema_p} {ema_v:.5f}"
                else:  # bb_rev
                    bp = CFG["operacion"].get("bb_period", 20)
                    bk = CFG["operacion"].get("bb_k", 2.0)
                    bb = bb_last(closes, bp, bk)
                    if bb is None:
                        continue
                    cn, cp, lo1, lo0, up1, up0 = bb
                    if cp >= lo0 and cn < lo1:
                        lado = "call"
                    elif cp <= up0 and cn > up1:
                        lado = "put"
                    info_txt = f"px {cn:.5f} BB[{lo1:.5f},{up1:.5f}]"
                    log(f"  {par:8s} | {info_txt} | {'SENAL ' + lado.upper() if lado else 'sin senal'}")

                if lado is None:
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
    log(f"Balance: {api.get_balance()}")

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
