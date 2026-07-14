# main.py - Bot MACD-crossover multi-activo MULTI-HILO en IQ Option.
#
# Estrategia: MACD(12,26,9) en velas 5-min CERRADAS.
#   CALL cuando MACD cruza signal de abajo-arriba.
#   PUT  cuando MACD cruza signal de arriba-abajo.
#   Multi-hilo: analiza todos los activos y abre trades en paralelo.
#   Max trades abiertos configurable en config.json -> "max_trades".
#
#   .venv314\Scripts\python.exe main.py            # DEMO
#   .venv314\Scripts\python.exe main.py --dry      # solo loguea senales
#   .venv314\Scripts\python.exe main.py --real     # CUIDADO
import argparse
import json
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


def ema(c, span):
    c = np.asarray(c, float)
    a = 2.0 / (span + 1)
    out = np.copy(c)
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def macd_last(closes, fast=None, slow=None, sig_p=None):
    if fast is None:
        fast = CFG["macd"]["fast"]
    if slow is None:
        slow = CFG["macd"]["slow"]
    if sig_p is None:
        sig_p = CFG["macd"]["signal"]
    c = np.asarray(closes, dtype=float)
    if len(c) < slow + sig_p + 2:
        return None
    ema_f = ema(c, fast)
    ema_s = ema(c, slow)
    macd_line = ema_f - ema_s
    sig_line = ema(macd_line, sig_p)
    return (macd_line[-1], sig_line[-1], macd_line[-2], sig_line[-2])


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


def ejecutar_trade(api, par, lado, macd, signal, payout, stake, expiry, vela_id):
    sumar_trade()
    with _lock:
        _activos_ref["abiertos"] = _trades_abiertos
    try:
        ok, oid = api.buy(stake, f"{par}-op", lado, expiry)
        if not ok:
            motivo = str(oid)[:60] if oid else "desconocido"
            log(f"[SKIP] {par} {lado.upper()}: {motivo}")
            with _lock:
                _cruces_fallidos.add(f"{par}-{vela_id}")
            return
        log(f"[ENTRADA] {par} {lado.upper()} MACD {macd:.5f} / Sig {signal:.5f} | "
            f"payout {payout:.0%} | id={oid} | exp {expiry}m")

        res = api.check_win_v4(oid)
        gano, profit = _parse_result(res, stake, payout)

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
    log(f"=== MACD Bot | {len(activos)} activos | MACD({CFG['macd']['fast']},{CFG['macd']['slow']},{CFG['macd']['signal']}) | "
        f"{_instrumento(CFG['operacion']['expiry_min'])} {CFG['operacion']['expiry_min']}m | stake ${CFG['operacion']['stake']} | max {CFG.get('max_trades', 1)} trades | {'DRY-RUN' if dry else 'OPERANDO'} ===")

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

            if time.time() - _ultimo_reload > 30:
                try:
                    with open("config.json", encoding="utf-8") as f:
                        nuevo = json.load(f)
                    # Actualizar solo params operacionales (no email/password)
                    for k in ("macd", "operacion", "max_trades", "filtro_hora", "riesgo"):
                        if k in nuevo:
                            CFG[k] = nuevo[k]
                    _ultimo_reload = time.time()
                except Exception:
                    pass

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

            for par, payout in activos:
                stake = CFG["operacion"]["stake"]
                expiry = CFG["operacion"]["expiry_min"]
                max_trades = CFG.get("max_trades", 1)

                if not dry and not hay_capacidad():
                    log(f"[LLENO] {max_trades} trades abiertos, esperando...")
                    time.sleep(10)
                    break

                try:
                    p = api.get_all_profit().get(_profit_key(par), {}).get(_instrumento(expiry))
                except Exception:
                    p = None
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
                    n_velas = max(CFG["macd"]["slow"] + CFG["macd"]["signal"] + 2,
                                  CFG.get("operacion", {}).get("ema_trend", 0) + 5, 120)
                    velas = api.get_candles(par, CFG["operacion"]["timeframe_seg"], n_velas, time.time())
                except Exception:
                    continue
                if not velas or len(velas) < CFG["macd"]["slow"] + CFG["macd"]["signal"] + 2:
                    continue

                vela_cerrada = int(velas[-2]["from"])
                if ultimas_velas.get(par) == vela_cerrada:
                    continue
                ultimas_velas[par] = vela_cerrada

                closes = [float(v["close"]) for v in velas[:-1]]
                r = macd_last(closes)
                if r is None:
                    continue

                macd_val, signal_val, prev_macd, prev_signal = r
                lado = None
                if prev_macd <= prev_signal and macd_val > signal_val:
                    lado = "call"
                elif prev_macd >= prev_signal and macd_val < signal_val:
                    lado = "put"

                diff = macd_val - signal_val
                diff_prev = prev_macd - prev_signal
                cruzo = "CRUCE" if lado else "sin cruce"
                log(f"  {par:8s} | MACD {macd_val:+.6f} / Sig {signal_val:+.6f} | "
                    f"diff {diff:+.6f} | prev {diff_prev:+.6f} | {cruzo}")

                if lado is None:
                    continue

                # Filtro tendencia: solo operar A FAVOR de la EMA
                ema_p = CFG.get("operacion", {}).get("ema_trend", 0)
                if ema_p and len(closes) >= ema_p:
                    ema_t = float(ema(np.asarray(closes, dtype=float), ema_p)[-1])
                    precio = closes[-1]
                    if (lado == "call" and precio <= ema_t) or (lado == "put" and precio >= ema_t):
                        log(f"  [FILTRO-EMA{ema_p}] {par} {lado.upper()} descartado "
                            f"(contra tendencia: precio {precio:.5f} vs EMA {ema_t:.5f})")
                        continue

                clave = f"{par}-{vela_cerrada}"
                with _lock:
                    if clave in _cruces_fallidos:
                        continue

                if dry:
                    log(f"[DRY] {par} {lado.upper()} MACD {macd_val:.5f} / Sig {signal_val:.5f} | payout {p:.0%}")
                    continue

                t = threading.Thread(
                    target=ejecutar_trade,
                    args=(api, par, lado, macd_val, signal_val, p, stake, expiry, vela_cerrada),
                    daemon=True,
                )
                t.start()

            time.sleep(POLL)

        except Exception as e:
            log(f"[WARN] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(POLL)


def main():
    global CFG, _balance_mode
    ap = argparse.ArgumentParser(description="Bot MACD-crossover multi-activo MULTI-HILO.")
    ap.add_argument("--real", action="store_true", help="Cuenta REAL (default: demo)")
    ap.add_argument("--dry", action="store_true", help="No opera, solo loguea senales")
    args = ap.parse_args()

    _balance_mode = "REAL" if args.real else "PRACTICE"

    with open("config.json", encoding="utf-8") as f:
        CFG = json.load(f)
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
