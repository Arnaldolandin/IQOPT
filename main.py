# main.py - Bot MACD-crossover sobre TODOS los activos binarios reales de IQ Option.
#
# Estrategia: MACD(12,26,9) en velas 5-min CERRADAS.
#   CALL cuando MACD cruza signal de abajo-arriba.
#   PUT  cuando MACD cruza signal de arriba-abajo.
#   Una posicion a la vez. Scanea todos los activos binarios reales.
#
# Backtest (50 activos, 16k velas, train/test 70/30, walk-forward):
#   Top OOS: GBPCAD 5m MACD(8,17,9) exp 15m -> WR 67.2% p=0.022
#   NIKE, COKE, AMAZON tambien performaron bien.
#
# Mecanica IQ: candles "<par>", compra "<par>-op".
# Requiere api.update_ACTIVES_OPCODE() tras conectar.
#
#   .venv314\Scripts\python.exe main.py            # DEMO
#   .venv314\Scripts\python.exe main.py --dry      # solo loguea senales
#   .venv314\Scripts\python.exe main.py --real     # CUIDADO
import argparse
import json
import time
from datetime import datetime

import numpy as np
from iqoptionapi.stable_api import IQ_Option

CFG = {}  # cargado desde config.json en main()
POLL = 5


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open("rsi_iq.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def obtener_activos_binarios(api):
    """Devuelve lista (subyacente, payout_binary) de los activos del config con payout vivo."""
    configurados = CFG.get("pares_binarios", [])
    if not configurados:
        log("No hay pares_binarios en config.json")
        return []
    try:
        profits = api.get_all_profit()
    except Exception:
        return []
    activos = []
    for par in configurados:
        key = f"{par}-op"
        info = profits.get(key, {})
        payout = None
        if isinstance(info, dict):
            payout = info.get("binary")
        if payout is None or payout <= 0:
            continue
        activos.append((par, payout))
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
    """Devuelve (macd, signal, prev_macd, prev_signal) o None."""
    c = np.asarray(closes, dtype=float)
    if len(c) < slow + sig_p + 2:
        return None
    ema_f = ema(c, fast)
    ema_s = ema(c, slow)
    macd_line = ema_f - ema_s
    sig_line = ema(macd_line, sig_p)
    return (macd_line[-1], sig_line[-1], macd_line[-2], sig_line[-2])


def _instrumento(expiry):
    return "turbo" if expiry <= 5 else "binary"


def run(api, activos, dry=False):
    stake = CFG["operacion"]["stake"]
    expiry = CFG["operacion"]["expiry_min"]
    prod = "turbo" if expiry <= 5 else "binary"
    log(f"=== MACD Bot | {len(activos)} activos | MACD({CFG['macd']['fast']},{CFG['macd']['slow']},{CFG['macd']['signal']}) | "
        f"{prod} {expiry}m | stake ${stake} | {'DRY-RUN' if dry else 'OPERANDO'} ===")

    sesion = {"trades": 0, "wins": 0, "pnl": 0.0}
    ultimas_velas = {}     # par -> timestamp de ultima vela evaluada

    while True:
        try:
            for par, payout in activos:
                # Payout vivo actualizado
                try:
                    p = api.get_all_profit().get(f"{par}-op", {}).get(_instrumento(expiry))
                except Exception:
                    p = None
                payout_ok = p is not None and p >= CFG["operacion"]["min_payout"]
                if not payout_ok:
                    continue

                # Descargar velas 5m
                try:
                    velas = api.get_candles(par, CFG["operacion"]["timeframe_seg"], 60, time.time())
                except Exception:
                    continue
                if not velas or len(velas) < CFG["macd"]["slow"] + CFG["macd"]["signal"] + 2:
                    continue

                # Solo evaluar en vela CERRADA nueva
                vela_cerrada = int(velas[-2]["from"])
                if ultimas_velas.get(par) == vela_cerrada:
                    continue
                ultimas_velas[par] = vela_cerrada

                closes = [float(v["close"]) for v in velas[:-1]]
                r = macd_last(closes)
                if r is None:
                    continue

                macd, signal, prev_macd, prev_signal = r
                lado = None
                if prev_macd <= prev_signal and macd > signal:
                    lado = "call"
                elif prev_macd >= prev_signal and macd < signal:
                    lado = "put"
                if lado is None:
                    continue

                if dry:
                    log(f"[DRY] {par} {lado.upper()} MACD {macd:.5f} / Sig {signal:.5f} | payout {p:.0%}")
                    continue

                # Comprar. check_win_v4 bloquea hasta cerrar = una posicion a la vez.
                asset_buy = f"{par}-op"
                ok, oid = api.buy(stake, asset_buy, lado, expiry)
                if not ok:
                    log(f"[ERROR] {par} buy {lado.upper()}: {oid}")
                    continue
                log(f"[ENTRADA] {par} {lado.upper()} MACD {macd:.5f} / Sig {signal:.5f} | "
                    f"payout {p:.0%} | id={oid} | exp {expiry}m")

                res = api.check_win_v4(oid)
                gano, profit = _parse_result(res, stake, p)
                sesion["trades"] += 1
                sesion["pnl"] += profit
                if gano:
                    sesion["wins"] += 1
                wr = sesion["wins"] / sesion["trades"] * 100
                log(f"[CIERRE] {par} {lado.upper()} {'GANADA' if gano else 'PERDIDA'} | "
                    f"profit ${profit:+.2f} | sesion: {sesion['trades']} ops, WR {wr:.1f}%, "
                    f"PnL ${sesion['pnl']:+.2f}")

            time.sleep(POLL)

        except Exception as e:
            log(f"[WARN] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(POLL)


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


def main():
    global CFG
    ap = argparse.ArgumentParser(description="Bot MACD-crossover multi-activo en IQ Option.")
    ap.add_argument("--real", action="store_true", help="Cuenta REAL (default: demo)")
    ap.add_argument("--dry", action="store_true", help="No opera, solo loguea senales")
    args = ap.parse_args()

    with open("config.json", encoding="utf-8") as f:
        CFG = json.load(f)
    api = IQ_Option(CFG["email"], CFG["password"])
    log("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}")
        return
    api.change_balance("REAL" if args.real else "PRACTICE")
    api.update_ACTIVES_OPCODE()
    if args.real:
        log("MODO REAL — dinero real")

    activos = obtener_activos_binarios(api)
    if not activos:
        log("No se encontraron activos binarios reales.")
        return

    log(f"Activos ({len(activos)}): {', '.join(f'{n}({p*100:.0f}%)' for n, p in activos)}")
    log(f"Balance: {api.get_balance()}")

    run(api, activos, dry=args.dry)


if __name__ == "__main__":
    main()
