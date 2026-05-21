# rsi_iq_bot.py - Bot enfocado: RSI-reversion sobre USDJPY REAL en IQ Option.
#
# Unico edge validado en TODA la investigacion (Deriv frxUSDJPY + backtest IQ confirman):
#   RSI(14) sobre velas 1-min CERRADAS -> CALL si RSI<35, PUT si RSI>65.
#   Una posicion a la vez. Solo USDJPY (EURUSD/GBPJPY no tienen edge).
#
# Backtest en feed de IQ (8000 velas, des-solapado, robusto 5/6 ventanas):
#   turbo 5m  -> 56.0% WR (payout 83%, BE 54.6%, margen +1.4pt, p=0.30)
#   binary 10m-> 58.8% WR (payout 84%, BE 54.3%, margen +4.5pt, p=0.062) <- EN USO
# OJO: el binary de IQ usa expiraciones a horas fijas; el tiempo real a expiracion puede
# variar (~10-15m). El backtest asumio 10m exactos -> el WR real lo confirma el demo.
#
# Mecanismo IQ (importante): candles usan "USDJPY"; la compra usa "USDJPY-op" y REQUIERE
# api.update_ACTIVES_OPCODE() tras conectar (si no, "asset not found on consts").
#
#   .venv314\Scripts\python.exe rsi_iq_bot.py            # DEMO, opera binary 10m
#   .venv314\Scripts\python.exe rsi_iq_bot.py --dry      # solo loguea señales, no compra
#   .venv314\Scripts\python.exe rsi_iq_bot.py --real     # cuenta REAL (CUIDADO)
import argparse
import json
import time
from datetime import datetime

import numpy as np
from iqoptionapi.stable_api import IQ_Option

ASSET_CANDLES = "USDJPY"      # subyacente real para el RSI
ASSET_BUY = "USDJPY-op"       # opcion turbo a comprar (id se resuelve con update_ACTIVES_OPCODE)
RSI_PERIOD = 14
RSI_LOW, RSI_HIGH = 35, 65
EXPIRY_MIN = 10               # binary (>5m). El backtest da 58.8% a 10m (el mejor margen)
                              # nota: el binary de IQ usa expiraciones a horas fijas, asi que
                              # el tiempo real a expiracion puede variar ~10-15m (el demo lo confirma)
STAKE = 1.0                   # USD por operacion (demo)
MIN_PAYOUT = 0.80            # no operar si el payout cae por debajo
POLL = 5                      # segundos entre chequeos


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open("rsi_iq.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def rsi(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    val = np.nan
    for i in range(period, len(closes)):
        ag = (ag * (period - 1) + gain[i - 1]) / period
        al = (al * (period - 1) + loss[i - 1]) / period
        rs = ag / al if al > 0 else 999
        val = 100 - 100 / (1 + rs)
    return val


def _instrumento(expiry):
    return "turbo" if expiry <= 5 else "binary"


def payout_actual(api, expiry):
    """Payout (fraccion) de USDJPY-op para el instrumento segun expiracion, o None."""
    try:
        return api.get_all_profit().get(ASSET_BUY, {}).get(_instrumento(expiry))
    except Exception:
        return None


def mercado_abierto(api, expiry):
    """True si USDJPY-op admite compras ahora. Las binarias sobre pares REALES en IQ
    solo se ofrecen en horario de mercado (limitado); fuera de el, open=False."""
    try:
        ot = api.get_all_open_time()
        return bool(ot.get(_instrumento(expiry), {}).get(ASSET_BUY, {}).get("open"))
    except Exception:
        return False


def run(api, dry=False, stake=STAKE, expiry=EXPIRY_MIN):
    prod = "turbo" if expiry <= 5 else "binary"
    log(f"=== RSI IQ Bot | {ASSET_BUY} | RSI {RSI_LOW}/{RSI_HIGH} | {prod} {expiry}m | "
        f"stake ${stake} | {'DRY-RUN' if dry else 'OPERANDO'} ===")
    sesion = {"trades": 0, "wins": 0, "pnl": 0.0}
    ultima_vela = 0

    while True:
        try:
            velas = api.get_candles(ASSET_CANDLES, 60, RSI_PERIOD + 40, time.time())
            if not velas or len(velas) < RSI_PERIOD + 2:
                time.sleep(POLL)
                continue

            # Evaluar solo en cada nueva vela CERRADA (la ultima esta en formacion)
            vela_cerrada = int(velas[-2]["from"])
            if vela_cerrada == ultima_vela:
                time.sleep(POLL)
                continue
            ultima_vela = vela_cerrada

            closes = [float(v["close"]) for v in velas[:-1]]   # solo cerradas
            r = rsi(closes, RSI_PERIOD)
            if r is None:
                continue

            lado = None
            if r < RSI_LOW:
                lado = "call"
            elif r > RSI_HIGH:
                lado = "put"
            if lado is None:
                log(f"RSI {r:.1f} (sin señal)")
                continue

            # USDJPY-op solo abre en horario de mercado (binarias reales = ventana limitada)
            if not mercado_abierto(api, expiry):
                log(f"[CERRADO] {lado.upper()} RSI {r:.1f} -> USDJPY-op fuera de horario, sin operar")
                continue

            # Verificar payout vivo
            p = payout_actual(api, expiry)
            if p is None or p < MIN_PAYOUT:
                log(f"[SKIP] {lado.upper()} RSI {r:.1f} -> payout {p} < {MIN_PAYOUT:.0%}")
                continue

            if dry:
                log(f"[DRY] {lado.upper()} RSI {r:.1f} | payout {p:.0%} (no se opera)")
                continue

            # Comprar (turbo). Bloquea hasta cerrar via check_win_v4.
            ok, oid = api.buy(stake, ASSET_BUY, lado, expiry)
            if not ok:
                log(f"[ERROR] buy {lado.upper()}: {oid}")
                time.sleep(POLL)
                continue
            log(f"[ENTRADA] {lado.upper()} RSI {r:.1f} | payout {p:.0%} | id={oid} | exp {expiry}m")

            res = api.check_win_v4(oid)   # espera el cierre
            gano, profit = _parse_result(res, stake, p)
            sesion["trades"] += 1
            sesion["pnl"] += profit
            if gano:
                sesion["wins"] += 1
            wr = sesion["wins"] / sesion["trades"] * 100
            log(f"[CIERRE] {lado.upper()} {'GANADA' if gano else 'PERDIDA'} | profit ${profit:+.2f} | "
                f"sesion: {sesion['trades']} ops, WR {wr:.1f}%, PnL ${sesion['pnl']:+.2f}")

        except Exception as e:
            log(f"[WARN] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(POLL)


def _parse_result(res, stake, payout):
    """check_win_v4 devuelve normalmente (status, profit) o solo profit. Normaliza."""
    win_flag, amount = None, None
    if isinstance(res, (list, tuple)):
        if len(res) >= 1:
            win_flag = res[0]
        if len(res) >= 2:
            amount = res[1]
    else:
        amount = res
    # Determinar gano: por amount si esta, si no por win_flag
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
    global RSI_LOW, RSI_HIGH
    ap = argparse.ArgumentParser(description="Bot RSI-reversion USDJPY en IQ Option.")
    ap.add_argument("--real", action="store_true", help="Cuenta REAL (default: demo)")
    ap.add_argument("--dry", action="store_true", help="No opera, solo loguea señales")
    ap.add_argument("--stake", type=float, default=STAKE)
    ap.add_argument("--expiry", type=int, default=EXPIRY_MIN, help="Minutos de expiracion (<=5 turbo, >5 binary)")
    ap.add_argument("--rsi-low", type=float, default=RSI_LOW, help="Umbral CALL (default 35, validado)")
    ap.add_argument("--rsi-high", type=float, default=RSI_HIGH, help="Umbral PUT (default 65, validado)")
    args = ap.parse_args()

    # Umbrales configurables (aflojar -> mas operaciones pero edge diluido; solo demo)
    RSI_LOW, RSI_HIGH = args.rsi_low, args.rsi_high

    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    api = IQ_Option(cfg["email"], cfg["password"])
    log("Conectando a IQ Option...")
    ok, reason = api.connect()
    if not ok:
        log(f"NO CONECTO: {reason}")
        return
    api.change_balance("REAL" if args.real else "PRACTICE")
    api.update_ACTIVES_OPCODE()   # IMPRESCINDIBLE para poder comprar USDJPY-op
    if args.real:
        log("⚠️  MODO REAL — dinero real")
    log(f"Conectado ({'REAL' if args.real else 'DEMO'}). Balance: {api.get_balance()}")
    run(api, dry=args.dry, stake=args.stake, expiry=args.expiry)


if __name__ == "__main__":
    main()
