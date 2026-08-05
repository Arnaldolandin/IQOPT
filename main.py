# main.py - Bot SECUENCIAL (LSTM / Transformer) multi-activo MULTI-HILO en IQ Option.
#
# Estrategia UNICA: 'seq'. Un modelo secuencial mira las ultimas L velas cerradas y
# devuelve P(sube) a horizonte 2 velas (expiry 10m, binary). Regla simetrica:
#   CALL si P >= seq_threshold ; PUT si P <= 1 - seq_threshold ; nada en el medio.
# No hay primario (bbrev/stoch) ni indicadores: el modelo decide solo.
#   Filtro ATR opcional (volatilidad minima). Multi-hilo, hasta max_trades a la vez.
#   El modelo y su ventana viven en seq_model.py; se entrena con train_seq_save.py.
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


# ── Control de correlacion ────────────────────────────────────────────────
# max_trades cuenta POSICIONES, no apuestas independientes, y esa diferencia se pago en
# vivo: el 2026-08-04, sobre 98 operaciones, hubo 55 parejas simultaneas compartiendo
# divisa y un grupo de 8 posiciones liquidando en el MISMO tick de 15 min. AUDNZD PUT +
# EURNZD PUT + GBPNZD PUT no son tres apuestas: las tres son "el NZD sube", con el stake
# al triple. Si falla, fallan las tres.
#
# Aqui cada operacion se descompone en exposicion CON SIGNO: CALL de XXXYYY es +XXX/-YYY,
# PUT al reves. Se limita cuantas posiciones abiertas pueden compartir la MISMA exposicion
# firmada. Apostar +NZD en un par y -NZD en otro no cuenta: son apuestas opuestas, no
# repetidas.
#
# 'max_por_divisa: 0' lo desactiva (comportamiento anterior). Ante cualquier duda deja
# pasar la orden: un fallo aqui nunca debe bloquear la operativa.
_exposicion = {}          # oid -> {"+NZD", "-AUD"}


def _exposicion_de(par, lado):
    """{'+EUR','-USD'} para EURUSD CALL. Para no-forex (ETHUSD, XAUUSD) usa el par entero:
    no tiene sentido descomponer 'ETH'/'USD' como si fueran cruces."""
    base = par.split("-")[0]
    if len(base) == 6 and base.isalpha() and base.isupper():
        a, b = base[:3], base[3:]
    else:
        a, b = base, None
    signo = "+" if lado.lower() == "call" else "-"
    otro = "-" if signo == "+" else "+"
    e = {signo + a}
    if b:
        e.add(otro + b)
    return e


def correlacion_excedida(par, lado):
    """(True, motivo) si abrir esta orden repetiria una apuesta ya viva."""
    try:
        tope = int(CFG.get("operacion", {}).get("max_por_divisa", 0) or 0)
        if tope <= 0:
            return False, ""
        nueva = _exposicion_de(par, lado)
        with _lock:
            vivas = list(_exposicion.values())
        for d in nueva:
            n = sum(1 for v in vivas if d in v)
            if n >= tope:
                return True, f"ya hay {n} posicion(es) abierta(s) apostando {d}"
        return False, ""
    except Exception:
        return False, ""       # nunca bloquear por un fallo del propio control


def sumar_trade(oid=None, par=None, lado=None):
    global _trades_abiertos
    with _lock:
        _trades_abiertos += 1
        if oid is not None and par and lado:
            _exposicion[oid] = _exposicion_de(par, lado)


def restar_trade(oid=None):
    global _trades_abiertos
    with _lock:
        _trades_abiertos -= 1
        if oid is not None:
            _exposicion.pop(oid, None)


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


# ── Persistencia de operaciones en vuelo ──────────────────────────────────
# Cuando un proceso muere con una orden abierta (os._exit(3), watchdog, WS caido
# con check_win_v2 colgado), el resultado se pierde para el log aunque IQ si
# resuelve la orden (el balance se mueve). Medido 2026-08-01: entrada ETHUSD
# 12:15 id=14126336632, balance 10013.21->10012.21, y NINGUN [CIERRE] en el log.
# Al entrar se persiste el oid; al arrancar se reclama su resultado.
_PENDIENTES_FILE = "operaciones_pendientes.json"


def _cargar_pendientes():
    try:
        with open(_PENDIENTES_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _guardar_pendientes(d):
    try:
        tmp = _PENDIENTES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, _PENDIENTES_FILE)
    except Exception:
        pass


def persistir_apertura(oid, par, lado, stake, payout, vela_id):
    with _lock:
        d = _cargar_pendientes()
        d[str(oid)] = {
            "par": par, "lado": lado, "stake": stake, "payout": payout,
            "vela_id": vela_id, "ts": time.time(),
        }
        _guardar_pendientes(d)


def quitar_pendiente(oid):
    with _lock:
        d = _cargar_pendientes()
        if d.pop(str(oid), None) is not None:
            _guardar_pendientes(d)


def _consultar_resultado_seguro(api, oid, timeout=12.0):
    """Consulta el resultado de una orden por el canal crudo (api.api.get_betinfo),
    SIN el bucle de reconexion de check_win_v2/get_betinfo de la libreria.

    La get_betinfo() de stable_api hace `while True:` y, si IQ no responde en ~10s,
    llama `self.connect()` INTERNAMENTE (stable_api.py:716-721) sin timeout externo.
    Ese connect() pisaba la conexion recien creada y la dejaba rota. Medido el
    2026-08-01: cada arranque con una orden pendiente vieja terminaba en "WebSocket
    caido" a los ~38s y en bucle infinito de reinicios (13:21 -> 21:00 sin parar).

    Devuelve ("listo", win, profit) si IQ cerro la orden, ("en_vuelo", None, None) si
    respondio pero aun no cierra, o None si no respondio en `timeout` (orden vieja ya
    resuelta: el balance lo refleja)."""
    gb = api.api.game_betinfo
    gb.isSuccessful = None
    api.api.get_betinfo(oid)
    t0 = time.time()
    while gb.isSuccessful is None and time.time() - t0 < timeout:
        time.sleep(0.2)
    if not gb.isSuccessful:
        return None
    try:
        data = gb.dict["result"]["data"][str(oid)]
        win = data["win"]
        if win == "" or win is None:
            return ("en_vuelo", None, None)
        profit = float(data["profit"]) - float(data["deposit"])
        return ("listo", win, profit)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def recuperar_pendientes(api):
    """Reclama el resultado de operaciones que quedaron en vuelo cuando un proceso
    murio (os._exit(3), watchdog, check_win_v2 colgado con el WS caido). Consulta UNA
    vez con el metodo crudo (sin el connect() interno de la libreria, que rompia la
    conexion y entraba en bucle de reinicios). Si IQ no responde, la orden es vieja y
    ya esta resuelta (el balance lo refleja): se descarta para no reintentarla en cada
    arranque."""
    with _lock:
        d = _cargar_pendientes()
    if not d:
        return
    log(f"[RECUP] {len(d)} operacion(es) pendiente(s) de un proceso anterior; "
        "consultando resultados...")
    for oid_s, info in list(d.items()):
        oid = int(oid_s)
        estado = _consultar_resultado_seguro(api, oid)
        if estado is None:
            log(f"[RECUP] {info.get('par')} id={oid}: IQ no responde (orden vieja ya "
                "resuelta; el balance lo refleja). Se descarta.")
            quitar_pendiente(oid)
            continue
        if estado[0] == "en_vuelo":
            log(f"[RECUP] {info.get('par')} id={oid}: orden en vuelo, queda pendiente "
                "para el proximo arranque")
            continue
        _, win, profit = estado
        gano = win is True or str(win).lower() in ("win", "true")
        registrar_resultado(profit)
        with _lock:
            _sesion["trades"] += 1
            _sesion["pnl"] += profit
            if gano:
                _sesion["wins"] += 1
            tr = _sesion["trades"]
            wr = _sesion["wins"] / tr * 100
            pnl = _sesion["pnl"]
        log(f"[CIERRE-RECUP] {info.get('par')} {info.get('lado')} "
            f"{'GANADA' if gano else 'PERDIDA'} | profit ${profit:+.2f} | "
            f"sesion: {tr} ops, WR {wr:.1f}%, PnL ${pnl:+.2f}")
        quitar_pendiente(oid)


def registrar_resultado(profit):
    with _lock:
        _reset_dia_si_cambia()
        _riesgo["pnl_dia"] += profit


def _profit_key(par):
    return par if "-OTC" in par else f"{par}-op"


def _instrumento(expiry):
    return "turbo" if expiry <= 5 else "binary"


_open_time_ok = [True]
_open_time_cache = {"t": 0.0, "datos": {}}


def _open_time_binario(api):
    """Disponibilidad de binary/turbo leyendo SOLO get_all_init_v2().

    NO se usa api.get_all_open_time(). Medido el 2026-07-30: esa funcion lanza tres
    hilos y dos de ellos estan rotos en esta cuenta:

      __get_binary_open  -> get_all_init_v2()                  OK, es lo unico que
                                                               necesitamos (operamos
                                                               binarias)
      __get_digital_open -> get_digital_underlying_list_data() devuelve None ->
                            'TypeError: NoneType is not subscriptable' en un hilo de
                            la libreria, fuera de nuestro try/except
      __get_other_open   -> get_instruments(cfd/forex/crypto)  son los endpoints de
                            MARGEN retirados; ya se comprobo que IQ responde
                            'Invalid contract' a los 159 nombres probados

    Como get_all_open_time() hace join() de los tres, los dos rotos se llevaban por
    delante al bueno: la llamada superaba el timeout y el filtro quedaba desactivado
    para toda la sesion (paso a las 16:59:55 del 2026-07-30). Leyendo el init de
    binarias directamente se evitan ambos.

    Devuelve el mismo formato que espera _mercado_abierto():
        {"binary": {"EURUSD": {"open": True}, ...}, "turbo": {...}}
    """
    out = {"binary": {}, "turbo": {}}
    datos = api.get_all_init_v2()
    if not datos:
        return {}
    for opcion in ("binary", "turbo"):
        bloque = datos.get(opcion)
        if not isinstance(bloque, dict):
            continue
        for activo in bloque.get("actives", {}).values():
            try:
                nombre = str(activo["name"]).split(".")[1]
            except (KeyError, IndexError, TypeError):
                continue
            # 'enabled' False = el activo no existe para esta cuenta;
            # 'is_suspended' True = existe pero ahora no acepta ordenes.
            abierto = bool(activo.get("enabled")) and not activo.get("is_suspended")
            out[opcion][nombre] = {"open": abierto}
    return out


def _open_time_ciclo(api):
    """Disponibilidad de activos con timeout, CACHE y auto-desactivacion.

    Si la consulta falla o se cuelga una vez, se desactiva para toda la sesion y
    volvemos al comportamiento anterior (dejar que IQ rechace en el buy). Nunca
    bloquear la operativa por esto.

    CACHE (2026-07-30): el bucle gira cada POLL=3 s y esta funcion se llamaba en CADA
    vuelta. Consultar el endpoint 20 veces por minuto es absurdo -- un mercado no abre
    y cierra en segundos -- y ademas ANADE latencia al ciclo, que es precisamente lo
    que castiga el WR (inmediatas 59.68% vs demoradas 51.09%). Se cachea
    'open_time_ttl_seg': con 300 s la consulta pasa de ~1200/h a 12/h y el peor caso es
    operar 5 min contra un horario obsoleto, que el propio buy corrige rechazando.

    POR QUE IMPORTA (medido el 2026-07-30 sobre 2.816 intentos del log): el bot pedia
    ordenes sobre los 49 activos hubiera o no mercado y solo ejecutaba el 28%. Los
    1.675 intentos contra activos cerrados no se pierden solos: cada uno gasta 5-6
    reintentos y ~55 s de cola, y las ordenes BUENAS esperan detras hasta caducar. De
    ahi los 361 rechazos por timing, que de 13h a 22h se comen entre el 27% y el 83%
    de lo ejecutable.
    """
    if not CFG.get("operacion", {}).get("usar_open_time", False):
        return {}
    if not _open_time_ok[0]:
        return {}
    ttl = float(CFG.get("operacion", {}).get("open_time_ttl_seg", 300))
    ahora = time.time()
    if _open_time_cache["datos"] and ahora - _open_time_cache["t"] < ttl:
        return _open_time_cache["datos"]
    res = [None]

    def _c():
        try:
            res[0] = _open_time_binario(api)
        except Exception:
            res[0] = None

    t = threading.Thread(target=_c, daemon=True)
    t.start()
    t.join(timeout=35)          # get_all_init_v2 espera hasta 30 s por dentro
    if t.is_alive() or not isinstance(res[0], dict) or not res[0]:
        _open_time_ok[0] = False
        log("[OPEN-TIME] init de binarias fallo o se colgo -> desactivado por esta "
            "sesion; IQ decidira en el buy")
        return {}
    inst = _instrumento(CFG.get("operacion", {}).get("expiry_min", 10))
    n_abiertos = sum(1 for v in res[0].get(inst, {}).values() if v.get("open"))
    # Si sale 0 el filtro bloquearia TODO. Antes que dejar el bot mudo una sesion
    # entera, se descarta el dato y se opera como siempre.
    if n_abiertos == 0:
        _open_time_ok[0] = False
        log(f"[OPEN-TIME] 0 activos abiertos en '{inst}' -> dato sospechoso, "
            "filtro desactivado por esta sesion")
        return {}
    _open_time_cache["t"] = ahora
    _open_time_cache["datos"] = res[0]
    log(f"[OPEN-TIME] {n_abiertos} activos abiertos en '{inst}' (cache {ttl:.0f}s)")
    return res[0]


def minutos_al_vencimiento(expiry_min, grilla_min):
    """Minutos reales hasta que liquidara la opcion si se compra AHORA.

    Las binarias de IQ no vencen 'expiry_min' despues de la compra: vencen en marcas
    fijas de reloj (grilla de 15 min: :00, :15, :30, :45). Comprando a las 20:00 con
    expiry 10 la opcion liquida 20:15 -> el horizonte REAL es 15 min, no 10.

    Medido en vivo: dos entradas a las 20:00:01 y 20:05:05 liquidaron AMBAS a las
    20:15, o sea sobre el mismo tick: no eran dos operaciones sino una con doble
    stake.

    'expiry_min' NO interviene en el calculo, y esto es una correccion del 2026-07-31.
    Antes esta funcion saltaba a la marca siguiente mientras la primera estuviera a
    menos de expiry_min ("IQ asigna la primera marca que este AL MENOS a esa
    distancia"). Es falso: IQ asigna la marca siguiente y punto, sin respetar el
    minimo pedido. Medido sobre 395 entradas emparejadas con su cierre, agrupadas por
    el minuto de cierre de vela en que se decidio (mod 15), pidiendo SIEMPRE 10 min:

        mod 15 == 0  -> n=108  duracion mediana 14.57 min
        mod 15 == 5  -> n=228  duracion mediana  9.01 min   <- el horizonte entrenado
        mod 15 == 10 -> n= 59  duracion mediana  4.80 min

    Con la formula vieja, pedir 10 en el bucket de :05 daba 24.9 (la marca de :15 esta
    a 9.9, "menos de 10", asi que saltaba a :30). La realidad son 9.01. O sea que la
    formula erraba por 15 minutos justo en el unico bucket que sirve, y activar
    'alinear_expiry' bloqueaba precisamente lo que habia que dejar pasar -- de ahi las
    "9 señales sin ejecutar" que menciona expiry_alineado().

    Se conserva el parametro por compatibilidad de firma; el payout SI depende de lo
    que se pide (>5 -> binary 87%), asi que expiry_min sigue mandando en el buy.
    """
    ahora = datetime.now(timezone.utc)
    m = ahora.minute + ahora.second / 60.0
    prox = (int(m // grilla_min) + 1) * grilla_min
    return prox - m


def horizonte_modelo_min(par):
    """Horizonte (minutos) al que el modelo del par predice.

    Lee 'meta.H' (velas de 5m) del .pt.json del modelo: es la fuente de verdad, no la
    config. Si un modelo se entrena a H=3 predice a 15 min, y comparar su senal contra
    el horizonte_modelo_min global (10) la descartaria siempre. El global queda solo de
    respaldo para pares sin modelo legible.
    """
    try:
        path = modelo_de(par)
        if path:
            with open(path + ".json", encoding="utf-8") as f:
                cfg = json.load(f)
            h = cfg.get("meta", {}).get("H")
            if h:
                return float(h) * 5.0
    except Exception:
        pass
    return float(CFG.get("operacion", {}).get("horizonte_modelo_min", 10))


def expiry_alineado(op, par=None):
    """True si conviene operar AHORA: el horizonte real coincide con el ENTRENADO.

    Hay DOS numeros distintos y confundirlos costo 9 señales sin ejecutar:

      horizonte_modelo_min : lo que el modelo predice (2 velas de 5m = 10 min). Es lo
                             que hay que hacer coincidir con la duracion real.
      expiry_min           : lo que se le PIDE a IQ. IQ asigna la primera marca de la
                             grilla que este AL MENOS a esa distancia.

    Si se piden 10 min, comprando a las :20:05 faltan 9.92 para la marca de :30 -- por
    apenas 5 segundos no alcanza el minimo, IQ salta a :45 y la opcion dura 24.9 min.
    La ventana buena tenia ancho CERO: habia que comprar en el segundo exacto.

    Pidiendo menos (7 min) el margen absorbe la latencia: a las :20:05 los 9.92 minutos
    hasta :30 superan el minimo de 7, IQ asigna :30 y la duracion real es ~10, que es
    lo que el modelo predice.
    """
    if not op.get("alinear_expiry", True):
        return True, 0.0
    grilla = float(op.get("expiry_grilla_min", 15))
    tol = float(op.get("expiry_tolerancia_min", 1.5))
    pedido = float(op.get("expiry_min", 7))
    objetivo = horizonte_modelo_min(par) if par else float(op.get("horizonte_modelo_min", 10))
    real = minutos_al_vencimiento(pedido, grilla)
    return abs(real - objetivo) <= tol, real


def _mercado_abierto(abiertos, par, expiry):
    """True si el activo acepta ordenes ahora, segun _open_time_ciclo().

    Las claves son el nombre pelado ("EURUSD", "AIG-OTC"), sin el "-op" de
    get_all_profit(). Si la consulta fallo o el activo no aparece, devolvemos True:
    ante la duda dejamos que IQ decida en el buy (comportamiento anterior) en vez de
    bloquear operativa por un fallo de la API.
    """
    if not abiertos:
        return True
    info = abiertos.get(_instrumento(expiry), {}).get(par)
    if not isinstance(info, dict) or "open" not in info:
        return True
    return bool(info["open"])


def _llamar_timeout(fn, timeout, default=None):
    """Ejecuta fn() en un hilo y devuelve (resultado, exito). exito=True SOLO si fn
    retorno limpio dentro de 'timeout' seg; False si se colgo (timeout) O lanzo excepcion.

    La API de iqoptionapi puede BLOQUEAR indefinidamente cuando el WebSocket muere (no
    tiene timeout interno, no lanza excepcion, solo espera): sin esto una llamada colgada
    congela el bucle principal PARA SIEMPRE. Paso el 2026-07-24: WS caido ('Connection is
    already closed') -> get_candles colgado -> bot frozen 6.5 h sin operar, solo el hilo
    de Telegram girando. El worker queda como daemon: si de verdad esta colgado, no
    bloquea el cierre del proceso."""
    caja = {}
    def _run():
        try:
            caja["r"] = fn()
        except Exception:
            caja["e"] = True
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or "e" in caja:
        return default, False
    return caja.get("r", default), True


def verificar_conexion(api):
    global _conectado, _ultima_ping
    ahora = time.time()
    if ahora - _ultima_ping < 30:
        return True
    _ultima_ping = ahora
    # get_balance CON timeout: si el WS murio, la llamada cuelga y sin esto
    # verificar_conexion nunca retornaria -> el bot no detectaria la caida.
    _, exito = _llamar_timeout(api.get_balance, 15)
    if exito:
        _conectado = True
        return True
    if _conectado:
        log("[RECONNECT] WebSocket caido, reconectando...")
        _conectado = False

    # connect() SIN timeout era el ultimo agujero de esta clase, y justo en la ruta que
    # se activa cuando el WS ya esta muerto. El 2026-07-28 el WS cayo a las 14:56, la
    # llamada se quedo dentro y el bot no volvio hasta que alguien lo relanzo A MANO 2h23
    # despues: el log no tiene ni "Reconectado OK" ni "no se pudo reconectar", o sea que
    # ni siquiera completo el primer intento. _llamar_timeout se escribio para esto y la
    # reconexion se habia quedado fuera.
    #
    # 45 s por intento + backoff = 315 s en el peor caso, holgadamente por debajo de los
    # MAX_SILENCIO=600 s del watchdog: una tanda completa de reintentos NO debe parecerle
    # un bucle congelado. Si se suben estos numeros, subir tambien MAX_SILENCIO.
    def _intento_connect():
        # connect() puede REVENTAR en vez de devolver (False, motivo): stable_api hace
        # json.loads() del motivo y con un fallo de red el motivo no es JSON. Se captura
        # aqui dentro para no perder el texto (_llamar_timeout solo diria "fallo").
        try:
            return api.connect()
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:80]}"

    for intento in range(5):
        res, exito = _llamar_timeout(_intento_connect, 45, (False, ""))
        if not exito:
            log(f"[RECONNECT] Intento {intento + 1} colgado (>45s), abandonado")
        else:
            ok, reason = res if isinstance(res, (list, tuple)) else (False, str(res))
            if ok:
                # tambien con timeout: si el WS vuelve a caerse a mitad, estas dos
                # colgarian el bucle igual que colgaba connect()
                _llamar_timeout(lambda: api.change_balance(_balance_mode), 15)
                log(f"[RECONNECT] Reconectado OK (intento {intento + 1})")
                _llamar_timeout(api.get_ALL_Binary_ACTIVES_OPCODE, 30)
                _conectado = True
                return True
            log(f"[RECONNECT] Intento {intento + 1} fallo: {reason}")
        time.sleep(3 * (intento + 1))
    log("[RECONNECT] No se pudo reconectar tras 5 intentos")
    return False


def obtener_activos_binarios(api):
    configurados = CFG.get("pares_binarios", [])
    op = CFG.get("operacion", {})
    # Con 'modelos_por_par' la lista efectiva son los pares QUE TIENEN MODELO. Escanear
    # los demas es gasto puro: modelo_de() les devuelve None y nunca se operan, pero
    # igual se les pediria payout y velas, ensuciando el log y las llamadas a la API.
    mapa = op.get("modelos_por_par") or {}
    if mapa:
        configurados = [p for p in configurados if p in mapa]
        faltan = [p for p in mapa if p not in configurados]
        if faltan:
            log(f"[AVISO] con modelo pero fuera de pares_binarios: {', '.join(faltan)}")
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


def adx(highs, lows, closes, period=14):
    """ADX (Wilder) de la ultima vela: fuerza de tendencia (0-100). Bajo (~<20) = rango,
    alto = tendencia fuerte. Devuelve None si faltan velas.

    Filtro de REGIMEN: el modelo es de reversion (apuesta contra la tendencia), y la
    reversion falla cuando hay tendencia fuerte (el precio sigue en vez de revertir).
    Medido 2026-07-28: operar solo con ADX bajo sube el WR de 52.55% a 53.29% (roza BE).
    """
    n = len(closes)
    if n < 2 * period + 1:
        return None
    pdm, mdm, tr = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))

    def suav(x, p):                       # suavizado de Wilder: primer valor = suma inicial
        s = [sum(x[:p])]
        for i in range(p, len(x)):
            s.append(s[-1] - s[-1] / p + x[i])
        return s

    atr_s, pdm_s, mdm_s = suav(tr, period), suav(pdm, period), suav(mdm, period)
    dx = []
    for a, pd, md in zip(atr_s, pdm_s, mdm_s):
        a = a if a > 0 else 1e-12
        pdi, mdi = 100 * pd / a, 100 * md / a
        suma = pdi + mdi
        dx.append(100 * abs(pdi - mdi) / (suma if suma > 0 else 1e-12))
    if len(dx) < period:
        return None
    adx_val = sum(dx[:period]) / period   # ADX = Wilder smooth de DX
    for x in dx[period:]:
        adx_val = (adx_val * (period - 1) + x) / period
    return adx_val


def modelo_de(par):
    """Ruta del modelo que le corresponde a este par.

    Cada par se evalua con SU modelo: uno entrenado con EURUSD no tiene ningun
    respaldo aplicado a BTCUSD, la dinamica es distinta. Si el par no figura en
    'modelos_por_par' se devuelve None y el par NO se opera: es preferible no operar
    a operar con un modelo ajeno.
    """
    op = CFG.get("operacion", {})
    mapa = op.get("modelos_por_par") or {}
    if par in mapa:
        return mapa[par]
    if not mapa:                      # compat: config viejo con un solo modelo
        return op.get("seq_model")
    return None


def umbral_de(par):
    """Umbral del par. Cada modelo tiene su propia escala de confianza: el de BTCUSD
    llega a P=0.64 y dispara en ~50% de las velas a 0.54, mientras el de EURUSD apenas
    pasa de 0.52. Un umbral unico obliga a elegir entre inundar un par u apagar el
    otro. 'umbrales_por_par' manda; si el par no figura, se usa seq_threshold.
    """
    op = CFG.get("operacion", {})
    mapa = op.get("umbrales_por_par") or {}
    return float(mapa.get(par, op.get("seq_threshold", 0.54)))


def predecir_seq(velas, par=None):
    """Estrategia 'seq': modelo secuencial puro, sin primario bbrev/stoch.
    Devuelve (lado, P, info). Regla simetrica sobre P(sube).

    Si al lado del .pt existe el _hgb.pkl (HistGradientBoosting sobre la MISMA ventana
    aplanada), P final = promedio de P(LSTM) y P(HGB). El peso lo da 'hgb_peso'
    (default 0.5). Sin .pkl, queda el LSTM solo (retrocompat).

    OJO: el modelo esta atado a la version de seq_model.ventana_features() con la que
    se entreno. Si se cambia esa funcion hay que reentrenar (train_seq_save.py).
    """
    op = CFG["operacion"]
    path = modelo_de(par) if par else op.get("seq_model", "models/seq_lstm_EURUSD.pt")
    if not path:
        return None, 0.0, f"(sin modelo para {par}; no se opera)"
    thr = umbral_de(par)
    try:
        import seq_model
        # El volumen es parte del vector con el que se entreno. Si no se pasa,
        # ventana_features rellena esas columnas con CERO y el modelo recibe algo
        # distinto de lo que vio entrenando, devolviendo probabilidades sin sentido
        # y sin ningun error visible. get_candles ya lo trae en cada vela.
        vol = [float(v.get("volume", 0) or 0) for v in velas[:-1]]
        p = seq_model.predecir_p(velas, path, extras={"vol": vol})
        # HGB combinado: si existe el .pkl, promediar. Fallo del HGB (pkl corrupto,
        # sklearn ausente) = queda el LSTM solo, NUNCA tumbar la operativa.
        pkl = path.replace(".pt", "_hgb.pkl")
        if os.path.isfile(pkl):
            try:
                ph = seq_model.predecir_hgb(velas, pkl, extras={"vol": vol})
                if ph is not None:
                    w = float(op.get("hgb_peso", 0.5))
                    p = w * p + (1.0 - w) * ph
            except Exception:
                pass
    except Exception as e:
        return None, 0.0, f"(err seq: {type(e).__name__}: {str(e)[:40]})"
    if p is None:
        return None, 0.0, "(ventana insuficiente/con huecos)"
    if p >= thr:
        return "call", p, f"seq P {p:.3f}"
    if p <= 1 - thr:
        return "put", p, f"seq P {p:.3f}"
    return None, p, f"seq P {p:.3f}"


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


def _esperar_cierre(api, oid, timeout=1000.0):
    """Espera el push 'option-closed' de IQ sin consultar get_betinfo.

    Medido el 2026-08-01: IQ dejo de responder api_game_betinfo y
    check_win_v2 -> get_betinfo -> self.connect() interno (sin timeout)
    mataba el WebSocket ~10-12s tras cada [ENTRADA], con el bot en bucle de
    muerte. El push 'option-closed' (client.py) llega solo mientras la
    conexion esta viva: un buy NO la mata (medido: WS vivo 83s+ tras entrar).
    Devuelve profit neto (profit_amount - amount) o None si el push no llega
    en `timeout` segundos (duracion real maxima de una opcion: 15 min)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ao = api.api.order_async.get(oid)
            if ao and "option-closed" in ao:
                msg = ao["option-closed"]["msg"]
                return float(msg["profit_amount"]) - float(msg["amount"])
        except Exception:
            pass
        time.sleep(2.0)
    return None


def ejecutar_trade(api, par, lado, payout, stake, expiry, vela_id, info_txt=""):
    # El cupo de max_trades se toma AQUI, cuando la orden entra, no al lanzar el hilo:
    # esperar no es operar. 'cupo' recuerda si llegamos a tomarlo, para no soltar en el
    # finally un hueco que nunca ocupamos.
    cupo = False
    try:
        # REINTENTO SOSTENIDO. Medido el 2026-07-22: IQ no acepta ordenes de forma
        # continua, sino en VENTANAS de ~3-4 min que abren unos 6-7 min pasado cada
        # cuarto de hora (:06-:10, :21-:25, :36-:40, :51-:55). Cinco aperturas seguidas
        # lo confirmaron, dos de ellas predichas de antemano.
        #
        # El bot actua al cerrar cada vela de 5m, o sea en +0:05, +5:05 y +10:05 desde
        # la marca del cuarto: NINGUNO cae dentro de la ventana. Por eso fallaba casi
        # todas las compras -- no por el activo ni por el expiry, sino porque sus
        # horarios de escaneo estan desalineados con los del broker por construccion.
        #
        # Antes se rendia en ~2 s y, peor aun, 'not available' cortaba el bucle sin
        # reintentar: justo el mensaje que da una ventana cerrada, el unico caso en que
        # el reintento SI sirve. Ahora se insiste hasta 'reintento_max_seg'.
        #
        # OJO: la senal envejece mientras se reintenta. El modelo predice a 10 min DESDE
        # EL CIERRE DE VELA; entrar 6 min tarde es una apuesta distinta de la calculada.
        op_ = CFG.get("operacion", {})
        max_seg = float(op_.get("reintento_max_seg", 240))
        pausa = float(op_.get("reintento_pausa_seg", 10))
        # 'suspended' = el producto no existe para este activo (BTCUSD): no se reintenta
        PERMANENTES = ("suspended", "not enough money", "balance")

        ok, oid, motivo = False, None, ""
        t_ini = time.time()
        intentos = 0
        ultimo_log = 0.0
        while True:
            intentos += 1
            # el tope de posiciones se comprueba justo antes de comprar: si esta lleno,
            # se sigue esperando en vez de abrir la 11a
            if not hay_capacidad():
                if time.time() - t_ini + pausa > max_seg:
                    motivo = f"max_trades lleno durante toda la espera"
                    break
                time.sleep(pausa)
                continue
            # Correlacion: se comprueba AQUI, junto al buy, no al lanzar el hilo. Durante
            # la espera del reintento pueden haberse abierto otras posiciones sobre la
            # misma divisa, y lo que importa es la foto del momento de comprar.
            corr, por_que = correlacion_excedida(par, lado)
            if corr:
                motivo = f"correlacion: {por_que}"
                break
            try:
                ok, oid = api.buy(stake, f"{par}-op", lado, expiry)
            except Exception as e:
                ok, oid = False, f"excepcion: {str(e)[:60]}"
            if ok:
                sumar_trade(oid, par, lado)
                cupo = True
                registrar_apertura()
                with _lock:
                    _activos_ref["abiertos"] = _trades_abiertos
                break
            motivo = str(oid)[:80] if oid else "buy rechazado (timing/'buy late')"
            bajo = motivo.lower()
            if any(p in bajo for p in PERMANENTES):
                break
            transcurrido = time.time() - t_ini
            if transcurrido + pausa > max_seg:
                break
            # un log cada 60 s, no en cada intento: si no, inunda rsi_iq.log
            if transcurrido - ultimo_log >= 60:
                log(f"[ESPERA] {par} {lado.upper()}: {motivo} "
                    f"({intentos} intentos, {transcurrido:.0f}s de {max_seg:.0f})")
                ultimo_log = transcurrido
            time.sleep(pausa)
        if not ok:
            log(f"[SKIP] {par} {lado.upper()} tras {intentos} intentos en "
                f"{time.time()-t_ini:.0f}s: {motivo}")
            with _lock:
                _cruces_fallidos.add(f"{par}-{vela_id}")
            return
        if intentos > 1:
            log(f"[ENTRO TRAS ESPERAR] {par} {lado.upper()}: {intentos} intentos, "
                f"{time.time()-t_ini:.0f}s")
        log(f"[ENTRADA] {par} {lado.upper()} | payout {payout:.0%} | id={oid} | "
            f"exp {expiry}m | {info_txt}")
        # Persistir la operacion: si este proceso muere antes del cierre (os._exit(3),
        # watchdog, WS caido), el proximo arranque reclama su resultado con
        # recuperar_pendientes() en vez de perder el [CIERRE] para siempre.
        persistir_apertura(oid, par, lado, stake, payout, vela_id)

        # Resolucion del resultado por PUSH 'option-closed', NO por check_win_v2.
        # Medido 2026-08-01: check_win_v2 -> get_betinfo -> self.connect() interno
        # (sin timeout) mataba el WS ~10-12s tras cada ENTRADA (hoy IQ no responde
        # api_game_betinfo), entrando el bot en bucle de muerte con el watchdog
        # relanzandolo cada ~6 min. El push llega solo y no toca la conexion.
        # 'option-closed' viene de client.py; check_win_v3 ya lo usaba, pero en
        # while True sin timeout: aqui con timeout (max duracion real 15 min).
        res = _esperar_cierre(api, oid)
        if res is None:
            log(f"[SIN-CIERRE] {par} {lado.upper()} id={oid}: no llego el push "
                f"option-closed; la persistencia lo reclamara en el proximo arranque")
            return
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
        quitar_pendiente(oid)
    except Exception as e:
        log(f"[ERROR] {par}: {type(e).__name__}: {str(e)[:60]}")
    finally:
        # solo se devuelve el cupo si de verdad se tomo: los hilos que se rindieron
        # esperando nunca lo ocuparon, y restarlo aqui dejaria el contador en negativo.
        # 'oid' puede no existir si se salio antes del buy: sin el, la exposicion de esta
        # orden quedaria colgada para siempre y bloquearia su divisa.
        if cupo:
            restar_trade(locals().get("oid"))
        with _lock:
            _activos_ref["abiertos"] = _trades_abiertos


ESTADO_VELAS = "estado_velas.json"


def cargar_ultimas_velas():
    """Ultima vela ya procesada por activo, PERSISTIDA entre arranques.

    Vivia solo en memoria, y eso duplicaba operaciones: al reiniciar, el diccionario
    arrancaba vacio, el bot veia la ultima vela cerrada como nueva y repetia una señal
    que ya habia comprado. El 2026-07-22, con el vigilante reiniciando cada 3-5 min para
    incorporar modelos, eso convirtio 3 señales en 7 posiciones en veinte minutos --
    EURJPY llego a comprarse tres veces sobre la misma vela, triple stake sobre una
    unica prediccion.
    """
    try:
        with open(ESTADO_VELAS, encoding="utf-8") as f:
            return {k: int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


HEARTBEAT = "heartbeat.json"


def _escribir_heartbeat():
    """Escribe el timestamp del ultimo ciclo del bucle de trading. Lo lee watchdog.py.
    Nunca debe tumbar el bot: si falla el disco, se ignora.

    ATOMICO a proposito. Antes era open(HEARTBEAT, "w"), y "w" TRUNCA el archivo a 0 bytes
    antes de escribir: si el watchdog leia en esa rendija encontraba un archivo vacio,
    json.load reventaba, heartbeat_edad() devolvia None y el watchdog reiniciaba un bot
    que estaba perfectamente vivo. Paso dos veces la noche del 2026-07-28 al 29 (01:11 y
    05:35), con el bot operando con normalidad segun su propio log.

    Se agravo al dejar de rebajar velas ya procesadas: el bucle pasa a girar en vacio
    entre cierres, asi que el heartbeat se escribe cada ~3 s en vez de cada ~47 s, unas
    15 veces mas de oportunidades de colisionar con la lectura del watchdog.

    Es el MISMO tropiezo que ya documenta tomar_cerrojo() en seq_model.py, donde "w"
    truncaba el .lock y dejaba pasar dos cerrojos. os.replace() es atomico en Windows
    dentro del mismo volumen: el watchdog ve el archivo viejo o el nuevo, nunca a medias.
    """
    tmp = HEARTBEAT + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time()}, f)
    except Exception:
        return
    # os.replace REINTENTADO, no a la primera: en Windows no se puede reemplazar un
    # archivo que otro proceso tiene abierto, asi que si el watchdog esta leyendo justo
    # en ese instante salta PermissionError (WinError 5). Medido: con un lector en bucle,
    # a la primera falla. El lector solo lo tiene abierto microsegundos, asi que un par de
    # reintentos bastan.
    for _ in range(3):
        try:
            os.replace(tmp, HEARTBEAT)
            return
        except Exception:
            time.sleep(0.05)
    # Si aun asi no se pudo, da igual: el latido se escribe cada ~3 s y el watchdog
    # tolera 600 s. Perder uno no acerca ni de lejos a un reinicio.


def guardar_ultimas_velas(d):
    # Nunca debe tumbar el ciclo de operativa: si falla el disco, se sigue operando y
    # como mucho se reprocesa una vela tras el proximo reinicio.
    try:
        with open(ESTADO_VELAS, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass


def run(api, activos, dry=False):
    op = CFG["operacion"]
    _mods = op.get("modelos_por_par") or {"(unico)": op.get("seq_model")}
    _det = ", ".join(f"{k}->{os.path.basename(v or '?')}@{umbral_de(k)}"
                     for k, v in _mods.items())
    log(f"=== Bot SEQ ({_det}) | {len(activos)} activos | "
        f"ATR min {op.get('min_atr', 0)} | {_instrumento(op['expiry_min'])} {op['expiry_min']}m | "
        f"stake ${op['stake']} | max {CFG.get('max_trades', 1)} trades | {'DRY-RUN' if dry else 'OPERANDO'} ===")

    filtro = CFG.get("filtro_hora", {})
    if filtro.get("habilitado"):
        horas_cfg = filtro.get("horas_por_par", {})
        offset = filtro.get("timezone_offset", 0)
        log(f"Filtro de hora ACTIVADO (UTC, Chile={offset}h) - {len(horas_cfg)} pares con horarios")
    else:
        log("Filtro de hora DESACTIVADO - opera 24/7 en todos los pares")

    ultimas_velas = cargar_ultimas_velas()
    if ultimas_velas:
        log(f"[ESTADO] {len(ultimas_velas)} activos con vela ya procesada: no se "
            f"repetiran sus señales tras este reinicio")
    _ultima_limpieza = time.time()
    _ultimo_reload = time.time()

    while True:
        try:
            # Latido del BUCLE DE TRADING (no del proceso): el watchdog lo vigila. Si el
            # bucle se congela (p.ej. una llamada de API colgada), el heartbeat envejece
            # aunque el hilo de Telegram siga vivo -> el watchdog reinicia. Detecta el
            # fallo del 2026-07-24, que el mtime del log NO detectaba (Telegram lo movia).
            _escribir_heartbeat()
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
                # Reconexion in-process ROTA en iqoptionapi: tras una caida del WS,
                # connect() cuelga >45s en todos los intentos (medido 2026-08-01:
                # 8 intentos seguidos, todos colgados). Quedarse aqui en bucle no
                # arregla nada y, peor, el heartbeat se escribe al inicio de cada
                # ciclo asi que el watchdog NUNCA ve la congelacion: el bot queda
                # caido horas con proceso "vivo". Un proceso NUEVO conecta en ~2s.
                # Salir para que el watchdog lo relance limpio.
                log("[FATAL] Sin conexion tras 5 intentos. Saliendo para que el "
                    "watchdog relance un proceso limpio...")
                os._exit(3)

            # Una sola consulta de payouts por ciclo (evita 231 llamadas/ciclo).
            # CON timeout: si el WS murio esta llamada colgaria el bucle (ver
            # _llamar_timeout). Si falla, {} y seguimos: el buy decidira disponibilidad.
            profits_ciclo, _ = _llamar_timeout(api.get_all_profit, 15, {})
            if profits_ciclo is None:
                profits_ciclo = {}

            # Idem para el horario del mercado. OJO: el payout NO sirve de proxy, IQ
            # lo devuelve tambien con el activo cerrado -> el par pasaba el filtro y el
            # cierre solo se descubria al comprar ("the asset is not available").
            abiertos_ciclo = _open_time_ciclo(api)

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

                # Mercado cerrado -> ni bajar velas ni calcular indicadores.
                if not _mercado_abierto(abiertos_ciclo, par, expiry):
                    continue

                filtro = CFG.get("filtro_hora", {})
                if filtro.get("habilitado") and "-OTC" not in par:
                    hora_utc = datetime.now(timezone.utc).hour
                    horas_par = filtro.get("horas_por_par", {}).get(par)
                    if horas_par is None or hora_utc not in horas_par:
                        continue

                # ── No bajar velas de un par cuya vela actual YA se proceso.
                # La vela de 5m solo cambia cada timeframe_seg, pero el bucle gira cada
                # POLL segundos. Antes se bajaban las 49 ventanas en CADA pasada y el
                # duplicado se descartaba DESPUES (con el `ultimas_velas` de mas abajo),
                # o sea que ya se habia pagado la llamada de red.
                #
                # Y esa llamada es TODO el coste: medido el 2026-07-28, la mediana por par
                # es 0.96 s, de los cuales la inferencia del modelo son 22 ms (2%) y el
                # resto es get_candles. Una pasada completa son ~47 s.
                #
                # El efecto sobre el RETRASO, que es lo que importa: sin esto, si el cierre
                # de vela caia a mitad de pasada, los ultimos pares se evaluaban hasta 94 s
                # tarde (lo que quedaba de la pasada en curso + la siguiente entera).
                # Saltando lo ya procesado, entre cierre y cierre el bucle gira en vacio y
                # la pasada arranca justo en el cierre: el peor caso baja a ~47 s. Importa
                # porque el retraso cuesta WR (inmediatas 59.68%, demoradas 51.09%).
                #
                # El calculo es por reloj y coincide con velas[-2]["from"]: velas[-1] es la
                # vela en formacion, que empieza en (ahora // tf) * tf.
                #
                # OJO: si el activo esta CERRADO, sus velas son viejas y la guardada nunca
                # alcanza a 'esperada', asi que se sigue bajando en cada pasada -- que es lo
                # que hace falta para enterarse de la reapertura.
                _tf = int(CFG["operacion"].get("timeframe_seg", 300))
                _esperada = (int(time.time()) // _tf) * _tf - _tf
                if ultimas_velas.get(par, 0) >= _esperada:
                    continue

                try:
                    op_ = CFG["operacion"]
                    # Las 300 velas venian de la estrategia vieja (5x EMA50 para que
                    # convergiera). El modelo 'seq' solo necesita L + ATR_P + 1 = 79,
                    # mas la vela en formacion. Bajar menos acorta el tiempo entre el
                    # cierre de vela y la compra, que es el desfase de ~5s entre el
                    # precio que el modelo tomo de referencia y el strike real.
                    import seq_model as _sm
                    minimo = _sm.L_DEFECTO + _sm.ATR_P + 2
                    n_velas = max(int(op_.get("n_velas", 100)), minimo)
                    # CON timeout: get_candles es la llamada que colgó el bot el
                    # 2026-07-24 cuando el WS murio. Si se cuelga o falla -> exito=False.
                    # NO continuar al siguiente par: si el WS murio, los otros 48 se
                    # colgaran igual y cada uno paga sus 20s de timeout -> 16 min dentro
                    # del for con el heartbeat congelado, y el watchdog no ve la caida
                    # (solo hay que salir para que verificar_conexion actue al inicio
                    # del proximo ciclo; medido el 2026-08-01: WS cayo ~12:14:45 y el
                    # bot recien murio a las 12:25:30 martillando get_candles).
                    velas, exito = _llamar_timeout(
                        lambda: api.get_candles(par, op_["timeframe_seg"], n_velas, time.time()), 20)
                    if not exito:
                        break
                except Exception:
                    continue
                if not velas or len(velas) < minimo:
                    continue

                vela_cerrada = int(velas[-2]["from"])
                # '>=' y no '==': si por lo que sea la vela guardada fuese posterior,
                # con '==' se reprocesaria igualmente. Asi solo pasan las de verdad nuevas.
                if ultimas_velas.get(par, 0) >= vela_cerrada:
                    continue
                ultimas_velas[par] = vela_cerrada
                guardar_ultimas_velas(ultimas_velas)

                closes = [float(v["close"]) for v in velas[:-1]]
                highs = [float(v.get("max", v.get("high", v["close"]))) for v in velas[:-1]]
                lows = [float(v.get("min", v.get("low", v["close"]))) for v in velas[:-1]]
                # ── Estrategia UNICA: 'seq' - modelo secuencial (LSTM / Transformer).
                # Predice P(sube) sobre la vela cerrada; regla simetrica: CALL si
                # P>=thr, PUT si P<=1-thr. La ventana la construye
                # seq_model.ventana_features(), la MISMA funcion que uso el
                # entrenamiento: si divergen, el bot alimenta al modelo con features
                # distintas y falla en silencio.
                lado, score, info_txt = predecir_seq(velas, par)
                thr = umbral_de(par)
                cumple = lado is not None
                log(f"  {par:8s} | {info_txt} | " + (
                    ("SENAL " + lado.upper()) if cumple
                    else f"sin senal seq (|P-0.5| < {thr-0.5:.3f})"))
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

                # Filtro ADX (regimen): el modelo es de REVERSION y la reversion falla en
                # tendencia fuerte. Con adx_max>0, solo se opera cuando ADX < adx_max
                # (mercado en rango). adx_max=0 -> desactivado. Ver adx() y la memoria
                # optimizar-reversion. Es filtro de EJECUCION: no toca el modelo.
                adx_max = CFG.get("operacion", {}).get("adx_max", 0)
                if adx_max and len(closes) > 2 * 14:
                    adx_v = adx(highs, lows, closes, 14)
                    if adx_v is not None:
                        if adx_v >= adx_max:
                            log(f"  [FILTRO-ADX] {par} {lado.upper()} descartado "
                                f"(ADX {adx_v:.1f} >= max {adx_max}, tendencia fuerte)")
                            continue
                        info_txt = info_txt + f" | ADX {adx_v:.1f}"

                clave = f"{par}-{vela_cerrada}"
                with _lock:
                    if clave in _cruces_fallidos:
                        continue

                # Alinear el horizonte real con el que se entreno (ver expiry_alineado).
                ok_exp, mins = expiry_alineado(CFG["operacion"], par)
                if not ok_exp:
                    # 'mins' es la duracion REAL (distancia a la proxima marca de 15);
                    # 'expiry' es lo que se le PIDE a IQ y solo fija el payout. Loguear
                    # el segundo hacia parecer que el filtro comparaba contra el numero
                    # equivocado.
                    _obj = horizonte_modelo_min(par)
                    log(f"  [EXPIRY] {par} {lado.upper()} descartado: duracion real "
                        f"{mins:.1f} min, el modelo predice a {_obj} min")
                    continue

                if dry:
                    log(f"[DRY] {par} {lado.upper()} | {info_txt} | payout {p:.0%}")
                    continue

                if ops_hora_excedidas():
                    max_ops = CFG.get("riesgo", {}).get("max_operaciones_hora")
                    log(f"[RIESGO] {max_ops} ops/hora alcanzadas. Esperando...")
                    break

                # El cupo de max_trades lo toma ejecutar_trade cuando la orden ENTRA de
                # verdad, no aqui. Con el reintento sostenido, un hilo puede pasarse
                # minutos esperando a que abra la ventana del broker; si reservara el
                # cupo al arrancar, diez hilos en espera dejaban el bot en [LLENO] sin
                # una sola posicion abierta -- que es justo lo que paso al estrenarlo.
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

    # INSTANCIA UNICA DEL BOT: defensa real contra ordenes duplicadas. Un segundo main.py
    # (lanzado por fuera del watchdog) operaria sobre la misma senal con el stake al doble,
    # invisible porque _lock no cruza procesos. En --dry no opera, asi que ahi se permite.
    if not args.dry:
        from seq_model import tomar_cerrojo
        _AQUI = os.path.dirname(os.path.abspath(__file__))
        if tomar_cerrojo(os.path.join(_AQUI, "bot.lock")) is None:
            log("YA HAY OTRO BOT vivo (bot.lock tomado): no arranco, evito ordenes duplicadas.")
            raise SystemExit(0)

    _balance_mode = "REAL" if args.real else "PRACTICE"

    # Cargar .env si existe (sin dependencia python-dotenv)
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_env):
        with open(_env, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

    with open("config.json", encoding="utf-8") as f:
        CFG = json.load(f)

    # Credenciales: variables de entorno tienen prioridad sobre config.json.
    # Permite NO guardar secretos en el repo (config.json esta gitignored/untracked).
    CFG["email"] = os.getenv("IQ_EMAIL") or None
    CFG["password"] = os.getenv("IQ_PASSWORD") or None
    tg = CFG.setdefault("telegram", {})
    tg["token"] = os.getenv("TELEGRAM_TOKEN") or tg.get("token")
    tg["chat_id"] = os.getenv("TELEGRAM_CHAT_ID") or tg.get("chat_id")
    if not CFG.get("email") or not CFG.get("password"):
        log("FALTAN CREDENCIALES: define IQ_EMAIL/IQ_PASSWORD en .env")
        return

    api = IQ_Option(CFG["email"], CFG["password"])
    log("Conectando a IQ Option...")
    # connect() puede REVENTAR en vez de devolver (False, motivo): stable_api.py hace
    # json.loads(reason) dando por hecho que el motivo es JSON, y cuando el fallo es de
    # red el motivo es el string 'Websocket connection closed.' -> JSONDecodeError.
    # Sin este try la traza se va por stderr y rsi_iq.log se queda en la linea de
    # arriba: el 2026-07-22 hubo nueve arranques que no dejaron ni una pista del motivo.
    # connect() CON timeout: si IQ esta caido al arrancar, la llamada se colgaria sin
    # esto y el proceso quedaria vivo colgado (mismo agujero que la reconexion). Con
    # timeout, falla y 'return' termina el proceso -> el watchdog lo reintenta.
    # _llamar_timeout traga la excepcion del login KO (JSONDecodeError de stable_api
    # cuando el motivo no es JSON); exito=False cubre timeout y reviente.
    res, _exito = _llamar_timeout(api.connect, 45, (False, ""))
    if not _exito:
        ok, reason = False, "timeout o fallo de red (connect no devolvio)"
    else:
        ok, reason = res if isinstance(res, (list, tuple)) else (False, str(res))
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

    # Reclamar operaciones pendientes de un proceso anterior. En hilo daemon: si hay
    # varias y cada una tarda hasta 60s de timeout, el arranque no debe esperar (el
    # watchdog mediria un arranque lento). El resultado llega como [CIERRE-RECUP].
    # Medido 2026-08-01: entrada ETHUSD 12:15 sin [CIERRE], balance -1.00 en IQ.
    if not args.dry:
        threading.Thread(target=recuperar_pendientes, args=(api,), daemon=True).start()

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

        # Aviso de ARRANQUE. No es cosmetico: el 2026-07-24 el PC se reinicio solo y el
        # bot estuvo 27 min caido sin que nadie lo notara. Un mensaje por cada arranque
        # hace visible el reinicio que no pediste -- si llega uno a deshora, algo paso.
        modo = "DRY (sin operar)" if args.dry else ("REAL" if args.real else "DEMO")
        try:
            commander._enviar(
                tg_cfg["chat_id"],
                f"Bot IQOPT iniciado\nModo: {modo}\n"
                f"Balance: {_sesion.get('balance_inicial')}\n"
                f"Activos: {len(activos)}\n"
                f"Hora: {datetime.now():%Y-%m-%d %H:%M:%S}",
                tg_cfg["token"])
        except Exception as e:
            log(f"[TELEGRAM] no se pudo avisar del arranque: {type(e).__name__}")

    run(api, activos, dry=args.dry)


if __name__ == "__main__":
    main()
