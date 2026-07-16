# telegram_commands.py - Control del bot via Telegram
import threading
import json
import time
import os
import requests
from datetime import datetime, timezone


class TelegramCommanderSimple:
    """Controla parametros del bot IQ Option via Telegram."""

    def __init__(self, api, cfg, sesion_ref, activos_ref):
        self.api = api
        self.cfg = cfg
        self.sesion = sesion_ref
        self.activos = activos_ref
        self.ultimo_comando = 0
        self.cooldown_comandos = 2

    def procesar_comando(self, texto):
        ahora = time.time()
        if ahora - self.ultimo_comando < self.cooldown_comandos:
            return None
        self.ultimo_comando = ahora

        partes = texto.strip().split()
        if not partes:
            return None
        cmd = partes[0].lower()

        if cmd in ("/status", "/estado"):
            return self._status()
        elif cmd in ("/balance", "/saldo"):
            return self._balance()
        elif cmd == "/activas":
            return self._activas()
        elif cmd in ("/pares", "/symbols"):
            return self._pares()
        elif cmd == "/modo":
            return self._modo()
        elif cmd == "/estrategia":
            return self._estrategia()
        elif cmd in ("/ayuda", "/help"):
            return self._ayuda()
        elif cmd in ("/reiniciar", "/restart"):
            return self._reiniciar()
        elif cmd == "/pnl":
            return self._pnl()
        elif cmd == "/config":
            return self._ver_config()
        elif cmd == "/pausar":
            return self._pausar()
        elif cmd == "/reanudar":
            return self._reanudar()
        elif len(partes) >= 2 and cmd == "/setstake":
            return self._set_param("operacion", "stake", partes[1],
                                   lambda v: v.replace(".", "", 1).isdigit() and float(v) > 0)
        elif len(partes) >= 2 and cmd == "/setmaxtrades":
            return self._set_param_global("max_trades", partes[1],
                                          lambda v: v.isdigit() and int(v) >= 1)
        elif len(partes) >= 2 and cmd == "/setexpiry":
            return self._set_param("operacion", "expiry_min", partes[1],
                                   lambda v: v.isdigit() and 1 <= int(v) <= 60)
        elif len(partes) >= 2 and cmd == "/setpayout":
            return self._set_param("operacion", "min_payout", partes[1],
                                   lambda v: v.replace(".", "", 1).isdigit() and 0.5 <= float(v) <= 1.0)
        elif len(partes) >= 2 and cmd == "/filtrar":
            val = partes[1].lower()
            if val in ("on", "si", "true", "1"):
                return self._toggle_filtro(True)
            elif val in ("off", "no", "false", "0"):
                return self._toggle_filtro(False)
            return "[ERROR] Uso: /filtrar on|off"
        elif cmd in ("/bb", "/bollinger"):
            return self._ver_bb()
        elif len(partes) >= 2 and cmd == "/setbb":
            return self._set_bb(partes[1], partes[2] if len(partes) >= 3 else None)
        elif len(partes) >= 2 and cmd == "/setestrategia":
            return self._set_estrategia(partes[1])
        elif cmd == "/macd":
            return self._ver_macd()
        elif len(partes) >= 3 and cmd == "/setmacd":
            return self._set_macd(partes[1], partes[2], partes[3] if len(partes) >= 4 else None)
        elif len(partes) >= 2 and cmd == "/setmacdema":
            return self._set_macdema(partes[1])
        elif cmd in ("/atr", "/volatilidad"):
            return self._ver_atr()
        elif len(partes) >= 2 and cmd == "/setatr":
            return self._set_atr(partes[1], partes[2] if len(partes) >= 3 else None)
        elif len(partes) >= 2 and cmd == "/setpoll":
            return self._set_global("poll_seg", partes[1],
                                    lambda v: v.isdigit() and int(v) >= 1)
        else:
            return "[ERROR] Comando no reconocido. Usa /ayuda"

    # ── Comandos principales ─────────────────────────────────────────────

    def _status(self):
        try:
            balance = self.api.get_balance()
        except Exception:
            balance = "?"
        s = self.sesion
        tr = s["trades"]
        wr = (s["wins"] / tr * 100) if tr > 0 else 0
        real = self._pnl_real()
        pnl_txt = (f"PnL real {self._fmt(real)}" if real is not None
                   else f"PnL calc {self._fmt(s['pnl'])}")
        filtro = self.cfg.get("filtro_hora", {})
        filtro_on = filtro.get("habilitado", False)
        horas_cfg = filtro.get("horas_por_par", {})
        offset = filtro.get("timezone_offset", 0)
        hora_utc = datetime.now(timezone.utc).hour
        hora_chile = (hora_utc + offset) % 24

        op = self.cfg.get("operacion", {})
        est = op.get("estrategia", "bb_rev")
        est_txt = (f"MACD-cruce({op.get('macd_fast',6)},{op.get('macd_slow',13)},{op.get('macd_signal',5)})"
                   if est == "macd" else f"BB-reversion({op.get('bb_period',20)},{op.get('bb_k',2.0)})")
        lines = [
            "[STATUS] Bot IQ Option\n",
            f"Modo: {'REAL' if self._es_real() else 'DEMO'}",
            f"Balance: ${balance}",
            f"Estrategia: {est_txt}",
            f"Exp {op['expiry_min']}m | Stake ${op['stake']}",
            f"Max trades: {self.cfg.get('max_trades', 1)} | ATR min: {op.get('min_atr', 0)}",
            f"Filtro hora: {'ON' if filtro_on else 'OFF'} ({len(horas_cfg)} pares)",
            f"Hora UTC: {hora_utc} | Chile: {hora_chile}",
            f"\nSesion: {tr} ops | WR {wr:.1f}% | {pnl_txt}",
        ]
        return "\n".join(lines)

    def _balance(self):
        try:
            b = self.api.get_balance()
            return f"[BALANCE] ${b}"
        except Exception as e:
            return f"[ERROR] No se pudo obtener balance: {e}"

    def _activas(self):
        return f"[ACTIVAS] Trades abiertos: {self.activos['abiertos']} / Max: {self.cfg.get('max_trades', 1)}"

    def _pares(self):
        pares = self.cfg.get("pares_binarios", [])
        payout = self.cfg.get("operacion", {}).get("min_payout", 0)
        lines = [f"[PARES] {len(pares)} configurados (min payout {payout:.0%}):\n"]
        for i in range(0, len(pares), 6):
            lines.append(", ".join(pares[i:i + 6]))
        return "\n".join(lines)

    def _modo(self):
        modo = "REAL" if self._es_real() else "DEMO"
        try:
            b = self.api.get_balance()
        except Exception:
            b = "?"
        return f"[MODO] {modo} | Balance: ${b}"

    def _estrategia(self):
        op = self.cfg["operacion"]
        filtro = self.cfg.get("filtro_hora", {})
        horas = filtro.get("horas_por_par", {})
        tf = op['timeframe_seg'] // 60 if op['timeframe_seg'] >= 60 else op['timeframe_seg']
        est = op.get("estrategia", "bb_rev")
        if est == "macd":
            ema_p = op.get("macd_ema", 0)
            filt = f" + filtro EMA{ema_p}" if ema_p else " (sin EMA)"
            senal = (f"CRUCE MACD ({op.get('macd_fast',6)},{op.get('macd_slow',13)},{op.get('macd_signal',5)}){filt}\n"
                     f"CALL: MACD cruza sobre signal | PUT: cruza bajo signal"
                     + (f"\nEMA{ema_p}: CALL solo si precio>EMA, PUT si precio<EMA" if ema_p else ""))
        else:
            senal = (f"REVERSION Bollinger ({op.get('bb_period',20)}, {op.get('bb_k',2.0)})\n"
                     f"CALL: precio cruza bajo banda inferior | PUT: sobre banda superior")
        return (
            f"[ESTRATEGIA] {est}\n{senal}\n"
            f"Velas {tf}m | Expiracion {op['expiry_min']}m | Stake ${op['stake']}\n"
            f"Filtro ATR: min {op.get('min_atr',0)} (period {op.get('atr_period',14)})\n"
            f"Filtro hora: {'ON' if filtro.get('habilitado') else 'OFF'} ({len(horas)} pares)\n"
            f"Multi-hilo: max {self.cfg.get('max_trades', 1)} trades simultaneos"
        )

    def _ayuda(self):
        return (
            "[AYUDA] Comandos disponibles:\n\n"
            "/status - Estado completo del bot\n"
            "/balance - Balance actual\n"
            "/modo - Modo DEMO/REAL\n"
            "/activas - Trades abiertos\n"
            "/pares - Lista de activos\n"
            "/estrategia - Detalle de la estrategia\n"
            "/pnl - PnL de la sesion\n"
            "/config - Ver configuracion actual\n\n"
            "Configuracion:\n"
            "/setstake [monto] - Stake por trade\n"
            "/setmaxtrades [n] - Max trades simultaneos\n"
            "/setexpiry [min] - Expiracion\n"
            "/setpayout [0.80] - Min payout\n"
            "/setestrategia bb_rev|macd - Cambiar estrategia\n"
            "/bb - Ver Bollinger (periodo, k)\n"
            "/setbb [periodo] [k] - Ajustar Bollinger\n"
            "/macd - Ver cruce MACD (fast/slow/signal + EMA)\n"
            "/setmacd [fast] [slow] [signal] - Ajustar MACD\n"
            "/setmacdema [periodo] - Filtro EMA del MACD (0 desactiva)\n"
            "/atr - Ver filtro de volatilidad ATR\n"
            "/setatr [min] [period] - ATR minimo (0 desactiva)\n"
            "/filtrar on|off - Filtro de hora\n"
            "/setpoll [seg] - Intervalo de escaneo\n\n"
            "Control:\n"
            "/pausar - Pausar trading\n"
            "/reanudar - Reanudar trading\n"
            "/reiniciar - Reiniciar bot"
        )

    def _reiniciar(self):
        import sys
        threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
        return "[OK] Reiniciando bot..."

    def _pnl_real(self):
        """PnL autoritativo del broker: balance actual - balance inicial. None si no disponible."""
        bi = self.sesion.get("balance_inicial")
        if bi is None:
            return None
        try:
            return float(self.api.get_balance()) - float(bi)
        except Exception:
            return None

    @staticmethod
    def _fmt(x):
        return f"{'+' if x >= 0 else '-'}${abs(x):.2f}"

    def _pnl(self):
        s = self.sesion
        tr = s["trades"]
        wr = (s["wins"] / tr * 100) if tr > 0 else 0
        real = self._pnl_real()
        lines = [
            "[PNL] Sesion",
            f"Operaciones: {tr}",
            f"Wins: {s['wins']} ({wr:.1f}% WR)",
            f"PnL calculado: {self._fmt(s['pnl'])}",
        ]
        if real is not None:
            lines.append(f"PnL REAL (balance): {self._fmt(real)}")
            if abs(real - s["pnl"]) > 0.01:
                lines.append(f"(ojo: difieren {self._fmt(real - s['pnl'])} -> se cayo algun cierre)")
        return "\n".join(lines)

    def _ver_config(self):
        op = self.cfg["operacion"]
        filtro = self.cfg.get("filtro_hora", {})
        tf = op['timeframe_seg'] // 60 if op['timeframe_seg'] >= 60 else op['timeframe_seg']
        est = op.get("estrategia", "bb_rev")
        return (
            f"[CONFIG]\n"
            f"Estrategia: {est}\n"
            f"BB periodo/k: {op.get('bb_period', 20)}/{op.get('bb_k', 2.0)}\n"
            f"MACD: {op.get('macd_fast',6)}/{op.get('macd_slow',13)}/{op.get('macd_signal',5)} | EMA filtro: {op.get('macd_ema',0)}\n"
            f"Timeframe: {tf}m\n"
            f"Expiracion: {op['expiry_min']}m\n"
            f"Stake: ${op['stake']}\n"
            f"Min payout: {op['min_payout']:.0%}\n"
            f"ATR min/period: {op.get('min_atr', 0)}/{op.get('atr_period', 14)}\n"
            f"Max trades: {self.cfg.get('max_trades', 1)}\n"
            f"Filtro hora: {'ON' if filtro.get('habilitado') else 'OFF'}\n"
            f"Timezone offset: {filtro.get('timezone_offset', 0)}h\n"
            f"Poll: {self.cfg.get('poll_seg', 3)}s"
        )

    def _pausar(self):
        self.cfg.setdefault("riesgo", {})["pausado"] = True
        self._guardar_cfg()
        return "[OK] Bot PAUSADO. Trades abiertos se cierran normalmente."

    def _reanudar(self):
        if self.cfg.get("riesgo", {}).get("pausado"):
            self.cfg["riesgo"]["pausado"] = False
            self._guardar_cfg()
            return "[OK] Bot REANUDADO."
        return "[INFO] El bot no esta pausado."

    # ── Setters ──────────────────────────────────────────────────────────

    def _set_param(self, seccion, clave, valor_str, validator):
        if not validator(valor_str):
            return f"[ERROR] Valor invalido: {valor_str}"
        actual = self.cfg[seccion][clave]
        if isinstance(actual, float):
            nuevo = float(valor_str)
        elif isinstance(actual, int):
            nuevo = int(valor_str)
        else:
            nuevo = valor_str
        self.cfg[seccion][clave] = nuevo
        self._guardar_cfg()
        return f"[OK] {seccion}.{clave} = {nuevo}"

    def _set_param_global(self, clave, valor_str, validator):
        if not validator(valor_str):
            return f"[ERROR] Valor invalido: {valor_str}"
        self.cfg[clave] = int(valor_str)
        self._guardar_cfg()
        return f"[OK] {clave} = {int(valor_str)}"

    def _set_global(self, clave, valor_str, validator):
        if not validator(valor_str):
            return f"[ERROR] Valor invalido: {valor_str}"
        self.cfg[clave] = int(valor_str)
        self._guardar_cfg()
        return f"[OK] {clave} = {int(valor_str)}"

    def _set_estrategia(self, val):
        val = val.lower()
        validas = ("bb_rev", "macd")
        if val not in validas:
            return f"[ERROR] Estrategia invalida. Opciones: {', '.join(validas)}"
        self.cfg.setdefault("operacion", {})["estrategia"] = val
        self._guardar_cfg()
        desc = {"bb_rev": "Reversion Bollinger (CALL bajo banda / PUT sobre banda)",
                "macd": "Cruce MACD puro (CALL cruza arriba / PUT cruza abajo, sin EMAs ni pendiente)"}[val]
        return f"[OK] estrategia = {val}\n{desc}\nSe aplicara en el proximo ciclo (hot-reload)."

    def _ver_macd(self):
        op = self.cfg.get("operacion", {})
        ema_p = op.get("macd_ema", 0)
        filtro = f"filtro EMA{ema_p} (solo a favor de tendencia)" if ema_p else "sin filtro EMA"
        return (f"[MACD] fast={op.get('macd_fast',6)} slow={op.get('macd_slow',13)} signal={op.get('macd_signal',5)} | {filtro}\n"
                f"CALL cuando MACD cruza sobre signal, PUT al reves.\n"
                f"Con EMA{ema_p}: CALL solo si precio>EMA, PUT solo si precio<EMA.\n"
                f"Ajustar: /setmacd [fast] [slow] [signal] | /setmacdema [periodo] (0 desactiva).")

    def _set_macdema(self, per_str):
        if not per_str.isdigit() or int(per_str) > 500:
            return "[ERROR] /setmacdema [periodo]  (0 desactiva, max 500)"
        self.cfg.setdefault("operacion", {})["macd_ema"] = int(per_str)
        self._guardar_cfg()
        p = int(per_str)
        estado = "sin filtro EMA" if p == 0 else f"filtro EMA{p} (solo a favor de tendencia)"
        return f"[OK] macd_ema = {p} ({estado})\nSe aplicara en el proximo ciclo (hot-reload)."

    def _set_macd(self, fast_str, slow_str, signal_str):
        if not (fast_str.isdigit() and slow_str.isdigit()):
            return "[ERROR] /setmacd [fast] [slow] [signal]"
        fast, slow = int(fast_str), int(slow_str)
        sig = int(signal_str) if signal_str and signal_str.isdigit() else self.cfg["operacion"].get("macd_signal", 5)
        if not (2 <= fast <= 100 and 3 <= slow <= 400):
            return "[ERROR] fast 2-100, slow 3-400"
        if fast >= slow:
            return "[ERROR] fast debe ser menor que slow"
        op = self.cfg.setdefault("operacion", {})
        op["macd_fast"] = fast
        op["macd_slow"] = slow
        op["macd_signal"] = sig
        self._guardar_cfg()
        return f"[OK] MACD {fast}/{slow}/{sig}\nSe aplicara en el proximo ciclo (hot-reload)."

    def _ver_bb(self):
        op = self.cfg.get("operacion", {})
        per = op.get("bb_period", 20)
        k = op.get("bb_k", 2.0)
        return (f"[BB] periodo={per}  k={k}\n"
                f"Reversion Bollinger: CALL si precio cruza bajo la banda inferior, "
                f"PUT si cruza sobre la superior.\n"
                f"Usa /setbb [periodo] [k]  (ej: /setbb 20 2.0).")

    def _set_bb(self, per_str, k_str):
        if not per_str.isdigit() or not (5 <= int(per_str) <= 200):
            return "[ERROR] periodo debe ser entero 5-200"
        op = self.cfg.setdefault("operacion", {})
        op["bb_period"] = int(per_str)
        if k_str is not None:
            try:
                k = float(k_str)
            except ValueError:
                return "[ERROR] k invalido. Ej: /setbb 20 2.0"
            if not (0.5 <= k <= 5):
                return "[ERROR] k debe estar entre 0.5 y 5"
            op["bb_k"] = k
        self._guardar_cfg()
        return f"[OK] BB periodo={op['bb_period']} k={op.get('bb_k',2.0)}\nSe aplicara en el proximo ciclo (hot-reload)."

    def _ver_atr(self):
        op = self.cfg.get("operacion", {})
        ma = op.get("min_atr", 0.0)
        per = op.get("atr_period", 14)
        modo = f"solo opera si ATR({per}) >= {ma:.4%} del precio" if ma else "SIN filtro ATR"
        return (f"[ATR] min_atr={ma} (={ma:.4%})  period={per}\n{modo}\n"
                f"Filtro de volatilidad (ATR normalizado por precio).\n"
                f"0 = sin filtro | tipico 0.0005-0.003 (0.05%-0.3%)\n"
                f"Usa /setatr [min_atr] [period]  (ej: /setatr 0.001 14).")

    def _set_atr(self, val_str, per_str):
        try:
            val = float(val_str)
        except ValueError:
            return "[ERROR] Valor invalido. Ejemplo: /setatr 0.001 14"
        if val < 0:
            return "[ERROR] min_atr debe ser >= 0"
        op = self.cfg.setdefault("operacion", {})
        op["min_atr"] = val
        if per_str is not None:
            if not per_str.isdigit() or not (2 <= int(per_str) <= 200):
                return "[ERROR] period debe ser entero 2-200"
            op["atr_period"] = int(per_str)
        self._guardar_cfg()
        estado = "DESACTIVADO" if val == 0 else f"solo ATR >= {val:.4%}"
        return f"[OK] min_atr={val} atr_period={op.get('atr_period',14)} ({estado})\nSe aplicara en el proximo ciclo (hot-reload)."

    def _toggle_filtro(self, on):
        self.cfg.setdefault("filtro_hora", {})["habilitado"] = on
        self._guardar_cfg()
        estado = "ACTIVADO" if on else "DESACTIVADO"
        return f"[OK] Filtro de hora {estado}"

    # ── Helpers ──────────────────────────────────────────────────────────

    def _es_real(self):
        try:
            return self.api.get_balance_mode() == "REAL"
        except Exception:
            return False

    def _guardar_cfg(self):
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ── Telegram polling ─────────────────────────────────────────────────

    def run(self, token="", chat_id=""):
        if not token:
            token = self.cfg.get("telegram", {}).get("token", "")
        if not chat_id:
            chat_id = self.cfg.get("telegram", {}).get("chat_id", "")
        if not token or not chat_id:
            print("[TELEGRAM] No hay token/chat_id en config.json")
            return

        base = f"https://api.telegram.org/bot{token}"
        # Limpiar webhook y conflictos previos
        try:
            requests.post(f"{base}/deleteWebhook", data={"drop_pending_updates": True}, timeout=10)
        except Exception:
            pass

        print(f"[TELEGRAM] Bot activo. Chat ID: {chat_id}")
        ultimo_update = 0
        consecutive_409 = 0

        while True:
            resp = None
            try:
                url = f"{base}/getUpdates"
                params = {"offset": ultimo_update + 1, "timeout": 20}
                resp = requests.get(url, params=params, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            ultimo_update = update["update_id"]
                            if "message" in update and "text" in update["message"]:
                                texto = update["message"]["text"]
                                respuesta = self.procesar_comando(texto)
                                if respuesta:
                                    self._enviar(chat_id, respuesta, token)
                    consecutive_409 = 0
                elif resp.status_code == 409:
                    consecutive_409 += 1
                    if consecutive_409 <= 2:
                        print(f"[TELEGRAM] Conflict (409), reintentando...")
                        time.sleep(2)
                    else:
                        # Limpiar webhook y forzar limpieza
                        try:
                            requests.post(f"{base}/deleteWebhook",
                                          data={"drop_pending_updates": True}, timeout=10)
                        except Exception:
                            pass
                        print(f"[TELEGRAM] 409 persistente, limpiando webhook...")
                        time.sleep(5)
                        consecutive_409 = 0
                else:
                    print(f"[TELEGRAM] HTTP {resp.status_code}")
                    time.sleep(5)
            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.ConnectionError as e:
                print(f"[TELEGRAM] Conexion perdida: {str(e)[:60]}")
                time.sleep(5)
            except Exception as e:
                print(f"[TELEGRAM] Error: {type(e).__name__}: {str(e)[:60]}")
                time.sleep(3)
            finally:
                if resp is not None:
                    resp.close()

    def _enviar(self, chat_id, texto, token):
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {"chat_id": chat_id, "text": texto}
            requests.post(url, data=data, timeout=5)
        except Exception:
            pass
