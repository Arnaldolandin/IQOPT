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
        elif len(partes) >= 2 and cmd == "/setconf":
            return self._set_conf(partes[1])
        elif len(partes) >= 2 and cmd in ("/setumbral", "/setbbml"):
            return self._set_umbral(partes[1])
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
        _est = op.get("estrategia", "meta")
        _est_txt = (f"Meta-labeling bbrev (umbral P>={op.get('bb_ml_threshold', 0.6)})"
                    if _est == "meta"
                    else f"Predictor optimizado (min_conf {op.get('min_conf', 0.02)})")
        lines = [
            "[STATUS] Bot IQ Option\n",
            f"Modo: {'REAL' if self._es_real() else 'DEMO'}",
            f"Balance: ${balance}",
            f"Estrategia: {_est_txt}",
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
        if op.get("estrategia", "meta") == "meta":
            senal = (f"META-LABELING (portado de Deriv): bbrev (reversion Bollinger "
                     f"{op.get('bb_std', 2.0)}sigma, periodo {op.get('bb_period', 20)}) elige el lado, y un "
                     f"meta-modelo (32 features) predice P(la senal gana).\n"
                     f"Opera solo si hay extremo 2sigma Y meta P >= {op.get('bb_ml_threshold', 0.6)}.")
        else:
            senal = (f"PREDICTOR optimizado (13 indicadores, pesos aprendidos por logistica)\n"
                     f"MACD, EMA50/200, RSI, Stochastic, Williams%R, CCI, ADX/DI, ROC, Bollinger, ult3.\n"
                     f"Opera CALL/PUT solo si la confianza |p-0.5| >= {op.get('min_conf', 0.02)}.")
        return (
            f"[ESTRATEGIA] {senal}\n"
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
            "/setumbral [x] - Umbral P del meta-modelo (ej 0.55-0.65) [estrategia meta]\n"
            "/setconf [x] - Umbral de confianza |p-0.5| [estrategia predictor]\n"
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
        _est = op.get("estrategia", "meta")
        if _est == "meta":
            _est_lines = (f"Estrategia: meta-labeling bbrev\n"
                          f"bb_ml_threshold: {op.get('bb_ml_threshold', 0.6)}\n"
                          f"bb_std: {op.get('bb_std', 2.0)} | bb_period: {op.get('bb_period', 20)}\n")
        else:
            _est_lines = (f"Estrategia: predictor confluencia\n"
                          f"min_conf: {op.get('min_conf', 0.02)}\n")
        return (
            f"[CONFIG]\n"
            f"{_est_lines}"
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

    def _set_conf(self, val_str):
        try:
            v = float(val_str)
        except ValueError:
            return "[ERROR] /setconf [x]  con x tipo 0.02 (umbral de confianza |p-0.5|)"
        if not (0.0 <= v <= 0.49):
            return "[ERROR] confianza entre 0 y 0.49"
        self.cfg.setdefault("operacion", {})["min_conf"] = v
        self._guardar_cfg()
        return (f"[OK] min_conf = {v}\nEl bot opera solo si |p-0.5| >= {v} "
                f"(mas alto = menos ops, mas selectivo).\nSe aplicara en el proximo ciclo (hot-reload).")

    def _set_umbral(self, val_str):
        """Umbral del meta-modelo (bb_ml_threshold): opera si P(gana) >= umbral."""
        try:
            v = float(val_str)
        except ValueError:
            return "[ERROR] /setumbral [x]  con x tipo 0.60 (umbral P del meta-modelo)"
        if not (0.5 <= v <= 0.9):
            return "[ERROR] umbral entre 0.5 y 0.9 (tipico 0.55-0.65)"
        self.cfg.setdefault("operacion", {})["bb_ml_threshold"] = v
        self._guardar_cfg()
        return (f"[OK] bb_ml_threshold = {v}\nEl bot opera si el meta-modelo da P(gana) >= {v} "
                f"(mas alto = menos ops, mas selectivo; mas bajo = mas ops, menos calidad).\n"
                f"Se aplica en el proximo ciclo (hot-reload).")

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
