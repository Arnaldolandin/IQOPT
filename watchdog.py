# watchdog.py - Supervisor del bot IQOPT. Lo relanza si (a) el proceso muere o (b) el
# BUCLE DE TRADING se congela (heartbeat viejo).
#
#   .venv314\Scripts\python.exe watchdog.py            # supervisa main.py (demo)
#   .venv314\Scripts\python.exe watchdog.py --real     # pasa flags al bot
#
# POR QUE HEARTBEAT y no el mtime del log: el 2026-07-24 el WebSocket de IQ murio, una
# llamada de API se colgo y el bucle de trading quedo frozen 6.5 h, PERO el hilo daemon
# de Telegram siguio escribiendo al log (spam 409) -> el log parecia "vivo". El heartbeat
# lo escribe SOLO el bucle de trading (main._escribir_heartbeat), asi que si ese bucle se
# congela, el heartbeat envejece aunque el proceso siga "vivo". Esto lo detecta.
import json
import os
import subprocess
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(AQUI, ".venv314", "Scripts", "python.exe")
HEARTBEAT = os.path.join(AQUI, "heartbeat.json")
LOG = os.path.join(AQUI, "watchdog.log")
BOT_OUT = os.path.join(AQUI, "bot_stdout.log")

MAX_SILENCIO = 600      # seg sin latido -> bucle congelado -> reiniciar (holgado: cubre
                        # los ~3 min que puede tardar una tanda de reconexiones)
CHECK = 60              # cada cuanto vigila
GRACIA = 120            # tras (re)lanzar, no exige heartbeat durante este tiempo (arranque)


def log(msg):
    linea = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
    except Exception:
        pass


def heartbeat_edad():
    """Segundos desde el ultimo latido, o None si el archivo no existe/ilegible."""
    try:
        with open(HEARTBEAT, encoding="utf-8") as f:
            return time.time() - float(json.load(f)["ts"])
    except Exception:
        return None


def lanzar(flags):
    # borra el latido viejo para no confundirlo con el del proceso anterior
    try:
        os.remove(HEARTBEAT)
    except Exception:
        pass
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    out = open(BOT_OUT, "a", encoding="utf-8")
    p = subprocess.Popen([PY, "main.py"] + flags, cwd=AQUI,
                         stdout=out, stderr=subprocess.STDOUT, env=env)
    log(f"Bot lanzado (PID {p.pid}) main.py {' '.join(flags)}")
    return p


def matar(p):
    try:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    except Exception as e:
        log(f"Error matando bot: {e}")


def main():
    flags = sys.argv[1:]
    log(f"Watchdog iniciado. MAX_SILENCIO={MAX_SILENCIO}s CHECK={CHECK}s GRACIA={GRACIA}s")
    proc = lanzar(flags)
    t_lanzado = time.time()
    while True:
        time.sleep(CHECK)
        if proc.poll() is not None:
            log(f"Bot MUERTO (exit {proc.returncode}). Relanzando...")
            proc = lanzar(flags); t_lanzado = time.time(); continue
        if time.time() - t_lanzado < GRACIA:
            continue
        edad = heartbeat_edad()
        if edad is None:
            log("Sin heartbeat pese a la gracia -> algo va mal. Reiniciando bot...")
        elif edad > MAX_SILENCIO:
            log(f"Bucle CONGELADO: heartbeat {edad:.0f}s > {MAX_SILENCIO}s. Reiniciando bot...")
        else:
            continue
        matar(proc)
        proc = lanzar(flags); t_lanzado = time.time()


if __name__ == "__main__":
    main()
