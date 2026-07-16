# colector_sentimiento.py - Registra el SENTIMIENTO (traders_mood) + precio, cada N seg.
# NO opera. Solo colecta datos en vivo para poder testear despues si fadear a la multitud
# tiene edge. Guarda en sentiment_log.jsonl (append, resumible). Dejar corriendo dias.
#   .venv314\Scripts\python.exe colector_sentimiento.py            # pares reales, cada 60s
#   .venv314\Scripts\python.exe colector_sentimiento.py 30 all     # cada 30s, todos los pares
import json, time, sys, threading
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option

INTERVALO = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MODO = sys.argv[2] if len(sys.argv) > 2 else "real"   # real | all
LOG = "sentiment_log.jsonl"
CFG = json.load(open("config.json", encoding="utf-8"))


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def conectar():
    api = IQ_Option(CFG["email"], CFG["password"])
    ok, r = api.connect()
    if not ok:
        raise Exception(f"connect: {r}")
    api.change_balance("PRACTICE")
    done = [False]
    def _u():
        try:
            api.get_ALL_Binary_ACTIVES_OPCODE()
        except Exception:
            pass
        done[0] = True
    threading.Thread(target=_u, daemon=True).start()
    t0 = time.time()
    while not done[0] and time.time() - t0 < 45:
        time.sleep(1)
    time.sleep(1)
    return api


def main():
    pares = list(dict.fromkeys(CFG.get("pares_binarios", [])))
    if MODO == "real":
        pares = [p for p in pares if "-OTC" not in p]
    log(f"Colector de sentimiento | {len(pares)} pares ({MODO}) | cada {INTERVALO}s | -> {LOG}")
    api = conectar()

    # activar streams de mood
    activos = []
    for p in pares:
        try:
            api.start_mood_stream(p)
            activos.append(p)
        except Exception:
            pass
    log(f"Streams de mood activados: {len(activos)}. Calentando 12s...")
    time.sleep(12)

    ciclos = filas = 0
    ultimo_ping = time.time()
    fh = open(LOG, "a", encoding="utf-8")
    while True:
        try:
            # verificar conexion cada ~5 min
            if time.time() - ultimo_ping > 300:
                ultimo_ping = time.time()
                try:
                    api.get_balance()
                except Exception:
                    log("[RECONNECT] reconectando...")
                    try:
                        del api
                    except Exception:
                        pass
                    time.sleep(3)
                    api = conectar()
                    for p in activos:
                        try:
                            api.start_mood_stream(p)
                        except Exception:
                            pass
                    time.sleep(10)

            t_ciclo = time.time()
            n_ok = 0
            ahora = int(time.time())
            for p in activos:
                try:
                    m = api.get_traders_mood(p)
                except Exception:
                    continue
                if m is None:
                    continue
                fh.write(json.dumps({"ts": ahora, "par": p, "mood_call": round(float(m), 4)}) + "\n")
                fh.flush()
                n_ok += 1
                filas += 1
            ciclos += 1
            if ciclos % 3 == 0:
                log(f"ciclo {ciclos} | {n_ok} pares con mood | {filas} filas totales")
            dt = time.time() - t_ciclo
            time.sleep(max(1, INTERVALO - dt))
        except KeyboardInterrupt:
            log("Detenido por usuario.")
            break
        except Exception as e:
            log(f"[WARN] {type(e).__name__}: {str(e)[:60]}")
            time.sleep(5)
    fh.close()


if __name__ == "__main__":
    main()
