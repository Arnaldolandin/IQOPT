# analizar_sentimiento.py - Analiza sentiment_log.jsonl. Descarga velas 1m para el periodo
# colectado y empareja cada lectura de mood con el precio en t y en t+EXPIRY. Testea si fadear
# (o seguir) a la multitud supera break-even, por umbral de sentimiento extremo.
#   .venv314\Scripts\python.exe analizar_sentimiento.py [expiry_seg]
import json, sys, time, threading
from collections import defaultdict
import numpy as np
from iqoptionapi.stable_api import IQ_Option

EXPIRY = int(sys.argv[1]) if len(sys.argv) > 1 else 120
BE = 1.0 / 1.87


def conectar():
    cfg = json.load(open("config.json", encoding="utf-8"))
    api = IQ_Option(cfg["email"], cfg["password"])
    api.connect(); api.change_balance("PRACTICE")
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


def bajar(api, par, desde, hasta):
    todas = {}; end = hasta + 120
    for _ in range(200):
        try:
            lote = api.get_candles(par, 60, 1000, end)
        except Exception:
            break
        if not lote:
            break
        for v in lote:
            todas[v["from"]] = float(v["close"])
        old = min(v["from"] for v in lote)
        if old <= desde:
            break
        end = old - 1
    return todas


def main():
    rows = defaultdict(list)
    try:
        for line in open("sentiment_log.jsonl", encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            rows[d["par"]].append((d["ts"], d["mood_call"]))
    except FileNotFoundError:
        print("No hay sentiment_log.jsonl. Corre colector_sentimiento.py primero."); return
    n = sum(len(v) for v in rows.values())
    if n == 0:
        print("Log vacio."); return
    tmin = min(t for v in rows.values() for t, _ in v)
    tmax = max(t for v in rows.values() for t, _ in v)
    dias = (tmax - tmin) / 86400
    print(f"Lecturas: {n} | pares: {len(rows)} | span {dias:.1f} dias | expiry {EXPIRY}s")
    if dias < 0.02:
        print("Muy poca coleccion aun. Deja el colector corriendo mas tiempo.")

    print("Descargando velas 1m para los precios...")
    api = conectar()
    obs = []  # (mood, fade_gana)
    for par, lst in sorted(rows.items()):
        precios = bajar(api, par, tmin, tmax)
        if not precios:
            continue
        keys = np.array(sorted(precios))
        def px(ts):
            i = np.searchsorted(keys, ts, side="right") - 1
            return precios[keys[i]] if 0 <= i < len(keys) else None
        for ts, mood in lst:
            p0 = px(ts); p1 = px(ts + EXPIRY)
            if p0 is None or p1 is None or p1 == p0:
                continue
            subio = p1 > p0
            crowd_call = mood > 0.5
            fade_gana = (not subio) if crowd_call else subio
            obs.append((mood, fade_gana))
    if not obs:
        print("Sin observaciones emparejables aun."); return

    M = np.array([o[0] for o in obs]); G = np.array([o[1] for o in obs], float)
    print(f"\nObservaciones: {len(obs)} | mood medio {M.mean()*100:.1f}% | dispersion {M.std()*100:.1f}%")
    print(f"{'FILTRO':>26} | {'FADE crowd':>14} | {'SEGUIR crowd':>13}")
    print("-" * 60)
    for thr in (0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        mask = np.abs(M - 0.5) >= (thr - 0.5) if thr > 0 else np.ones(len(M), bool)
        if mask.sum() < 20:
            continue
        wf = G[mask].mean() * 100; ws = (1 - G[mask]).mean() * 100
        etq = "todas" if thr == 0 else f"|mood-50%|>={(thr-0.5)*100:.0f}pt"
        mk = "  <-- >BE" if wf > BE * 100 else ""
        print(f"{etq:>26} | {wf:>6.2f}% ({int(mask.sum()):>5}){mk} | {ws:>6.2f}%")
    print(f"\nbreak-even {BE*100:.1f}%. FADE = contra la mayoria. Para concluir hace falta muestra")
    print("grande (miles) y split temporal; con pocas horas de datos es solo un vistazo.")


if __name__ == "__main__":
    main()
