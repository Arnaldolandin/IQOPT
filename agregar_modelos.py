"""Monitorea models/ y agrega al config.json los modelos nuevos de 11 features.
Solo agrega el modelo principal (sin sufijo _sN). El bot hot-reload cada 30s."""

import json, time, os, glob
import numpy as np

CONFIG = "config.json"

def modelos_ready():
    """Set de pares LISTOS = 5 semillas .npz de 11 features EN DISCO. Se verifican los
    .npz reales, no el .pt.json: el meta puede decir semillas=5 cuando el reentrenamiento
    quedo a medias (2-4 semillas escritas), y fiarse del JSON metia pares incompletos."""
    ready = set()
    for j in glob.glob("models/seq_lstm_*.pt.json"):
        par = os.path.basename(j).replace("seq_lstm_", "").replace(".pt.json", "")
        if "_s" in par:
            continue
        sem = sorted(glob.glob(f"models/seq_lstm_{par}_s*.npz"))
        if len(sem) != 5:
            continue
        try:
            if all("W_ih" in np.load(s).files and int(np.load(s)["W_ih"].shape[1]) == 11
                   for s in sem):
                ready.add(par)
        except Exception:
            pass
    return ready

def main():
    print("[agregar_modelos] Monitoreando...")
    while True:
        try:
            ready = modelos_ready()
            with open(CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            op = cfg.setdefault("operacion", {})
            mpp = op.setdefault("modelos_por_par", {})
            upp = op.setdefault("umbrales_por_par", {})

            to_remove = [k for k in mpp if "_s" in k]
            for k in to_remove:
                del mpp[k]
                upp.pop(k, None)

            added = []
            for par in ready:
                if par not in mpp:
                    mpp[par] = f"models/seq_lstm_{par}.pt"
                    upp.setdefault(par, 0.54)
                    added.append(par)

            if added or to_remove:
                # Escritura ATOMICA: el bot hot-reloadea config.json cada 30s; escribir
                # in situ deja una ventana en que leeria un JSON truncado. tmp + replace
                # es atomico en el mismo disco.
                tmp = CONFIG + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                os.replace(tmp, CONFIG)
                if added:
                    print(f"[+] Agregados: {', '.join(sorted(added))} (total {len(mpp)})")
                if to_remove:
                    print(f"[-] Eliminadas {len(to_remove)} entradas semilla")

            all_expected = set(json.load(open("config.json")).get("entrenamiento", {}).get("pares", []))
            missing = all_expected - ready
            if not missing:
                print("[OK] Todos los modelos listos!")
                break

        except Exception as e:
            print(f"[!] {e}")

        time.sleep(30)

if __name__ == "__main__":
    main()
