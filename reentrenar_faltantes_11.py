# -*- coding: utf-8 -*-
# reentrenar_faltantes_11.py - Reentrena a 11 features SOLO los pares cuyo modelo real
# (los .npz de semillas en disco) no esta completo. Deteccion por SEMILLAS reales, no por
# el .pt.json: varios pares tienen el JSON en n_feats=11 pero sin las 5 semillas de 11,
# y retrain_faltantes.py (que mira el JSON) los daba por listos. Cada par: 5 semillas.
import subprocess, sys, os, glob, json
import numpy as np

os.environ["PYTHONIOENCODING"] = "utf-8"

def completo(par):
    """True si el par ya tiene 5 semillas .npz de 11 features en disco."""
    base = f"models/seq_lstm_{par}"
    sem = sorted(glob.glob(base + "_s*.npz"))
    if len(sem) != 5:
        return False
    for s in sem:
        z = np.load(s)
        if "W_ih" not in z.files or int(z["W_ih"].shape[1]) != 11:
            return False
    return True

cfg = json.load(open("config.json", encoding="utf-8"))
todos = cfg["entrenamiento"]["pares"]
faltan = [p for p in todos if not completo(p)]
print(f"[reentrenar] {len(faltan)} faltantes: {', '.join(faltan)}", flush=True)

for i, par in enumerate(faltan, 1):
    print(f"\n{'='*60}\n[{i}/{len(faltan)}] reentrenando {par} a 11 feats, 5 semillas\n{'='*60}", flush=True)
    r = subprocess.run([sys.executable, "train_seq_save.py", "--par", par, "--semillas", "5"])
    ok = completo(par)
    print(f"[reentrenar] {par}: {'OK (5 semillas de 11)' if ok else 'FALLO/incompleto'} (exit {r.returncode})", flush=True)

print(f"\n[reentrenar] TERMINADO. Faltaban {len(faltan)}.", flush=True)
