# retrain_faltantes.py - Entrena solo los pares que quedaron con 9 features
import subprocess, sys, json, os

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

# Pares que ya tienen 11 features
ok = set()
for fn in os.listdir("models"):
    if fn.startswith("seq_lstm_") and fn.endswith(".pt.json") and "_s" not in fn:
        with open(f"models/{fn}", encoding="utf-8") as f:
            c = json.load(f)
        if c.get("n_feats", 9) == 11:
            ok.add(c["meta"]["par"])

todos = cfg["entrenamiento"]["pares"]
faltan = [p for p in todos if p not in ok]
print(f"Ya entrenados: {len(ok)} | Faltantes: {len(faltan)}")
print(f"Faltantes: {', '.join(faltan)}")

for i, par in enumerate(faltan, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(faltan)}] {par}")
    print("=" * 60, flush=True)
    subprocess.run([sys.executable, "train_seq_save.py", "--par", par, "--semillas", "5"])
