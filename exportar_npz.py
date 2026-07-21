# exportar_npz.py - Exporta un modelo .pt ya entrenado a .npz para inferir sin torch.
#
#   .venv314\Scripts\python.exe exportar_npz.py [models/seq_lstm_EURUSD.pt]
#
# Verifica que la implementacion en numpy coincida con torch antes de dar el .npz por
# bueno: si difirieran, el bot en el servidor operaria con un modelo distinto al
# validado, y no habria forma de notarlo mirando los logs.
import sys

import numpy as np
import torch

import seq_model as S

path = sys.argv[1] if len(sys.argv) > 1 else "models/seq_lstm_EURUSD.pt"

net, L = S.cargar(path)
import json
cfg = json.load(open(path + ".json", encoding="utf-8"))
arq = cfg["arq"]

if not S.exportar_npz(net, arq, path):
    print(f"[ERROR] no se pudo exportar (arq={arq}; solo se soporta lstm)")
    sys.exit(1)
npz = path.replace(".pt", "") + ".npz"
cfg["npz"] = True
json.dump(cfg, open(path + ".json", "w", encoding="utf-8"))
print(f"[OK] {npz}")

# --- verificacion: numpy vs torch sobre entradas aleatorias ---
rng = np.random.default_rng(0)
peor = 0.0
for _ in range(200):
    f = rng.normal(0, 1, (L, S.N_FEATS)).astype(np.float32)
    with torch.no_grad():
        p_t = float(torch.sigmoid(net(torch.tensor(f).unsqueeze(0))).item())
    p_n = S.predecir_npz(f, npz)
    peor = max(peor, abs(p_t - p_n))
print(f"maxima diferencia numpy vs torch en 200 entradas: {peor:.2e}")
if peor > 1e-5:
    print("[ERROR] las implementaciones NO coinciden; no usar el .npz")
    sys.exit(1)
print("[OK] coinciden; el servidor puede correr sin torch")
