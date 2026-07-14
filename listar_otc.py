# listar_otc.py - Lista todos los pares OTC disponibles
import json
import threading
import time
from iqoptionapi.stable_api import IQ_Option
import iqoptionapi.constants as OP_code

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

api = IQ_Option(cfg["email"], cfg["password"])
ok, reason = api.connect()
if not ok:
    print(f"NO CONECTO: {reason}")
    exit()
api.change_balance("PRACTICE")

# Cargar opcode
done = [False]
def _upd():
    try:
        api.get_ALL_Binary_ACTIVES_OPCODE()
    except:
        pass
    done[0] = True
t = threading.Thread(target=_upd, daemon=True)
t.start()
t.join(timeout=45)

from iqoptionapi.api import OP_code

# Buscar pares con "-otc" en el nombre
otc_pairs = []
for k, v in sorted(OP_code.ACTIVES.items()):
    if "-otc" in k.lower() or "otc" in k.lower():
        otc_pairs.append((k, v))

# Buscar pares que tengan payout OTC
profits = api.get_all_profit()
otc_con_payout = []
for key, info in sorted(profits.items()):
    if isinstance(info, dict):
        otc_val = info.get("otc", info.get("turbo"))
        if otc_val and otc_val > 0 and "otc" in key.lower():
            otc_con_payout.append((key, otc_val))

print(f"\n=== Pares OTC en OP_code.ACTIVES: {len(otc_pairs)} ===")
for k, v in otc_pairs[:30]:
    print(f"  {k}: {v}")

print(f"\n=== Pares OTC con payout: {len(otc_con_payout)} ===")
for k, p in otc_con_payout[:50]:
    print(f"  {k}: {p}")

# Buscar todos los pares binary disponibles
print(f"\n=== Todos los pares binary con payout ===")
all_binary = []
for key, info in sorted(profits.items()):
    if isinstance(info, dict):
        for ptype in ("binary", "turbo", "otc"):
            val = info.get(ptype)
            if val and val > 0:
                all_binary.append((key, ptype, val))
                break

# Separar normales vs OTC
normales = [(k, t, p) for k, t, p in all_binary if "otc" not in k.lower()]
otc = [(k, t, p) for k, t, p in all_binary if "otc" in k.lower()]

print(f"\nNormales: {len(normales)}, OTC: {len(otc)}")
print(f"\n--- OTC con payout ---")
for k, t, p in otc:
    par = k.replace("-op", "")
    print(f"  {par:20s} type={t:8s} payout={p:.0%}")

# Guardar pares OTC limpios
otc_pars = list(set(k.replace("-op", "").replace("-otc", "") for k, t, p in otc))
otc_pars.sort()
print(f"\n=== Pares OTC limpios ({len(otc_pars)}) ===")
for p in otc_pars:
    print(f"  {p}")

with open("otc_pairs.json", "w") as f:
    json.dump(otc_pars, f, indent=2)
print("Guardado en otc_pairs.json")
