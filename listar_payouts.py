# listar_payouts.py - Conecta a IQ Option (DEMO) y lista los activos de binarias
# abiertos con su payout, separando pares REALES vs OTC. No opera.
#
#   .venv314\Scripts\python.exe listar_payouts.py
import json
import sys

from iqoptionapi.stable_api import IQ_Option

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)

api = IQ_Option(cfg["email"], cfg["password"])
print("Conectando a IQ Option...")
ok, reason = api.connect()
if not ok:
    print(f"NO CONECTO: {reason}")
    sys.exit(1)
api.change_balance("PRACTICE")
print(f"Conectado (DEMO). Balance: {api.get_balance()}\n")

# Activos abiertos por tipo de instrumento
open_time = api.get_all_open_time()
# Payouts (profit) por activo para binary y turbo
profits = api.get_all_profit()

def be(p):  # break-even WR dado payout p (fraccion)
    return 1.0 / (1.0 + p) * 100 if p else 0


for instrumento in ("turbo", "binary"):
    if instrumento not in open_time:
        continue
    print(f"===== {instrumento.upper()} =====")
    filas = []
    for activo, info in open_time[instrumento].items():
        if not info.get("open"):
            continue
        # payout: profits[activo][instrumento] suele ser fraccion (0.0-1.0)
        pinfo = profits.get(activo, {})
        p = pinfo.get(instrumento)
        es_otc = "OTC" in activo.upper()
        filas.append((activo, es_otc, p))
    # primero reales, luego OTC, ordenado por payout desc
    filas.sort(key=lambda x: (x[1], -(x[2] or 0)))
    print(f"  {'ACTIVO':18} {'TIPO':6} {'PAYOUT':>7} {'BREAK-EVEN WR':>14}")
    for activo, es_otc, p in filas:
        tipo = "OTC" if es_otc else "REAL"
        if p is None:
            print(f"  {activo:18} {tipo:6} {'?':>7}")
        else:
            print(f"  {activo:18} {tipo:6} {p*100:6.1f}% {be(p):13.1f}%")
    reales = [f for f in filas if not f[1] and f[2]]
    if reales:
        mejor = max(reales, key=lambda x: x[2])
        print(f"  -> mejor REAL: {mejor[0]} payout {mejor[2]*100:.1f}% (break-even {be(mejor[2]):.1f}% WR)")
    print()
