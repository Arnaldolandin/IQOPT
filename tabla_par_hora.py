# tabla_par_hora.py - Tabla de horas rentables por par
import json

with open("analisis_par_hora.json", encoding="utf-8") as f:
    data = json.load(f)

por_par = data["por_par"]
for par in sorted(por_par.keys()):
    rentables = [c for c in por_par[par] if c["ev"] > 0 and c["tr"] >= 10]
    if not rentables:
        continue
    rentables.sort(key=lambda x: -x["ev"])
    print(f"\n{par}")
    for c in rentables:
        print(f"  hora {c['hora']:2d}  WR {c['wr']*100:5.1f}%  EV {c['ev']:+.4f}  tr={c['tr']}")
