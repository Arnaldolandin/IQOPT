import json
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
offset = cfg["filtro_hora"]["timezone_offset"]
hpp = cfg["filtro_hora"]["horas_por_par"]
hpp_local = {}
for par, horas in hpp.items():
    hpp_local[par] = sorted([(h + offset) % 24 for h in horas])
cfg["filtro_hora"]["horas_por_par"] = hpp_local
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("Horas convertidas de UTC a hora local (Chile UTC-4)")
