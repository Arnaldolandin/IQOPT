# -*- coding: utf-8 -*-
"""Reentrena los 18 modelos de ACCIONES sin volumen y exporta los .npz que usa el bot.

POR QUE
-------
IQ dejo de enviar `volume` en las acciones: desde 2026-03 la mayoria y 100% cero desde
2026-05 (verificado en `cache_ohlc_5m_v2` y en vivo con get_candles). Los modelos se
entrenaron con el volumen historico real -- `seq_lstm_APPLE.pt.json` tiene corte
2026-03-31, o sea datos que hasta febrero traian volumen -- y en produccion reciben esas
dos columnas a CERO. main.py hace `float(v.get("volume", 0) or 0)`, que con la clave a 0
produce una lista de ceros de la longitud correcta: pasa el `len(vol)==n` de
ventana_features() y las features mueren sin ningun error. El fallo en silencio.

Reentrenar con --sin-volumen alinea el entrenamiento con lo que hay en vivo. NO es una
fuente de edge: es higiene. El muro (BE 53.48%) sigue donde estaba.

DETALLE QUE IMPORTA: el bot NO usa los .pt
------------------------------------------
seq_model.predecir_p() prefiere, en este orden: ensemble `{base}_s1..s9.npz` -> `{base}.npz`
-> torch. Los 18 pares tienen ensemble de 5 semillas en .npz, asi que reentrenar solo el
.pt no cambiaria NADA en produccion. Por eso aqui se exporta cada semilla a .npz.

Se escribe todo en `models_nuevos/` y NO se toca `models/`: el bot esta vivo y leyendo de
ahi. El intercambio es un paso aparte y manual, con el bot parado.
"""
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(AQUI, ".venv314", "Scripts", "python.exe")
DEST = os.path.join(AQUI, "models_nuevos")
SEMILLAS = 5          # los 18 pares corren hoy con ensemble de 5 semillas

ACCIONES = ["AIG", "ALIBABA", "AMAZON", "APPLE", "BAIDU", "CISCO", "CITI", "COKE",
            "FACEBOOK", "GOOGLE", "GS", "INTEL", "JPM", "MCDON", "MORSTAN", "MSFT",
            "NIKE", "TESLA"]


def main():
    solo = sys.argv[1:] or ACCIONES
    os.makedirs(DEST, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # dejar aire de CPU al bot: el escaneo de 49 pares ya arrastra hasta 100 s de retraso
    # y saturar la maquina 2 h lo empeoraria justo mientras opera
    env.setdefault("OMP_NUM_THREADS", "4")

    t0 = time.time()
    hechos, fallidos = [], []
    for i, par in enumerate(solo, 1):
        salida = os.path.join(DEST, f"seq_lstm_{par}.pt")
        # ya entrenado (p.ej. el piloto): no repetir
        if all(os.path.isfile(salida.replace(".pt", f"_s{k}.pt")) for k in range(1, SEMILLAS + 1)):
            print(f"\n[{i}/{len(solo)}] {par}: ya estaba entrenado, salto", flush=True)
            hechos.append(par)
            continue
        print(f"\n{'='*60}\n[{i}/{len(solo)}] ENTRENANDO {par} "
              f"({(time.time()-t0)/60:.0f} min transcurridos)\n{'='*60}", flush=True)
        r = subprocess.run([PY, "train_seq_save.py", "--par", par,
                            "--semillas", str(SEMILLAS), "--sin-volumen",
                            "--salida", salida], cwd=AQUI, env=env)
        (hechos if r.returncode == 0 else fallidos).append(par)

    print(f"\n{'='*60}\nEXPORTANDO .npz (es lo que el bot lee de verdad)\n{'='*60}", flush=True)
    n_ok = n_ko = 0
    for par in hechos:
        for k in range(1, SEMILLAS + 1):
            pt = os.path.join(DEST, f"seq_lstm_{par}_s{k}.pt")
            if not os.path.isfile(pt):
                continue
            r = subprocess.run([PY, "exportar_npz.py", pt], cwd=AQUI, env=env,
                               capture_output=True, text=True)
            # exportar_npz VERIFICA numpy vs torch y sale != 0 si no coinciden;
            # un .npz que no coincide con el .pt validado seria operar a ciegas
            if r.returncode == 0:
                n_ok += 1
            else:
                n_ko += 1
                print(f"  [FALLO] {par} s{k}: {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}")
        print(f"  {par}: exportado", flush=True)

    print(f"\n{'='*60}\nRESUMEN ({(time.time()-t0)/60:.0f} min)\n{'='*60}")
    print(f"  entrenados : {len(hechos)}/{len(solo)}")
    print(f"  .npz OK    : {n_ok}   fallidos: {n_ko}")
    if fallidos:
        print(f"  FALLARON   : {', '.join(fallidos)}")
    print(f"\nTodo en {DEST}. models/ NO se ha tocado: el bot sigue con los viejos.")
    print("Para activar: parar el bot, mover los archivos, arrancar el watchdog.")


if __name__ == "__main__":
    main()
