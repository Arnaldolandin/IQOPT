# lead_lag.py - Prediccion CROSS-ASSET: ¿el retorno de un par LIDER predice el del REZAGADO
# en la barra siguiente? Alinea por timestamp comun, mide correlacion cruzada a lag +1 y
# testea si operar el rezagado en la direccion del lider supera break-even.
#   python lead_lag.py

import json, glob, os, math
import numpy as np

CACHE = "cache_ohlc_5m"
BE = 1.0 / 1.87


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json")))
             if "-OTC" not in os.path.basename(f)]
    series = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float); t = np.asarray(d["times"], float)
        except Exception:
            continue
        if len(c) > 3000:
            series[os.path.basename(f)[:-5]] = (t, c)
    nombres = list(series.keys())
    print(f"Pares: {len(nombres)} | analisis lead-lag a lag +1 barra (5m)")

    # retornos por par indexados por timestamp
    ret = {}
    for name, (t, c) in series.items():
        r = np.zeros(len(c)); r[1:] = np.diff(np.log(np.maximum(c, 1e-12)))
        ret[name] = dict(zip(t.astype(np.int64), r))

    resultados = []
    N = len(nombres)
    for i in range(N):
        A = nombres[i]; ta, ca = series[A]
        ra_map = ret[A]
        for j in range(N):
            if i == j:
                continue
            B = nombres[j]
            rb_map = ret[B]
            # timestamps comunes de B; predecir r_B(t) con r_A(t-300s) (A lidera 1 barra=300s)
            xs = []; ys = []
            for tb in rb_map:
                ta_prev = tb - 300
                if ta_prev in ra_map:
                    xs.append(ra_map[ta_prev]); ys.append(rb_map[tb])
            if len(xs) < 2000:
                continue
            xs = np.array(xs); ys = np.array(ys)
            # correlacion cruzada lag+1
            xa = xs - xs.mean(); yb = ys - ys.mean()
            den = math.sqrt((xa * xa).sum() * (yb * yb).sum())
            cc = (xa * yb).sum() / den if den > 0 else 0
            # direccion: signo del lider predice signo del rezagado?
            s = (np.sign(xs) == np.sign(ys))
            v = (np.sign(xs) != 0) & (np.sign(ys) != 0)
            acc = s[v].mean() * 100 if v.sum() else 0
            resultados.append((A, B, cc, acc, int(v.sum())))

    resultados.sort(key=lambda r: abs(r[2]), reverse=True)
    print("\nTOP 15 relaciones lead-lag por |correlacion cruzada| (lag+1):")
    print(f"{'LIDER':>10} -> {'REZAGADO':>10} | {'crossCorr':>9} {'dirAcc':>7} {'n':>7}")
    print("-" * 60)
    for A, B, cc, acc, n in resultados[:15]:
        mk = "  <-- >BE" if acc > BE * 100 else ""
        print(f"{A:>10} -> {B:>10} | {cc:>+9.4f} {acc:>6.2f}% {n:>7}{mk}")

    # mejor por accuracy direccional
    resultados.sort(key=lambda r: r[3], reverse=True)
    print("\nTOP 10 por accuracy direccional (el signo del lider predice el rezagado):")
    print(f"{'LIDER':>10} -> {'REZAGADO':>10} | {'dirAcc':>7} {'crossCorr':>9} {'n':>7}")
    print("-" * 60)
    for A, B, cc, acc, n in resultados[:10]:
        mk = "  <-- >BE 53.5%" if acc > BE * 100 else ""
        print(f"{A:>10} -> {B:>10} | {acc:>6.2f}% {cc:>+9.4f} {n:>7}{mk}")

    mejor_acc = resultados[0][3] if resultados else 0
    mejor_cc = max(abs(r[2]) for r in resultados) if resultados else 0
    print("\n" + "=" * 60)
    print(f"Mejor accuracy direccional cross-asset: {mejor_acc:.2f}% (BE {BE*100:.1f}%)")
    print(f"Mayor |correlacion cruzada lag+1|: {mejor_cc:.4f}")
    if mejor_acc <= BE * 100 + 0.3:
        print("  Ninguna relacion lider->rezagado supera break-even. Sin edge cross-asset a 5m.")
    print("  (Nota: el lead-lag real vive a nivel de tick/segundos; a 5m los pares ya se movieron juntos.)")


if __name__ == "__main__":
    main()
