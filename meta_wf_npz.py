# meta_wf_npz.py - Walk-forward de umbral FIJO sobre las features ya cacheadas por
# meta_feats_cache.py. Segundos en vez de 45 min, asi que permite iterar.
#   python meta_wf_npz.py [prefijo_cache]      (default cache_ohlc_5m)
# Responde: la meseta 0.54-0.58 aguanta fuera de muestra, y en QUE activos vive.
import os, sys, glob
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
PREF = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m"
PERMUTAR = "--permuta" in sys.argv             # control negativo
BE = 0.532; UMBRALES = [0.54, 0.56, 0.58]; NFOLDS = 5
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def cargar():
    ts, Xs, ys, fz, tg = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join("cache_feats", f"{PREF}__*.npz"))):
        d = np.load(f); tag = os.path.basename(f).split("__")[1][:-4]
        if len(d["y"]) == 0:
            continue
        ts.append(d["t"]); Xs.append(d["X"]); ys.append(d["y"]); fz.append(d["fz"])
        tg.append(np.full(len(d["y"]), tag))
    t = np.concatenate(ts); X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
    f = np.concatenate(fz).astype(int); g = np.concatenate(tg)
    o = np.argsort(t, kind="stable")           # cronologico global (clave para el WF)
    return t[o], X[o], y[o], f[o], g[o]


def stats(wr, n):
    se = (wr * (1 - wr) / n) ** 0.5
    return se, (wr - BE) / se if se > 0 else 0.0


def main():
    t, X, y, fz, tags = cargar()
    n = len(y)
    if PERMUTAR:
        # Control negativo: baraja las etiquetas en bloques de 500 (mantiene la estructura
        # temporal gruesa, destruye la relacion features->resultado). Si aqui sale un edge,
        # el montaje del walk-forward esta roto, no hay nada que interpretar.
        rng = np.random.default_rng(42); y = y.copy()
        for a in range(0, n, 500):
            b = min(a + 500, n); rng.shuffle(y[a:b])
        print("[CONTROL NEGATIVO] etiquetas permutadas por bloques -> se espera ~50%\n")
    print(f"WF UMBRAL FIJO sobre features cacheadas | {PREF} | {len(set(tags))} activos")
    print(f"{n} senales | BE {BE*100:.1f}% | base {y.mean()*100:.2f}% | "
          f"congeladas (fut==base) {fz.mean()*100:.2f}%\n")

    P = np.full(n, np.nan)
    for k in range(NFOLDS):
        tr1 = int(n * (0.40 + 0.10 * k)); te1 = int(n * (0.50 + 0.10 * k))
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[:tr1], y[:tr1])
        P[tr1:te1] = m.predict_proba(X[tr1:te1])[:, 1]
        print(f"  fold {k+1}: train {tr1} -> test [{tr1}:{te1}] base {y[tr1:te1].mean()*100:.2f}%", flush=True)
    ev_m = ~np.isnan(P)

    for u in UMBRALES:
        sel = ev_m & (P >= u)
        if not sel.sum():
            print(f"\n=== umbral {u} === sin senales"); continue
        wr = float(y[sel].mean()); nn = int(sel.sum()); se, z = stats(wr, nn)
        print(f"\n=== umbral {u} ===")
        print(f"  GLOBAL  {wr*100:.2f}% +-{se*100:.2f} ({nn} ops) EV {ev(wr)*100:+.1f}% "
              f"z={z:.2f} {'SIGNIFICATIVO' if z > 1.64 else 'no signif.'}")
        # sin velas congeladas: si el edge se cae aqui, era artefacto de feed parado
        s2 = sel & (fz == 0)
        if s2.sum():
            wr2 = float(y[s2].mean()); se2, z2 = stats(wr2, int(s2.sum()))
            print(f"  sin congeladas {wr2*100:.2f}% +-{se2*100:.2f} ({int(s2.sum())} ops) "
                  f"EV {ev(wr2)*100:+.1f}% z={z2:.2f}")
        # por fold, para ver si es consistente o lo carga una sola ventana
        linea = []
        for k in range(NFOLDS):
            a = int(n * (0.40 + 0.10 * k)); b = int(n * (0.50 + 0.10 * k))
            s3 = np.zeros(n, bool); s3[a:b] = True; s3 &= sel
            linea.append(f"f{k+1} {y[s3].mean()*100:.1f}%({int(s3.sum())})" if s3.sum() else f"f{k+1} -")
        print("  por fold: " + " | ".join(linea))

    u = 0.56; sel = ev_m & (P >= u)
    print(f"\n[POR ACTIVO] umbral {u}, min 50 ops")
    filas = []
    for tg in sorted(set(tags)):
        s = sel & (tags == tg)
        if s.sum() >= 50:
            wr = float(y[s].mean()); _, z = stats(wr, int(s.sum()))
            filas.append((wr, int(s.sum()), tg, z))
    for wr, nn, tg, z in sorted(filas, reverse=True):
        print(f"  {tg:<10} {wr*100:6.2f}% {nn:5d} ops EV {ev(wr)*100:+6.1f}% z={z:5.2f}")
    if filas:
        gan = sum(1 for w, _, _, _ in filas if w > BE)
        print(f"  -> {gan}/{len(filas)} activos por encima de BE "
              f"(si fuera ruido puro se esperaria ~{len(filas)//2})")


if __name__ == "__main__":
    main()
