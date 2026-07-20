# meta_train_npz.py - Entrena el meta-modelo desde las features cacheadas por
# meta_feats_cache.py y guarda models/meta_bbrev_iq.pkl.
#
# Equivale a meta_train_iq.py (mismas PARAMS, misma validacion OOF, mismo pickle), pero
# lee de cache_feats/ en vez de recomputar ~50 min de features: los procesos largos mueren
# en este entorno y meta_train_iq.py no es reanudable.
#
# Para reproducir la configuracion de PRODUCCION hay que haber cacheado con --combo --gap0
# (2do primario stoch + etiqueta desde la vela de decision).
#   python meta_feats_cache.py cache_ohlc_5m --combo --gap0
#   python meta_train_npz.py cache_ohlc_5m_COMBO_G0
import os, sys, glob, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
PREF = sys.argv[1] if len(sys.argv) > 1 else "cache_ohlc_5m_COMBO_G0"
DEST = "models/meta_bbrev_iq.pkl"
BE = 0.532
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def main():
    fs = sorted(glob.glob(os.path.join("cache_feats", f"{PREF}__*.npz")))
    if not fs:
        print(f"[ERROR] no hay cache_feats/{PREF}__*.npz. Corre antes:")
        print(f"        python meta_feats_cache.py <cache> --combo --gap0")
        return
    ts, Xs, ys = [], [], []
    for f in fs:
        d = np.load(f)
        if len(d["y"]) == 0:
            continue
        ts.append(d["t"]); Xs.append(d["X"]); ys.append(d["y"])
    t = np.concatenate(ts); X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(int)
    o = np.argsort(t, kind="stable"); X, y = X[o], y[o]     # cronologico
    print(f"{PREF}: {len(fs)} activos | {len(y)} senales | WR base {y.mean()*100:.1f}% "
          f"(BE {BE*100:.1f}%)")

    oof = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); yo, po = y[mask], oof[mask]

    print("\n=== Validacion out-of-fold ===")
    print(f"  baseline: WR {yo.mean()*100:.1f}% n={len(yo)} EV {ev(yo.mean())*100:+.1f}%")
    min_n = max(150, int(0.05 * len(yo))); best = None
    for thr in np.arange(0.50, 0.70, 0.01):
        mm = po >= thr; n = int(mm.sum())
        if n < min_n:
            continue
        wr = yo[mm].mean()
        if best is None or ev(wr) > best[1]:
            best = (round(float(thr), 2), ev(wr), wr, n)
    for tt in (0.54, 0.56, 0.58, 0.60, 0.64):
        mm = po >= tt; n = int(mm.sum())
        if n:
            wr = yo[mm].mean()
            print(f"    P>={tt:.2f}: WR {wr*100:.1f}% n={n} EV {ev(wr)*100:+.1f}%"
                  f"{'  (n bajo)' if n < 300 else ''}")
    if best:
        w, e, wr, n = best
        print(f"  mejor EV: P>={w} -> WR {wr*100:.1f}% n={n} ({n/len(yo)*100:.0f}%) EV {e*100:+.1f}%")

    os.makedirs("models", exist_ok=True)
    final = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
    with open(DEST, "wb") as f:
        pickle.dump(final, f)
    print(f"\n[SAVE] {DEST}  (features SIN look-ahead MTF)")
    print(f"[RECOMENDACION] bb_ml_threshold: {best[0] if best else 0.60}")
    print("Recordatorio: el WF limpio deja esto en break-even. Reentrenar arregla la")
    print("coherencia train/produccion, no crea rentabilidad. Mantener en DEMO.")


if __name__ == "__main__":
    main()
