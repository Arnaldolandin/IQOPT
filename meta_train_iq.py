# meta_train_iq.py - Porta el META-LABELING de Deriv a IQ Option.
# Primario: bbrev (reversion Bollinger 2sigma) genera la senal (CALL/PUT).
# Meta (este): HistGradientBoosting predice P(la senal GANA) sobre 32 features
# (ml_features.extract_features, el mismo de Deriv) y filtra las de baja calidad.
# Datos: cache_ohlc_5m (IQ, 5m). Payout IQ ~88% -> break-even 53.2%.
# Valida OOF (TimeSeriesSplit). Guarda models/meta_bbrev_iq.pkl y recomienda umbral.
#   .venv314\Scripts\python.exe meta_train_iq.py
import os, json, glob, pickle, warnings
import numpy as np
from ml_features import extract_features, mtf_hasta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

CACHE = "cache_ohlc_5m"
NCON = 2          # velas de holding (expiry 10m / tf 5m) - config IQ
K = 2.0           # sigma de bbrev (DEBE coincidir con bb_std del bot)
PERIOD = 20
BE = 0.532        # break-even IQ (payout ~88%)
def ev(wr): return 1.88 * wr - 1
PARAMS = dict(max_iter=250, learning_rate=0.03, max_depth=4, l2_regularization=2.0,
              min_samples_leaf=40, random_state=42)


def cargar_velas(d):
    """arrays IQ -> lista [epoch, o, h, l, c] (formato de ml_features)."""
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    t = d.get("times", list(range(len(c))))
    return [[float(t[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i])] for i in range(len(c))]


def build_signals(V):
    # COMBO mr_combo: bbrev (Bollinger 2sigma) primero; si no dispara, stoch (%K extremo).
    closes = [v[4] for v in V]; highs = [v[2] for v in V]; lows = [v[3] for v in V]
    N = len(V); out = []
    for i in range(max(PERIOD, 60), N - NCON):
        w = closes[i - PERIOD + 1:i + 1]; sma = np.mean(w); sd = np.std(w)
        if sd <= 0:
            continue
        z = (closes[i] - sma) / sd
        side = "CALL" if z <= -K else "PUT" if z >= K else None
        if side is None:                                  # stoch como 2do primario
            pk = 14; hh = max(highs[i - pk + 1:i + 1]); ll = min(lows[i - pk + 1:i + 1])
            kk = 100 * (closes[i] - ll) / (hh - ll) if hh > ll else 50.0
            side = "CALL" if kk < 20 else "PUT" if kk > 80 else None
        if side is None:
            continue
        win = V[max(0, i - 99):i + 1]; ep = win[-1][0]
        # MTF anclado a la vela de decision i (ver mtf_hasta: antes se seleccionaba por
        # el INICIO de la barra y colaba las 2 velas siguientes -> look-ahead).
        cmtf = mtf_hasta(V[max(0, i - 179):i + 1], factor=3, max_barras=60)
        cmtf = cmtf if len(cmtf) >= 2 else None
        fv, _ = extract_features(win, velas_mtf=cmtf)
        if len(fv) == 0:
            continue
        won = int(closes[i + NCON] > closes[i]) if side == "CALL" else int(closes[i + NCON] < closes[i])
        out.append((ep, fv, won))
    return out


def main():
    files = [f for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))) if "-OTC" not in os.path.basename(f)]
    if not files:
        print(f"[ERROR] No hay datos en '{CACHE}/'. Ese cache es gitignored: NO viene con git pull.")
        print(f"        - Para OPERAR no necesitas reentrenar: models/meta_bbrev_iq.pkl ya esta commiteado.")
        print(f"        - Si igual queres reentrenar aca, baja los datos primero:")
        print(f"              python download_ohlc_5m.py     (baja ~6 meses de 5m a {CACHE}/)")
        print(f"          y despues:  python meta_train_iq.py")
        return
    rows = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if len(d.get("close", [])) < 400:
            continue
        V = cargar_velas(d)
        sig = build_signals(V)
        rows.extend(sig)
        print(f"{os.path.basename(f)[:-5]}: {len(V)} velas -> {len(sig)} senales bbrev", flush=True)

    rows.sort(key=lambda r: r[0])
    X = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    print(f"\nTotal senales: {len(rows)}  WR base={y.mean()*100:.1f}%  (BE {BE*100:.1f}%)")

    oof = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = HistGradientBoostingClassifier(**PARAMS); m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof); yo, po = y[mask], oof[mask]

    print("\n=== Validacion out-of-fold (meta-labeling en IQ) ===")
    print(f"  baseline (todas): WR {yo.mean()*100:.1f}%  n={len(yo)}  EV {ev(yo.mean())*100:+.1f}%")
    min_n = max(150, int(0.05 * len(yo))); best = None
    for thr in np.arange(0.50, 0.70, 0.01):
        mm = po >= thr; n = int(mm.sum())
        if n < min_n:
            continue
        wr = yo[mm].mean()
        if best is None or ev(wr) > best[1]:
            best = (round(float(thr), 2), ev(wr), wr, n)
    for t in (0.55, 0.60, 0.65):
        mm = po >= t; n = int(mm.sum())
        if n:
            print(f"    P>={t:.2f}: WR {yo[mm].mean()*100:.1f}% n={n} EV {ev(yo[mm].mean())*100:+.1f}%")
    if best:
        w, e, wr, n = best
        print(f"  META-FILTRO P>={w}: WR {wr*100:.1f}%  n={n} ({n/len(yo)*100:.0f}%)  EV {e*100:+.1f}%")

    os.makedirs("models", exist_ok=True)
    final = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
    with open("models/meta_bbrev_iq.pkl", "wb") as f:
        pickle.dump(final, f)
    thr = best[0] if best else 0.60
    print(f"\n[SAVE] models/meta_bbrev_iq.pkl")
    print(f"[RECOMENDACION] bb_ml_threshold: {thr}")
    print(f"Recordatorio: en IQ el meta-labeling suele quedar AL break-even (~54%), no sobre.")


if __name__ == "__main__":
    main()
