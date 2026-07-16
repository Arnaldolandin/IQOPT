# valida_meta.py - Validacion RIGUROSA del meta-labeling con split 3-vias temporal:
#   train (60%) -> ajusta primario + meta
#   val   (20%) -> elige el umbral meta (congela) buscando WR>BE con volumen minimo
#   test  (20%) -> mide ESE umbral en datos ciegos. Si no supera 53.5%, es ruido.
import warnings, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import mejora_prediccion as MP
warnings.filterwarnings("ignore")
BE = 53.5


def main():
    print("Cargando..."); XV, XC, trend, Y, Hs = MP.cargar()
    yg = Y[2]; n = len(XV)
    a = int(n * 0.60); b = int(n * 0.80)   # train / val / test
    print(f"train {a} | val {b-a} | test {n-b} | BE {BE}%\n")

    clf = LogisticRegression(C=1.0, max_iter=1000).fit(XV[:a], yg[:a])
    p_tr = clf.predict_proba(XV[:a])[:, 1]; pred_tr = (p_tr > 0.5).astype(float)
    acierto_tr = (pred_tr == yg[:a]).astype(float)
    Xm_tr = np.column_stack([XC[:a], np.abs(p_tr - 0.5)])
    meta = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4).fit(Xm_tr, acierto_tr)

    def evalu(sl):
        p = clf.predict_proba(XV[sl])[:, 1]; pred = (p > 0.5).astype(float)
        Xm = np.column_stack([XC[sl], np.abs(p - 0.5)]); pm = meta.predict_proba(Xm)[:, 1]
        return pred, yg[sl], pm

    # elegir umbral en VAL: el menor umbral con WR>=BE+0.5 y >=500 ops
    pv, yv, pmv = evalu(slice(a, b))
    elegido = None
    for thr in np.arange(0.50, 0.75, 0.01):
        m = pmv >= thr
        if m.sum() >= 500:
            w = (pv[m] == yv[m]).mean() * 100
            if w >= BE + 0.5:
                elegido = round(float(thr), 2); wv, nv = w, int(m.sum()); break
    print("[VAL] umbral elegido:", elegido)
    if elegido is None:
        print("  ningun umbral supera BE+0.5 con >=500 ops en validacion -> sin edge.")
        # aun asi mostrar el mejor de val como referencia
        best = max(((float(t), (pv[pmv>=t]==yv[pmv>=t]).mean()*100 if (pmv>=t).sum() else 0, int((pmv>=t).sum()))
                    for t in np.arange(0.50,0.70,0.01)), key=lambda z: z[1])
        print(f"  (mejor val: umbral {best[0]:.2f} -> {best[1]:.2f}% en {best[2]} ops)")
        return
    print(f"  en VAL: {wv:.2f}% ({nv} ops)")

    pt, yt, pmt = evalu(slice(b, n))
    m = pmt >= elegido
    wt = (pt[m] == yt[m]).mean() * 100 if m.sum() else 0
    print(f"\n[TEST ciego] umbral {elegido}: {wt:.2f}%  ({int(m.sum())} ops)")
    print(f">>> {'SUPERA' if wt>BE else 'NO supera'} break-even {BE}%  ->",
          "candidato real, validar mas" if wt > BE else "ruido / sobreajuste confirmado")


if __name__ == "__main__":
    main()
