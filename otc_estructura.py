# otc_estructura.py - Busca ESTRUCTURA PROFUNDA en el feed sintetico OTC (que la autocorrelacion
# simple no ve): exponente de Hurst (reversion/tendencia), variance-ratio a horizontes largos,
# periodicidad via FFT, y test directo de reversion a la media. Compara OTC vs REAL como control.
#   python otc_estructura.py

import json, glob, os, math
import numpy as np

BE = 1.0 / 1.87


def cargar(patron_otc):
    files = glob.glob("cache_ohlc/*.json")
    files = [f for f in files if (("-OTC" in os.path.basename(f)) == patron_otc)]
    out = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
            c = np.asarray(d["close"], float)
        except Exception:
            continue
        if len(c) > 3000:
            out.append(c)
    return out


def hurst(ts):
    # via variance-ratio de retornos a distintos lags -> pendiente en log-log
    r = np.diff(np.log(np.maximum(ts, 1e-12)))
    lags = [2, 4, 8, 16, 32, 64]
    tau = []
    for k in lags:
        m = len(r) // k
        if m < 20:
            return np.nan
        blk = r[:m * k].reshape(m, k).sum(1)
        tau.append(np.sqrt(blk.var()))
    lags = np.array(lags[:len(tau)], float); tau = np.array(tau)
    if (tau <= 0).any():
        return np.nan
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]   # 0.5=random, <0.5 revierte, >0.5 tiende


def vr_curve(ts, qs=(2, 5, 10, 20, 50)):
    r = np.diff(np.log(np.maximum(ts, 1e-12)))
    v1 = r.var()
    out = []
    for q in qs:
        m = len(r) // q
        if m < 20 or v1 == 0:
            out.append(np.nan); continue
        blk = r[:m * q].reshape(m, q).sum(1)
        out.append(blk.var() / (q * v1))
    return out


def fft_peak(ts):
    # detrend y buscar frecuencia dominante en el precio
    x = ts - np.linspace(ts[0], ts[-1], len(ts))
    x = x - x.mean()
    n = len(x)
    sp = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    sp[0] = 0
    if len(sp) < 3:
        return np.nan, np.nan
    k = np.argmax(sp[1:]) + 1
    periodo = 1.0 / freqs[k] if freqs[k] > 0 else np.nan   # en barras
    # ratio del pico vs energia media (cuan dominante)
    dominancia = sp[k] / (sp[1:].mean() + 1e-12)
    return periodo, dominancia


def test_reversion_media(series, N=20, H=2):
    # ¿si el precio esta lejos de su SMA(N), revierte? entrada OPERABLE (+1)
    tr = [0, 0]
    for c in series:
        n = len(c)
        cs = np.cumsum(c); sma = np.full(n, np.nan)
        sma[N:] = (cs[N:] - cs[:-N]) / N
        z = (c - sma)
        for i in range(N + 2, n - 2 - H):
            if not np.isfinite(z[i]):
                continue
            # lejos por encima -> PUT (revierte abajo); lejos por debajo -> CALL
            std = c[i - N:i].std()
            if std == 0:
                continue
            zz = z[i] / std
            if zz > 2:
                lado = "put"
            elif zz < -2:
                lado = "call"
            else:
                continue
            gg = (c[i + 1 + H] > c[i + 1]) if lado == "call" else (c[i + 1 + H] < c[i + 1])
            tr[0] += 1; tr[1] += int(gg)
    return tr[1] / tr[0] * 100 if tr[0] else 0, tr[0]


def analizar(series, etq):
    H = [hurst(c) for c in series]
    H = [x for x in H if not np.isnan(x)]
    VR = np.array([vr_curve(c) for c in series], float)
    per = []; dom = []
    for c in series:
        p, dd = fft_peak(c)
        if not np.isnan(p):
            per.append(p); dom.append(dd)
    rev_wr, rev_n = test_reversion_media(series)
    print(f"\n[{etq}]  ({len(series)} pares)")
    print(f"  Hurst medio: {np.mean(H):.3f}   (0.50=random walk; <0.45 revierte; >0.55 tiende)")
    print(f"  Variance ratio a q=2,5,10,20,50: " + " ".join(f"{np.nanmean(VR[:,i]):.3f}" for i in range(VR.shape[1])))
    print(f"  FFT dominancia media del pico: {np.mean(dom):.1f}x  (1x=ruido blanco; alto=periodicidad)")
    print(f"  Test reversion a SMA20 (|z|>2), OPERABLE: {rev_wr:.2f}% en {rev_n} ops  (BE {BE*100:.1f}%)")
    return np.mean(H), rev_wr


def main():
    print("=" * 80)
    print("ESTRUCTURA PROFUNDA — OTC (sintetico) vs REAL (control)")
    print("=" * 80)
    otc = cargar(True); real = cargar(False)
    ho, ro = analizar(otc, "OTC sintetico")
    hr, rr = analizar(real, "REAL (control)")
    print("\n" + "=" * 80)
    print("VEREDICTO:")
    if abs(ho - 0.5) < 0.03 and ro < BE * 100 + 0.5:
        print(f"  OTC: Hurst {ho:.3f} ~ 0.5 y reversion {ro:.1f}% < BE -> random walk sin estructura explotable.")
    else:
        print(f"  OTC muestra desviacion (Hurst {ho:.3f}, reversion {ro:.1f}%) -> investigar mas.")
    print(f"  (real de control: Hurst {hr:.3f}, reversion {rr:.1f}%)")


if __name__ == "__main__":
    main()
