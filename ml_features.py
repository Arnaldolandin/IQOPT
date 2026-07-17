import numpy as np

FEATURE_NAMES = [
    "adx", "di_plus", "di_minus", "rsi", "atr_pct",
    "ema9", "ema21", "ema50", "ema9_pct", "ema21_pct",
    "stoch_k", "stoch_d",
    "macd", "macd_signal", "macd_hist",
    "bb_upper_pct", "bb_lower_pct", "bb_width",
    "roc_3", "roc_5",
    "hour_sin", "hour_cos",
    "consecutive",
    "pullback",
    "pivot_dist",
    "vol_regime",
    "session_asia",
    "session_europe",
    "roc_1",
    "roc_8",
    "rsi_mtf",
    "adx_mtf",
]


def _ema(data, periodo):
    alpha = 2 / (periodo + 1)
    ema = np.mean(data[:periodo])
    for p in data[periodo:]:
        ema = (p - ema) * alpha + ema
    return ema


def _rsi(velas, periodo=14):
    if len(velas) < periodo + 2:
        return None
    closes = np.array([v[4] for v in velas])
    diffs = np.diff(closes)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = np.mean(gains[-periodo:])
    avg_loss = np.mean(losses[-periodo:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(velas, periodo=14):
    if len(velas) < periodo + 1:
        return None
    trs = []
    for i in range(1, len(velas)):
        h, l, pc = velas[i][2], velas[i][3], velas[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return np.mean(trs[-periodo:])


def _adx(velas, periodo=14):
    if len(velas) < periodo * 2 + 1:
        return None, None, None
    n = len(velas)
    tr = np.zeros(n)
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    for i in range(1, n):
        h, l, pc = velas[i][2], velas[i][3], velas[i - 1][4]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up_move[i] = velas[i][2] - velas[i - 1][2]
        down_move[i] = velas[i - 1][3] - velas[i][3]
    atr_vals = np.zeros(n)
    atr_vals[periodo] = np.mean(tr[1:periodo + 1])
    for i in range(periodo + 1, n):
        atr_vals[i] = (atr_vals[i - 1] * (periodo - 1) + tr[i]) / periodo
    plus = np.zeros(n)
    minus = np.zeros(n)
    for i in range(1, n):
        plus[i] = max(up_move[i], 0) / max(tr[i], 1e-10) * 100
        minus[i] = max(down_move[i], 0) / max(tr[i], 1e-10) * 100
    di_plus_vals = np.zeros(n)
    di_minus_vals = np.zeros(n)
    di_plus_vals[periodo] = np.mean(plus[1:periodo + 1])
    di_minus_vals[periodo] = np.mean(minus[1:periodo + 1])
    for i in range(periodo + 1, n):
        di_plus_vals[i] = (di_plus_vals[i - 1] * (periodo - 1) + plus[i]) / periodo
        di_minus_vals[i] = (di_minus_vals[i - 1] * (periodo - 1) + minus[i]) / periodo
    dx = np.abs(di_plus_vals - di_minus_vals) / np.maximum(di_plus_vals + di_minus_vals, 1e-10) * 100
    adx_vals = np.zeros(n)
    idx = periodo * 2
    if idx >= n:
        return None, None, None
    adx_vals[idx] = np.mean(dx[periodo:idx])
    for i in range(idx + 1, n):
        adx_vals[i] = (adx_vals[i - 1] * (periodo - 1) + dx[i]) / periodo
    return float(adx_vals[-1]), float(di_plus_vals[-1]), float(di_minus_vals[-1])


def _stoch(velas, k_period=14, d_period=3):
    if len(velas) < k_period + d_period:
        return None, None
    closes = np.array([v[4] for v in velas])
    highs = np.array([v[2] for v in velas])
    lows = np.array([v[3] for v in velas])
    k_vals = []
    for i in range(k_period - 1, len(velas)):
        hh = np.max(highs[i - k_period + 1:i + 1])
        ll = np.min(lows[i - k_period + 1:i + 1])
        if hh - ll < 1e-10:
            k_vals.append(50)
        else:
            k_vals.append((closes[i] - ll) / (hh - ll) * 100)
    if len(k_vals) < d_period:
        return None, None
    k = k_vals[-1]
    d = np.mean(k_vals[-d_period:])
    return float(k), float(d)


def _macd(velas, fast=12, slow=26, signal=9):
    if len(velas) < slow + signal:
        return None, None, None
    closes = [v[4] for v in velas]
    n = len(closes)

    def _ema_series(data, periodo):
        alpha = 2 / (periodo + 1)
        out = np.full(len(data), np.nan)
        out[periodo - 1] = np.mean(data[:periodo])
        for i in range(periodo, len(data)):
            out[i] = (data[i] - out[i - 1]) * alpha + out[i - 1]
        return out

    ema_f = _ema_series(closes, fast)
    ema_s = _ema_series(closes, slow)
    macd_line = ema_f - ema_s
    current_macd = float(macd_line[-1])

    macd_vals = macd_line[~np.isnan(macd_line)]
    if len(macd_vals) < signal:
        return current_macd, None, None

    ema_signal_vals = _ema_series(macd_vals, signal)
    signal_line = float(ema_signal_vals[-1])
    hist = current_macd - signal_line
    return current_macd, signal_line, hist


def _bb(velas, periodo=20, desviacion=2.0):
    if len(velas) < periodo:
        return None, None, None
    closes = [v[4] for v in velas]
    recent = closes[-periodo:]
    mid = np.mean(recent)
    std = np.std(recent, ddof=1)
    upper = mid + desviacion * std
    lower = mid - desviacion * std
    return float(upper), float(mid), float(lower)


def extract_features(velas, velas_mtf=None):
    n = len(velas)
    if n < 60:
        return np.array([]), 0

    closes = [v[4] for v in velas]
    epoch = velas[-1][0]
    dt = np.datetime64(int(epoch), "s")
    hour = int((dt - dt.astype("datetime64[D]")) / np.timedelta64(1, "h"))

    px = float(closes[-1])
    if px <= 0:
        return np.array([]), 0

    adx, di_plus, di_minus = _adx(velas)
    rsi = _rsi(velas)
    atr = _atr(velas)
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    ema50 = _ema(closes, 50)
    stoch_k, stoch_d = _stoch(velas)
    macd_line, macd_signal, macd_hist = _macd(velas)
    bb_upper, bb_mid, bb_lower = _bb(velas)
    roc_1 = ((closes[-1] - closes[-2]) / closes[-2]) if n >= 2 else 0
    roc_3 = ((closes[-1] - closes[-4]) / closes[-4]) if n >= 4 else 0
    roc_5 = ((closes[-1] - closes[-6]) / closes[-6]) if n >= 6 else 0
    roc_8 = ((closes[-1] - closes[-9]) / closes[-9]) if n >= 9 else 0

    consecutive = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if consecutive < 0: break
            consecutive += 1
        elif closes[i] < closes[i - 1]:
            if consecutive > 0: break
            consecutive -= 1
        else:
            break

    atr_safe = max(atr, 1e-10) if atr and atr > 0 else 1e-10
    recent_10_c = len(closes) if len(closes) < 10 else 10
    half = recent_10_c // 2
    if half < 1:
        pullback = 0
    else:
        rc = closes[-recent_10_c:]
        trend_dir = np.mean(rc[half:]) - np.mean(rc[:half])
        highs_n = max(v[2] for v in velas[-recent_10_c:])
        lows_n = min(v[3] for v in velas[-recent_10_c:])
        if trend_dir > 0:
            pullback = (highs_n - closes[-1]) / atr_safe
        else:
            pullback = (closes[-1] - lows_n) / atr_safe

    pivot_dist = min(max(v[2] for v in velas[-10:]) - px, px - min(v[3] for v in velas[-10:])) / atr_safe

    trs = []
    for i in range(1, len(velas)):
        h, l, pc = velas[i][2], velas[i][3], velas[i - 1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) >= 50:
        short_tr = np.mean(trs[-14:])
        long_tr = np.mean(trs[-50:])
        vol_regime = short_tr / max(long_tr, 1e-10) - 1
    else:
        vol_regime = 0

    features = {}
    features["adx"] = adx if adx is not None else 0
    features["di_plus"] = di_plus if di_plus is not None else 0
    features["di_minus"] = di_minus if di_minus is not None else 0
    features["rsi"] = rsi if rsi is not None else 50
    features["atr_pct"] = atr / px if atr and atr > 0 else 0

    features["ema9"] = ema9 / px if ema9 else 1
    features["ema21"] = ema21 / px if ema21 else 1
    features["ema50"] = ema50 / px if ema50 else 1
    features["ema9_pct"] = (px - ema9) / ema9 if ema9 else 0
    features["ema21_pct"] = (px - ema21) / ema21 if ema21 else 0

    features["stoch_k"] = stoch_k if stoch_k is not None else 50
    features["stoch_d"] = stoch_d if stoch_d is not None else 50

    features["macd"] = macd_line if macd_line is not None else 0
    features["macd_signal"] = macd_signal if macd_signal is not None else 0
    features["macd_hist"] = macd_hist if macd_hist is not None else 0

    if bb_upper and bb_lower and bb_upper > bb_lower:
        features["bb_upper_pct"] = (bb_upper - px) / (bb_upper - bb_lower)
        features["bb_lower_pct"] = (px - bb_lower) / (bb_upper - bb_lower)
        features["bb_width"] = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
    else:
        features["bb_upper_pct"] = 0
        features["bb_lower_pct"] = 0
        features["bb_width"] = 0

    features["roc_1"] = roc_1
    features["roc_3"] = roc_3
    features["roc_5"] = roc_5
    features["roc_8"] = roc_8

    features["consecutive"] = float(consecutive)
    features["pullback"] = float(pullback)
    features["pivot_dist"] = float(pivot_dist)
    features["vol_regime"] = float(vol_regime)
    hour_rad = 2 * np.pi * hour / 24
    features["hour_sin"] = np.sin(hour_rad)
    features["hour_cos"] = np.cos(hour_rad)

    features["session_asia"] = 1.0 if 0 <= hour < 8 else 0.0
    features["session_europe"] = 1.0 if 8 <= hour < 16 else 0.0

    if velas_mtf is not None and len(velas_mtf) >= 30:
        rsi_mtf = _rsi(velas_mtf)
        adx_mtf, _, _ = _adx(velas_mtf)
        features["rsi_mtf"] = rsi_mtf if rsi_mtf is not None else 50
        features["adx_mtf"] = adx_mtf if adx_mtf is not None else 0
    else:
        features["rsi_mtf"] = 50
        features["adx_mtf"] = 0

    vec = np.array([features[n] for n in FEATURE_NAMES], dtype=np.float64)
    return vec, px


def build_dataset(velas, horizon=3, velas_mtf=None):
    rows, targets, prices = [], [], []
    n = len(velas)
    for i in range(60, n - horizon):
        chunk = velas[:i + 1]
        if velas_mtf is not None:
            chunk_epoch = chunk[-1][0]
            mtf_idx = sum(1 for v in velas_mtf if v[0] <= chunk_epoch)
            chunk_mtf = velas_mtf[:mtf_idx] if mtf_idx >= 2 else None
        else:
            chunk_mtf = None
        fut_close = float(velas[i + horizon][4])
        curr_close = float(velas[i][4])
        if curr_close <= 0:
            continue
        feat_vec, _ = extract_features(chunk, velas_mtf=chunk_mtf)
        if len(feat_vec) == 0:
            continue
        target = 1 if fut_close > curr_close else 0
        rows.append(feat_vec)
        targets.append(target)
        prices.append(curr_close)
    if not rows:
        return np.array([]), np.array([]), np.array([])
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.int32), np.array(prices)
