"""
Offset estimation: when the mouth opens, the sound gets louder. Put both signals on the
same 10 ms grid, slide the audio against the lips, and take the shift where they match best.

Two signal pairs are matched and their match curves averaged (v4, chosen with scripts/bench.py):
  - mouth opening (MAR)          vs  speech-band energy (300-3400 Hz)
  - mouth OPENING speed (dMAR+)  vs  sound onsets (onset strength)
Speech is rhythmic, so a single pair sometimes also "matches" one syllable-beat away; the two
pairs make different mistakes, and averaging cancels most of them. Both signals are partly
whitened first (PHAT-beta, beta = 0.3), which sharpens the peak.

Sign convention: offset > 0  ->  the audio is LATE (heard after the lips move).
"""
import numpy as np

GRID_S = 0.010
PAIRS = [("mar", "band"), ("mar_open_vel", "onset")]
WHITEN = 0.3


def _fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN (frames without a face)."""
    x = np.asarray(x, float).copy()
    bad = np.isnan(x)
    if bad.all():
        raise ValueError("no face found in any frame")
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _znorm(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x * 0.0


def lip_signals(lips) -> dict:
    """lips: a MAR array, or a dict of lip tracks (mar/area/width). Adds the opening speed
    (positive part of the derivative) of each track as <name>_open_vel."""
    if not isinstance(lips, dict):
        lips = {"mar": lips}
    out = {}
    for name, x in lips.items():
        x = _fill_gaps(x)
        out[name] = x
        out[name + "_open_vel"] = np.clip(np.gradient(x), 0, None)
    return out


def _whiten(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """PHAT-style whitening (Knapp & Carter 1976): flatten the spectrum so every rhythm
    frequency counts equally -> a sharper correlation peak. beta < 1 = partial (PHAT-beta);
    beta = 1 is full PHAT, which on 3-5 s clips mostly amplifies noise (see README)."""
    X = np.fft.rfft(x - x.mean())
    mag = np.abs(X)
    return np.fft.irfft(X / (mag + 1e-3 * mag.max()) ** beta, n=len(x))


def _corr_curve(v_t, lip, a_t, aud, max_lag_s, smooth_s, whiten=False):
    """Correlation for every lag on the 10 ms grid. Fixed-length lip window (the middle part)
    compared with the audio at every shift — with the full overlap instead, big shifts can win
    on a short overlap by chance."""
    start, end = max(v_t[0], a_t[0]), min(v_t[-1], a_t[-1])
    grid = np.arange(start, end, GRID_S)
    max_lag = int(round(max_lag_s / GRID_S))
    if len(grid) < 2 * max_lag + 50:
        raise ValueError(f"video too short for a ±{max_lag_s:.1f} s search")
    m = np.interp(grid, v_t, lip)
    e = np.interp(grid, a_t, aud)
    k = max(1, int(round(smooth_s / GRID_S)))
    kernel = np.ones(k) / k
    m = np.convolve(m, kernel, mode="same")
    e = np.convolve(e, kernel, mode="same")
    if whiten:                                   # True = full PHAT, a number = PHAT-beta
        m, e = _whiten(m, float(whiten)), _whiten(e, float(whiten))
    m, e = _znorm(m), _znorm(e)

    lags = np.arange(-max_lag, max_lag + 1)
    w0, w1 = max_lag, len(m) - max_lag
    a = m[w0:w1]
    # audio sample at t+L lines up with lip sample at t  ->  L > 0 means audio is late
    corr = np.array([np.corrcoef(a, e[w0 + L:w1 + L])[0, 1] for L in lags])
    return lags, np.nan_to_num(corr, nan=-1.0)


def _peak(lags, corr, allowed=None):
    """allowed: optional boolean mask of lags the peak may be taken from (hybrid search)."""
    best = int(np.argmax(np.where(allowed, corr, -np.inf))) if allowed is not None else int(np.argmax(corr))
    shift = 0.0                      # sub-sample refinement: parabola through the peak
    if 0 < best < len(corr) - 1:
        y0, y1, y2 = corr[best - 1], corr[best], corr[best + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
    offset_ms = (lags[best] + shift) * GRID_S * 1000.0
    # confidence: how far the best match stands out from all other shifts
    others = np.delete(corr, slice(max(0, best - 5), best + 6))
    confidence = float((corr[best] - np.median(others)) / (others.std() + 1e-9)) if len(others) else 0.0
    return {"offset_ms": float(offset_ms), "confidence": confidence, "peak_corr": float(corr[best]),
            "at_search_edge": best in (0, len(corr) - 1),
            "lags_ms": lags * GRID_S * 1000.0, "corr": corr}


def estimate_offset(v_t, lip, a_t, aud, max_lag_s: float = 0.5, smooth_s: float = 0.04,
                    whiten: bool = False):
    """One signal pair (lip, audio). Returns offset_ms, confidence, peak_corr, lags_ms, corr."""
    lags, corr = _corr_curve(v_t, _fill_gaps(lip), a_t, aud, max_lag_s, smooth_s, whiten)
    return _peak(lags, corr)


def estimate_offset_multi(v_t, lips, a_t, audio: dict, pairs=PAIRS,
                          max_lag_s: float = 0.5, smooth_s: float = 0.04, whiten=WHITEN,
                          near_ms: float = None, window_ms: float = 60.0):
    """Main method: average the match curves of several (lip, audio) pairs, then take the peak.
    lips: MAR array or dict of lip tracks (see lip_signals).
    near_ms: hybrid mode - take the peak only within +-window_ms of a coarse estimate (e.g.
    SyncNet's, 40 ms steps), so a wrong syllable-beat peak far away can't win."""
    lips = lip_signals(lips)
    curves = [_corr_curve(v_t, lips[ln], a_t, audio[an], max_lag_s, smooth_s, whiten)
              for ln, an in pairs]
    lags = curves[0][0]
    allowed = None
    if near_ms is not None:
        allowed = np.abs(lags * GRID_S * 1000.0 - near_ms) <= window_ms
    result = _peak(lags, np.mean([c for _, c in curves], axis=0), allowed)
    result["pairs"] = [f"{ln}/{an}" for ln, an in pairs]
    return result
