"""
Drift: an offset that GROWS over time, e.g. a video written at the wrong frame rate
(LivePortrait writes int(29.97) = 29 fps -> 3.2% slow -> lips fall ~32 ms/s behind the voice).

Model:  offset(t) = offset0 + drift * t        (drift in ms per second of video)
A wrong frame rate scales the video clock: speed = 1 + drift / 1000, and re-timing the video
frames by that speed removes the drift (correct.retime_video).

Two estimators:
  windows  - offset in short sliding windows, then a robust (outlier-rejecting) line fit
  grid     - search speed x offset together; keep the speed whose correlation peak is best
"""
import numpy as np

from .sync import GRID_S, PAIRS, estimate_offset_multi, lip_signals, _fill_gaps, _znorm


def _grid_signals(v_t, lips, a_t, audio, pairs, smooth_s=0.04):
    """All pairs on one shared 10 ms grid (full length, smoothed, not yet normalised)."""
    start, end = max(v_t[0], a_t[0]), min(v_t[-1], a_t[-1])
    grid = np.arange(start, end, GRID_S)
    k = max(1, int(round(smooth_s / GRID_S)))
    kernel = np.ones(k) / k
    L = lip_signals(lips)
    out = []
    for ln, an in pairs:
        m = np.convolve(np.interp(grid, v_t, _fill_gaps(L[ln])), kernel, mode="same")
        e = np.convolve(np.interp(grid, a_t, audio[an]), kernel, mode="same")
        out.append((m, e))
    return grid, out


def local_offsets(v_t, lips, a_t, audio, win_s=1.6, hop_s=0.2, max_lag_s=0.3, pairs=PAIRS):
    """Offset (ms) in sliding windows. Returns times_s (window centres), offsets_ms, peak corr."""
    grid, sig = _grid_signals(v_t, lips, a_t, audio, pairs)
    w, hop, ml = int(win_s / GRID_S), int(hop_s / GRID_S), int(max_lag_s / GRID_S)
    lags = np.arange(-ml, ml + 1)
    cs, offs, peaks = [], [], []
    for s0 in range(ml, len(grid) - w - ml + 1, hop):
        curve = np.zeros(len(lags))
        for m, e in sig:
            a = _znorm(m[s0:s0 + w])
            curve += [np.dot(a, _znorm(e[s0 + L:s0 + L + w])) / w for L in lags]
        curve /= len(sig)
        b = int(np.argmax(curve))
        frac = 0.0
        if 0 < b < len(curve) - 1:
            y0, y1, y2 = curve[b - 1], curve[b], curve[b + 1]
            d = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0
        cs.append(grid[s0 + w // 2])
        offs.append((lags[b] + frac) * GRID_S * 1000)
        peaks.append(curve[b])
    return np.array(cs), np.array(offs), np.array(peaks)


def _robust_line(t, y, w, iters=4, tol_ms=50.0):
    """Weighted least squares, dropping points more than tol_ms off the line (syllable slips)."""
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        if keep.sum() < 3:
            break
        A = np.vstack([np.ones(keep.sum()), t[keep]]).T
        W = np.sqrt(np.clip(w[keep], 1e-3, None))
        coef, *_ = np.linalg.lstsq(A * W[:, None], y[keep] * W, rcond=None)
        new = np.abs(y - (coef[0] + coef[1] * t)) <= tol_ms
        if (new == keep).all():
            break
        keep = new
    A = np.vstack([np.ones(keep.sum()), t[keep]]).T
    coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None) if keep.sum() >= 2 else (np.array([np.median(y), 0.0]),)
    return float(coef[0]), float(coef[1]), keep


def estimate_drift_windows(v_t, lips, a_t, audio, **kw):
    t, off, peak = local_offsets(v_t, lips, a_t, audio, **kw)
    if len(t) < 3:
        raise ValueError("video too short for drift estimation")
    off0, drift_ms_per_ms, keep = _robust_line(t, off, peak)
    drift = drift_ms_per_ms                     # ms of offset per s of video
    return {"method": "windows", "drift_ms_per_s": drift, "offset0_ms": off0,
            "speed": 1 + drift / 1000.0, "inliers": int(keep.sum()), "windows": len(t),
            "t": t, "offsets_ms": off, "peaks": peak}


def estimate_drift_grid(v_t, lips, a_t, audio, speeds=None, **kw):
    """Re-time the lip clock by each candidate speed; keep the one with the best peak."""
    if speeds is None:
        speeds = np.round(np.arange(0.95, 1.0501, 0.0025), 4)
    best = None
    for s in speeds:
        r = estimate_offset_multi(v_t * s, lips, a_t, audio, **kw)
        if best is None or r["peak_corr"] > best[1]["peak_corr"]:
            best = (s, r)
    s, r = best
    return {"method": "grid", "speed": float(s), "drift_ms_per_s": (s - 1) * 1000.0,
            "offset0_ms": r["offset_ms"], "peak_corr": r["peak_corr"], "confidence": r["confidence"]}


def estimate_drift_reference(v_t, lips, ref_t, ref_lips, track="mar", max_lag_s=0.3,
                             speeds=None):
    """Lip-to-lip: compare a generated/dubbed video with the video it was made from (e.g. the
    reenacted video vs the therapist's driving video). Both show the SAME mouth motion, so the
    match is far sharper than lips-vs-audio and works on short clips.
    Returns the speed that maps this video's clock onto the reference's, and the lag (ms;
    > 0 = this video's lips are later than the reference)."""
    if speeds is None:
        speeds = np.round(np.arange(0.95, 1.0501, 0.0005), 4)
    x = _fill_gaps(lips[track] if isinstance(lips, dict) else lips)
    r = _fill_gaps(ref_lips[track] if isinstance(ref_lips, dict) else ref_lips)
    ml = int(max_lag_s / GRID_S)
    best = (-2.0, 1.0, 0)
    for s in speeds:
        tt = v_t * s
        start, end = max(tt[0], ref_t[0]) + max_lag_s, min(tt[-1], ref_t[-1]) - max_lag_s
        if end - start < 1.0:
            continue
        g = np.arange(start, end, GRID_S)
        a = _znorm(np.interp(g, tt, x))
        for L in range(-ml, ml + 1):
            c = float(np.dot(a, _znorm(np.interp(g + L * GRID_S, ref_t, r))) / len(g))
            if c > best[0]:
                best = (c, s, L)
    c, s, L = best
    return {"method": "reference", "speed": float(s), "drift_ms_per_s": (s - 1) * 1000.0,
            "lag_ms": -L * GRID_S * 1000.0, "corr": c}
