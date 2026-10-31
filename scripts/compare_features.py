"""
Quick experiment: which lip / audio features give the most reliable offset?
Lip tracks are computed once and cached (work/mar_cache); the audio shift is simulated on the
energy signal itself (exact, fast), so many variants can be compared in seconds.

    python -B scripts/compare_features.py
"""
import glob
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from avsync import sync  # noqa: E402
from avsync.audio import HOP_S, SR, load_audio  # noqa: E402
from avsync.visual import mar_track  # noqa: E402

SHIFTS = [-300, -200, -120, -80, -40, 0, 40, 80, 120, 200, 300]
CACHE = os.path.join(ROOT, "work", "mar_cache")


def cached_mar(clip):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, os.path.basename(clip) + ".npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["t"], d["mar"]
    t, mar, _ = mar_track(clip)
    np.savez(f, t=t, mar=mar)
    return t, mar


def audio_features(y):
    import librosa
    hop = int(SR * HOP_S)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    n = min(len(rms), len(onset))
    return {"logrms": np.log1p(rms[:n] * 100), "onset": onset[:n]}


def lip_variants(mar):
    m = sync._fill_gaps(mar)
    vel = np.gradient(m)
    return {"mar": m, "mar_open_vel": np.clip(vel, 0, None), "mar_abs_vel": np.abs(vel)}


COMBOS = [("mar", "logrms"), ("mar_open_vel", "onset"), ("mar_abs_vel", "onset"),
          ("mar_open_vel", "logrms"), ("mar", "onset")]


def main():
    clips = sorted(glob.glob(os.path.join(ROOT, "data", "clips", "*.mp4")))
    data = []
    for c in clips:
        t, mar = cached_mar(c)
        data.append((os.path.basename(c), t, lip_variants(mar), load_audio(c)))
    print(f"{len(clips)} clips\n")
    print(f"{'lip feature':14} {'audio feature':10} {'MAE calib':>10} {'within 40ms':>12} {'peak corr':>10}")
    for lip_name, audio_name in COMBOS:
        errs, peaks = [], []
        for name, t, lips, y in data:
            base = None
            for s in sorted(SHIFTS, key=lambda x: x != 0):
                # shift the audio itself: s > 0 -> prepend silence (audio later)
                ys = np.concatenate([np.zeros(int(SR * s / 1000)), y]) if s >= 0 else y[int(SR * -s / 1000):]
                a = audio_features(ys)[audio_name]
                a_t = np.arange(len(a)) * HOP_S
                r = sync.estimate_offset(t, lips[lip_name], a_t, a)
                if s == 0:
                    base = r["offset_ms"]
                errs.append(r["offset_ms"] - base - s)
                peaks.append(r["peak_corr"])
        errs = np.abs(errs)
        print(f"{lip_name:14} {audio_name:10} {errs.mean():>8.1f}ms {np.mean(errs <= 40) * 100:>11.0f}% "
              f"{np.mean(peaks):>10.2f}")

    # ensemble: average the correlation curves of several pairs, then take the peak
    for members in ([("mar", "logrms"), ("mar_open_vel", "onset")],
                    [("mar", "logrms"), ("mar_open_vel", "onset"), ("mar", "onset")]):
        errs = []
        for name, t, lips, y in data:
            base = None
            for s in sorted(SHIFTS, key=lambda x: x != 0):
                ys = np.concatenate([np.zeros(int(SR * s / 1000)), y]) if s >= 0 else y[int(SR * -s / 1000):]
                feats = audio_features(ys)
                a_t = np.arange(len(feats["logrms"])) * HOP_S
                rs = [sync.estimate_offset(t, lips[ln], a_t, feats[an]) for ln, an in members]
                curve = np.mean([r["corr"] for r in rs], axis=0)
                off = rs[0]["lags_ms"][int(np.argmax(curve))]
                if s == 0:
                    base = off
                errs.append(off - base - s)
        errs = np.abs(errs)
        label = " + ".join(f"{ln}/{an}" for ln, an in members)
        print(f"ENSEMBLE {label}: MAE {errs.mean():.1f}ms, within 40ms {np.mean(errs <= 40) * 100:.0f}%")


if __name__ == "__main__":
    main()
