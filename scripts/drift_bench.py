"""
Drift (wrong frame rate) detection benchmark.

    python -B scripts/drift_bench.py --set dev
    python -B scripts/drift_bench.py --set heldout
    python -B scripts/drift_bench.py --real       # app's raw LivePortrait outputs (true speed 29/29.97)
    python -B scripts/drift_bench.py --duration --set heldout   # accuracy vs video length

Simulation: a video written at the wrong rate has every frame time scaled, so the cached lip
track's timestamps are divided by `speed` (exact). True drift = (speed - 1) * 1000 ms/s.
Metric: sync left at the END of the clip after correcting with the estimated speed,
|drift error| * duration; "ok" = at most 40 ms (1 frame at 25 fps).
"""
import argparse
import glob
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import bench  # noqa: E402
from avsync import drift  # noqa: E402
from avsync.audio import HOP_S, SR, audio_tracks, features, load_audio  # noqa: E402
from avsync.visual import lip_track  # noqa: E402

SPEEDS = [29 / 29.97, 0.98, 0.99, 1.0, 1.01, 1.02, 29.97 / 29]
REAL = r"C:\Users\ashis\Vid-Re-en APP\backend\data\practice\*\lp_output\*.mp4"   # read only
REAL_DRIVING = r"C:\Users\ashis\Vid-Re-en APP\backend\data\library\en_dogs_are_sitting_by_the__fa7d.mp4"

ESTIMATORS = {
    "windows 1.6s": lambda *a: drift.estimate_drift_windows(*a, win_s=1.6),
    "windows 2.0s": lambda *a: drift.estimate_drift_windows(*a, win_s=2.0),
    "grid": lambda *a: drift.estimate_drift_grid(*a),
}


def run_sim(set_name):
    clips = sorted(glob.glob(os.path.join(bench.SETS[set_name], "**", "*.mp4"), recursive=True))
    data = []
    for c in clips:
        t, lips, found, frames = bench.cached_lips(c, set_name)
        if found < 0.5 * frames:
            continue
        a_t, au = audio_tracks(c)
        data.append((c, bench.group_of(c, set_name), t, lips, a_t, au))
    groups = sorted({d[1] for d in data})
    print(f"{set_name}: {len(data)} clips x {len(SPEEDS)} speeds\n")
    print(f"{'estimator':14} {'MAE drift':>10} {'end err':>8} {'ok<=40ms':>9}   "
          + "  ".join(f"{g[:9]:>9}" for g in groups))
    for name, est in ESTIMATORS.items():
        errs, ends, grp = [], [], []
        for c, g, t, lips, a_t, au in data:
            for s in SPEEDS:
                tt = t / s
                try:
                    r = est(tt, lips, a_t, au)
                    e = r["drift_ms_per_s"] - (s - 1) * 1000
                except ValueError:
                    e = np.nan
                errs.append(abs(e))
                ends.append(abs(e) * tt[-1])
                grp.append(g)
        errs, ends, grp = np.array(errs), np.array(ends), np.array(grp)
        ok = np.nan_to_num(ends, nan=1e9) <= 40
        print(f"{name:14} {np.nanmean(errs):>6.1f}ms/s {np.nanmedian(ends):>6.0f}ms {ok.mean() * 100:>8.0f}%   "
              + "  ".join(f"{ok[grp == g].mean() * 100:>8.0f}%" for g in groups))


def long_recording(set_name, groups=None):
    """Join in-sync clips into one long in-sync recording (lips and audio cut to the same
    length per clip, then placed one after another). Returns v_t, lips, a_t, audio."""
    clips = sorted(glob.glob(os.path.join(bench.SETS[set_name], "**", "*.mp4"), recursive=True))
    ts, ls, ys, t0 = [], {k: [] for k in ("mar", "area", "width")}, [], 0.0
    for c in clips:
        if groups and bench.group_of(c, set_name) not in groups:
            continue
        t, lips, found, frames = bench.cached_lips(c, set_name)
        if found < 0.9 * frames:
            continue
        y = load_audio(c)
        dur = min(len(y) / SR, t[-1])
        keep = t < dur
        ts.append(t[keep] + t0)
        for k in ls:
            ls[k].append(lips[k][keep])
        ys.append(y[:int(dur * SR)])
        t0 += dur
    y = np.concatenate(ys)
    f = features(y)
    return (np.concatenate(ts), {k: np.concatenate(v) for k, v in ls.items()},
            np.arange(len(f["logrms"])) * HOP_S, f, t0)


def run_duration(set_name, groups):
    v_t, lips, a_t, au, total = long_recording(set_name, groups)
    print(f"{set_name} {groups or ''}: joined recording {total:.0f} s; segments x {len(SPEEDS)} speeds\n")
    print(f"{'length':>6}  " + "  ".join(f"{n:>22}" for n in ESTIMATORS) + "   (MAE ms/s | end err <=40ms)")
    for L in (4, 8, 16, 32):
        if L > total:
            break
        starts = np.arange(0, total - L + 1e-6, max(2.0, (total - L) / 8))
        cells = []
        for name, est in ESTIMATORS.items():
            errs, ok = [], []
            for st in starts:
                vm = (v_t >= st) & (v_t < st + L)
                am = (a_t >= st) & (a_t < st + L)
                seg_l = {k: x[vm] for k, x in lips.items()}
                seg_a = {k: x[am] for k, x in au.items()}
                for s in SPEEDS:
                    tt = (v_t[vm] - st) / s
                    try:
                        r = est(tt, seg_l, a_t[am] - st, seg_a)
                        e = abs(r["drift_ms_per_s"] - (s - 1) * 1000)
                    except ValueError:
                        e = np.inf
                    errs.append(e)
                    ok.append(e * L <= 40)
            errs = np.array(errs)
            cells.append(f"{np.mean(errs[np.isfinite(errs)]):>8.1f} | {np.mean(ok) * 100:>4.0f}%")
        print(f"{L:>5}s  " + "  ".join(f"{c:>22}" for c in cells))


def run_real():
    print(f"Real case: app's raw LivePortrait outputs, true speed {29 / 29.97:.4f} "
          f"(drift {(29 / 29.97 - 1) * 1000:+.1f} ms/s)\n")
    vids = [REAL_DRIVING] + [v for v in sorted(glob.glob(REAL)) if "concat" not in v]
    ref_t, ref_lips, _ = lip_track(REAL_DRIVING)
    for v in vids:
        t, lips, _ = lip_track(v)
        a_t, au = audio_tracks(v)
        label = "therapist (reference)" if v == REAL_DRIVING else v.split(os.sep)[-3]
        line = f"{label:22}"
        for name, est in ESTIMATORS.items():
            r = est(t, lips, a_t, au)
            line += f"  {name}: {r['drift_ms_per_s']:+6.1f}"
        r = drift.estimate_drift_reference(t, lips, ref_t, ref_lips)
        line += (f"  | REFERENCE: {r['drift_ms_per_s']:+6.1f} ms/s (speed {r['speed']:.4f}), "
                 f"lag {r['lag_ms']:+.0f} ms, corr {r['corr']:.2f}")
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(bench.SETS), default="dev")
    p.add_argument("--real", action="store_true")
    p.add_argument("--duration", action="store_true")
    p.add_argument("--groups", nargs="*", help="held-out groups to join, e.g. phone_en phone_hi")
    a = p.parse_args()
    if a.real:
        run_real()
    elif a.duration:
        run_duration(a.set, a.groups)
    else:
        run_sim(a.set)


if __name__ == "__main__":
    main()
