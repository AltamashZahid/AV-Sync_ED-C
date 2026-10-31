"""
SyncNet baseline on exactly the same tests as scripts/bench.py (same clips, same 11 shifts,
same numpy audio shift, same metrics).

    python -B scripts/bench_syncnet.py --set dev
    python -B scripts/bench_syncnet.py --set heldout

Video embeddings are cached per clip (work/syncnet_cache); the GPU is used if present.
Per-test rows go to results/tests_<set>_syncnet.csv (for scripts/coverage_curve.py).
Note SyncNet's resolution is 1 frame = 40 ms (25 fps).
"""
import argparse
import csv
import glob
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import bench  # noqa: E402
from avsync import syncnet_baseline as sn  # noqa: E402
from avsync.audio import load_audio  # noqa: E402

MIN_CONF = 3.0          # usual SyncNet rule of thumb for "this is a reliable talking face"


def cached_vfeat(clip, set_name):
    d = os.path.join(ROOT, "work", "syncnet_cache", set_name)
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, os.path.relpath(clip, bench.SETS[set_name]).replace(os.sep, "__") + ".npy")
    if os.path.exists(f):
        return np.load(f)
    v = sn.video_embeddings(sn.face_crops(clip))
    np.save(f, v)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(bench.SETS), default="dev")
    p.add_argument("--fine", action="store_true", help="shifts between frames (bench.FINE_SHIFTS)")
    a = p.parse_args()
    if a.fine:
        bench.SHIFTS = bench.FINE_SHIFTS
    clips = sorted(glob.glob(os.path.join(bench.SETS[a.set], "**", "*.mp4"), recursive=True))
    rows = []
    for c in clips:
        _, _, found, frames = bench.cached_lips(c, a.set)     # same clip filter as bench.py
        if found < 0.5 * frames:
            continue
        v = cached_vfeat(c, a.set)
        y = (np.clip(load_audio(c), -1, 1) * 32767).astype(np.int16)
        res = {}
        for s in bench.SHIFTS:
            n = int(round(16000 * abs(s) / 1000))
            ys = np.concatenate([np.zeros(n, np.int16), y]) if s >= 0 else y[n:]
            res[s] = sn.offset_from_embeddings(v, sn.audio_embeddings(ys, len(v)))
        for s in bench.SHIFTS:
            e = res[s]["offset_ms"] - res[0]["offset_ms"] - s
            rows.append({"clip": os.path.basename(c), "group": bench.group_of(c, a.set), "shift_ms": s,
                         "error_ms": e, "confidence": min(res[s]["confidence"], res[0]["confidence"]),
                         "natural_ms": res[0]["offset_ms"]})
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"tests_{a.set}_syncnet{'_fine' if a.fine else ''}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    err = np.abs([r["error_ms"] for r in rows])
    res_ = -np.array([r["error_ms"] for r in rows])
    conf = np.array([r["confidence"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    ok = conf >= MIN_CONF
    groups = sorted(set(grp))
    print(f"{a.set}: {len(rows) // len(bench.SHIFTS)} clips x {len(bench.SHIFTS)} shifts")
    print(f"SyncNet: MAE {err.mean():.1f} ms, median {np.median(err):.1f} ms, <=40ms {np.mean(err <= 40) * 100:.0f}%, "
          f"ITU ok {np.mean((res_ >= -45) & (res_ <= 125)) * 100:.0f}%, covered {ok.mean() * 100:.0f}%, "
          f"conf<=40 {np.mean(err[ok] <= 40) * 100 if ok.any() else float('nan'):.0f}%")
    if len(groups) > 1:
        print("  per group <=40ms: " + ", ".join(f"{g} {np.mean(err[grp == g] <= 40) * 100:.0f}%" for g in groups))
    nat = {g: [r["natural_ms"] for r in rows if r["group"] == g and r["shift_ms"] == 0] for g in groups}
    print("  natural offset: " + ", ".join(f"{g} median {np.median(v):+.0f} ms" for g, v in nat.items()))
    print("rows ->", out)


if __name__ == "__main__":
    main()
