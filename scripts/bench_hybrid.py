"""
Hybrid: SyncNet gives the coarse offset (40 ms steps, rarely a wrong syllable), our correlation
refines it on the 10 ms grid, searching only +-WINDOW ms around SyncNet's answer + LEAD.

LEAD: SyncNet learned to ignore the natural lip lead, our curve includes it, so the centre is
moved by LEAD ms. LEAD and WINDOW are chosen on dev (--set dev, prints a small grid), then
run once on held-out with the chosen values (--set heldout).

    python -B scripts/bench_syncnet.py --set dev       # first: SyncNet rows (cached embeddings)
    python -B scripts/bench_hybrid.py --set dev
    python -B scripts/bench_hybrid.py --set heldout
    python -B scripts/bench_hybrid.py --set heldout --fine   # shifts between frames (after
                                                             # bench_syncnet.py --fine)

Same clips, shifts and metrics as bench.py. Confidence = SyncNet's (it decides the region).
Per-test rows go to results/tests_<set>_hybrid.csv (for scripts/coverage_curve.py).
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
from avsync import sync  # noqa: E402
from avsync.audio import HOP_S, features, load_audio, stream_start_gap  # noqa: E402

LEAD_MS, WINDOW_MS = 80.0, 60.0        # chosen on dev (see README)
DEV_GRID = {"lead": [0, 40, 60, 80, 100, 120], "window": [40, 60, 80, 120]}


def syncnet_offsets(set_name, suffix=""):
    """(group, clip) -> {shift: (offset_ms, confidence)} from bench_syncnet.py's rows."""
    out = {}
    with open(os.path.join(ROOT, "results", f"tests_{set_name}_syncnet{suffix}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = int(r["shift_ms"])
            off = float(r["error_ms"]) + float(r["natural_ms"]) + s
            out.setdefault((r["group"], r["clip"]), {})[s] = (off, float(r["confidence"]))
    return out


def score(data, sn, lead, window, hybrid=True):
    rows = []
    for c, g, t, lips, gap, feats in data:
        key = (g, os.path.basename(c))
        off = {}
        for s in bench.SHIFTS:
            f = feats[s]
            a_t = np.arange(len(f["logrms"])) * HOP_S + gap
            near = sn[key][s][0] + lead if hybrid else None
            r = sync.estimate_offset_multi(t, lips, a_t, f, near_ms=near, window_ms=window)
            off[s] = r["offset_ms"]
        for s in bench.SHIFTS:
            rows.append({"clip": key[1], "group": g, "shift_ms": s, "error_ms": off[s] - off[0] - s,
                         "confidence": min(sn[key][s][1], sn[key][0][1]), "natural_ms": off[0]})
    return rows


def summary(rows):
    err = np.abs([r["error_ms"] for r in rows])
    res = -np.array([r["error_ms"] for r in rows])
    ok = np.array([r["confidence"] for r in rows]) >= bench.MIN_CONF
    return (f"MAE {err.mean():5.1f} ms, median {np.median(err):4.1f} ms, <=40ms {np.mean(err <= 40) * 100:3.0f}%, "
            f"<=20ms {np.mean(err <= 20) * 100:3.0f}%, ITU ok {np.mean((res >= -45) & (res <= 125)) * 100:3.0f}%, "
            f"covered {ok.mean() * 100:3.0f}%, conf<=40 {np.mean(err[ok] <= 40) * 100:3.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(bench.SETS), default="dev")
    p.add_argument("--fine", action="store_true", help="shifts between frames (bench.FINE_SHIFTS)")
    a = p.parse_args()
    suffix = "_fine" if a.fine else ""
    if a.fine:
        bench.SHIFTS = bench.FINE_SHIFTS
    sn = syncnet_offsets(a.set, suffix)
    clips = sorted(glob.glob(os.path.join(bench.SETS[a.set], "**", "*.mp4"), recursive=True))
    data = []
    for c in clips:
        t, lips, found, frames = bench.cached_lips(c, a.set)
        if found < 0.5 * frames:
            continue
        y = load_audio(c)
        feats = {s: features(bench.shifted(y, s)) for s in bench.SHIFTS}
        data.append((c, bench.group_of(c, a.set), t, lips, stream_start_gap(c), feats))
    print(f"{a.set}: {len(data)} clips x {len(bench.SHIFTS)} shifts")

    if a.set == "dev" and not a.fine:
        for lead in DEV_GRID["lead"]:
            for window in DEV_GRID["window"]:
                print(f"  lead {lead:3d}  window {window:3d}:  {summary(score(data, sn, lead, window))}")
    rows = score(data, sn, LEAD_MS, WINDOW_MS)
    print(f"\nHybrid (lead {LEAD_MS:.0f}, window {WINDOW_MS:.0f}): {summary(rows)}")
    groups = sorted({r["group"] for r in rows})
    if len(groups) > 1:
        print("  per group <=40ms: " + ", ".join(
            f"{g} {np.mean([abs(r['error_ms']) <= 40 for r in rows if r['group'] == g]) * 100:.0f}%"
            for g in groups))
    with open(os.path.join(ROOT, "results", f"tests_{a.set}_syncnet{suffix}.csv"), encoding="utf-8") as fh:
        other = [{**r, "error_ms": float(r["error_ms"]), "confidence": float(r["confidence"])}
                 for r in csv.DictReader(fh)]
    print(f"  for comparison, SyncNet : {summary(other)}")
    ours = score(data, sn, LEAD_MS, WINDOW_MS, hybrid=False)   # confidence column here = SyncNet's
    err = np.abs([r["error_ms"] for r in ours])
    print(f"  for comparison, ours v4 : MAE {err.mean():5.1f} ms, median {np.median(err):4.1f} ms, "
          f"<=40ms {np.mean(err <= 40) * 100:3.0f}%, <=20ms {np.mean(err <= 20) * 100:3.0f}%")
    out = os.path.join(ROOT, "results", f"tests_{a.set}_hybrid{suffix}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("rows ->", out)


if __name__ == "__main__":
    main()
