"""
Evaluation with synthetic ground truth: take videos that are in sync, shift their audio by
known amounts, and check how close the detected offset is.

    python -B scripts/evaluate.py                       # all videos in data/clips
    python -B scripts/evaluate.py --shifts -200 0 200   # custom shifts (ms)

Writes results/eval_<time>.csv, a summary table and a plot.
"Raw error" assumes each original clip is perfectly in sync. "Calibrated error" first measures
the clip's own natural lip-to-sound lead (shift 0) and subtracts it — what matters for
correction is detecting the CHANGE in sync.
"""
import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from avsync.correct import correct_sync, shift_audio  # noqa: E402
from avsync.pipeline import detect  # noqa: E402
from avsync.visual import lip_track  # noqa: E402

DEFAULT_SHIFTS = [-300, -200, -120, -80, -40, 0, 40, 80, 120, 200, 300]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"))
    p.add_argument("--shifts", type=int, nargs="+", default=DEFAULT_SHIFTS)
    p.add_argument("--max-lag", type=float, default=0.5)
    p.add_argument("--keep", action="store_true", help="keep the shifted/corrected videos in work/")
    a = p.parse_args()

    clips = sorted(glob.glob(os.path.join(a.clips, "**", "*.mp4"), recursive=True))
    if not clips:
        sys.exit(f"No .mp4 clips in {a.clips}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(ROOT, "work", stamp)
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for clip in clips:
        name = os.path.splitext(os.path.basename(clip))[0]
        t = time.time()
        visual = lip_track(clip)                      # slow part, done once per clip
        base = None
        for s in sorted(a.shifts, key=lambda x: x != 0):    # shift 0 first (calibration)
            shifted = os.path.join(work, f"{name}_shift{s:+d}.mp4")
            shift_audio(clip, s / 1000.0, shifted)
            r = detect(shifted, max_lag_s=a.max_lag, visual=visual)
            if s == 0:
                base = r["offset_ms"]
            # correction: undo the detected change. The true shift is known, so the sync left
            # after correction is exact (re-detecting with the same detector would just agree
            # with its own mistakes).
            applied = r["offset_ms"] - base
            fixed = os.path.join(work, f"{name}_shift{s:+d}_fixed.mp4")
            correct_sync(shifted, applied, fixed)
            if not a.keep:                       # ~35 MB per clip and shift
                os.remove(shifted)
                os.remove(fixed)
            rows.append({"clip": name, "true_shift_ms": s, "detected_ms": round(r["offset_ms"], 1),
                         "raw_error_ms": round(r["offset_ms"] - s, 1),
                         "calibrated_error_ms": round(r["offset_ms"] - base - s, 1),
                         "after_correction_ms": round(s - applied, 1),
                         "confidence": round(r["confidence"], 2), "peak_corr": round(r["peak_corr"], 3),
                         "face_frames": f"{visual[2]['face_found']}/{visual[2]['frames']}"})
        print(f"{name}: natural offset {base:+.0f} ms, done in {time.time() - t:.0f}s")

    csv_path = os.path.join(out_dir, f"eval_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary per shift
    print(f"\n{'shift':>7} {'MAE raw':>9} {'MAE calib':>10} {'within 40ms':>12} {'after fix':>10}")
    lines = []
    for s in sorted(a.shifts):
        rs = [r for r in rows if r["true_shift_ms"] == s]
        raw = np.mean([abs(r["raw_error_ms"]) for r in rs])
        cal = np.mean([abs(r["calibrated_error_ms"]) for r in rs])
        ok = np.mean([abs(r["calibrated_error_ms"]) <= 40 for r in rs]) * 100
        fix = np.mean([abs(r["after_correction_ms"]) for r in rs])
        line = f"{s:>+7d} {raw:>8.1f}ms {cal:>9.1f}ms {ok:>11.0f}% {fix:>9.1f}ms"
        print(line)
        lines.append(line)
    allcal = [abs(r["calibrated_error_ms"]) for r in rows]
    total = (f"\nAll: calibrated MAE {np.mean(allcal):.1f} ms, median {np.median(allcal):.1f} ms, "
             f"within 40 ms {np.mean(np.array(allcal) <= 40) * 100:.0f}% "
             f"({len(clips)} clips x {len(a.shifts)} shifts)")
    print(total)
    with open(os.path.join(out_dir, f"eval_{stamp}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("shift   MAE raw   MAE calib  within 40ms  after fix\n" + "\n".join(lines) + total + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [r["true_shift_ms"] for r in rows]
        base_by_clip = {r["clip"]: r["detected_ms"] for r in rows if r["true_shift_ms"] == 0}
        ys = [r["detected_ms"] - base_by_clip[r["clip"]] for r in rows]
        ax.scatter(xs, ys, alpha=0.6, label="detected (calibrated)")
        lim = max(a.shifts) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="perfect")
        ax.set_xlabel("true audio shift (ms)")
        ax.set_ylabel("detected shift (ms)")
        ax.set_title("AV offset detection")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"eval_{stamp}.png"), dpi=130)
    except Exception as e:                                   # plot is optional
        print("plot skipped:", e)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
