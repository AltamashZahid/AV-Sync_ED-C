"""
Coverage vs accuracy: let each method answer only when its confidence is above a threshold,
and sweep the threshold. Coverage = share of tests answered; accuracy = share of answered
tests within 40 ms. The confidence scales of the methods differ (our z-score vs SyncNet's
distance margin), so methods are compared at EQUAL COVERAGE, not at equal threshold.

    python -B scripts/coverage_curve.py --set heldout
Reads results/tests_<set>_*.csv (written by bench.py / bench_syncnet.py).
"""
import argparse
import csv
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

# label, csv slug, default threshold, colour (validated categorical slots 1-3, light surface)
METHODS = [("Ours v3", "v3_mar_logrms_mar_vel_onset", 3.0, "#2a78d6"),
           ("Ours v4", "mar_band_mar_vel_onset_phat_0_3", 3.0, "#eb6834"),
           ("SyncNet", "syncnet", 3.0, "#1baf7a")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(set_name, slug):
    with open(os.path.join(ROOT, "results", f"tests_{set_name}_{slug}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return (np.abs([float(r["error_ms"]) for r in rows]), np.array([float(r["confidence"]) for r in rows]),
            np.array([r["group"] for r in rows]))


def curve(err, conf):
    order = np.argsort(-conf)                      # most confident first
    ok = (err[order] <= 40).astype(float)
    k = np.arange(1, len(ok) + 1)
    return k / len(ok) * 100, np.cumsum(ok) / k * 100, conf[order]


def at_coverage(cov, acc, c):
    return acc[min(np.searchsorted(cov, c), len(acc) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="heldout")
    p.add_argument("--font", type=float, default=10, help="base font size (bigger for slides)")
    p.add_argument("--exclude", nargs="*", default=[], help="groups to leave out, e.g. reenacted")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": a.font})
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    marks = [25, 50, 75, 90, 100]
    print(f"{a.set}{' without ' + ','.join(a.exclude) if a.exclude else ''}: accuracy (<=40 ms) at coverage")
    print(f"{'method':10} " + " ".join(f"{m:>6}%" for m in marks) + "   at conf>=3")
    ends = []
    for label, slug, thr, color in METHODS:
        err, conf, grp = load(a.set, slug)
        keep = ~np.isin(grp, a.exclude)
        cov, acc, cs = curve(err[keep], conf[keep])
        ax.plot(cov, acc, color=color, lw=2, label=label, solid_capstyle="round")
        i = max(np.searchsorted(-cs, -thr, side="right") - 1, 0)        # last point with conf >= thr
        ax.plot(cov[i], acc[i], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ends.append((acc[-1], label, color))
        print(f"{label:10} " + " ".join(f"{at_coverage(cov, acc, m):>6.1f}%" for m in marks)
              + f"   covered {cov[i]:.0f}%, acc {acc[i]:.1f}%")
    # direct labels at the right end, nudged apart so they never collide
    ends.sort()
    last, step = -1e9, 2.2 * a.font / 10
    for y, label, color in ends:
        y = max(y, last + step)
        last = y
        ax.annotate(label, (100, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=a.font - 1, annotation_clip=False)
    ax.set_xlim(0, 100)
    lo = min(80, ax.get_ylim()[0])
    ax.set_ylim(lo, 101)
    ax.set_xlabel("Coverage: share of tests the method answers (%)", color=INK2)
    ax.set_ylabel("Accuracy: within 40 ms (%)", color=INK2)
    title = "Answer less, be right more — held-out clips" if a.set == "heldout" else f"Coverage vs accuracy — {a.set}"
    if a.exclude:
        title += f" (without {', '.join(a.exclude)})"
    slide = a.font != 10                    # slide version: the slide carries title and note
    if not slide:
        ax.set_title(title, color=INK, loc="left", fontsize=a.font + 1)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    if not slide:
        ax.text(0, -0.2, "Dot = default threshold (confidence ≥ 3). Lines: 297 tests (27 clips × 11 shifts).",
                transform=ax.transAxes, color=INK2, fontsize=a.font - 2)
    fig.tight_layout()
    suffix = ("_no_" + "_".join(a.exclude) if a.exclude else "") + ("_large" if a.font != 10 else "")
    out = os.path.join(ROOT, "results", f"coverage_{a.set}{suffix}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("plot ->", out)


if __name__ == "__main__":
    main()
