"""
Fast method comparison with a proper dev / held-out split.

    python -B scripts/bench.py --set dev                 # choose methods here (11 RAVDESS clips)
    python -B scripts/bench.py --set heldout --final     # run ONCE on unseen clips

Lip tracks are computed once per clip and cached (work/lip_cache). The audio shift is applied
to the samples in numpy (sample-exact, like ffmpeg's adelay/atrim), so all methods are scored on
exactly the same inputs in seconds. scripts/evaluate.py does the same end-to-end with ffmpeg.

Metrics (calibrated: the clip's own offset at shift 0 is taken as "in sync"):
  MAE        mean |detected change - true shift|
  <=40ms     share within 40 ms (= 1 video frame at 25 fps, the usual criterion, e.g. VocaLiST)
  ITU ok     share where the sync LEFT after correction would not be noticed:
             ITU-R BT.1359 detectability, audio early <= 45 ms and audio late <= 125 ms
  covered    share of tests with confidence >= MIN_CONF (the detector may say "can't tell")
  conf<=40   <=40ms among the covered tests only
"""
import argparse
import csv
import glob
import re
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from avsync import sync  # noqa: E402
from avsync.audio import HOP_S, SR, features, load_audio, stream_start_gap  # noqa: E402
from avsync.visual import lip_track  # noqa: E402

SHIFTS = [-300, -200, -120, -80, -40, 0, 40, 80, 120, 200, 300]
# --fine: shifts between video frames (not multiples of 40 ms) - tests precision, not robustness
FINE_SHIFTS = [-150, -110, -70, -50, -30, -10, 0, 10, 30, 50, 70, 110, 150]
MIN_CONF = 3.0                 # same threshold as pipeline.detect_and_correct (fixed on dev)
SETS = {"dev": os.path.join(ROOT, "data", "clips"), "heldout": os.path.join(ROOT, "data", "heldout")}

# name -> (pairs, whiten). v3 is the method chosen before this script existed.
METHODS = {
    "v3  mar/logrms + mar_vel/onset": ([("mar", "logrms"), ("mar_open_vel", "onset")], False),
    "    mar/band + mar_vel/onset": ([("mar", "band"), ("mar_open_vel", "onset")], False),
    "    area/logrms + area_vel/onset": ([("area", "logrms"), ("area_open_vel", "onset")], False),
    "    area/band + area_vel/onset": ([("area", "band"), ("area_open_vel", "onset")], False),
    "    mar+area 4 pairs (logrms)": ([("mar", "logrms"), ("mar_open_vel", "onset"),
                                       ("area", "logrms"), ("area_open_vel", "onset")], False),
    "    mar+area 4 pairs (band)": ([("mar", "band"), ("mar_open_vel", "onset"),
                                     ("area", "band"), ("area_open_vel", "onset")], False),
    "    v3 + whitening": ([("mar", "logrms"), ("mar_open_vel", "onset")], True),
    "    mar+area 4 pairs (band) + whitening": ([("mar", "band"), ("mar_open_vel", "onset"),
                                                 ("area", "band"), ("area_open_vel", "onset")], True),
    "    mar/band + mar_vel/onset + PHAT-0.3": ([("mar", "band"), ("mar_open_vel", "onset")], 0.3),
    "    mar/band + mar_vel/onset + PHAT-0.5": ([("mar", "band"), ("mar_open_vel", "onset")], 0.5),
}


def cached_lips(clip, set_name):
    d = os.path.join(ROOT, "work", "lip_cache", set_name)
    os.makedirs(d, exist_ok=True)
    rel = os.path.relpath(clip, SETS[set_name]).replace(os.sep, "__")
    f = os.path.join(d, rel + ".npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["t"], {k: z[k] for k in ("mar", "area", "width")}, int(z["found"]), int(z["frames"])
    t, lips, info = lip_track(clip)
    np.savez(f, t=t, found=info["face_found"], frames=info["frames"], **lips)
    return t, lips, info["face_found"], info["frames"]


def shifted(y, s_ms):
    """s > 0: audio later (silence in front); s < 0: audio earlier (start cut)."""
    n = int(round(SR * abs(s_ms) / 1000))
    return np.concatenate([np.zeros(n), y]) if s_ms >= 0 else y[n:]


def group_of(clip, set_name):
    rel = os.path.relpath(clip, SETS[set_name])
    return rel.split(os.sep)[0] if os.sep in rel else set_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=list(SETS), default="dev")
    p.add_argument("--final", action="store_true", help="held-out: only the chosen methods")
    p.add_argument("--methods", nargs="*", help="substring filter on method names")
    a = p.parse_args()

    clips = sorted(glob.glob(os.path.join(SETS[a.set], "**", "*.mp4"), recursive=True))
    methods = METHODS
    if a.methods:
        methods = {k: v for k, v in METHODS.items() if any(m in k for m in a.methods)}

    data = []
    for c in clips:
        t, lips, found, frames = cached_lips(c, a.set)
        if found < 0.5 * frames:
            print(f"skip {os.path.basename(c)}: face in only {found}/{frames} frames")
            continue
        y = load_audio(c)
        gap = stream_start_gap(c)
        feats = {s: features(shifted(y, s)) for s in SHIFTS}
        data.append((c, group_of(c, a.set), t, lips, gap, feats))
    groups = sorted({g for _, g, *_ in data})
    print(f"{a.set}: {len(data)} clips ({', '.join(f'{g} {sum(d[1] == g for d in data)}' for g in groups)})"
          f" x {len(SHIFTS)} shifts\n")

    head = (f"{'method':42} {'MAE':>7} {'median':>7} {'<=40ms':>7} {'ITU ok':>7} "
            f"{'covered':>8} {'conf<=40':>9}")
    if len(groups) > 1:
        head += "   " + "  ".join(f"{g[:10]:>10}" for g in groups)
    print(head + ("   (per group: <=40ms)" if len(groups) > 1 else ""))
    natural = {}
    for name, (pairs, whiten) in methods.items():
        errs, res, grp, confs, rows = [], [], [], [], []
        for c, g, t, lips, gap, feats in data:
            off, conf = {}, {}
            for s in SHIFTS:
                f = feats[s]
                a_t = np.arange(len(f["logrms"])) * HOP_S + gap
                r = sync.estimate_offset_multi(t, lips, a_t, f, pairs=pairs, whiten=whiten)
                off[s], conf[s] = r["offset_ms"], r["confidence"]
            natural.setdefault(name, []).append((g, off[0]))
            for s in SHIFTS:
                e = off[s] - off[0] - s                 # detection error of the change
                errs.append(e)
                res.append(-e)                           # sync left after correcting
                grp.append(g)
                confs.append(min(conf[s], conf[0]))     # both detections must be confident
                rows.append({"clip": os.path.basename(c), "group": g, "shift_ms": s, "error_ms": e,
                             "confidence": confs[-1], "natural_ms": off[0]})
        errs, res, grp = np.abs(np.array(errs)), np.array(res), np.array(grp)
        ok = np.array(confs) >= MIN_CONF
        slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        with open(os.path.join(ROOT, "results", f"tests_{a.set}_{slug}.csv"), "w", newline="",
                  encoding="utf-8") as f:                 # per-test rows for coverage_curve.py
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        itu = np.mean((res >= -45) & (res <= 125)) * 100
        line = (f"{name:42} {errs.mean():>5.1f}ms {np.median(errs):>5.1f}ms "
                f"{np.mean(errs <= 40) * 100:>6.0f}% {itu:>6.0f}% {ok.mean() * 100:>7.0f}% "
                f"{(np.mean(errs[ok] <= 40) * 100 if ok.any() else float('nan')):>8.0f}%")
        if len(groups) > 1:
            line += "   " + "  ".join(f"{np.mean(errs[grp == g] <= 40) * 100:>9.0f}%" for g in groups)
        print(line)

    print("\nNatural offset of the ORIGINAL clips (shift 0, no calibration; + = lips lead):")
    for name, vals in natural.items():
        by = {g: [v for gg, v in vals if gg == g] for g in groups}
        print(f"  {name.strip():40} " + "  ".join(
            f"{g}: median {np.median(v):+.0f} ms [{min(v):+.0f}..{max(v):+.0f}]" for g, v in by.items()))
        if a.final:
            break


if __name__ == "__main__":
    main()
