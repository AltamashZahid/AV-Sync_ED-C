"""
Detect (and optionally correct) the audio-video offset of one or more videos.

    python -B scripts/detect.py video.mp4 [more.mp4 ...]
    python -B scripts/detect.py video.mp4 --correct fixed.mp4
    python -B scripts/detect.py reenacted.mp4 --drift --reference driving.mp4 --correct fixed.mp4

offset > 0 means the audio is LATE (heard after the lips move).
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avsync.pipeline import detect, detect_and_correct, detect_and_correct_drift, detect_drift  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--max-lag", type=float, default=0.5, help="largest offset searched (s)")
    p.add_argument("--correct", help="write a corrected copy here (single video only)")
    p.add_argument("--natural", type=float, default=0.0,
                   help="normal lip lead (ms) to count as in sync, e.g. 80 for studio video")
    p.add_argument("--drift", action="store_true", help="look for a growing offset (wrong fps)")
    p.add_argument("--reference", help="--drift: the video this one was made from (lip-to-lip)")
    a = p.parse_args()

    if a.correct and len(a.videos) != 1:
        sys.exit("--correct works with one video")
    for v in a.videos:
        if a.drift:
            r = (detect_and_correct_drift(v, a.correct, a.reference) if a.correct
                 else detect_drift(v, a.reference))
            print(f"{os.path.basename(v)}: drift {r['drift_ms_per_s']:+.1f} ms/s (speed {r['speed']:.4f}, "
                  f"{r['method']}) -> {r['drift_total_ms']:+.0f} ms over {r['duration_s']:.1f} s"
                  f" -> {'NEEDS FIX' if r['needs_fix'] else 'ok'}")
            if a.correct:
                print("   corrected ->", r.get("output", "not needed"))
            continue
        r = (detect_and_correct(v, a.correct, max_lag_s=a.max_lag, natural_offset_ms=a.natural)
             if a.correct else detect(v, max_lag_s=a.max_lag, natural_offset_ms=a.natural))
        late = "audio LATE" if r["offset_ms"] > 0 else "audio EARLY"
        print(f"{os.path.basename(v)}: offset {r['offset_ms']:+.0f} ms ({late}), "
              f"confidence {r['confidence']:.1f}, peak corr {r['peak_corr']:.2f}, "
              f"face in {r['video']['face_found']}/{r['video']['frames']} frames -> {r['verdict'].upper()}")
        if a.correct:
            print("   corrected ->", r["output"] if r["corrected"] else "not needed / not confident")


if __name__ == "__main__":
    main()
