"""One call: video -> offset, verdict (and optionally a corrected copy)."""
from .audio import audio_tracks
from .correct import correct_sync, retime_video
from .drift import estimate_drift_grid, estimate_drift_reference
from .sync import estimate_offset_multi
from .visual import lip_track

# ITU-R BT.1359-1 (1998), in OUR sign convention (+ = audio late). The recommendation's own
# sign is the opposite (+ = sound leads picture).
DETECTABLE_EARLY_MS, DETECTABLE_LATE_MS = 45.0, 125.0
ACCEPTABLE_EARLY_MS, ACCEPTABLE_LATE_MS = 90.0, 185.0
MIN_CONFIDENCE = 3.0                    # below this the estimate is reported as "uncertain"


def verdict(error_ms: float, confidence: float) -> str:
    """error_ms: offset beyond the natural lip lead. Viewers notice audio that is early much
    sooner than audio that is late (sound never arrives before the sight in real life)."""
    if confidence < MIN_CONFIDENCE:
        return "uncertain"
    if -DETECTABLE_EARLY_MS <= error_ms <= DETECTABLE_LATE_MS:
        return "in sync"
    if -ACCEPTABLE_EARLY_MS <= error_ms <= ACCEPTABLE_LATE_MS:
        return "noticeable"
    return "unacceptable"


def detect(video_path: str, max_lag_s: float = 0.5, visual=None, natural_offset_ms: float = 0.0):
    """visual: a cached (times, lips, info) from lip_track() — the slow part — so the same
    video frames can be tested against many audio versions.
    natural_offset_ms: the normal lip lead to treat as "in sync" (lips move before sound)."""
    v_t, lips, info = visual if visual is not None else lip_track(video_path)
    a_t, audio = audio_tracks(video_path)
    result = estimate_offset_multi(v_t, lips, a_t, audio, max_lag_s=max_lag_s)
    result["video"] = info
    result["error_ms"] = result["offset_ms"] - natural_offset_ms
    result["verdict"] = verdict(result["error_ms"], result["confidence"])
    return result


def detect_and_correct(video_path: str, out_path: str, min_confidence: float = MIN_CONFIDENCE,
                       min_offset_ms: float = 20.0, natural_offset_ms: float = 0.0, **kw):
    """Corrects only when the estimate is confident and the offset is worth fixing."""
    r = detect(video_path, natural_offset_ms=natural_offset_ms, **kw)
    r["corrected"] = bool(r["confidence"] >= min_confidence and abs(r["error_ms"]) >= min_offset_ms
                          and not r["at_search_edge"])
    if r["corrected"]:
        correct_sync(video_path, r["error_ms"], out_path)
        r["output"] = out_path
    return r


MIN_DRIFT_MS = 40.0                     # fix drift only if it would move the lips by > 1 frame


def detect_drift(video_path: str, reference_path: str = None):
    """Offset that grows over time (wrong frame rate). With a reference video (the video this
    one was generated/dubbed from) lips are compared to lips: works on short clips. Without
    one, lips vs audio: needs >= ~16-32 s of speech (see README)."""
    v_t, lips, info = lip_track(video_path)
    if reference_path:
        r_t, r_lips, _ = lip_track(reference_path)
        r = estimate_drift_reference(v_t, lips, r_t, r_lips)
    else:
        a_t, audio = audio_tracks(video_path)
        r = estimate_drift_grid(v_t, lips, a_t, audio)
    r["duration_s"] = float(v_t[-1])
    r["drift_total_ms"] = r["drift_ms_per_s"] * r["duration_s"]
    r["needs_fix"] = abs(r["drift_total_ms"]) > MIN_DRIFT_MS
    return r


def detect_and_correct_drift(video_path: str, out_path: str, reference_path: str = None):
    r = detect_drift(video_path, reference_path)
    if r["needs_fix"]:
        retime_video(video_path, r["speed"], out_path)
        r["output"] = out_path
    return r
