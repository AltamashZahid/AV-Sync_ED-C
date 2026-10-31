"""
AV sync error detection & correction.

    visual.mar_track(video)        -> mouth opening (MAR) per frame, with exact timestamps
    audio.audio_tracks(video)      -> speech loudness + sound onsets over time
    sync.estimate_offset_multi(..) -> how late the audio is (ms) + confidence
    correct.shift_audio(...)       -> writes a copy with the audio moved
    detect(video)                  -> all of the above for one video
"""
from .pipeline import detect  # noqa: F401
