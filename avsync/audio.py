"""
Audio tracks from the video's own audio, every 10 ms:
  logrms - speech loudness (log-compressed RMS energy)
  onset  - sound onsets (librosa onset strength: where new sounds start)
  band   - log energy in the speech band 300-3400 Hz (ignores rumble/hiss outside it,
           and that is where vowel formants F1/F2 — i.e. mouth opening — show up)
"""
import os
import subprocess
import tempfile

import numpy as np

SR = 16000
HOP_S = 0.010                       # one value every 10 ms


def load_audio(video_path: str, sr: int = SR) -> np.ndarray:
    """Mono float audio of a video (ffmpeg -> wav -> librosa)."""
    import librosa
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "audio.wav")
        r = subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
                            "-ar", str(sr), "-c:a", "pcm_s16le", wav],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError(f"No usable audio in {video_path}:\n{r.stderr[-800:]}")
        y, _ = librosa.load(wav, sr=sr, mono=True)
    return y


def stream_start_gap(video_path: str) -> float:
    """audio start_time - video start_time (s). A container can delay the audio just by
    timestamps; extracting to WAV drops that, so it is added back to the audio times."""
    starts = {}
    for kind in ("v", "a"):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"{kind}:0",
                            "-show_entries", "stream=start_time", "-of", "csv=p=0", str(video_path)],
                           capture_output=True, text=True)
        try:
            starts[kind] = float(r.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            starts[kind] = 0.0
    return starts["a"] - starts["v"]


def features(y: np.ndarray, sr: int = SR) -> dict:
    import librosa
    hop = int(sr * HOP_S)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    spec = np.abs(librosa.stft(y, n_fft=512, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    band = np.log1p(spec[(freqs >= 300) & (freqs <= 3400)].sum(axis=0) * 10.0)
    n = min(len(rms), len(onset), len(band))
    return {"logrms": np.log1p(rms[:n] * 100.0), "onset": onset[:n], "band": band[:n]}


def audio_tracks(video_path: str, sr: int = SR):
    """Returns (times_s, {"logrms": ..., "onset": ...}) on the video's timeline."""
    f = features(load_audio(video_path, sr), sr)
    times = np.arange(len(f["logrms"])) * HOP_S + stream_start_gap(video_path)
    return times, f
