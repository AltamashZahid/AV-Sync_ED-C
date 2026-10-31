"""
Correction: write a copy of the video with its audio moved in time (video frames untouched).
The shift is baked into the samples (silence added / audio trimmed), not just into container
timestamps, so every player and every tool sees the same result.
"""
import subprocess


def shift_audio(video_in: str, delay_s: float, video_out: str):
    """delay_s > 0: audio plays LATER;  delay_s < 0: audio plays EARLIER.
    Video stream is copied as-is; audio is re-encoded (AAC)."""
    if delay_s >= 0:
        af = f"adelay={delay_s * 1000:.1f}:all=1"                     # prepend silence
    else:
        af = f"atrim=start={-delay_s:.4f},asetpts=PTS-STARTPTS"       # cut the start
    cmd = ["ffmpeg", "-y", "-i", str(video_in), "-map", "0:v:0", "-map", "0:a:0",
           "-c:v", "copy", "-af", af, "-c:a", "aac", "-b:a", "192k", str(video_out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])
    return video_out


def correct_sync(video_in: str, offset_ms: float, video_out: str):
    """Undo a detected offset (offset > 0 = audio late -> move it earlier)."""
    return shift_audio(video_in, -offset_ms / 1000.0, video_out)


def retime_video(video_in: str, speed: float, video_out: str, fps: str = None):
    """Undo drift (wrong frame rate): frame times are multiplied by `speed` (from avsync.drift).
    Same frames in the same order; audio copied untouched. fps: output rate, e.g. '30000/1001'
    (default: the old rate divided by speed)."""
    from fractions import Fraction
    if fps is None:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=r_frame_rate", "-of", "csv=p=0", str(video_in)],
                           capture_output=True, text=True)
        new = float(Fraction(r.stdout.strip().splitlines()[0])) / speed
        standard = ["24000/1001", "24", "25", "30000/1001", "30", "50", "60000/1001", "60"]
        near = [f for f in standard if abs(float(Fraction(f)) / new - 1) < 0.005]
        fps = min(near, key=lambda f: abs(float(Fraction(f)) - new)) if near else f"{new:.5f}"
    cmd = ["ffmpeg", "-y", "-i", str(video_in), "-map", "0:v:0", "-map", "0:a?",
           "-vf", f"setpts=PTS*{speed:.6f}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(video_out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])
    return video_out
