"""
SyncNet baseline (Chung & Zisserman, "Out of time: automated lip sync in the wild", 2016).

Official weights (models/syncnet_v2.model, sha256 961e8696…, Hugging Face mirror of the Oxford
file) loaded as plain data (weights_only) into our own copy of the layers (syncnet_model.py);
no downloaded code is run. Never trained or changed.
Follows the official run_pipeline.py / SyncNetInstance.evaluate():
  - video at 25 fps, face crop with the same formula (crop_scale 0.4, median-smoothed box),
    224x224 BGR, 5-frame windows
  - audio 16 kHz, python_speech_features MFCC (13), 20 MFCC frames per 5 video frames
  - offset = argmin of the mean embedding distance over +-15 frames (40 ms steps),
    confidence = median distance - min distance
One difference: the face box comes from MediaPipe landmarks instead of the S3FD detector.

Sign: converted to OUR convention (offset > 0 = audio late).
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np

from .syncnet_model import build_syncnet, mfcc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "syncnet_v2.model")
FPS = 25
VSHIFT = 15
CROP_SCALE = 0.4

_model = None


def model():
    global _model
    if _model is None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = build_syncnet()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)   # data only
        net.load_state_dict(state, strict=True)                              # every layer must match
        net = net.to(dev).eval()
        _model = (net, dev)
    return _model


def face_crops(video_path: str) -> np.ndarray:
    """(N, 224, 224, 3) BGR face crops at 25 fps, official crop formula."""
    import mediapipe as mp
    from scipy import signal
    with tempfile.TemporaryDirectory() as d:
        v25 = os.path.join(d, "v25.mp4")      # ffmpeg also applies the phone rotation flag
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-an",
                        "-r", str(FPS), "-qscale:v", "2", v25], check=True)
        cap = cv2.VideoCapture(v25)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    boxes = []
    for f in frames:
        h, w = f.shape[:2]
        res = mesh.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
            boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        else:
            boxes.append([np.nan] * 4)
    mesh.close()
    b = np.array(boxes, float)
    if np.isnan(b).all():
        raise ValueError("no face found")
    idx = np.arange(len(b))
    for j in range(4):                                  # fill missed frames
        bad = np.isnan(b[:, j])
        b[bad, j] = np.interp(idx[bad], idx[~bad], b[~bad, j])
    s = signal.medfilt(np.maximum(b[:, 3] - b[:, 1], b[:, 2] - b[:, 0]) / 2, kernel_size=13)
    y = signal.medfilt((b[:, 1] + b[:, 3]) / 2, kernel_size=13)
    x = signal.medfilt((b[:, 0] + b[:, 2]) / 2, kernel_size=13)
    crops = []
    for i, f in enumerate(frames):                       # == run_pipeline.crop_video
        bs = s[i]
        bsi = int(bs * (1 + 2 * CROP_SCALE))
        fr = np.pad(f, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = y[i] + bsi, x[i] + bsi
        face = fr[int(my - bs):int(my + bs * (1 + 2 * CROP_SCALE)),
                  int(mx - bs * (1 + CROP_SCALE)):int(mx + bs * (1 + CROP_SCALE))]
        crops.append(cv2.resize(face, (224, 224)))
    return np.stack(crops)


def video_embeddings(crops: np.ndarray, batch=40) -> np.ndarray:
    """One 1024-d embedding per 5-frame window (window i = frames i..i+4)."""
    import torch
    net, dev = model()
    im = torch.from_numpy(np.transpose(crops, (3, 0, 1, 2))[None].astype(np.float32))  # 1,3,T,H,W
    out = []
    with torch.no_grad():
        for i in range(0, len(crops) - 4, batch):
            x = torch.cat([im[:, :, v:v + 5] for v in range(i, min(len(crops) - 4, i + batch))], 0)
            out.append(net.forward_lip(x.to(dev)).cpu().numpy())
    return np.concatenate(out)


def audio_embeddings(audio_int16: np.ndarray, n_windows: int, batch=100) -> np.ndarray:
    import torch
    net, dev = model()
    feats = mfcc(audio_int16, 16000).T                                   # 13 x T (100 Hz)
    cc = torch.from_numpy(feats[None, None].astype(np.float32))
    # == official: min_length = min(frames, len // 640); windows 0 .. min_length - 6
    n = min(n_windows - 1, len(audio_int16) // 640 - 5)
    out = []
    with torch.no_grad():
        for i in range(0, n, batch):
            x = torch.cat([cc[:, :, :, v * 4:v * 4 + 20] for v in range(i, min(n, i + batch))], 0)
            out.append(net.forward_aud(x.to(dev)).cpu().numpy())
    return np.concatenate(out)


def offset_from_embeddings(vfeat: np.ndarray, afeat: np.ndarray, vshift=VSHIFT):
    """== SyncNetInstance.calc_pdist + evaluate (mean distance per shift)."""
    n = min(len(vfeat), len(afeat))
    v, a = vfeat[:n], afeat[:n]
    ap = np.pad(a, ((vshift, vshift), (0, 0)))
    dists = np.stack([np.linalg.norm(v[i] - ap[i:i + 2 * vshift + 1] + 1e-6, axis=1) for i in range(n)])
    mdist = dists.mean(axis=0)
    j = int(np.argmin(mdist))
    off_frames = vshift - j                         # SyncNet's own "AV offset"
    return {"offset_ms": -off_frames * 1000.0 / FPS,  # our sign: + = audio late
            "syncnet_offset_frames": off_frames,
            "confidence": float(np.median(mdist) - mdist[j]), "min_dist": float(mdist[j])}


def detect(video_path: str):
    from .audio import load_audio
    crops = face_crops(video_path)
    vfeat = video_embeddings(crops)
    y = (np.clip(load_audio(video_path), -1, 1) * 32767).astype(np.int16)
    return offset_from_embeddings(vfeat, audio_embeddings(y, len(vfeat)))
