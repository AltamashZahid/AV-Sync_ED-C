"""
Our own re-implementation of the pieces SyncNet needs, so no downloaded code is executed:
  - SyncNet network layers, written to match SyncNetModel.py of github.com/joonson/syncnet_python
    (MIT, (c) Joon Son Chung) so the official weights load into it as plain data
  - MFCC with the defaults of python_speech_features (MIT, (c) James Lyons), which SyncNet
    was trained with: 25 ms / 10 ms frames, 512-pt FFT, 26 mel filters, 13 coeffs,
    pre-emphasis 0.97, lifter 22, c0 replaced by log frame energy
"""
import decimal
import math

import numpy as np


# ---------------------------------------------------------------- MFCC (python_speech_features)
def _round_half_up(x):
    return int(decimal.Decimal(x).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


def _hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def _filterbanks(nfilt=26, nfft=512, sr=16000, low=0, high=None):
    high = high or sr / 2
    mel = np.linspace(_hz2mel(low), _hz2mel(high), nfilt + 2)
    b = np.floor((nfft + 1) * _mel2hz(mel) / sr)
    fb = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(int(b[j]), int(b[j + 1])):
            fb[j, i] = (i - b[j]) / (b[j + 1] - b[j])
        for i in range(int(b[j + 1]), int(b[j + 2])):
            fb[j, i] = (b[j + 2] - i) / (b[j + 2] - b[j + 1])
    return fb


def mfcc(signal, sr=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, preemph=0.97,
         ceplifter=22):
    from scipy.fftpack import dct
    signal = np.asarray(signal, dtype=float)
    nfft = 1
    while nfft < winlen * sr:
        nfft *= 2
    sig = np.append(signal[0], signal[1:] - preemph * signal[:-1])
    flen, fstep = _round_half_up(winlen * sr), _round_half_up(winstep * sr)
    n = 1 if len(sig) <= flen else 1 + int(math.ceil((len(sig) - flen) / fstep))
    pad = np.concatenate([sig, np.zeros((n - 1) * fstep + flen - len(sig))])
    idx = np.arange(flen)[None, :] + (np.arange(n) * fstep)[:, None]
    frames = pad[idx]
    pspec = np.square(np.abs(np.fft.rfft(frames, nfft))) / nfft
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feat = pspec @ _filterbanks(nfilt, nfft, sr).T
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = dct(np.log(feat), type=2, axis=1, norm="ortho")[:, :numcep]
    feat = feat * (1 + (ceplifter / 2.0) * np.sin(np.pi * np.arange(numcep) / ceplifter))
    feat[:, 0] = np.log(energy)
    return feat


# ---------------------------------------------------------------- SyncNet layers
def build_syncnet(dim=1024):
    """Layer names/shapes identical to SyncNetModel.S, so the state dict loads 1:1."""
    import torch.nn as nn

    class S(nn.Module):
        def __init__(self):
            super().__init__()
            self.netcnnaud = nn.Sequential(
                nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d((1, 1), (1, 1)),
                nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1)), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
                nn.MaxPool2d((3, 3), (1, 2)),
                nn.Conv2d(192, 384, (3, 3), padding=(1, 1)), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, (3, 3), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.MaxPool2d((3, 3), (2, 2)),
                nn.Conv2d(256, 512, (5, 4), padding=(0, 0)), nn.BatchNorm2d(512), nn.ReLU())
            self.netfcaud = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                          nn.Linear(512, dim))
            self.netfclip = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                          nn.Linear(512, dim))
            self.netcnnlip = nn.Sequential(
                nn.Conv3d(3, 96, (5, 7, 7), (1, 2, 2), 0), nn.BatchNorm3d(96), nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2)),
                nn.Conv3d(96, 256, (1, 5, 5), (1, 2, 2), (0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
                nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2)),
                nn.Conv3d(256, 512, (1, 6, 6), padding=0), nn.BatchNorm3d(512), nn.ReLU(inplace=True))

        def forward_aud(self, x):
            return self.netfcaud(self.netcnnaud(x).flatten(1))

        def forward_lip(self, x):
            return self.netfclip(self.netcnnlip(x).flatten(1))

    return S()
