# Audio-Visual Synchronization Error Detection & Correction

BTP project (follow-up to the speech-therapy app). Detect when a video's audio is out of
sync with the lips, measure the offset in milliseconds, decide whether a viewer would notice
(ITU-R BT.1359), and correct it automatically.

**Idea:** when the mouth opens, speech gets louder. We track mouth opening (MAR) per video
frame and speech energy over time, slide one against the other, and the shift where they
match best is the offset. Correction moves the audio by that amount. No training, no GPU,
no new packages.

## Rules
- The professor's model (`Video_Reenactment_Therapy\LivePortrait`) is **never changed**.
  We only import its `mouth_aspect_ratio()` from `face_features.py` (read-only, no bytecode).
- No packages are installed: everything runs with the model PC's existing venv
  (OpenCV, MediaPipe, librosa, scipy, matplotlib).

## Background (literature)
| Work | Idea | Relation to us |
|---|---|---|
| Hershey & Movellan, *Audio Vision*, NIPS 1999 | mutual information between audio energy and pixels | same principle: loudness co-varies with the moving face |
| Slaney & Covell, *FaceSync*, NIPS 2000 | canonical correlation of face images and audio | linear correlation measure, like ours, but learned from pixels |
| Knapp & Carter, IEEE TASSP 1976 | generalized cross-correlation, PHAT weighting | our peak search + partial PHAT whitening (PHAT-β) |
| Chung & Zisserman, *SyncNet* ("Out of time"), ACCV-W 2016 | CNN embeddings of mouth crops and MFCCs, offset = best embedding distance | standard deep baseline; >99% (clip-level) on BBC TV, needs weights + training data |
| Kadandale et al., *VocaLiST*, Interspeech 2022 | cross-modal transformer, LRS2 | state of the art; counts a clip correct when within ±1 frame (we use ±40 ms) |
| ITU-R BT.1359-1 (1998) | subjective tests: detectable at sound **+45 ms early / −125 ms late**, acceptable at +90 / −185 ms | our `verdict()`; note the asymmetry — early audio is noticed much sooner |

Our method is a classical, explainable signal-processing approach (GCC on lip and speech
envelopes). It needs no training data, runs on CPU, and every number can be explained, at the cost
of the accuracy deep models reach on hard "in the wild" video.

## Layout
| Path | What |
|---|---|
| `avsync/visual.py` | `lip_track()`: MAR (professor's formula) + inner-lip area + mouth width for **every** frame with its real timestamp (NaN = no face). Applies phone rotation flags. |
| `avsync/audio.py` | Every 10 ms: log RMS, onset strength, **speech-band (300–3400 Hz) energy**; adds the container's audio/video start gap |
| `avsync/sync.py` | Cross-correlation over ±500 ms, 2-pair ensemble, PHAT-β whitening, sub-frame peak, confidence |
| `avsync/correct.py` | `shift_audio`: copy with the audio delayed/advanced; `retime_video`: undo drift (same frames, new timing) |
| `avsync/drift.py` | Drift = offset that grows over time (wrong frame rate): `grid` (lips vs audio), `reference` (lips vs lips of the source video), `windows` (rejected) |
| `avsync/pipeline.py` | `detect(video)` → offset, confidence, **verdict** (in sync / noticeable / unacceptable / uncertain); `detect_and_correct()`; `detect_drift()` / `detect_and_correct_drift()` |
| `scripts/detect.py` | CLI for one or more videos |
| `scripts/bench.py` | Fast method comparison with a **dev / held-out split** (cached lip tracks, exact numpy shifts) |
| `scripts/drift_bench.py` | Drift benchmark: simulated wrong frame rates, accuracy vs video length, real LivePortrait outputs |
| `avsync/syncnet_model.py`, `avsync/syncnet_baseline.py` | SyncNet baseline: our own copy of the official layers + MFCC; official weights loaded as data |
| `scripts/bench_syncnet.py` | SyncNet on exactly the same tests as `bench.py` (`--fine`: shifts between frames) |
| `scripts/bench_hybrid.py` | Hybrid: SyncNet coarse + our correlation refined (`--fine` too) |
| `scripts/coverage_curve.py` | Coverage vs accuracy curves (`results/coverage_*.png`) |
| `models/syncnet_v2.model` | Official SyncNet weights (not in git) |
| `scripts/evaluate.py` | End-to-end: real ffmpeg shifts (AAC re-encode) → detect → correct |
| `scripts/compare_features.py` | Earlier feature experiment (v1–v3) |
| `data/clips/` | **Dev set**: 11 RAVDESS clips (2 actors) — methods are chosen here only |
| `data/heldout/` | **Held-out set**, 27 clips never used for choices: `actors/` (5, other RAVDESS actors), `phone_en/` (6, phone, English), `phone_hi/` (5, phone, Hindi), `reenacted/` (11, LivePortrait outputs) |
| `results/` | CSV, summary and plot from each evaluation |
| `report/` | BTP report (`REPORT.md`, `REPORT.docx` made with pandoc) |

**Sign convention:** offset > 0 → audio is **late** (heard after the lips move). ITU's own
convention is the opposite (+ = sound leads).

## Run
```powershell
cd "C:\Users\ashis\BTP_AVsync_ER&C"
$py = "C:\Users\ashis\Video_Reenactment_Therapy\LivePortrait\venv\Scripts\python.exe"
& $py -B scripts\detect.py some_video.mp4 --natural 80          # verdict vs a normal 80 ms lip lead
& $py -B scripts\detect.py some_video.mp4 --natural 80 --correct fixed.mp4
& $py -B scripts\bench.py --set dev                              # compare methods (dev only)
& $py -B scripts\bench.py --set heldout --methods v3 PHAT-0.3    # final test
& $py -B scripts\evaluate.py --clips data\heldout                # end-to-end with ffmpeg
& $py -B scripts\detect.py reenacted.mp4 --drift --reference driving.mp4 --correct fixed.mp4
& $py -B scripts\drift_bench.py --duration --set heldout --groups phone_en phone_hi
```
Always use `-B` (no `__pycache__` anywhere, including the model folder).

## Evaluation protocol
1. Start from clips that are (assumed) in sync.
2. Shift the audio by −300 … +300 ms (11 shifts) → 121 tests on dev, 297 on held-out.
3. **Calibrated error** = detected change − true shift (the clip's own offset at shift 0 is its
   "in sync" point; lips naturally lead the sound).
4. Metrics: MAE; share within **40 ms** (1 frame at 25 fps, the VocaLiST-style criterion);
   **ITU ok** = sync left after correction would not be noticed (≤45 ms early, ≤125 ms late);
   **coverage** = share with confidence ≥ 3 (otherwise the detector says "uncertain").
5. Methods are compared and chosen on the dev set only, then run **once** on held-out.

## Results (2026-10-30)

**Dev set (11 clips × 11 shifts)** — method development:

| Version | MAE | ≤ 40 ms |
|---|---|---|
| v1: MAR vs loudness, full-overlap correlation | 53.9 ms | 91% |
| v2: + fixed-length window | 51.6 ms | 92% |
| v3: + 2-pair ensemble (MAR/loudness + opening speed/onsets) | 19.5 ms | 97% |
| inner-lip area instead of MAR | 58.2 ms | 90% |
| full PHAT whitening | 92.2 ms | 81% |
| MAR/**speech band** + opening speed/onsets | 14.4 ms | 98% |
| **v4: + PHAT-β (β = 0.3)** | **10.4 ms** | **98%** |

Negative results are kept on purpose: full PHAT whitening hurts badly on 3–5 s clips (it
amplifies noise in a short envelope), and the inner-lip area is worse than the simple MAR.

**Held-out set (27 unseen clips × 11 shifts), v3 vs v4:**

| | ≤ 40 ms all | ITU ok | covered | ≤ 40 ms when confident | actors | phone EN | phone HI | reenacted |
|---|---|---|---|---|---|---|---|---|
| v3 | 84% | 86% | 42% | 100% | 93% | 98% | 100% | 64% |
| **v4** | **85%** | **86%** | **58%** | **98%** | 93% | 98% | **100%** | 66% |

- End-to-end check (`evaluate.py`, real ffmpeg shifts + AAC re-encode, v4): **84%** within
  40 ms, matching the fast numpy benchmark (85%). So the benchmark is faithful.
  (`results/eval_20261030_193821*`)
- **Works on real phone video and on Hindi** (98–100%), though it was developed on English
  studio clips.
- v4's big dev gain shrinks on unseen data (85% vs 84%). The improvement is mainly in
  confidence (coverage 42% → 58%). This is the reason for a held-out set.
- The reenacted group is the weak spot. In 6 of 11 old LivePortrait outputs, the lips do not follow
  the audio at **any** shift (peak correlation 0.03–0.08). The detector reports these as
  "uncertain" (confidence < 3), which is the right answer, not a wrong offset.

## Findings on real data
1. **Phone recordings carry extra audio delay.** Natural lip lead: studio RAVDESS +65…+97 ms,
   phone clips (English and Hindi) median **≈ +145 ms**, and the app's patient recordings
   +167…+176 ms. So phone capture adds roughly 60–90 ms of audio latency on top of the natural lead.
2. **Bug found in the app's practice videos (and fixed).** The detector measured the app's
   reenacted videos at +2…+12 ms against +84 ms for the therapist video they were made
   from. A lip-to-lip comparison (reenacted vs therapist, no audio) showed frame *i* =
   therapist frame *i* exactly, but the file plays at **29 fps instead of 29.97**. Cause (read
   only, not changed): `live_portrait_pipeline.py` uses `output_fps = int(get_fps(driving))`,
   so 29.97 becomes 29. The video is 3.2% slow and the lips drift up to ~150 ms behind the voice.
   **Fix** in the app's own adapter (`backend/pipeline.py`, `_match_driving_frame_rate`): re-time
   the same frames to the driving rate. Result: +12 → **+68 ms**, confidence 4.4 → 5.5,
   verdict "noticeable" → "in sync".
3. **Seed-VC voice conversion keeps sync:** +68 ms before, +66 ms after the patient-voice swap.
4. Searching a playback speed together with the offset (audio vs lips) is too noisy on 4 s
   clips: even correct videos "prefer" ±2%. Drift is better found lip-to-lip when a reference
   video exists (finding 2).

**Caveats (say these in the report):** only 2 dev actors; held-out phone clips are all one
speaker; "in sync" for the originals is assumed, not measured with a clapper; synthetic shifts
are sample-exact, which is why the median error is ~0 ms. Coverage at the fixed threshold of 3
is modest on short clips. A coverage/accuracy curve would show the trade-off.

## Baseline: SyncNet (Chung & Zisserman 2016)
- **Weights:** the official `syncnet_v2.model` (54.6 MB). The Oxford server was unreachable, so it
  came from Hugging Face (`lithiumice/syncnet`). A second mirror (`nawta/syncnet`) has the same
  SHA-256 `961e8696…5442`. It is loaded as plain data (`torch.load(weights_only=True)`,
  `strict=True`, so every layer must match).
- **No downloaded code is run.** The network layers and the MFCC (python_speech_features
  defaults) are re-typed in `avsync/syncnet_model.py` from the official source (MIT licence).
- **Pipeline follows the official one:** 25 fps, the official crop formula (crop scale 0.4,
  median-smoothed box, 224×224), 5-frame windows, 20 MFCC frames per window, ±15 frames,
  offset = minimum mean distance, confidence = median − minimum. **One difference:** the face
  box comes from MediaPipe instead of S3FD. SyncNet resolution: 1 frame = 40 ms.

| Same tests | Dev ≤40 ms | Held-out ≤40 ms | Held-out MAE | Covered (conf ≥ 3) | Accurate when covered |
|---|---|---|---|---|---|
| Ours v3 | 97% | 84% | 98 ms | 42% | 100% |
| Ours v4 | 98% | 85% | 101 ms | 58% | 98% |
| **SyncNet** | **100%** | **87%** | **28 ms** | **74%** | **100%** |

Per group (held-out, ≤40 ms), ours v4 / SyncNet: actors 93 / 100, phone EN 98 / 100,
phone HI 100 / 100, reenacted 66 / 68.

**Coverage vs accuracy** (`results/coverage_heldout.png`). Each method answers only above a
confidence threshold. The confidence scales differ, so methods are compared at equal coverage:

| Accuracy (≤40 ms) at coverage | 50% | 75% | 90% | 100% |
|---|---|---|---|---|
| Ours v3 | 98.7% | 91.5% | 84.0% | 83.5% |
| Ours v4 | 98.7% | 96.9% | 86.9% | 84.5% |
| SyncNet | 100% | 100% | 93.7% | 86.9% |
| *without the reenacted group:* ours v4 | 98.9% | 98.5% | 98.1% | 97.2% |
| *without the reenacted group:* SyncNet | 100% | 100% | 100% | 100% |

**Reading it honestly:**
- SyncNet is better, as expected from a CNN trained on hundreds of hours of TV.
- The gap is small on clean talking faces (97% vs 100% when every test is answered).
- Both methods fail on the same broken reenacted videos.
- Ours needs no training data, no GPU and no face detector network, gives sub-frame
  (10 ms) offsets, and every step can be explained.
- SyncNet's "natural offset" is ~0 because it learned the human lip lead. It also measures the phone
  clips as **+40 ms** late, which independently confirms finding 1 (phone audio delay).

## Hybrid: SyncNet + our correlation
SyncNet rarely picks a wrong syllable, but it only has 40 ms steps. Our correlation is precise
(10 ms grid + parabola) but sometimes jumps to a wrong syllable far away. The hybrid takes the
best of both: our match curve, with the peak searched only within **±60 ms of SyncNet's answer
+ 80 ms** (SyncNet ignores the natural lip lead, our curve includes it).
`estimate_offset_multi(..., near_ms=, window_ms=)`, `scripts/bench_hybrid.py`.

Lead and window were chosen on dev: every lead 40–100 ms with a window ≥ 60 ms scores the same
(100%), so the middle of that range was taken; then held-out was run once.

| Held-out, 27 clips | MAE | median | ≤ 40 ms | ≤ 20 ms | ITU ok |
|---|---|---|---|---|---|
| *Standard shifts (11, mostly multiples of 40 ms)* | | | | | |
| Ours v4 | 101 ms | 0.5 ms | 85% | 83% | 86% |
| SyncNet | 28 ms | 0.0 ms | 87% | 85% | 89% |
| **Hybrid** | 31 ms | 0.4 ms | **87%** | **86%** | 89% |
| *Fine shifts (13, between frames: ±10, 30, 50, 70, 110, 150 ms)* | | | | | |
| Ours v4 | 63 ms | 0.3 ms | 90% | 89% | — |
| SyncNet | 25 ms | **10 ms** | 89% | 71% | 92% |
| **Hybrid** | **18 ms** | **0.3 ms** | 88% | **85%** | **93%** |

- On the standard shifts the hybrid ties SyncNet. Those shifts are almost all multiples of
  40 ms, so SyncNet's coarse steps are never punished.
- On shifts between frames SyncNet is typically 10 ms off (median). The hybrid brings this to
  0.3 ms and has the lowest mean error (18 ms), while keeping SyncNet's robustness
  (no far-away wrong-syllable jumps like ours v4, MAE 63 ms).
- The hybrid can't fix what SyncNet gets wrong: the reenacted group stays at ~70%.

## Drift: errors that grow over time
A constant offset is one error type. The other is **drift**: the video plays at the wrong speed,
so the lips slip further from the voice every second. Example: a video written at 29 fps
that should be 29.97 fps (finding 2 above) drifts by 32 ms/s. Model:
`offset(t) = offset0 + drift·t`, speed = 1 + drift/1000, fixed by re-timing the frames.

Simulation: dividing the cached lip timestamps by a speed is exactly what a wrong frame rate
does. 7 speeds: 29/29.97, 0.98, 0.99, 1, 1.01, 1.02, 29.97/29. Clips are joined into
long in-sync recordings to test longer videos. "ok" = sync left at the end after correction ≤ 40 ms.

| Length | dev (grid) | held-out phone EN+HI (grid) | held-out actors (grid) |
|---|---|---|---|
| 4 s | 37 ms/s err, 6% ok | 35 ms/s, 27% | 23 ms/s, 24% |
| 8 s | 15 ms/s, 32% | 12 ms/s, 29% | 8 ms/s, 56% |
| 16 s | 1.3 ms/s, 89% | 2.7 ms/s, 59% | 1.6 ms/s, 86% |
| 32 s | **0.2 ms/s, 94%** | **0.2 ms/s, 100%** | (only 25 s of data) |

- **Lips vs audio needs ≥ 16–32 s of speech** to measure drift. On short clips the natural
  lip lead jitters (tens of ms per syllable) as much as the drift itself.
- The sliding-window + line-fit estimator was rejected on dev. Once the drift adds up to
  more than its ±300 ms window search, it fails (worse at 32 s than at 16 s).
- **With a reference video, short clips are fine.** For a reenacted or dubbed video we have
  the source video, which shows the same mouth motion. Lips-to-lips matching (`reference`) on the 6 real
  4.6 s LivePortrait outputs: −35…−36 ms/s found vs −32.4 true (therapist vs itself: exactly
  0). The end error is ≤ 17 ms, under one frame, so all 6 are correctly fixed.
- Full chain on a real output, BTP code only: drift −164 ms over the clip → `retime_video`
  → offset +9 ms ("noticeable") becomes **+68 ms ("in sync")**, therapist original +84 ms.

## Next steps
- A confidence threshold that adapts to clip length
- Hybrid confidence: combine SyncNet's confidence with how sharp our peak is inside the window
- Drift on real long recordings (not joined clips); variable frame rate phone video
- More speakers for the phone/Hindi sets; a clapper-board recording for true zero
