"""
Visual track: mouth opening (MAR) for EVERY frame, with its real timestamp.

The professor's face_features.extract_features() only keeps frames where a face was found,
so a missed frame silently shifts every later value — fine for scoring, wrong for timing.
Here we run our own frame loop (NaN where no face, timestamps from the video itself) but
reuse the professor's mouth_aspect_ratio() so the lip measurement is identical.
The professor's code is only imported, never changed.
"""
import os
import sys

sys.dont_write_bytecode = True          # never write __pycache__ into the model folder

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", r"C:\Users\ashis\Video_Reenactment_Therapy\LivePortrait")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def _rotate(frame, rotation):
    """Phones store portrait video sideways + a rotation flag that OpenCV ignores."""
    r = int(round(rotation)) % 360
    if r == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# MediaPipe FaceMesh indices: inner-lip contour (in order), outer eye corners (face scale)
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
             324, 318, 402, 317, 14, 87, 178, 88, 95]
EYE_CORNERS = (33, 263)
LIP_CORNERS = (61, 291)


def _shape_features(lm):
    """Inner-lip area and mouth width, scaled by eye distance (so face size/zoom cancels).
    Area grows with any opening (also rounded vowels that barely change MAR)."""
    pts = np.array([(lm.landmark[i].x, lm.landmark[i].y) for i in INNER_LIP])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))   # shoelace
    e1, e2 = (np.array([lm.landmark[i].x, lm.landmark[i].y]) for i in EYE_CORNERS)
    c1, c2 = (np.array([lm.landmark[i].x, lm.landmark[i].y]) for i in LIP_CORNERS)
    scale = np.linalg.norm(e1 - e2) + 1e-9
    return area / scale ** 2, np.linalg.norm(c1 - c2) / scale


def mar_track(video_path: str):
    """Returns (times_s, mar, info). mar[i] is NaN when no face was found in frame i."""
    times, feats, info = lip_track(video_path)
    return times, feats["mar"], info


def lip_track(video_path: str):
    """Returns (times_s, {"mar", "area", "width"}, info); NaN where no face was found.
    mar uses the professor's mouth_aspect_ratio(); area/width are our own extras."""
    import mediapipe as mp
    from face_features import mouth_aspect_ratio      # professor's formula, read-only

    mesh = mp.solutions.face_mesh.FaceMesh(            # same settings as face_features.py
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0

    times, mars, areas, widths = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # real timestamp (phones often record variable frame rate); fall back to index / fps
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0 if t_ms > 0 or i == 0 else i / fps
        res = mesh.process(cv2.cvtColor(_rotate(frame, rotation), cv2.COLOR_BGR2RGB))
        times.append(t)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            area, width = _shape_features(lm)
            mars.append(float(mouth_aspect_ratio(lm)))
            areas.append(area)
            widths.append(width)
        else:
            mars.append(np.nan)
            areas.append(np.nan)
            widths.append(np.nan)
        i += 1
    cap.release()
    mesh.close()

    mar = np.array(mars, dtype=float)
    info = {"frames": i, "fps": fps, "rotation": rotation,
            "face_found": int(np.sum(~np.isnan(mar)))}
    feats = {"mar": mar, "area": np.array(areas, float), "width": np.array(widths, float)}
    return np.array(times, dtype=float), feats, info
