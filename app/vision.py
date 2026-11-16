"""表情・視線の分析 (MediaPipe Face Landmarker)。

すべての閾値は経験的な目安。照明・カメラ位置・眼鏡などで値は変わるため、
絶対評価ではなく「前回の自分との比較」に使うのがおすすめ。
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .media import iter_video_frames

HEAD_YAW_LIMIT = 15.0     # 正面とみなす頭の左右角度 (度)
HEAD_PITCH_LIMIT = 15.0   # 正面とみなす頭の上下角度 (度)
GAZE_LIMIT = 0.35         # 黒目の偏り (blendshape 値) の許容量
SMILE_THRESHOLD = 0.35
BLINK_THRESHOLD = 0.5


def _model_path() -> Path:
    path = config.MODELS_DIR / "face_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(config.FACE_LANDMARKER_URL, tmp)
        tmp.rename(path)
    return path


def _head_angles(matrix: np.ndarray) -> tuple[float, float]:
    """顔の変換行列から (yaw, pitch) を度で返す。0 がカメラ正面。"""
    forward = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    return yaw, pitch


def _gaze(bs: dict[str, float]) -> tuple[float, float]:
    """目の向き (水平, 垂直)。0 に近いほど正面。"""
    horizontal = ((bs["eyeLookOutLeft"] - bs["eyeLookInLeft"])
                  - (bs["eyeLookOutRight"] - bs["eyeLookInRight"])) / 2
    vertical = ((bs["eyeLookUpLeft"] + bs["eyeLookUpRight"])
                - (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"])) / 2
    return horizontal, vertical


def analyze(video_path: Path, meta: dict, on_progress: Callable[[float], None] | None = None) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    fps = config.VISION_SAMPLE_FPS
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_model_path())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    frames = []
    duration = meta["duration"]
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for t, rgb in iter_video_frames(video_path, meta["width"], meta["height"], fps):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, int(t * 1000))
            if not result.face_landmarks:
                frames.append({"t": t, "face": False})
            else:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                yaw, pitch = _head_angles(np.array(result.facial_transformation_matrixes[0]))
                gh, gv = _gaze(bs)
                frames.append({
                    "t": t, "face": True, "yaw": yaw, "pitch": pitch, "gaze_h": gh, "gaze_v": gv,
                    "smile": (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2,
                    "blink": (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2,
                })
            if on_progress and duration > 0 and len(frames) % int(fps * 5) == 0:
                on_progress(min(t / duration, 1.0))
    return summarize(frames, fps)


def summarize(frames: list[dict], fps: float) -> dict:
    face = [f for f in frames if f["face"]]
    if not frames:
        return {"available": False, "reason": "映像フレームを読み込めませんでした。"}
    if len(face) < max(5, len(frames) * 0.1):
        return {"available": False, "reason": "顔をほとんど検出できませんでした (顔が映っているか確認してください)。"}

    for f in face:
        f["eye_contact"] = (abs(f["yaw"]) < HEAD_YAW_LIMIT and abs(f["pitch"]) < HEAD_PITCH_LIMIT
                            and abs(f["gaze_h"]) < GAZE_LIMIT and abs(f["gaze_v"]) < GAZE_LIMIT)

    yaw = np.array([f["yaw"] for f in face])
    pitch = np.array([f["pitch"] for f in face])
    smile = np.array([f["smile"] for f in face])

    # 瞬き: 目が閉じた状態に入った回数
    blinks, closed = 0, False
    for f in face:
        is_closed = f["blink"] > BLINK_THRESHOLD
        if is_closed and not closed:
            blinks += 1
        closed = is_closed
    face_minutes = len(face) / fps / 60

    # 視線が外れていた区間 (2 秒以上連続)
    look_away, start = [], None
    for f in frames:
        away = (not f["face"]) or (not f["eye_contact"])
        if away and start is None:
            start = f["t"]
        elif not away and start is not None:
            if f["t"] - start >= 2.0:
                look_away.append({"start": round(start, 1), "end": round(f["t"], 1)})
            start = None
    if start is not None and frames[-1]["t"] - start >= 2.0:
        look_away.append({"start": round(start, 1), "end": round(frames[-1]["t"], 1)})

    # 1 秒ごとのタイムライン
    timeline = []
    for sec in range(int(frames[-1]["t"]) + 1):
        fs = [f for f in face if sec <= f["t"] < sec + 1]
        timeline.append({
            "t": sec,
            "eye_contact": round(sum(f["eye_contact"] for f in fs) / len(fs), 2) if fs else None,
            "smile": round(float(np.mean([f["smile"] for f in fs])), 3) if fs else None,
            "yaw": round(float(np.mean([f["yaw"] for f in fs])), 1) if fs else None,
        })

    result = {
        "available": True,
        "sample_fps": fps,
        "face_detected_ratio": round(len(face) / len(frames), 3),
        "eye_contact_ratio": round(sum(f["eye_contact"] for f in face) / len(face), 3),
        "smile_mean": round(float(smile.mean()), 3),
        "smile_ratio": round(float((smile > SMILE_THRESHOLD).mean()), 3),
        "head_yaw_median": round(float(np.median(yaw)), 1),
        "head_pitch_median": round(float(np.median(pitch)), 1),
        # 一時的に横を向いた区間に引きずられないよう、中央値からの絶対偏差 (MAD) で揺れを測る
        "head_movement_std": round(float(1.4826 * np.hypot(
            np.median(np.abs(yaw - np.median(yaw))), np.median(np.abs(pitch - np.median(pitch))))), 1),
        "blinks_per_minute": round(blinks / face_minutes, 1) if face_minutes > 0 else None,
        "look_away_periods": look_away,
        "timeline": timeline,
    }
    result["evaluation"] = evaluate(result)
    return result


def evaluate(v: dict) -> list[dict]:
    items = []

    def add(level, title, value, advice):
        items.append({"level": level, "title": title, "value": value, "advice": advice})

    fd = v["face_detected_ratio"]
    if fd < 0.85:
        add("warn", "顔の映り", f"{fd:.0%}", "顔がカメラ枠から外れている時間があります。画面中央に顔が収まる位置で受けましょう。")

    ec = v["eye_contact_ratio"]
    if ec >= 0.6:
        add("good", "目線 (カメラ方向)", f"{ec:.0%}", "おおむねカメラ (面接官) の方を向いて話せています。")
    elif ec >= 0.4:
        add("warn", "目線 (カメラ方向)", f"{ec:.0%}",
            "目線が外れる時間がやや長めです。考える時も、話し始めと結論はカメラを見ることを意識しましょう。")
    else:
        add("bad", "目線 (カメラ方向)", f"{ec:.0%}",
            "目線が外れている時間が長いです。オンライン面接ではカメラのレンズを見ると『目が合っている』印象になります。")

    if v["smile_ratio"] < 0.05 and v["smile_mean"] < 0.15:
        add("warn", "表情 (笑顔)", f"笑顔の割合 {v['smile_ratio']:.0%}",
            "表情が硬めです。入退室の挨拶や、自己紹介の冒頭だけでも口角を上げると印象が柔らかくなります。")
    else:
        add("good", "表情 (笑顔)", f"笑顔の割合 {v['smile_ratio']:.0%}", "自然な笑顔が見られます。")

    hm = v["head_movement_std"]
    if hm > 8:
        add("warn", "頭の動き", f"ばらつき {hm:.1f}°", "頭や体の揺れが大きめです。背筋を伸ばし、うなずきは小さくゆっくりを意識しましょう。")
    else:
        add("good", "頭の動き", f"ばらつき {hm:.1f}°", "姿勢は安定しています。")

    bpm = v["blinks_per_minute"]
    if bpm is not None and bpm > 40:
        add("warn", "瞬き (参考値)", f"{bpm:.0f} 回/分", "瞬きがかなり多めです。緊張のサインに見えることがあります。")
    return items
