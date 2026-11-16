"""ffmpeg / ffprobe を使った動画・音声の前処理。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class MediaError(RuntimeError):
    pass


def _require(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        raise MediaError(f"{cmd} が見つかりません。ffmpeg をインストールしてください (例: brew install ffmpeg)")
    return path


def probe(video_path: Path) -> dict:
    """動画の長さ・解像度などを取得する。"""
    out = subprocess.run(
        [
            _require("ffprobe"), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaError("動画に音声トラックがありません。")
    width = height = None
    if video:
        width, height = int(video["width"]), int(video["height"])
        # スマホの縦動画は回転メタデータを持つ。ffmpeg は自動回転するので縦横を合わせる。
        rotation = 0
        for sd in video.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = int(float(sd["rotation"]))
        rotation = rotation or int(float(video.get("tags", {}).get("rotate", 0)))
        if abs(rotation) % 180 == 90:
            width, height = height, width
    return {
        "duration": float(info["format"].get("duration", 0.0)),
        "has_video": video is not None,
        "width": width,
        "height": height,
    }


def iter_video_frames(video_path: Path, width: int, height: int, fps: float, max_width: int = 640):
    """動画を fps 間隔で間引き、RGB の numpy 配列 (H, W, 3) を順に返す。"""
    scale = min(1.0, max_width / width)
    out_w = max(2, int(round(width * scale / 2)) * 2)
    out_h = max(2, int(round(height * scale / 2)) * 2)
    proc = subprocess.Popen(
        [
            _require("ffmpeg"), "-v", "error", "-i", str(video_path), "-an",
            "-vf", f"fps={fps},scale={out_w}:{out_h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame_bytes = out_w * out_h * 3
    index = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield index / fps, np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
            index += 1
    finally:
        proc.kill()
        proc.wait()


def extract_wav(video_path: Path, wav_path: Path) -> Path:
    """16kHz モノラル WAV を抽出する (ローカル Whisper と音響解析用)。"""
    subprocess.run(
        [
            _require("ffmpeg"), "-y", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
    )
    return wav_path


def extract_compressed_audio(video_path: Path, out_path: Path) -> Path:
    """OpenAI API 送信用の小さい音声ファイル (25MB 制限対策、約 90 分まで収まる)。"""
    subprocess.run(
        [
            _require("ffmpeg"), "-y", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", "32k",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def load_wav(wav_path: Path) -> np.ndarray:
    """WAV を float32 [-1, 1] の配列として読み込む。"""
    import wave

    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
