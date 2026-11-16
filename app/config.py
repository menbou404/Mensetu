"""アプリ全体の設定。値は環境変数 (.env) から読み込む。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.getenv("MENSETU_DATA_DIR", ROOT_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = Path(os.getenv("MENSETU_MODELS_DIR", ROOT_DIR / "models"))

# ローカル Whisper (faster-whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# OpenAI 文字起こし
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

# Claude 講評
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "high")

# 映像解析のサンプリングレート (fps)
VISION_SAMPLE_FPS = float(os.getenv("VISION_SAMPLE_FPS", "8"))

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
