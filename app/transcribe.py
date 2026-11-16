"""文字起こし。ローカル Whisper (faster-whisper) と OpenAI API の 2 方式に対応。

どちらも次の共通フォーマットを返す:
    {
      "engine": "local" | "openai",
      "model": str,
      "segments": [{"start": float, "end": float, "text": str}],
      "words":    [{"start": float, "end": float, "word": str}],
    }
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np

from . import config

# Whisper はフィラー (えー/あのー) を省略しがちなので、
# フィラーを含む文体のプロンプトを与えて書き起こしに残りやすくする。
FILLER_PROMPT = "えー、あのー、えっと、まあ、そうですね、はい。よろしくお願いします。"

ProgressFn = Callable[[float], None]

_model_cache: dict[str, object] = {}
_model_lock = Lock()


def _get_local_model(name: str):
    from faster_whisper import WhisperModel

    with _model_lock:
        if name not in _model_cache:
            _model_cache[name] = WhisperModel(
                name,
                device="cpu",
                compute_type=config.WHISPER_COMPUTE_TYPE,
                download_root=str(config.MODELS_DIR / "whisper"),
            )
        return _model_cache[name]


def transcribe_local(samples: np.ndarray, duration: float, model_name: str | None = None,
                     on_progress: ProgressFn | None = None) -> dict:
    """samples: 16kHz モノラル float32 の音声 (media.load_wav の戻り値)。"""
    model_name = model_name or config.WHISPER_MODEL
    model = _get_local_model(model_name)
    seg_iter, _info = model.transcribe(
        samples,
        language="ja",
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    segments, words = [], []
    for seg in seg_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        for w in seg.words or []:
            words.append({"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word})
        if on_progress and duration > 0:
            on_progress(min(seg.end / duration, 1.0))
    return {"engine": "local", "model": model_name, "segments": segments, "words": words}


def transcribe_openai(audio_path: Path, model_name: str | None = None) -> dict:
    if not config.has_openai_key():
        raise RuntimeError("OPENAI_API_KEY が設定されていません (.env を確認してください)")
    from openai import OpenAI

    model_name = model_name or config.OPENAI_TRANSCRIBE_MODEL
    client = OpenAI()
    with open(audio_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            language="ja",
            prompt=FILLER_PROMPT,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in (res.segments or [])
        if s.text.strip()
    ]
    words = [
        {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
        for w in (res.words or [])
    ]
    return {"engine": "openai", "model": model_name, "segments": segments, "words": words}
