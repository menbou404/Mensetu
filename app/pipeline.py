"""解析ジョブの実行とファイル保存。

data/jobs/<job_id>/
    input.<ext>     アップロードされた動画
    status.json     進捗
    result.json     解析結果
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, media, review, speech, transcribe, vision

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# 重い処理なので 1 本ずつ順番に実行する
_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


def job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise ValueError("invalid job id")
    return config.JOBS_DIR / job_id


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_input(job_id: str) -> Path | None:
    return next(job_dir(job_id).glob("input.*"), None)


class Status:
    """進捗を status.json に書き出す。"""

    WEIGHTS = {"extract": 0.05, "transcribe": 0.45, "vision": 0.25, "speech": 0.05, "review": 0.20}

    def __init__(self, job_id: str, data: dict):
        self.path = job_dir(job_id) / "status.json"
        self.data = data
        self.parts = {k: 0.0 for k in self.WEIGHTS}

    def save(self) -> None:
        with _lock:
            self.data["progress"] = round(sum(self.WEIGHTS[k] * v for k, v in self.parts.items()), 3)
            self.data["updated_at"] = time.time()
            _write_json(self.path, self.data)

    def part(self, name: str, value: float, step: str | None = None) -> None:
        self.parts[name] = value
        if step:
            self.data["step"] = step
        self.save()


def create_job(filename: str, options: dict) -> str:
    job_id = uuid.uuid4().hex
    d = job_dir(job_id)
    d.mkdir(parents=True)
    _write_json(d / "status.json", {
        "id": job_id, "filename": filename, "options": options, "state": "queued",
        "step": "待機中", "progress": 0.0, "error": None, "created_at": time.time(),
    })
    return job_id


def submit(job_id: str) -> None:
    _executor.submit(_run, job_id)


def submit_review(job_id: str) -> None:
    _executor.submit(_run_review_only, job_id)


def _run(job_id: str) -> None:
    d = job_dir(job_id)
    st = Status(job_id, read_json(d / "status.json"))
    opts = st.data["options"]
    st.data["state"] = "running"
    try:
        video_path = find_input(job_id)
        st.part("extract", 0.0, "音声を抽出中")
        meta = media.probe(video_path)
        wav = media.extract_wav(video_path, d / "audio.wav")
        samples = media.load_wav(wav)
        st.part("extract", 1.0)

        # 映像解析は文字起こしと並行して走らせる
        vision_result: dict = {}
        vision_thread = None
        if opts.get("use_vision", True) and meta["has_video"]:
            def run_vision():
                try:
                    vision_result.update(vision.analyze(
                        video_path, meta, on_progress=lambda p: st.part("vision", p)))
                except Exception as e:  # 映像解析が失敗しても他の結果は返す
                    traceback.print_exc()
                    vision_result.update({"available": False, "reason": f"映像解析に失敗しました: {e}"})
                st.part("vision", 1.0)
            vision_thread = threading.Thread(target=run_vision, daemon=True)
            vision_thread.start()
        else:
            vision_result.update({"available": False, "reason": "映像解析はスキップされました。"})
            st.parts["vision"] = 1.0

        if opts["engine"] == "openai":
            st.part("transcribe", 0.1, "文字起こし中 (OpenAI API)")
            mp3 = media.extract_compressed_audio(video_path, d / "audio.mp3")
            transcript = transcribe.transcribe_openai(mp3)
            mp3.unlink(missing_ok=True)
        else:
            model = opts.get("whisper_model") or config.WHISPER_MODEL
            st.part("transcribe", 0.0, f"文字起こし中 (ローカル Whisper {model}・初回はモデルをダウンロードします)")
            transcript = transcribe.transcribe_local(
                samples, meta["duration"], model,
                on_progress=lambda p: st.part("transcribe", p))
        st.part("transcribe", 1.0)

        st.part("speech", 0.0, "話し方を分析中")
        speech_result = speech.analyze(transcript["segments"], samples, meta["duration"])
        st.part("speech", 1.0)

        if vision_thread:
            st.data["step"] = "表情・視線を分析中"
            st.save()
            vision_thread.join()

        result = {
            "meta": meta,
            "transcript": transcript,
            "speech": speech_result,
            "vision": vision_result,
            "review": None,
            "review_error": None,
        }
        _write_json(d / "result.json", result)

        if opts.get("use_ai"):
            st.part("review", 0.1, "AI が回答内容を講評中 (1〜3 分ほどかかります)")
            _do_review(result, opts.get("context", ""))
            _write_json(d / "result.json", result)
        st.part("review", 1.0)

        wav.unlink(missing_ok=True)
        st.data.update({"state": "done", "step": "完了"})
        st.save()
    except Exception as e:
        traceback.print_exc()
        st.data.update({"state": "error", "error": str(e), "step": "エラー"})
        st.save()


def _do_review(result: dict, context: str) -> None:
    try:
        result["review"] = review.review(result["transcript"], result["speech"], result["vision"], context)
        result["review_error"] = None
    except Exception as e:
        traceback.print_exc()
        result["review_error"] = str(e)


def _run_review_only(job_id: str) -> None:
    d = job_dir(job_id)
    st = Status(job_id, read_json(d / "status.json"))
    result = read_json(d / "result.json")
    if result is None:
        return
    st.parts = {k: 1.0 for k in st.WEIGHTS}
    st.parts["review"] = 0.1
    st.data.update({"state": "running", "step": "AI が回答内容を講評中 (1〜3 分ほどかかります)"})
    st.data["options"]["use_ai"] = True
    st.save()
    _do_review(result, st.data["options"].get("context", ""))
    _write_json(d / "result.json", result)
    st.parts["review"] = 1.0
    st.data.update({"state": "done", "step": "完了"})
    st.save()


def list_jobs() -> list[dict]:
    if not config.JOBS_DIR.exists():
        return []
    jobs = [read_json(p / "status.json") for p in config.JOBS_DIR.iterdir() if p.is_dir()]
    return sorted([j for j in jobs if j], key=lambda j: j["created_at"], reverse=True)


def recover_interrupted() -> None:
    """サーバー再起動で中断されたジョブをエラー扱いにする。"""
    for job in list_jobs():
        if job["state"] in ("queued", "running"):
            job.update({"state": "error", "error": "サーバーが再起動されたため中断されました。", "step": "エラー"})
            _write_json(job_dir(job["id"]) / "status.json", job)
