"""FastAPI サーバー。`uv run mensetu` で起動し、http://127.0.0.1:8000 を開く。"""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, pipeline

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.recover_interrupted()
    yield


app = FastAPI(title="面接振り返りアナライザー", lifespan=lifespan)


def _job_dir_or_404(job_id: str) -> Path:
    try:
        d = pipeline.job_dir(job_id)
    except ValueError:
        raise HTTPException(404)
    if not d.exists():
        raise HTTPException(404)
    return d


@app.get("/api/config")
def get_config():
    return {
        "openai_available": config.has_openai_key(),
        "anthropic_available": config.has_anthropic_key(),
        "whisper_models": WHISPER_MODELS,
        "whisper_default": config.WHISPER_MODEL,
        "claude_model": config.CLAUDE_MODEL,
    }


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    engine: str = Form("local"),
    whisper_model: str = Form(""),
    use_ai: bool = Form(False),
    use_vision: bool = Form(True),
    context: str = Form(""),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"対応していない形式です ({', '.join(sorted(ALLOWED_EXT))})")
    if engine not in ("local", "openai"):
        raise HTTPException(400, "engine は local か openai を指定してください")
    if engine == "openai" and not config.has_openai_key():
        raise HTTPException(400, "OPENAI_API_KEY が設定されていません")
    if use_ai and not config.has_anthropic_key():
        raise HTTPException(400, "ANTHROPIC_API_KEY が設定されていません")
    if whisper_model and whisper_model not in WHISPER_MODELS:
        raise HTTPException(400, "不明な Whisper モデルです")

    options = {"engine": engine, "whisper_model": whisper_model or config.WHISPER_MODEL,
               "use_ai": use_ai, "use_vision": use_vision, "context": context[:2000]}
    job_id = pipeline.create_job(file.filename or f"video{ext}", options)
    with open(pipeline.job_dir(job_id) / f"input{ext}", "wb") as out:
        shutil.copyfileobj(file.file, out, length=1024 * 1024)
    pipeline.submit(job_id)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return pipeline.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    d = _job_dir_or_404(job_id)
    status = pipeline.read_json(d / "status.json")
    result = pipeline.read_json(d / "result.json") if status and status["state"] == "done" else None
    return {"status": status, "result": result}


@app.post("/api/jobs/{job_id}/review")
def rerun_review(job_id: str, context: str = Form("")):
    d = _job_dir_or_404(job_id)
    if not config.has_anthropic_key():
        raise HTTPException(400, "ANTHROPIC_API_KEY が設定されていません")
    status = pipeline.read_json(d / "status.json")
    if not status or status["state"] != "done":
        raise HTTPException(409, "解析が完了していません")
    if context:
        status["options"]["context"] = context[:2000]
        pipeline._write_json(d / "status.json", status)
    pipeline.submit_review(job_id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    _job_dir_or_404(job_id)
    path = pipeline.find_input(job_id)
    if not path:
        raise HTTPException(404)
    return FileResponse(path)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    d = _job_dir_or_404(job_id)
    status = pipeline.read_json(d / "status.json")
    if status and status["state"] in ("queued", "running"):
        raise HTTPException(409, "解析中のジョブは削除できません")
    shutil.rmtree(d)
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    import os

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  面接振り返りアナライザー: http://{host}:{port}\n")
    uvicorn.run("app.main:app", host=host, port=port)
