"""
app.py
------
FastAPI server exposing the Bangla speech-to-summary pipeline as a REST API.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                -> {"status": "ok"}
    POST /api/transcribe-summarize
         multipart/form-data with a single field "audio" (file upload: mp3/wav)
         -> {"transcript": "...", "summary": "...", "duration_sec": 4.2}
"""

import time
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import inference  # loads both models at import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bangla-api")

app = FastAPI(title="Bangla Speech-to-Summary API", version="1.0.0")

# CORS: restrict allow_origins to your actual website domain(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g. ["https://your-website.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB upload limit
ALLOWED_CONTENT_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/webm", "audio/ogg"}


class PipelineResponse(BaseModel):
    transcript: str
    summary: str
    duration_sec: float
    processing_time_sec: float


@app.get("/")
def serve_index():
    """Serves the website's homepage (index.html) from the same server/port
    as the API, so only one process/terminal is needed to run everything."""
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(inference.DEVICE)}


@app.post("/api/transcribe-summarize", response_model=PipelineResponse)
async def transcribe_summarize(audio: UploadFile = File(...)):
    if audio.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Rejected upload with content type: {audio.content_type}")
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {audio.content_type}")

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large (max 25MB).")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    start = time.time()
    try:
        result = inference.transcribe_and_summarize(audio_bytes)
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    elapsed = time.time() - start

    return PipelineResponse(
        transcript=result["transcript"],
        summary=result["summary"],
        duration_sec=result["duration_sec"],
        processing_time_sec=round(elapsed, 2),
    )
