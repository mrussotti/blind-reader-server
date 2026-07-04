"""Mail Reader server: a thin API over OpenAI vision + TTS.

Endpoints: /healthz, /frame-check, /transcribe, /speak.
Privacy rule: never log document content. It's someone's mail; log sizes
and timings only.
"""
import logging
import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_IMAGE_CHARS = 14_000_000  # ~10 MB of base64
MAX_SPEAK_CHARS = 1_200       # the app sends sentence-sized chunks
VALID_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

# Shared secret the app sends as X-App-Key. Optional: when unset (local dev),
# no auth is required. Set the same value in the server env and the EAS build
# env so a public deployment isn't an open proxy to the AI keys.
APP_KEY = os.environ.get("APP_KEY")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))

_request_log = defaultdict(deque)


def guard(request: Request) -> None:
    """Per-request app-key check + per-IP sliding-window rate limit."""
    if APP_KEY and request.headers.get("x-app-key") != APP_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = forwarded or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    if len(_request_log) > 10_000:
        _request_log.clear()
    window = _request_log[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests")
    window.append(now)

app = FastAPI(title="Mail Reader server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ImageBody(BaseModel):
    image: str


class SpeakBody(BaseModel):
    text: str
    voice: str = ai.TTS_VOICE_DEFAULT


def _validate_image(image: str) -> None:
    if not image:
        raise HTTPException(status_code=400, detail="Missing image")
    if len(image) > MAX_IMAGE_CHARS:
        raise HTTPException(status_code=400, detail="Image too large")


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": "v2"}


@app.post("/frame-check")
def frame_check(body: ImageBody, _: None = Depends(guard)):
    _validate_image(body.image)
    t0 = time.perf_counter()
    try:
        result = ai.frame_check(body.image)
    except Exception:
        logger.exception("[frame-check] failed")
        raise HTTPException(status_code=502, detail="Frame check failed")
    ms = round((time.perf_counter() - t0) * 1000)
    logger.info("[frame-check] %s needsLight=%s in %d ms", result["status"], result["needsLight"], ms)
    return result


@app.post("/transcribe")
def transcribe(body: ImageBody, _: None = Depends(guard)):
    _validate_image(body.image)
    t0 = time.perf_counter()
    try:
        result = ai.transcribe(body.image)
    except Exception:
        logger.exception("[transcribe] failed")
        raise HTTPException(status_code=502, detail="Transcription failed")
    ms = round((time.perf_counter() - t0) * 1000)
    logger.info(
        "[transcribe] type=%s text_chars=%d guidance=%s in %d ms",
        result["documentType"], len(result["text"]), "yes" if result["guidance"] else "no", ms,
    )
    return result


@app.post("/speak")
def speak(body: SpeakBody, _: None = Depends(guard)):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    if len(text) > MAX_SPEAK_CHARS:
        raise HTTPException(
            status_code=400, detail=f"Text too long (max {MAX_SPEAK_CHARS} chars per chunk)"
        )
    voice = body.voice if body.voice in VALID_VOICES else "nova"
    t0 = time.perf_counter()
    try:
        audio = ai.tts(text, voice)
    except Exception:
        logger.exception("[speak] failed")
        raise HTTPException(status_code=502, detail="Speech synthesis failed")
    ms = round((time.perf_counter() - t0) * 1000)
    logger.info("[speak] %d chars -> %d bytes in %d ms", len(text), len(audio), ms)
    return Response(
        content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-cache"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 4000)))
