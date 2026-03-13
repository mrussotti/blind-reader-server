import json
import logging
import os
import time

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── OpenAI client ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Flask app ──
app = Flask(__name__)
CORS(app, resources={
    r"/ocr": {"origins": "*"},
    r"/tts": {"origins": "*"},
    r"/healthz": {"origins": "*"},
})

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["30 per minute"],
    storage_uri="memory://",
)

# ── Validation limits ──
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_LENGTH = 5000


@app.get("/healthz")
@limiter.exempt
def healthz():
    return jsonify({"ok": True})


@app.post("/ocr")
def ocr():
    """Accepts JSON { image: <data URL or base64> } and returns { text, guidance? }."""
    ip = request.access_route[0] if request.access_route else request.remote_addr
    t0 = time.perf_counter()

    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    img_info = "<none>"
    if isinstance(image, str):
        img_info = f"len={len(image)} startswith_data={str(image).startswith('data:image')}"
    logger.info("[OCR] from %s body: %s", ip, img_info)

    if not image:
        return jsonify({"error": "Missing 'image' in body"}), 400

    # Input validation
    if isinstance(image, str) and len(image) > MAX_IMAGE_SIZE:
        return jsonify({"error": "Image too large (max 10 MB)"}), 400

    # Accept raw base64 and wrap as data URL if needed
    if not str(image).startswith("data:image"):
        image = f"data:image/jpeg;base64,{image}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You transcribe images of documents for a blind user. "
                        "Return a JSON object with two fields:\n"
                        '- "text": the document text in reading order. Do not add commentary or extra words. '
                        "If the image is unclear, do your best and indicate missing parts with ellipses.\n"
                        '- "guidance": positioning guidance for the user, or null if the image is clear. '
                        'Examples: "The image is too dark. Try turning on a light.", '
                        '"The document appears tilted. Try holding the phone more level.", '
                        '"The text is blurry. Try moving further from the document.", '
                        '"Only part of the document is visible. Try moving the phone to center the page."\n'
                        "Always return valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this document exactly."},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw = (resp.choices[0].message.content or "").strip()

        # Parse structured JSON response
        try:
            parsed = json.loads(raw)
            text = parsed.get("text", "").strip()
            guidance = parsed.get("guidance")
            if guidance:
                guidance = guidance.strip()
        except json.JSONDecodeError:
            # Fallback: treat entire response as plain text
            text = raw
            guidance = None

        t1 = time.perf_counter()
        preview = (text[:120] + ("\u2026" if len(text) > 120 else "")).replace("\n", " ")
        logger.info(
            "[OCR] -> %d chars, %d ms, guidance=%s, preview: %s",
            len(text), round((t1 - t0) * 1000),
            "yes" if guidance else "no", preview,
        )

        result = {"text": text}
        if guidance:
            result["guidance"] = guidance
        return jsonify(result)

    except Exception as e:
        t1 = time.perf_counter()
        logger.error("[OCR] ERROR after %d ms: %s", round((t1 - t0) * 1000), e)
        return jsonify({"error": str(e)}), 500


@app.post("/tts")
def tts():
    """
    Accepts JSON { text: <string>, voice: <optional> }
    Returns streaming audio (MP3).
    Voice options: alloy, echo, fable, onyx, nova, shimmer
    """
    ip = request.access_route[0] if request.access_route else request.remote_addr
    t0 = time.perf_counter()

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    voice = payload.get("voice", "nova")

    logger.info("[TTS] from %s text_len=%d voice=%s", ip, len(text), voice)

    if not text:
        return jsonify({"error": "Missing 'text' in body"}), 400

    # Input validation
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text too long (max {MAX_TEXT_LENGTH} chars)"}), 400

    # Validate voice
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        voice = "nova"

    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=text,
            response_format="mp3",
        )

        t1 = time.perf_counter()
        logger.info("[TTS] -> Generated in %d ms", round((t1 - t0) * 1000))

        def generate():
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk

        return Response(
            generate(),
            mimetype="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Cache-Control": "no-cache",
            },
        )

    except Exception as e:
        t1 = time.perf_counter()
        logger.error("[TTS] ERROR after %d ms: %s", round((t1 - t0) * 1000), e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True)
