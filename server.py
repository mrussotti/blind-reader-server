import os
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

app = Flask(__name__)
# Allow local dev devices to call the API
CORS(app, resources={
    r"/ocr": {"origins": "*"}, 
    r"/tts": {"origins": "*"}, 
    r"/healthz": {"origins": "*"}
})

client = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@app.post("/ocr")
def ocr():
    """Accepts JSON { image: <data URL or base64> } and returns { text }"""
    ip = request.access_route[0] if request.access_route else request.remote_addr
    t0 = time.perf_counter()

    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    img_info = "<none>"
    if isinstance(image, str):
        img_info = f"len={len(image)} startswith_data={str(image).startswith('data:image')}"
    print(f"[OCR] from {ip} body: {img_info}")

    if not image:
        return jsonify({"error": "Missing 'image' in body"}), 400

    # Accept raw base64 and wrap as data URL if needed
    if not str(image).startswith("data:image"):
        image = f"data:image/jpeg;base64,{image}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You transcribe images of documents for a blind user. "
                        "Return only the document text in reading order. "
                        "Do not add commentary or extra words. "
                        "If the image is unclear, do your best and indicate missing parts with ellipses."
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
        )
        text = (resp.choices[0].message.content or "").strip()
        t1 = time.perf_counter()
        preview = (text[:120] + ("…" if len(text) > 120 else "")).replace("\n", " ")
        print(f"[OCR] -> {len(text)} chars, {round((t1-t0)*1000)} ms, preview: {preview}")
        return jsonify({"text": text})
    except Exception as e:
        t1 = time.perf_counter()
        print(f"[OCR] ERROR after {round((t1-t0)*1000)} ms: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/tts")
def tts():
    """
    Accepts JSON { text: <string>, voice: <optional> }
    Returns streaming audio (MP3)
    Voice options: alloy, echo, fable, onyx, nova, shimmer
    """
    ip = request.access_route[0] if request.access_route else request.remote_addr
    t0 = time.perf_counter()

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    voice = payload.get("voice", "nova")  # nova is a warm female voice
    
    print(f"[TTS] from {ip} text_len={len(text)} voice={voice}")

    if not text:
        return jsonify({"error": "Missing 'text' in body"}), 400

    # Validate voice
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        voice = "nova"

    try:
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",  # tts-1 is faster, tts-1-hd is higher quality
            voice=voice,
            input=text,
            response_format="mp3"
        )
        
        t1 = time.perf_counter()
        print(f"[TTS] -> Generated in {round((t1-t0)*1000)} ms")
        
        # Stream the audio response
        def generate():
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk
        
        return Response(
            generate(),
            mimetype="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        t1 = time.perf_counter()
        print(f"[TTS] ERROR after {round((t1-t0)*1000)} ms: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True)