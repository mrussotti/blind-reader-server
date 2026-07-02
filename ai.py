"""AI calls and prompts for the Mail Reader server.

Vision (frame-check + transcribe) prefers Anthropic when ANTHROPIC_API_KEY is
set, falling back to OpenAI. TTS is OpenAI-only (Anthropic has no TTS API);
when it's unavailable the app falls back to the phone's local voice.

Privacy rule: nothing in this module logs document content. It's someone's
mail; log sizes and timings only.
"""
import json
import os

# ── Models ──
ANTHROPIC_FRAME_MODEL = os.environ.get("ANTHROPIC_FRAME_MODEL", "claude-haiku-4-5")
ANTHROPIC_TRANSCRIBE_MODEL = os.environ.get("ANTHROPIC_TRANSCRIBE_MODEL", "claude-opus-4-8")
OPENAI_FRAME_MODEL = os.environ.get("FRAME_MODEL", "gpt-4.1-mini")
OPENAI_TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4.1")
TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1-hd")
TTS_VOICE_DEFAULT = os.environ.get("TTS_VOICE", "nova")

FRAME_STATUSES = {"good", "adjust", "no_document"}
DOC_TYPES = {"letter", "bill", "statement", "advertisement", "envelope", "other"}

_anthropic_client = None
_openai_client = None


def vision_provider() -> str:
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"


def anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


# ── Image helpers ──

def as_data_url(image: str) -> str:
    """Accept a raw base64 string or a data URL; return a data URL."""
    if image.startswith("data:image"):
        return image
    return f"data:image/jpeg;base64,{image}"


def split_data_url(image: str) -> tuple:
    """Return (media_type, bare_base64) from a data URL or raw base64 string."""
    if image.startswith("data:") and ";base64," in image:
        header, data = image.split(";base64,", 1)
        media_type = header[5:] or "image/jpeg"
        return media_type, data
    return "image/jpeg", image


def downscale_image(image: str, max_edge: int = 640) -> str:
    """Shrink an image for the fast framing check.

    Phones send full-sensor-resolution snapshots; the framing verdict doesn't
    need that, and smaller images make the check dramatically faster. Returns
    a JPEG data URL; falls back to the original on any failure.
    """
    try:
        import base64 as b64
        import io

        from PIL import Image

        _, data = split_data_url(image)
        img = Image.open(io.BytesIO(b64.b64decode(data)))
        if max(img.size) <= max_edge:
            return image
        img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + b64.b64encode(buf.getvalue()).decode()
    except Exception:
        return image


def parse_json_object(raw: str):
    """Parse a model response that should be a JSON object.

    Tolerates markdown code fences. Returns None if it isn't a JSON object.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ── Prompts ──

FRAME_PROMPT = (
    "You are the eyes of a blind, elderly user who is holding their phone camera "
    "over a piece of mail so an app can read it aloud. You see one camera frame. "
    "Decide whether it is good enough to photograph and transcribe.\n\n"
    "Return a JSON object:\n"
    '{"status": "good" | "adjust" | "no_document", "cue": string or null, '
    '"needs_light": true or false}\n\n'
    '- "good": a page or envelope is clearly visible, fills most of the frame, '
    "and the text looks readable.\n"
    '- "adjust": a document is visible but the shot needs fixing. Set "cue" to ONE '
    "short spoken instruction, eight words or fewer, telling the user what to "
    'physically do. Examples: "Move the phone a little left.", '
    '"Raise the phone higher.", "Hold still."\n'
    '- "no_document": no page or envelope is in view. Set "cue" to null.\n'
    '- "needs_light": true only if the image is too dark to read.\n\n'
    "Cues must be plain spoken language for an elderly person. Never mention "
    "cameras, framing, pixels, focus, or images; say what to do with the phone "
    "or the letter."
)

TRANSCRIBE_PROMPT = (
    "You transcribe a photo of a piece of mail for a blind, elderly user who "
    "will hear the result read aloud.\n\n"
    "Return a JSON object:\n"
    '{"summary": string, "text": string, "guidance": string or null, '
    '"document_type": string}\n\n'
    '- "summary": ONE short spoken sentence saying what this is and who it is '
    'from. Examples: "This is a bill from National Grid.", "This looks like an '
    'advertisement from a car dealership.", "This is an envelope from the Social '
    'Security Administration." If a bill shows a clearly readable total due, you '
    "may include it.\n"
    '- "text": the document text, verbatim, in natural reading order. No '
    "commentary, no descriptions, no invented words. Mark unreadable parts with "
    '"...". Never guess at names, numbers, or dollar amounts you cannot actually '
    "read; this is someone's mail and mistakes matter. For an envelope, read who "
    "it is from and who it is addressed to.\n"
    '- "guidance": null if the photo is usable. Otherwise ONE short spoken '
    "instruction to fix it, in plain language.\n"
    '- "document_type": one of "letter", "bill", "statement", "advertisement", '
    '"envelope", "other".\n\n'
    'If the image is unusable, set "text" to "" and provide "guidance".'
)

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["good", "adjust", "no_document"]},
        "cue": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "needs_light": {"type": "boolean"},
    },
    "required": ["status", "cue", "needs_light"],
    "additionalProperties": False,
}

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "text": {"type": "string"},
        "guidance": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "document_type": {
            "type": "string",
            "enum": ["letter", "bill", "statement", "advertisement", "envelope", "other"],
        },
    },
    "required": ["summary", "text", "guidance", "document_type"],
    "additionalProperties": False,
}


# ── Vision backends ──

def _anthropic_vision_json(model, system, user_text, image, schema, max_tokens):
    """Vision request via Anthropic with structured output. Returns (parsed, raw)."""
    media_type, data = split_data_url(image)
    resp = anthropic_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    if resp.stop_reason == "refusal":
        return None, ""
    raw = next((b.text for b in resp.content if b.type == "text"), "").strip()
    return parse_json_object(raw), raw


def _openai_vision_json(model, system, user_text, image, detail, max_tokens=None):
    """Vision request via OpenAI. Returns (parsed, raw)."""
    kwargs = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    resp = openai_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": as_data_url(image), "detail": detail},
                    },
                ],
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
        **kwargs,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return parse_json_object(raw), raw


# ── Public API ──

def frame_check(image: str) -> dict:
    image = downscale_image(image)
    if vision_provider() == "anthropic":
        parsed, _ = _anthropic_vision_json(
            ANTHROPIC_FRAME_MODEL, FRAME_PROMPT, "Check this camera frame.",
            image, FRAME_SCHEMA, max_tokens=200,
        )
    else:
        parsed, _ = _openai_vision_json(
            OPENAI_FRAME_MODEL, FRAME_PROMPT, "Check this camera frame.",
            image, detail="low", max_tokens=80,
        )
    parsed = parsed or {}
    status = parsed.get("status")
    if status not in FRAME_STATUSES:
        status = "adjust"
    cue = parsed.get("cue")
    cue = cue.strip() if isinstance(cue, str) and cue.strip() else None
    return {"status": status, "cue": cue, "needsLight": bool(parsed.get("needs_light"))}


def transcribe(image: str) -> dict:
    if vision_provider() == "anthropic":
        parsed, raw = _anthropic_vision_json(
            ANTHROPIC_TRANSCRIBE_MODEL, TRANSCRIBE_PROMPT,
            "Transcribe this piece of mail.", image, TRANSCRIBE_SCHEMA,
            max_tokens=8000,
        )
    else:
        parsed, raw = _openai_vision_json(
            OPENAI_TRANSCRIBE_MODEL, TRANSCRIBE_PROMPT,
            "Transcribe this piece of mail.", image, detail="high",
        )
    if parsed is None:
        # Model broke the JSON contract; treat the raw response as the document
        # text rather than losing the user's letter.
        parsed = {"text": raw}
    text = (parsed.get("text") or "").strip()
    summary = (parsed.get("summary") or "").strip()
    guidance = parsed.get("guidance")
    guidance = guidance.strip() if isinstance(guidance, str) and guidance.strip() else None
    doc_type = parsed.get("document_type")
    if doc_type not in DOC_TYPES:
        doc_type = "other"
    return {"summary": summary, "text": text, "guidance": guidance, "documentType": doc_type}


def tts(text: str, voice: str) -> bytes:
    resp = openai_client().audio.speech.create(
        model=TTS_MODEL, voice=voice, input=text, response_format="mp3",
    )
    return resp.read()
