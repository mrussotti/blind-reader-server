import pytest
from fastapi.testclient import TestClient

import ai
import main

client = TestClient(main.app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["version"] == "v2"


def test_frame_check_requires_image():
    assert client.post("/frame-check", json={}).status_code == 422
    assert client.post("/frame-check", json={"image": ""}).status_code == 400


def test_frame_check_rejects_oversized_image(monkeypatch):
    monkeypatch.setattr(main, "MAX_IMAGE_CHARS", 10)
    resp = client.post("/frame-check", json={"image": "x" * 11})
    assert resp.status_code == 400


def test_frame_check_happy_path(monkeypatch):
    fake = {"status": "adjust", "cue": "Move the phone left.", "needsLight": False}
    monkeypatch.setattr(ai, "frame_check", lambda image: fake)
    resp = client.post("/frame-check", json={"image": "abc123"})
    assert resp.status_code == 200
    assert resp.json() == fake


def test_frame_check_upstream_failure(monkeypatch):
    def boom(image):
        raise RuntimeError("api down")

    monkeypatch.setattr(ai, "frame_check", boom)
    resp = client.post("/frame-check", json={"image": "abc"})
    assert resp.status_code == 502


def test_transcribe_happy_path(monkeypatch):
    fake = {
        "summary": "This is a bill from National Grid.",
        "text": "Amount due: $84.00",
        "guidance": None,
        "documentType": "bill",
    }
    monkeypatch.setattr(ai, "transcribe", lambda image: fake)
    resp = client.post("/transcribe", json={"image": "abc"})
    assert resp.status_code == 200
    assert resp.json() == fake


def test_speak_returns_audio(monkeypatch):
    monkeypatch.setattr(ai, "tts", lambda text, voice: b"mp3-bytes")
    resp = client.post("/speak", json={"text": "Hello.", "voice": "nova"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert resp.content == b"mp3-bytes"


def test_speak_falls_back_to_default_voice(monkeypatch):
    seen = {}

    def fake_tts(text, voice):
        seen["voice"] = voice
        return b"x"

    monkeypatch.setattr(ai, "tts", fake_tts)
    resp = client.post("/speak", json={"text": "Hi.", "voice": "not-a-voice"})
    assert resp.status_code == 200
    assert seen["voice"] == "nova"


def test_speak_validates_text():
    assert client.post("/speak", json={"text": "   "}).status_code == 400
    assert client.post("/speak", json={"text": "x" * (main.MAX_SPEAK_CHARS + 1)}).status_code == 400


def test_as_data_url():
    assert ai.as_data_url("abc") == "data:image/jpeg;base64,abc"
    assert ai.as_data_url("data:image/png;base64,abc") == "data:image/png;base64,abc"


def test_split_data_url():
    assert ai.split_data_url("data:image/png;base64,abc") == ("image/png", "abc")
    assert ai.split_data_url("data:image/jpeg;base64,xyz") == ("image/jpeg", "xyz")
    assert ai.split_data_url("rawbase64") == ("image/jpeg", "rawbase64")


def test_vision_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert ai.vision_provider() == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert ai.vision_provider() == "openai"


def test_app_key_enforced(monkeypatch):
    monkeypatch.setattr(main, "APP_KEY", "secret123")
    monkeypatch.setattr(ai, "frame_check", lambda image: {"status": "good", "cue": None, "needsLight": False})
    assert client.post("/frame-check", json={"image": "abc"}).status_code == 401
    ok = client.post("/frame-check", json={"image": "abc"}, headers={"X-App-Key": "secret123"})
    assert ok.status_code == 200
    # healthz stays open for platform health checks
    assert client.get("/healthz").status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(ai, "frame_check", lambda image: {"status": "good", "cue": None, "needsLight": False})
    main._request_log.clear()
    try:
        codes = [
            client.post("/frame-check", json={"image": "abc"}).status_code
            for _ in range(5)
        ]
        assert codes[:3] == [200, 200, 200]
        assert codes[3] == 429
    finally:
        main._request_log.clear()


def test_parse_json_object():
    assert ai.parse_json_object('{"a": 1}') == {"a": 1}
    assert ai.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert ai.parse_json_object("not json") is None
    assert ai.parse_json_object("[1, 2]") is None
