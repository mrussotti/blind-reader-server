import json
from unittest.mock import MagicMock, patch

import pytest

from server import MAX_IMAGE_SIZE, MAX_TEXT_LENGTH, app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── /healthz ──


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


# ── /ocr input validation ──


def test_ocr_missing_image(client):
    resp = client.post("/ocr", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_ocr_image_too_large(client):
    huge = "x" * (MAX_IMAGE_SIZE + 1)
    resp = client.post("/ocr", json={"image": huge})
    assert resp.status_code == 400
    assert "too large" in resp.get_json()["error"].lower()


# ── /tts input validation ──


def test_tts_missing_text(client):
    resp = client.post("/tts", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_tts_text_too_long(client):
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    resp = client.post("/tts", json={"text": long_text})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"].lower()


# ── /tts voice fallback ──


def test_tts_invalid_voice_defaults_to_nova(client):
    with patch("server.client") as mock_client:
        mock_response = MagicMock()
        mock_response.iter_bytes.return_value = [b"fake_audio"]
        mock_client.audio.speech.create.return_value = mock_response

        resp = client.post("/tts", json={"text": "hello", "voice": "invalid"})
        assert resp.status_code == 200
        mock_client.audio.speech.create.assert_called_once_with(
            model="tts-1-hd",
            voice="nova",
            input="hello",
            response_format="mp3",
        )


# ── /ocr success paths ──


@patch("server.client")
def test_ocr_success(mock_client, client):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        {"text": "Hello world", "guidance": None}
    )
    mock_client.chat.completions.create.return_value = mock_resp

    resp = client.post("/ocr", json={"image": "data:image/jpeg;base64,abc123"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["text"] == "Hello world"
    assert "guidance" not in data


@patch("server.client")
def test_ocr_with_guidance(mock_client, client):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        {"text": "Partial text", "guidance": "The image is too dark."}
    )
    mock_client.chat.completions.create.return_value = mock_resp

    resp = client.post("/ocr", json={"image": "data:image/jpeg;base64,abc123"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["text"] == "Partial text"
    assert data["guidance"] == "The image is too dark."


@patch("server.client")
def test_ocr_fallback_on_invalid_json(mock_client, client):
    """When the model returns non-JSON, treat it as plain text."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Just plain text response"
    mock_client.chat.completions.create.return_value = mock_resp

    resp = client.post("/ocr", json={"image": "data:image/jpeg;base64,abc123"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["text"] == "Just plain text response"


# ── /ocr error handling ──


@patch("server.client")
def test_ocr_api_error(mock_client, client):
    mock_client.chat.completions.create.side_effect = Exception("API down")

    resp = client.post("/ocr", json={"image": "data:image/jpeg;base64,abc123"})
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ── /tts success path ──


@patch("server.client")
def test_tts_success(mock_client, client):
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = [b"audio_data"]
    mock_client.audio.speech.create.return_value = mock_response

    resp = client.post("/tts", json={"text": "Hello"})
    assert resp.status_code == 200
    assert resp.content_type == "audio/mpeg"


# ── /tts error handling ──


@patch("server.client")
def test_tts_api_error(mock_client, client):
    mock_client.audio.speech.create.side_effect = Exception("TTS failed")

    resp = client.post("/tts", json={"text": "Hello"})
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ── /ocr accepts raw base64 (no data: prefix) ──


@patch("server.client")
def test_ocr_raw_base64(mock_client, client):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        {"text": "Some text", "guidance": None}
    )
    mock_client.chat.completions.create.return_value = mock_resp

    resp = client.post("/ocr", json={"image": "abc123base64data"})
    assert resp.status_code == 200

    # Verify the image was wrapped with data URL prefix
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    image_url = messages[1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
