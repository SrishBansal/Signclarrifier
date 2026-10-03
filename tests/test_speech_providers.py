"""Unit tests for ASR and TTS provider interfaces using mocks.

Verifies:
1. Python ASRProvider and MockASRProvider interface & contract
2. Python TTSProvider and MockTTSProvider interface & WAV synthesis
3. Server Whisper / Server TTS fallback interfaces
4. FastAPI endpoints (/api/asr, /api/tts, /api/debug/log_interaction, /api/debug/interactions)
5. Client-side JS speech providers structure and language mapping
"""
import os
import sys
import json
import pytest
from starlette.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.speech import (
    ASRProvider,
    ASRResult,
    MockASRProvider,
    ServerWhisperASRProvider,
    TTSProvider,
    TTSResult,
    MockTTSProvider,
    ServerTTSProvider,
)
from app.server import create_app


# ── 1. ASR Provider Interface Tests ───────────────────────────────────────────

def test_mock_asr_provider_transcribe_languages():
    asr = MockASRProvider(default_confidence=0.92)
    assert asr.is_available() is True

    # Test audio payload (16000 samples of 16-bit PCM = 32000 bytes = 1 sec)
    dummy_audio = b"\x00\x01" * 16000

    for lang in ["en-IN", "hi-IN", "ta-IN"]:
        res = asr.transcribe(dummy_audio, language=lang)
        assert isinstance(res, ASRResult)
        assert res.language == lang
        assert res.confidence == pytest.approx(0.92)
        assert len(res.text) > 0
        assert res.duration_sec > 0

    assert len(asr.call_history) == 3


def test_mock_asr_predefined_lookup():
    asr = MockASRProvider()
    asr.set_mock_transcript("hi-IN", "नमस्ते मुझे पानी चाहिए")

    audio = b"some_simulated_speech_bytes_here"
    res = asr.transcribe(audio, language="hi-IN")
    assert res.text == "नमस्ते मुझे पानी चाहिए"
    assert res.language == "hi-IN"


def test_mock_asr_empty_audio_rejection():
    asr = MockASRProvider()
    with pytest.raises(ValueError, match="No speech detected or empty audio payload"):
        asr.transcribe(b"", language="en-IN")


def test_server_whisper_provider_interface():
    whisper_asr = ServerWhisperASRProvider()
    # Test fallback stub behavior
    audio = b"fake_audio_stream_for_whisper"
    res = whisper_asr.transcribe(audio, language="ta-IN")
    assert isinstance(res, ASRResult)
    assert res.language == "ta-IN"
    assert res.is_fallback is True

    with pytest.raises(ValueError, match="Empty audio data"):
        whisper_asr.transcribe(b"")


# ── 2. TTS Provider Interface Tests ───────────────────────────────────────────

def test_mock_tts_provider_synthesize_and_wav_format():
    tts = MockTTSProvider()
    assert tts.is_available() is True

    # Check voices
    voices_en = tts.list_voices("en-IN")
    assert len(voices_en) >= 1
    assert "Ravi" in voices_en[0]["name"]

    voices_hi = tts.list_voices("hi-IN")
    assert len(voices_hi) >= 1

    voices_ta = tts.list_voices("ta-IN")
    assert len(voices_ta) >= 1

    # Synthesize phrase
    res = tts.synthesize("Hello, welcome to ClarifySign", language="en-IN")
    assert isinstance(res, TTSResult)
    assert res.mime_type == "audio/wav"
    assert res.language == "en-IN"
    assert res.duration_sec > 0
    assert len(res.audio_data) > 44  # Header + PCM payload

    # Validate RIFF / WAV binary header
    header = res.audio_data[:44]
    assert header[0:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert header[36:40] == b"data"


def test_mock_tts_empty_text_rejection():
    tts = MockTTSProvider()
    with pytest.raises(ValueError, match="Cannot synthesize empty text"):
        tts.synthesize("", language="hi-IN")


def test_server_tts_provider_fallback():
    server_tts = ServerTTSProvider(fallback_to_mock=True)
    assert server_tts.is_available() is True
    res = server_tts.synthesize("धन्यवाद", language="hi-IN")
    assert res.provider == "server-fallback"
    assert res.mime_type == "audio/wav"
    assert len(res.audio_data) > 44


# ── 3. FastAPI Server Endpoints Tests ──────────────────────────────────────────

def test_server_asr_and_tts_endpoints():
    app = create_app(reason="Test mock backend")
    client = TestClient(app)

    # 1. ASR endpoint with valid audio
    audio_data = b"dummy_sample_voice_bytes_long_enough"
    resp_asr = client.post("/api/asr?lang=hi-IN", content=audio_data)
    assert resp_asr.status_code == 200
    data_asr = resp_asr.json()
    assert "text" in data_asr
    assert "confidence" in data_asr
    assert data_asr["language"] == "hi-IN"

    # Empty audio rejection
    resp_asr_empty = client.post("/api/asr", content=b"")
    assert resp_asr_empty.status_code == 400

    # 2. TTS endpoint
    resp_tts = client.post("/api/tts", json={"text": "water bottle", "lang": "en-IN"})
    assert resp_tts.status_code == 200
    assert resp_tts.headers["content-type"] == "audio/wav"
    assert len(resp_tts.content) > 44
    assert resp_tts.content[:4] == b"RIFF"

    # Empty text rejection
    resp_tts_empty = client.post("/api/tts", json={"text": ""})
    assert resp_tts_empty.status_code == 400


def test_server_side_by_side_debug_logging():
    app = create_app(reason="Test mock backend")
    client = TestClient(app)

    log_entry = {
        "asr_text": "water bottle please",
        "confidence": 0.94,
        "edited_text": "water bottle please",
        "language": "en",
        "downstream_state": {"concepts": ["WATER", "BOTTLE"]}
    }

    # Submit side-by-side log entry
    resp_log = client.post("/api/debug/log_interaction", json=log_entry)
    assert resp_log.status_code == 200
    assert resp_log.json().get("logged") is True

    # Retrieve debug interactions
    resp_get = client.get("/api/debug/interactions")
    assert resp_get.status_code == 200
    interactions = resp_get.json().get("interactions", [])
    assert len(interactions) >= 1

    last = interactions[-1]
    assert last["asr_text"] == "water bottle please"
    assert last["confidence"] == pytest.approx(0.94)
    assert last["downstream_state"]["concepts"] == ["WATER", "BOTTLE"]


# ── 4. Client-side JS Speech Module Node Verification ──────────────────────────

def test_js_speech_providers_structure():
    import subprocess
    js_test_script = """
    const { LANG_TAGS, WebSpeechASRProvider, ServerFallbackASRProvider, WebSpeechTTSProvider, ServerFallbackTTSProvider } = require('./app/static/speech.js');
    console.assert(LANG_TAGS.en === 'en-IN', 'en mapping');
    console.assert(LANG_TAGS.hi === 'hi-IN', 'hi mapping');
    console.assert(LANG_TAGS.ta === 'ta-IN', 'ta mapping');

    const asr = new WebSpeechASRProvider();
    const serverAsr = new ServerFallbackASRProvider();
    const tts = new WebSpeechTTSProvider();
    const serverTts = new ServerFallbackTTSProvider();

    console.assert(typeof asr.start === 'function', 'asr.start');
    console.assert(typeof tts.speak === 'function', 'tts.speak');
    console.assert(typeof tts.cancel === 'function', 'tts.cancel');
    console.log('JS_SPEECH_OK');
    """
    cmd = ["node", "-e", js_test_script]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "JS_SPEECH_OK" in res.stdout
