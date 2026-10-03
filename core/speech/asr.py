"""ASR (Automated Speech Recognition) provider interfaces and implementations.

Decoupled provider interface supporting:
- Web Speech API proxy / mock
- Server-side ASR (e.g. Whisper fallback)
- Confidence scoring and word-level alternatives
- Error classification (permission, silence, network, unsupported)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time


@dataclass
class ASRResult:
    """Standardized speech recognition result."""
    text: str
    confidence: float  # 0.0 to 1.0
    language: str      # e.g. 'en-IN', 'hi-IN', 'ta-IN'
    is_fallback: bool = False
    duration_sec: float = 0.0
    raw_provider: str = "mock"
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class ASRProvider(ABC):
    """Abstract Base Class for Speech-to-Text providers."""

    @abstractmethod
    def transcribe(self, audio_data: bytes, language: str = "en-IN") -> ASRResult:
        """Transcribe audio bytes into text with confidence score."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available in current environment."""
        pass


class MockASRProvider(ASRProvider):
    """Mock ASR provider for testing, development, and offline environments."""

    def __init__(self, predefined_transcripts: Optional[Dict[str, str]] = None, default_confidence: float = 0.95):
        self.predefined = predefined_transcripts or {}
        self.default_confidence = default_confidence
        self.call_history: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        return True

    def set_mock_transcript(self, key: str, text: str):
        self.predefined[key] = text

    def transcribe(self, audio_data: bytes, language: str = "en-IN") -> ASRResult:
        self.call_history.append({
            "bytes_len": len(audio_data),
            "language": language,
            "timestamp": time.time()
        })

        if not audio_data or len(audio_data) < 10:
            raise ValueError("No speech detected or empty audio payload")

        # Check predefined lookup or synthesize sensible transcript
        key = str(len(audio_data))
        text = self.predefined.get(key, self.predefined.get(language, "ek bottle paani dena"))
        return ASRResult(
            text=text,
            confidence=self.default_confidence,
            language=language,
            is_fallback=False,
            duration_sec=round(len(audio_data) / 16000.0, 2),
            raw_provider="mock"
        )


class ServerWhisperASRProvider(ASRProvider):
    """Server-side ASR fallback interface (Whisper / faster-whisper)."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._loaded = False

    def is_available(self) -> bool:
        try:
            import whisper
            return True
        except ImportError:
            return False

    def _ensure_model(self):
        if not self._loaded and self.is_available():
            import whisper
            self._model = whisper.load_model(self.model_name)
            self._loaded = True

    def transcribe(self, audio_data: bytes, language: str = "en-IN") -> ASRResult:
        if not audio_data:
            raise ValueError("Empty audio data")

        # Map language code to Whisper code
        lang_code = language.split("-")[0].lower()  # 'en', 'hi', 'ta'

        if self.is_available():
            import tempfile, os
            self._ensure_model()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                tmp_path = f.name
            try:
                try:
                    res = self._model.transcribe(tmp_path, language=lang_code)
                    text = res.get("text", "").strip()
                    return ASRResult(
                        text=text,
                        confidence=0.90,
                        language=language,
                        is_fallback=True,
                        raw_provider="whisper"
                    )
                except Exception as e:
                    # Fallback stub if transcription fails (e.g., ffmpeg error)
                    return ASRResult(
                        text=f"[Server ASR error: {e}]",
                        confidence=0.50,
                        language=language,
                        is_fallback=True,
                        raw_provider="whisper-error"
                    )
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # Fallback stub when whisper package is not installed
            return ASRResult(
                text="[Server ASR stub: Whisper not installed]",
                confidence=0.50,
                language=language,
                is_fallback=True,
                raw_provider="whisper-stub"
            )
