"""TTS (Text-to-Speech) provider interfaces and implementations.

Decoupled provider interface supporting:
- Web Speech API proxy / mock
- Server-side TTS engines
- Voice selection per language (en-IN, hi-IN, ta-IN)
- Queue and cancel handling to prevent overlapping utterances
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time


@dataclass
class TTSResult:
    """Standardized speech synthesis output."""
    audio_data: bytes
    mime_type: str = "audio/wav"
    duration_sec: float = 1.0
    text: str = ""
    language: str = "en-IN"
    voice: Optional[str] = None
    provider: str = "mock"


class TTSProvider(ABC):
    """Abstract Base Class for Text-to-Speech providers."""

    @abstractmethod
    def synthesize(self, text: str, language: str = "en-IN", voice: Optional[str] = None) -> TTSResult:
        """Synthesize text to audio data."""
        pass

    @abstractmethod
    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        """List available voices for a given language tag."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available in current environment."""
        pass


class MockTTSProvider(TTSProvider):
    """Mock TTS provider generating valid synthetic WAV/PCM frames for tests."""

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []
        self._voices = {
            "en-IN": [{"id": "en-in-1", "name": "Ravi (Indian English)", "gender": "male"}],
            "hi-IN": [{"id": "hi-in-1", "name": "Swara (Hindi)", "gender": "female"}],
            "ta-IN": [{"id": "ta-in-1", "name": "Valluvar (Tamil)", "gender": "male"}]
        }

    def is_available(self) -> bool:
        return True

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        if language and language in self._voices:
            return self._voices[language]
        all_v = []
        for vlist in self._voices.values():
            all_v.extend(vlist)
        return all_v

    def synthesize(self, text: str, language: str = "en-IN", voice: Optional[str] = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        self.call_history.append({
            "text": text,
            "language": language,
            "voice": voice,
            "timestamp": time.time()
        })

        # Generate simple 44-byte standard RIFF/WAV header with 0.1s silence
        # for testing client playback / download without external binaries
        sample_rate = 16000
        duration = max(0.5, len(text) * 0.05)
        num_samples = int(sample_rate * duration)
        raw_pcm = b"\x00\x00" * num_samples
        data_size = len(raw_pcm)

        header = bytearray(44)
        header[0:4] = b"RIFF"
        header[4:8] = (data_size + 36).to_bytes(4, "little")
        header[8:12] = b"WAVE"
        header[12:16] = b"fmt "
        header[16:20] = (16).to_bytes(4, "little")  # Subchunk1Size
        header[20:22] = (1).to_bytes(2, "little")   # AudioFormat (PCM)
        header[22:24] = (1).to_bytes(2, "little")   # NumChannels
        header[24:28] = sample_rate.to_bytes(4, "little")
        header[28:32] = (sample_rate * 2).to_bytes(4, "little")
        header[32:34] = (2).to_bytes(2, "little")   # BlockAlign
        header[34:36] = (16).to_bytes(2, "little")  # BitsPerSample
        header[36:40] = b"data"
        header[40:44] = data_size.to_bytes(4, "little")

        wav_bytes = bytes(header) + raw_pcm

        return TTSResult(
            audio_data=wav_bytes,
            mime_type="audio/wav",
            duration_sec=duration,
            text=text,
            language=language,
            voice=voice or (self._voices.get(language, [{}])[0].get("id")),
            provider="mock"
        )


class ServerTTSProvider(TTSProvider):
    """Server-side TTS provider interface (e.g. eSpeak-ng / Edge-TTS / Coqui / Google TTS)."""

    def __init__(self, fallback_to_mock: bool = True):
        self.fallback = MockTTSProvider() if fallback_to_mock else None

    def is_available(self) -> bool:
        return self.fallback is not None

    def list_voices(self, language: Optional[str] = None) -> List[Dict[str, str]]:
        if self.fallback:
            return self.fallback.list_voices(language)
        return []

    def synthesize(self, text: str, language: str = "en-IN", voice: Optional[str] = None) -> TTSResult:
        if self.fallback:
            res = self.fallback.synthesize(text, language, voice)
            res.provider = "server-fallback"
            return res
        raise NotImplementedError("Server TTS backend not configured")
