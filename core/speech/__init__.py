"""Speech module providing provider interfaces for ASR and TTS.

Features:
- ASRProvider: Web Speech API / Whisper fallback / Mock
- TTSProvider: SpeechSynthesis / Server TTS / Mock
- Language tags: en-IN, hi-IN, ta-IN
- Queue and cancel management
"""

from .asr import ASRProvider, ASRResult, MockASRProvider, ServerWhisperASRProvider
from .tts import TTSProvider, TTSResult, MockTTSProvider, ServerTTSProvider

__all__ = [
    "ASRProvider",
    "ASRResult",
    "MockASRProvider",
    "ServerWhisperASRProvider",
    "TTSProvider",
    "TTSResult",
    "MockTTSProvider",
    "ServerTTSProvider",
]
