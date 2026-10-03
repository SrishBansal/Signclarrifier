/**
 * Speech Module: ASR and TTS Provider Interfaces for ClarifySign.
 *
 * Requirements:
 * 1. ASR Provider Interface:
 *    - Web Speech API for dev with language switching (en-IN, hi-IN, ta-IN)
 *    - Server-side fallback interface (/api/asr)
 *    - Shows transcript with confidence score
 *    - Allows shopkeeper to edit before sending
 *    - Logs ASR text and downstream state side-by-side
 * 2. TTS Provider Interface:
 *    - SpeechSynthesis provider with voice selection per language
 *    - Server TTS fallback interface (/api/tts)
 *    - Queue and cancel handling so overlapping utterances don't stack
 * 3. Robust error handling:
 *    - Mic permission denied
 *    - No speech detected
 *    - Network loss
 *    - Unsupported browser
 */

(function (global) {
  "use strict";

  const LANG_TAGS = {
    en: "en-IN",
    hi: "hi-IN",
    ta: "ta-IN"
  };

  /**
   * ==========================================
   * 1. ASR Provider Interface & Implementations
   * ==========================================
   */

  class BaseASRProvider {
    start(options) { throw new Error("start() must be implemented"); }
    stop() { throw new Error("stop() must be implemented"); }
    abort() { throw new Error("abort() must be implemented"); }
    isSupported() { return false; }
  }

  class WebSpeechASRProvider extends BaseASRProvider {
    constructor() {
      super();
      const root = typeof window !== "undefined" ? window : global;
      const SR = root.SpeechRecognition || root.webkitSpeechRecognition;
      this.SRClass = SR || null;
      this.activeRecognition = null;
      this.isListening = false;
    }

    isSupported() {
      return !!this.SRClass;
    }

    start(options) {
      const opts = Object.assign({
        lang: "hi",
        onStart: null,
        onResult: null,
        onError: null,
        onEnd: null
      }, options || {});

      if (!this.isSupported()) {
        const err = new Error("Web Speech API is not supported in this browser. Please use Chrome/Edge or type text.");
        err.code = "unsupported-browser";
        if (opts.onError) opts.onError(err);
        return;
      }

      this.abort();

      try {
        const root = typeof window !== "undefined" ? window : global;
        const recognition = new this.SRClass();
        recognition.lang = LANG_TAGS[opts.lang] || opts.lang || "hi-IN";
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.maxAlternatives = 3;

        let finalTranscript = "";
        let finalConfidence = 0.0;

        recognition.onstart = () => {
          this.isListening = true;
          if (opts.onStart) opts.onStart();
        };

        recognition.onresult = (ev) => {
          let interim = "";
          for (let i = ev.resultIndex; i < ev.results.length; ++i) {
            const res = ev.results[i];
            if (res.isFinal) {
              finalTranscript += res[0].transcript;
              finalConfidence = res[0].confidence || 0.90;
            } else {
              interim += res[0].transcript;
            }
          }
          if (opts.onResult) {
            opts.onResult({
              transcript: finalTranscript || interim,
              confidence: finalConfidence || 0.85,
              isFinal: !!finalTranscript,
              language: recognition.lang,
              provider: "web-speech"
            });
          }
        };

        recognition.onerror = (ev) => {
          this.isListening = false;
          let msg = "Speech recognition error.";
          const code = ev.error || "unknown";

          if (code === "not-allowed" || code === "permission-denied") {
            msg = "Microphone permission was denied. Please allow microphone access in your browser settings.";
          } else if (code === "no-speech") {
            msg = "No speech detected. Please speak closer to your microphone.";
          } else if (code === "network") {
            msg = "Speech recognition network error. Please check your internet connection or type.";
          } else if (code === "audio-capture") {
            msg = "No microphone hardware found on this device.";
          }

          const err = new Error(msg);
          err.code = code;
          if (opts.onError) opts.onError(err);
        };

        recognition.onend = () => {
          this.isListening = false;
          this.activeRecognition = null;
          if (opts.onEnd) opts.onEnd();
        };

        this.activeRecognition = recognition;
        recognition.start();
      } catch (ex) {
        this.isListening = false;
        if (opts.onError) opts.onError(ex);
      }
    }

    stop() {
      if (this.activeRecognition && this.isListening) {
        try { this.activeRecognition.stop(); } catch (e) {}
      }
      this.isListening = false;
    }

    abort() {
      if (this.activeRecognition) {
        try { this.activeRecognition.abort(); } catch (e) {}
      }
      this.activeRecognition = null;
      this.isListening = false;
    }
  }

  class ServerFallbackASRProvider extends BaseASRProvider {
    constructor() {
      super();
      this.mediaRecorder = null;
      this.chunks = [];
      this.isListening = false;
    }

    isSupported() {
      return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
    }

    async start(options) {
      const opts = Object.assign({
        lang: "hi",
        onStart: null,
        onResult: null,
        onError: null,
        onEnd: null
      }, options || {});

      if (!this.isSupported()) {
        const err = new Error("Audio recording is not supported on this browser.");
        err.code = "unsupported-browser";
        if (opts.onError) opts.onError(err);
        return;
      }

      this.abort();
      this.chunks = [];

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        this.mediaRecorder = new MediaRecorder(stream);

        this.mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) this.chunks.push(e.data);
        };

        this.mediaRecorder.onstart = () => {
          this.isListening = true;
          if (opts.onStart) opts.onStart();
        };

        this.mediaRecorder.onstop = async () => {
          this.isListening = false;
          stream.getTracks().forEach(t => t.stop());
          const blob = new Blob(this.chunks, { type: "audio/webm" });

          try {
            const langTag = LANG_TAGS[opts.lang] || opts.lang || "hi-IN";
            const resp = await fetch(`/api/asr?lang=${encodeURIComponent(langTag)}`, {
              method: "POST",
              body: blob
            });
            if (!resp.ok) {
              const j = await resp.json().catch(() => ({}));
              throw new Error(j.error || `Server ASR returned HTTP ${resp.status}`);
            }
            const res = await resp.json();
            if (opts.onResult) {
              opts.onResult({
                transcript: res.text,
                confidence: res.confidence,
                isFinal: true,
                language: res.language,
                provider: "server-fallback"
              });
            }
          } catch (ex) {
            if (opts.onError) opts.onError(ex);
          } finally {
            if (opts.onEnd) opts.onEnd();
          }
        };

        this.mediaRecorder.start();
      } catch (ex) {
        this.isListening = false;
        let msg = ex.message;
        let code = "unknown";
        if (ex.name === "NotAllowedError" || ex.name === "PermissionDeniedError") {
          msg = "Microphone permission denied. Please allow microphone access in your browser settings.";
          code = "not-allowed";
        }
        const err = new Error(msg);
        err.code = code;
        if (opts.onError) opts.onError(err);
      }
    }

    stop() {
      if (this.mediaRecorder && this.isListening) {
        try { this.mediaRecorder.stop(); } catch (e) {}
      }
      this.isListening = false;
    }

    abort() {
      if (this.mediaRecorder) {
        try {
          this.mediaRecorder.onstop = null;
          this.mediaRecorder.stop();
        } catch (e) {}
      }
      this.mediaRecorder = null;
      this.isListening = false;
    }
  }

  /**
   * ==========================================
   * 2. TTS Provider Interface & Implementations
   * ==========================================
   */

  class BaseTTSProvider {
    speak(text, options) { throw new Error("speak() must be implemented"); }
    cancel() { throw new Error("cancel() must be implemented"); }
    getVoices(lang) { return []; }
    isSupported() { return false; }
  }

  class WebSpeechTTSProvider extends BaseTTSProvider {
    constructor() {
      super();
      const root = typeof window !== "undefined" ? window : global;
      this.synth = root.speechSynthesis || null;
      this.isSpeaking = false;
      this.currentUtterance = null;
    }

    isSupported() {
      return !!this.synth;
    }

    getVoices(lang) {
      if (!this.synth) return [];
      const vs = this.synth.getVoices();
      if (!lang) return vs;
      const targetTag = LANG_TAGS[lang] || lang;
      const base = targetTag.split("-")[0].toLowerCase();

      return vs.filter(v => {
        const vLang = v.lang.replace("_", "-").toLowerCase();
        return vLang === targetTag.toLowerCase() || vLang.startsWith(base);
      });
    }

    getBestVoice(lang) {
      const candidates = this.getVoices(lang);
      if (!candidates.length) return null;
      const targetTag = (LANG_TAGS[lang] || lang).toLowerCase();
      // Prefer exact Indian locale tag e.g. hi-IN, en-IN, ta-IN
      const exact = candidates.find(v => v.lang.replace("_", "-").toLowerCase() === targetTag);
      return exact || candidates[0];
    }

    cancel() {
      if (this.synth) {
        // Immediate cancellation ensures overlapping utterances never stack
        this.synth.cancel();
      }
      this.isSpeaking = false;
      this.currentUtterance = null;
    }

    speak(text, options) {
      const opts = Object.assign({
        lang: "hi",
        voice: null,
        rate: 1.0,
        pitch: 1.0,
        cancelPrevious: true,
        onStart: null,
        onEnd: null,
        onError: null
      }, options || {});

      if (!this.isSupported()) {
        const err = new Error("Text-to-Speech is not supported in this browser.");
        err.code = "unsupported-browser";
        if (opts.onError) opts.onError(err);
        return Promise.resolve(false);
      }

      if (!text || !text.trim()) {
        return Promise.resolve(false);
      }

      // Preempt any pending or currently speaking audio
      if (opts.cancelPrevious) {
        this.cancel();
      }

      return new Promise((resolve) => {
        const utterance = new SpeechSynthesisUtterance(text);
        const voiceObj = opts.voice || this.getBestVoice(opts.lang);
        if (voiceObj) {
          utterance.voice = voiceObj;
          utterance.lang = voiceObj.lang;
        } else {
          utterance.lang = LANG_TAGS[opts.lang] || opts.lang || "hi-IN";
        }
        utterance.rate = opts.rate || 1.0;
        utterance.pitch = opts.pitch || 1.0;

        utterance.onstart = () => {
          this.isSpeaking = true;
          if (opts.onStart) opts.onStart();
        };

        utterance.onend = () => {
          this.isSpeaking = false;
          this.currentUtterance = null;
          if (opts.onEnd) opts.onEnd();
          resolve(true);
        };

        utterance.onerror = (ev) => {
          this.isSpeaking = false;
          this.currentUtterance = null;
          if (opts.onError) opts.onError(ev);
          resolve(false);
        };

        this.currentUtterance = utterance;
        this.synth.speak(utterance);
      });
    }
  }

  class ServerFallbackTTSProvider extends BaseTTSProvider {
    constructor() {
      super();
      this.activeAudio = null;
    }

    isSupported() {
      return typeof Audio !== "undefined";
    }

    cancel() {
      if (this.activeAudio) {
        try {
          this.activeAudio.pause();
          this.activeAudio.currentTime = 0;
        } catch (e) {}
        this.activeAudio = null;
      }
    }

    async speak(text, options) {
      const opts = Object.assign({
        lang: "hi",
        voice: null,
        onStart: null,
        onEnd: null,
        onError: null
      }, options || {});

      this.cancel();

      try {
        const langTag = LANG_TAGS[opts.lang] || opts.lang || "hi-IN";
        const resp = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, lang: langTag, voice: opts.voice })
        });
        if (!resp.ok) {
          throw new Error(`Server TTS returned HTTP ${resp.status}`);
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);

        return new Promise((resolve) => {
          audio.onplay = () => { if (opts.onStart) opts.onStart(); };
          audio.onended = () => {
            URL.revokeObjectURL(url);
            this.activeAudio = null;
            if (opts.onEnd) opts.onEnd();
            resolve(true);
          };
          audio.onerror = (ev) => {
            URL.revokeObjectURL(url);
            this.activeAudio = null;
            if (opts.onError) opts.onError(ev);
            resolve(false);
          };
          this.activeAudio = audio;
          audio.play().catch(ex => {
            if (opts.onError) opts.onError(ex);
            resolve(false);
          });
        });
      } catch (ex) {
        if (opts.onError) opts.onError(ex);
        return false;
      }
    }
  }

  // Exports
  global.WebSpeechASRProvider = WebSpeechASRProvider;
  global.ServerFallbackASRProvider = ServerFallbackASRProvider;
  global.WebSpeechTTSProvider = WebSpeechTTSProvider;
  global.ServerFallbackTTSProvider = ServerFallbackTTSProvider;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      LANG_TAGS,
      WebSpeechASRProvider,
      ServerFallbackASRProvider,
      WebSpeechTTSProvider,
      ServerFallbackTTSProvider
    };
  }

})(typeof window !== "undefined" ? window : global);
