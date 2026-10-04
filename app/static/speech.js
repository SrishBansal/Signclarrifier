/**
 * Speech Module: ASR and TTS using only browser-native Web Speech API.
 *
 * Server-side /api/asr and /api/tts are removed. If the browser does not
 * support SpeechRecognition or speechSynthesis an explicit message is shown.
 */

(function (global) {
  "use strict";

  const LANG_TAGS = {
    en: "en-IN",
    hi: "hi-IN",
    ta: "ta-IN",
    bn: "bn-IN",
    te: "te-IN"
  };

  /**
   * ==========================================
   * 1. ASR Provider — Web Speech API only
   * ==========================================
   */

  class WebSpeechASRProvider {
    constructor() {
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
        const err = new Error(
          "Speech recognition is not supported in this browser. " +
          "Please use Chrome or Edge, or type your message instead."
        );
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

  /**
   * ==========================================
   * 2. TTS Provider — Web Speech API only
   * ==========================================
   */

  class WebSpeechTTSProvider {
    constructor() {
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
      const exact = candidates.find(v => v.lang.replace("_", "-").toLowerCase() === targetTag);
      return exact || candidates[0];
    }

    cancel() {
      if (this.synth) {
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
        const err = new Error(
          "Text-to-speech is not supported in this browser. " +
          "Please use Chrome, Edge, or Safari."
        );
        err.code = "unsupported-browser";
        if (opts.onError) opts.onError(err);
        return Promise.resolve(false);
      }

      if (!text || !text.trim()) {
        return Promise.resolve(false);
      }

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

  // Exports
  global.WebSpeechASRProvider = WebSpeechASRProvider;
  global.WebSpeechTTSProvider = WebSpeechTTSProvider;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      LANG_TAGS,
      WebSpeechASRProvider,
      WebSpeechTTSProvider
    };
  }

})(typeof window !== "undefined" ? window : global);
