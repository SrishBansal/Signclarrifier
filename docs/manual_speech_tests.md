# ClarifySign Manual Speech Test Script (ASR & TTS)

This document provides step-by-step verification procedures for the decoupled ASR (Automated Speech Recognition) and TTS (Text-to-Speech) provider interfaces in ClarifySign.

---

## Prerequisites
1. Run `./run.sh` or `python -m uvicorn app.server:app_factory --factory --port 8001`.
2. Open `http://localhost:8001` in Google Chrome or Microsoft Edge.
3. Switch to the **Voice → Sign** tab.

---

## Test Case 1: Language Switching (en-IN, hi-IN, ta-IN)
- **Objective**: Ensure ASR and TTS configure correct regional BCP-47 locale tags without language crosstalk.
- **Steps**:
  1. Click the **Hindi** language tab (`hi`). Verify status/scope reflects Hindi and TTS voice status displays Hindi readiness.
  2. Click the **English** language tab (`en`). Verify tag switches to `en-IN`.
  3. Click the **Tamil** language tab (`ta`). Verify tag switches to `ta-IN`.
- **Expected Outcome**:
  - The language selector updates immediately.
  - Active ASR recognition uses `en-IN`, `hi-IN`, or `ta-IN`.
  - TTS selects matching regional voice.

---

## Test Case 2: ASR Transcript Preview, Confidence Score & Text Editing
- **Objective**: Verify the shopkeeper can review and edit recognized speech before submitting, isolating ASR inaccuracies from NLP/animation bugs.
- **Steps**:
  1. Select **English**.
  2. Click the microphone button (🎤). Status indicator turns to `Listening to you…`.
  3. Speak: *"water bottle please"*.
  4. Notice the textarea is populated with the recognized transcript.
  5. Check the `#asr-badge` below the textarea:
     - Displays `Heard: "water bottle please" (90% conf)`.
     - Displays prompt: `Edit text above if needed, then click 'Show sign'`.
  6. **Shopkeeper Correction**: Edit the textarea to say *"blue shirt"*.
  7. Click **Show sign**.
- **Expected Outcome**:
  - The avatar signs `BLUE` followed by `SHIRT`, confirming that the edited text was sent rather than the raw ASR output.
  - ASR errors are completely decoupled from downstream sign animations.

---

## Test Case 3: Side-by-Side Debug Logging
- **Objective**: Validate that raw ASR transcripts and downstream recognized states are logged side by side.
- **Steps**:
  1. Check the **Research** checkbox in the top header.
  2. Expand the `Research / diagnostics` disclosure panel.
  3. Speak or submit an utterance.
  4. Inspect the debug output in the panel:
     ```
     === SPEECH & DOWNSTREAM DEBUG ===
     ASR Heard:   "water bottle please" (90% conf)
     Shopkeeper:  "water bottle please"
     Downstream:  WATER -> BOTTLE
     ```
  5. Check `curl http://localhost:8001/api/debug/interactions`.
- **Expected Outcome**:
  - Both raw ASR input and downstream concepts are preserved side-by-side in real-time.

---

## Test Case 4: TTS Overlapping Utterance Preemption (Queue & Cancel)
- **Objective**: Verify that rapid sequential TTS requests cancel previous utterances rather than stacking or double-speaking.
- **Steps**:
  1. Click the **🔊** button repeatedly in rapid succession (3 times within 1 second).
  2. Listen to audio output.
- **Expected Outcome**:
  - The speech synthesis immediately cancels previous in-flight utterances.
  - No overlapping voices or audio stuttering occurs.

---

## Test Case 5: Error States & Fallbacks
- **5.1 Microphone Permission Denied**:
  - In browser site settings, block Microphone access for `localhost:8001`.
  - Click the 🎤 button.
  - *Expected*: Error banner displays: *"Microphone permission was denied. Please allow microphone access in your browser settings."* Textarea remains fully usable for typing.
- **5.2 No Speech Detected**:
  - Click 🎤 and remain silent for 5 seconds.
  - *Expected*: Message displayed: *"No speech detected. Please speak closer to your microphone."*
- **5.3 Unsupported Browser**:
  - In a browser without `SpeechRecognition` support (or simulated by disabling the API):
  - *Expected*: Notice displayed informing user that speech recognition is unavailable, and textarea remains active.
- **5.4 Server-Side Fallback Endpoints**:
  - Send audio POST to `http://localhost:8001/api/asr`:
    `curl -X POST http://localhost:8001/api/asr?lang=hi-IN --data-binary @some_audio.wav`
  - *Expected*: Returns JSON with `{"text": ..., "confidence": ..., "provider": ...}`.
  - Send payload to `http://localhost:8001/api/tts`:
    `curl -X POST http://localhost:8001/api/tts -H "Content-Type: application/json" -d '{"text":"namaste","lang":"hi-IN"}' --output out.wav`
  - *Expected*: Returns valid `audio/wav` file.
