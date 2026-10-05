# ClarifySign app
Sign -> Voice: webcam -> MediaPipe -> sign segmentation -> BiLSTM -> uncertainty -> EIG clarification -> 5 Indian languages (text + browser speech).
Voice -> Sign: browser speech recognition (or typing) -> ontology concept match -> skeleton replay of recorded signs. This is a finite isolated-sign research vocabulary, not general sentence translation.

Run: put the five model files in `models/` (including `holistic_landmarker.task` and `hand_landmarker.task`), then `./run.sh`, open http://localhost:8000 in Chrome. The hand task is an independent MediaPipe fallback used only when Holistic has pose but misses a hand; it does not replace or retrain the recognition model.
Without a trained model the app says so. It never invents predictions.

Layout: `clarifysign_ml/` features, model, training, streaming (shared by Colab and the server) | `app/` server, session, clarification (EIG), semantics, UI | `tests/`.
Known limits: translations are machine drafts needing native review; speech needs a device voice for the language (otherwise text only);
clarification thresholds are untuned starting values; the baseline is experimental and fails its configured demonstration-quality gate. See `CISLR_CLOUD_RUNBOOK.md` for the isolated-sign candidate path.
# Signclarrifier
