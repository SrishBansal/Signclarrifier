# ClarifySign app
Sign -> Voice: webcam -> MediaPipe -> sign segmentation -> BiLSTM -> uncertainty -> EIG clarification -> 5 Indian languages (text + browser speech).
Voice -> Sign: browser speech recognition (or typing) -> concept match -> skeleton replay of a real recorded sign. 17 supported signs only.

Run: put the 4 files from the Colab download into `models/`, then `./run.sh`, open http://localhost:8000 in Chrome.
Without a trained model the app says so. It never invents predictions.

Layout: `clarifysign_ml/` features, model, training, streaming (shared by Colab and the server) | `app/` server, session, clarification (EIG), semantics, UI | `tests/`.
Known limits: translations are machine drafts needing native review; speech needs a device voice for the language (otherwise text only);
clarification thresholds are untuned starting values; recognition accuracy on a live camera is untested until you train and try it.
# Signclarrifier
