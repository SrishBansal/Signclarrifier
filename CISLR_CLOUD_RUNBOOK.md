# CISLR candidate runbook

This is a **candidate-only** route for improving the current isolated-sign
recognizer. It leaves `models/` unchanged. CISLR is appropriate because it is
a word-level ISL recognition corpus; its published layout provides
`dataset.csv`, `prototype.csv`, `test.csv`, and videos. Do not use iSign or
ISLTranslate as a drop-in replacement here: they target continuous
video-to-text translation.

## Run in Colab or another ephemeral GPU machine

1. Accept the CISLR terms using your own account and place its data in your
   private cloud drive/bucket. Do not commit or redistribute the videos.
2. Clone this repository in the temporary runtime and install
   `requirements.txt`.
3. Make a reviewed input JSONL. Every row requires `dataset`, `sample_id`,
   `group_id`, `split`, `label`, and `video_path`. Use the source-video ID as
   `group_id` unless you have a stronger signer/session identifier. Map source
   labels only to the checkpoint's existing class names; do not guess semantic
   equivalents. For example:

```json
{"dataset":"cislr","sample_id":"source-id","group_id":"signer-or-video","split":"train","label":"shirt","video_path":"/content/drive/MyDrive/cislr/source-id.mp4"}
```

4. Extract features and write a validated manifest:

```bash
python scripts/extract_candidate_features.py \
  --input /content/drive/MyDrive/clarifysign/cislr_input.jsonl \
  --output-dir /content/drive/MyDrive/clarifysign/cislr-features
```

5. Train the candidate:

```bash
python scripts/train_candidate.py \
  --manifest /content/drive/MyDrive/clarifysign/cislr-features/manifest.jsonl \
  --output-dir /content/drive/MyDrive/clarifysign/candidates/run-001 \
  --device cuda
```

The command creates a new checkpoint and report outside `models/`. It evaluates
both the candidate and frozen baseline on the exact same test split, then
reports whether configured top-1, macro-F1, and calibration gates pass. A
passing report is not enough on its own: perform a real webcam test with people
not represented by the source groups before manually adopting the candidate.

The CISLR card describes it as a word-level ISL video-recognition dataset with
about 4,700 words. [Dataset card](https://huggingface.co/datasets/Exploration-Lab/CISLR/blob/main/README.md?code=true)
