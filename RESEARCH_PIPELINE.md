# Research data and continual-improvement pipeline

ClarifySign is a non-commercial research project. External datasets are not
committed to this repository, redistributed, or downloaded automatically.

## Dataset intake

1. A researcher accepts the source's terms with their own account.
2. Extract pose/features into a local, ignored directory.
3. Create a JSONL manifest using `ManifestSample` from
   `clarifysign_ml.research`.
4. Use the source video ID as `group_id` for iSign, so all clips from one video
   are in exactly one split. Use a signer/session identifier when that is the
   stronger leakage boundary.
5. Validate and fingerprint the manifest before any training run. Record the
   fingerprint and signer-independent metrics with the resulting model.

`data/research_sources.example.json` lists the approved research sources and
their intake rules. It contains metadata only, not licensed data.

## Learning queue

The default queue is a review queue, not online learning:

1. Ask for explicit opt-in.
2. Store only the 48 × 225 landmark feature sequence and its confirmed label.
3. Store no raw camera video.
4. A human reviewer checks label quality, duplicates, and consent before moving
   an example into a versioned training manifest.
5. Retrain periodically and compare against an untouched regression split.

This design improves reproducibility and privacy without treating unreviewed
live predictions as training truth.
