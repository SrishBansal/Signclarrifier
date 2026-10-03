# ClarifySign — Architectural Decisions

## Data contract
- `data/ontology.yaml` is the **single source** of vocabulary, cues, templates and labels.
- `data/include_classes.txt` lists the 262 model-output class names; every `sign_id` must be in it.
- No vocabulary, sentences, translations or cue words live in `.py` files.

## Recognition pipeline (Direction B — customer → shopkeeper)
- Frame → 225-dim landmark features (MediaPipe Holistic).
- Rolling `deque(maxlen=48)` of real frames; no `np.tile`.
- `Dialogue` (EIG clarification) resolves to a recognizer class (= `sign_id`).
- `concept_for_sign(sign_id)` maps to a concept; `signed_effect` drives state update.
- Boot refuses if any recognizer class lacks a matching `sign_id` in the ontology.

## NLU pipeline (Direction A — shopkeeper → customer)
- `NLUParser.parse` reads cue sets from `ontology.intent_cues`.
- `ISLPlanner.plan(state)` → ordered list of concept IDs → ISL gloss sequence.
- `/api/understand` returns `{heard, state, plan:[{concept,sign_id,gloss,kind}], text}`.

## Session management
- Browser provides a UUID `sid`; server keeps `{sid: DialogueManager}` with 30 min TTL.
- Both `/ws/sign?sid=` and `/api/understand?sid=` share the same `DialogueManager`.

## Clarification (EIG)
- `Dialogue` lives in `core/clarify/policy.py`; `update_posterior` is a standalone function.
- `resolution.py` and `calibration.py` deleted (unused / duplicated).

## Rendering (Direction A playback)
- `kind="sign"` → play library sign; `kind="marker"` → rest-pose hold; `kind="nosign"` → unknown marker.
- No `FS_` prefix anywhere in Python or JS.

## Port
- Default: **8000**. Set `PORT=` env var to override.
