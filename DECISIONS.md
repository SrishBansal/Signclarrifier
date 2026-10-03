# Architecture and Implementation Decisions (ClarifySign)

## 1. Web Application & Video Pipeline Robustness
- **WebSocket Error Isolation**:
  - `sess.on_frame` wrapped in `try/except` in `app/server.py` to catch extractor/model runtime errors and emit a structured JSON error event (`{"type": "error", "message": ...}`) rather than aborting or closing the socket unceremoniously.
  - JSON decode errors and malformed WebSocket client packets are caught and returned as error notifications to preserve the session.
- **Client-Side Inflight & Watchdog**:
  - In `app/static/index.html`, `inflight` is initialized to `false` and reset on `start()`, `stop()`, and `ws.onclose`.
  - Added a 2000ms watchdog timer: if an acknowledgement frame (`{"type": "ack"}`) is delayed beyond 2 seconds due to network lag or drop, `inflight` is automatically cleared to unfreeze the frame streaming loop.
- **Clarification Timeout**:
  - Added a 15-second timeout in `app/session.py` for pending clarifications. If the user does not respond within 15 seconds, the session resets to the `listening` state and notifies the user with a resign event.
- **Provisional Sign Feedback & Diagnostics**:
  - While signing, `provisional` top guess predictions and confidence percentages are displayed live in the status bar so the user receives continuous feedback.
  - Per-frame diagnostics (wrist coordinates, pose/hand detection flags, activity state, and FPS) are surfaced to the Research panel.
- **Port Conflict Safeguards**:
  - `run.sh` dynamically checks if the specified port (default 8001) is already in use via `lsof`, reporting the conflict clearly instead of failing with a confusing stack trace.

## 2. Shared Semantic Core (`/core`)
- **Web-Independent Architecture**:
  - The entire `/core` package has zero web dependencies (no FastAPI, Starlette, websockets, or Uvicorn imports).
- **Pydantic Models (`core/models.py`)**:
  - Models for `Intent`, `Item`, `Utterance`, and `SemanticState` support structured communication with intents (`REQUEST`, `QUESTION`, `CONFIRM`, `REJECT`, `GREET`, `INFORM`, `PAYMENT`, `DIRECTION`, `AVAILABILITY`).
  - Supports normalization, canonical signatures, and negation tracking.
- **Ontology & YAML Data Layer (`data/ontology.yaml`, `core/ontology.py`)**:
  - Concepts mapped across English, Hindi, Hinglish, and Tamil.
  - Keys for `YES` and `NO` quoted explicitly in YAML to avoid YAML boolean coercion (`True`/`False`).
  - Plural stemming (English -s, -es) and case-insensitive normalization added to fallback synonym index.
  - Unknown/out-of-vocabulary product requests fallback to fingerspelling (`FS_` prefix).
- **Grammar-Based ISL Planner (`core/planner.py`, `data/grammar_rules.yaml`)**:
  - Encodes Indian Sign Language grammar rules: Topic-first, attribute before quantity, action/predicate last, question markers at sentence end.
- **Multi-Turn Dialogue Management (`core/dialogue.py`)**:
  - Maintains shop context (`active_shop_counter`) and referent tracking across conversation turns (e.g., resolving "how much?" to the active product referent).
- **Natural Language Realization (`core/semantics.py`)**:
  - Template-based multi-lingual realization for English, Hindi, and Tamil without external API calls.

## 3. Data-Driven Decoupled Sign Library & Avatar Renderer
- **Sign Library (`core/sign_library/`)**:
  - Keyframe sequence representation: 48 frames per sign, 225 dimensions (left hand 63 + right hand 63 + pose 99) zero-centered and normalized.
  - Aggregation tools in `core/sign_library/builder.py`: supports Medoid selection (choosing the true recorded take minimizing pairwise Euclidean distances across takes to preserve natural biomechanics and coarticulation) and DTW-averaged sequences.
  - Strict fallback guarantee: missing signs are NEVER silently skipped. If a sign ID starts with `FS_` or represents fingerspelling, canonical letter handshapes are rendered; all other unrecognized signs trigger an animated 48-frame visible "UNKNOWN SIGN" shrug gesture with open palms facing upward.
  - Timeline compiler (`SignTimeline`): computes ease-in-out LERP (cubic hermite smoothstep) transitions between signs and rest poses, annotating every frame with sign IDs, progress, and gloss metadata.
- **Architectural Decoupling**:
  - Zero imports of NLU (`core.nlu`) or planner (`core.planner`) code within `core.sign_library`, `core.renderer`, or `app/static/renderer.js`.
  - Lazy loading implemented in `core/__init__.py` using PEP 562 `__getattr__` to ensure clean separation and avoid eager transitive imports when loading rendering utilities.
  - Proved via AST and runtime isolation tests in `tests/test_renderer_decoupling.py`.
- **Articulated Avatar Drawing (No Dot Clouds)**:
  - Real connection tables implemented for both pose (shoulders, torso, spine, hips, neck, head) and hands (21 landmarks with palm plate and individual articulated finger bones: thumb, index, middle, ring, pinky).
  - Drawn using rounded thick strokes (`lineCap="round"`, `lineJoin="round"`), distinct finger colors, and anatomical torso/head shapes rather than scattered point clouds.
- **Two-Bone Arm Inverse Kinematics (IK)**:
  - Analytical two-bone IK for arms implemented in both Python (`core/renderer.py`) and JavaScript (`app/static/renderer.js`).
  - Solves the elbow position given shoulder origin, wrist target, and natural outward bending direction using the law of cosines, ensuring rigid limb lengths and natural arm silhouettes.
- **Timeline Controls & Synchronized Gloss Strip**:
  - Playback controls supporting pause, resume, replay, and variable speed scaling (0.5× to 2.0×).
  - Synced gloss strip dynamically generates interactive pills for every sign in the sequence (`[HELLO] [THANKYOU] [PEN]`), actively highlighting the current sign during playback and showing transition states.

