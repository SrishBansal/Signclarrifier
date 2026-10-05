"""FastAPI shell. All logic lives in core/. No app/semantics or app/clarify imports."""
import io, os, json, asyncio, time, uuid, logging
import numpy as np, cv2
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from core.ontology import get_ontology
from core.nlu import NLUParser
from core.dialogue import DialogueManager
from core.planner import ISLPlanner, get_planner
from core.semantics import NaturalLanguageRealizer
from .session import SignSession

log = logging.getLogger("clarifysign.server")

HERE = os.path.dirname(__file__)
MODEL  = os.environ.get("MODEL_PATH",  "models/clarifysign_bilstm.pt")
TASK   = os.environ.get("TASK_PATH",   "models/holistic_landmarker.task")
SIGNS  = os.environ.get("SIGNS_PATH",  "models/signs.json")
RECOGNIZER_POLICY = os.environ.get("RECOGNIZER_POLICY_PATH", "data/recognizer_policy.yaml")
PORT   = int(os.environ.get("PORT", "8000"))

# Global last-frame store: sid -> (jpeg_bytes, landmarks_bgr)
_last_frames: dict = {}


def _json_native(value):
    """Convert NumPy values at the websocket boundary to JSON primitives.

    Feature extraction and probability calculations intentionally use NumPy.
    Starlette's ``send_json`` uses Python's standard JSON encoder, which cannot
    encode NumPy scalars.  Keeping conversion here makes every websocket event
    safe, including future diagnostics and clarification payloads.
    """
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_native(item) for item in value]
    return value


async def _send_event(ws: WebSocket, event: dict, *, sid: str, source: str) -> None:
    """Send one event with JSON conversion and actionable failure diagnostics."""
    payload = _json_native(event)
    try:
        # Preflight catches unsupported values before the websocket transport
        # turns them into an opaque per-frame error.
        json.dumps(payload, allow_nan=False)
        await ws.send_json(payload)
    except Exception:
        log.exception("websocket send failed sid=%s source=%s event_type=%r payload=%r",
                      sid, source, event.get("type"), payload)
        raise


def _boot_check(recognizer, ontology):
    """Refuse boot if any recognizer class is not an ontology sign_id. Lists offenders."""
    if recognizer is None:
        return
    ont_sign_ids = {d.get("sign_id") for d in ontology.concepts.values() if d.get("sign_id")}
    offenders = [c for c in recognizer.classes if c not in ont_sign_ids]
    if offenders:
        raise RuntimeError(
            f"Boot refused: {len(offenders)} recognizer class(es) have no matching ontology "
            f"sign_id: {offenders}. Add them to data/ontology.yaml or retrain the model."
        )


def load_backend():
    if not os.path.exists(MODEL):
        return None, f"No trained model at {MODEL}."
    if not os.path.exists(TASK):
        return None, f"Missing MediaPipe model at {TASK}."
    from clarifysign_ml.predict import Recognizer
    return Recognizer(MODEL), ""


def _recognition_audit(recognizer):
    """Audit installed assets once at boot; never turn an audit error into a crash."""
    if recognizer is None or not os.path.exists(SIGNS) or not os.path.exists(RECOGNIZER_POLICY):
        return None
    try:
        from clarifysign_ml.audit import audit_demo_library
        with open(RECOGNIZER_POLICY, encoding="utf-8") as handle:
            policy = (yaml.safe_load(handle) or {}).get("demo_audit", {})
        return audit_demo_library(recognizer, SIGNS, policy)
    except Exception as exc:
        log.warning("recognition audit unavailable: %s", exc)
        return {"passing": False, "error": str(exc)}


# Session store: sid -> (DialogueManager, last_access_time)
_sessions: dict = {}
_SESSION_TTL = 1800  # 30 min

def _get_dm(sid: str) -> DialogueManager:
    ont = get_ontology()
    now = time.time()
    if sid not in _sessions:
        _sessions[sid] = (DialogueManager(ont), now)
    dm, _ = _sessions[sid]
    _sessions[sid] = (dm, now)
    stale = [k for k, (_, t) in _sessions.items() if now - t > _SESSION_TTL]
    for k in stale:
        del _sessions[k]
    return dm


def _draw_landmarks(bgr, ext):
    """Draw the latest detected landmarks on a camera frame (best-effort)."""
    if bgr is None:
        return bgr
    out = bgr.copy()
    last = getattr(ext, "last", {})
    color = (0, 255, 0) if last.get("pose") else (0, 0, 255)
    h, w = out.shape[:2]
    label = f"pose={'Y' if last.get('pose') else 'N'} lh={'Y' if last.get('lh') else 'N'} rh={'Y' if last.get('rh') else 'N'}"
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    points = last.get("landmarks", {})
    colors = {"pose": (0, 255, 0), "lh": (255, 140, 0), "rh": (0, 180, 255)}
    for part, part_points in points.items():
        for x, y in part_points:
            cv2.circle(out, (round(x * w), round(y * h)), 2, colors.get(part, color), -1)
    return out


def create_app(recognizer=None, reason="", extractor_factory=None):
    logging.getLogger("clarifysign").setLevel(logging.INFO)
    diagnostic_log = os.environ.get("CLARIFYSIGN_DIAGNOSTIC_LOG")
    if diagnostic_log:
        diagnostic_logger = logging.getLogger("clarifysign")
        destination = os.path.abspath(diagnostic_log)
        if not any(getattr(handler, "baseFilename", None) == destination
                   for handler in diagnostic_logger.handlers):
            handler = logging.FileHandler(destination, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            diagnostic_logger.addHandler(handler)
    if recognizer is None and not reason:
        recognizer, reason = load_backend()
    ont = get_ontology()
    language_by_code = {entry["code"]: entry for entry in ont.languages()}
    default_language = next(entry for entry in ont.languages() if not entry.get("nlu_input_only"))
    try:
        _boot_check(recognizer, ont)
    except RuntimeError as e:
        reason = str(e); recognizer = None
    recognition_audit = _recognition_audit(recognizer)

    if extractor_factory is None:
        from clarifysign_ml.features import LandmarkExtractor
        extractor_factory = lambda: LandmarkExtractor(TASK)

    nlu = NLUParser(ont)
    planner = get_planner()

    app = FastAPI(title="ClarifySign")
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "static", "index.html"))

    @app.get("/api/config")
    def config():
        langs = [{"code": l["code"], "name": l["name"], "tag": l["speech_tag"]}
                 for l in ont.languages() if not l.get("nlu_input_only")]
        concepts = [{"id": cid, "sign_id": d.get("sign_id"),
                     "english": d.get("label", {}).get("en", cid),
                     "kind": ont.kind(cid)}
                    for cid, d in ont.concepts.items()]
        return {"languages": langs, "concepts": concepts,
                "recognition_ready": recognizer is not None, "reason": reason,
                "signs_available": os.path.exists(SIGNS),
                "recognition": {
                    "mode": "isolated_sign",
                    "class_count": len(recognizer.classes) if recognizer is not None else 0,
                    "demo_audit": recognition_audit,
                }}

    @app.get("/api/understand")
    def understand(text: str = Query(""), lang: str = Query(""), sid: str = Query("")):
        if not sid:
            sid = str(uuid.uuid4())
        dm = _get_dm(sid)
        language = language_by_code.get(lang, default_language)["name"]
        parsed = nlu.parse(text, language, dm.get_state())
        state = dm.update_from_shopkeeper(parsed)
        plan_items = planner.plan(state)
        plan_dicts = []
        for cid in plan_items:
            if cid is None: continue
            d = ont.concepts.get(cid.upper(), {})
            plan_dicts.append({
                "concept": cid,
                "sign_id": d.get("sign_id"),
                "gloss": d.get("label", {}).get("en", cid),
                "kind": ont.kind(cid),
            })
        real = NaturalLanguageRealizer.realize(state, language, ont)
        tokens = ont.normalize_text(text).split()
        cue_tokens = {token for cue in ont.intent_cues.values()
                      for field in ("tokens", "hi_tokens", "ta_tokens", "bn_tokens", "te_tokens")
                      for token in cue.get(field, [])}
        unmapped_tokens = [token for token in tokens
                           if ont.resolve_concept(token) is None and token not in cue_tokens]
        unrecorded = [entry["gloss"] for entry in plan_dicts if entry["kind"] != "sign"]
        return {"heard": text, "sid": sid,
                "state": {"intent": state.intent.value if state.intent else None,
                          "focus": state.current_focus_referent,
                          "items": [it.canonical_dict() for it in state.items]},
                "plan": plan_dicts, "text": real["text"],
                "coverage": {
                    "complete": not unmapped_tokens and not unrecorded,
                    "unmapped_tokens": unmapped_tokens,
                    "unrecorded_concepts": unrecorded,
                }}

    @app.get("/api/signs")
    def signs():
        if not os.path.exists(SIGNS):
            return JSONResponse({"error": "signs.json missing."}, 404)
        return FileResponse(SIGNS, media_type="application/json")

    @app.get("/api/debug/last_frame")
    def debug_last_frame(sid: str = Query("")):
        """Return the last received JPEG frame with landmarks drawn, for debugging."""
        entry = _last_frames.get(sid)
        if entry is None:
            # Return a blank 320x240 gray JPEG with a message
            blank = np.full((240, 320, 3), 80, dtype=np.uint8)
            cv2.putText(blank, "No frame yet", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            ok, buf = cv2.imencode(".jpg", blank)
            return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")
        annotated_bgr = entry
        ok, buf = cv2.imencode(".jpg", annotated_bgr)
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

    @app.websocket("/ws/sign")
    async def ws_sign(ws: WebSocket, sid: str = Query("")):
        await ws.accept()
        if not sid:
            sid = str(uuid.uuid4())
        log.info("websocket opened sid=%s", sid)
        if recognizer is None:
            await _send_event(ws, {"type": "error", "message": reason}, sid=sid, source="startup")
            await ws.close()
            return
        dm = _get_dm(sid)
        ext = extractor_factory()
        sess = SignSession(recognizer, ext, dm=dm, decode=dec)
        loop = asyncio.get_running_loop()
        await _send_event(ws, {"type": "state", "state": "listening"}, sid=sid, source="startup")
        _warn_shown = set()
        try:
            while True:
                try:
                    msg = await ws.receive()
                except WebSocketDisconnect as exc:
                    log.info("websocket disconnected sid=%s code=%s", sid, exc.code)
                    break
                except Exception:
                    log.exception("websocket receive failed sid=%s", sid)
                    break
                if msg["type"] == "websocket.disconnect":
                    log.info("websocket disconnect event sid=%s code=%s", sid, msg.get("code"))
                    break
                if msg.get("bytes"):
                    raw_bytes = msg["bytes"]
                    try:
                        img = dec(raw_bytes)
                        events = await loop.run_in_executor(None, sess.on_frame, raw_bytes) if img is not None else []
                        # Store annotated last frame for /api/debug/last_frame
                        if img is not None:
                            _last_frames[sid] = _draw_landmarks(img, ext)
                        for e in events:
                            await _send_event(ws, e, sid=sid, source="frame")
                    except Exception as ex:
                        log.exception("frame handling failed sid=%s error_type=%s", sid, type(ex).__name__)
                        warn_key = str(type(ex).__name__)
                        if warn_key not in _warn_shown:
                            _warn_shown.add(warn_key)
                            await _send_event(ws, {"type": "warn", "message": f"Frame processing issue: {ex}"},
                                              sid=sid, source="frame-warning")
                    await _send_event(ws, {"type": "ack"}, sid=sid, source="frame-ack")
                elif msg.get("text"):
                    try:
                        d = json.loads(msg["text"])
                        mtype = d.get("type")
                        if mtype == "config":
                            sess.set_lang(d.get("lang", "en"))
                        elif mtype == "answer":
                            for e in sess.on_answer(d.get("value")):
                                await _send_event(ws, e, sid=sid, source="answer")
                        elif mtype == "reset":
                            sess.reset()
                            await _send_event(ws, {"type": "state", "state": "listening"}, sid=sid, source="reset")
                        else:
                            await _send_event(ws, {"type": "warn", "message": f"Unknown type: {mtype}"},
                                              sid=sid, source="message")
                    except Exception as ex:
                        log.exception("ws msg error: %s", ex)
                        await _send_event(ws, {"type": "warn", "message": str(ex)}, sid=sid, source="message-error")
        finally:
            close = getattr(ext, "close", None)
            if close: close()
            _last_frames.pop(sid, None)
            log.info("websocket closed sid=%s", sid)

    return app


def app_factory():
    return create_app()
