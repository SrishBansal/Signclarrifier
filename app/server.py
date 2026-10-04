"""FastAPI shell. All logic lives in core/. No app/semantics or app/clarify imports."""
import io, os, json, asyncio, time, uuid, logging
import numpy as np, cv2
from fastapi import FastAPI, WebSocket, Query
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
PORT   = int(os.environ.get("PORT", "8000"))

# Global last-frame store: sid -> (jpeg_bytes, landmarks_bgr)
_last_frames: dict = {}


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
    if recognizer is None and not reason:
        recognizer, reason = load_backend()
    ont = get_ontology()
    language_by_code = {entry["code"]: entry for entry in ont.languages()}
    default_language = next(entry for entry in ont.languages() if not entry.get("nlu_input_only"))
    try:
        _boot_check(recognizer, ont)
    except RuntimeError as e:
        reason = str(e); recognizer = None

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
                "signs_available": os.path.exists(SIGNS)}

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
        return {"heard": text, "sid": sid,
                "state": {"intent": state.intent.value if state.intent else None,
                          "focus": state.current_focus_referent,
                          "items": [it.canonical_dict() for it in state.items]},
                "plan": plan_dicts, "text": real["text"]}

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
        if recognizer is None:
            await ws.send_json({"type": "error", "message": reason}); await ws.close(); return
        dm = _get_dm(sid)
        ext = extractor_factory()
        sess = SignSession(recognizer, ext, dm=dm, decode=dec)
        loop = asyncio.get_running_loop()
        await ws.send_json({"type": "state", "state": "listening"})
        _warn_shown = set()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
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
                            await ws.send_json(e)
                    except Exception as ex:
                        log.warning("frame error sid=%s: %s", sid, ex)
                        warn_key = str(type(ex).__name__)
                        if warn_key not in _warn_shown:
                            _warn_shown.add(warn_key)
                            await ws.send_json({"type": "warn", "message": f"Frame processing issue: {ex}"})
                    await ws.send_json({"type": "ack"})
                elif msg.get("text"):
                    try:
                        d = json.loads(msg["text"])
                        mtype = d.get("type")
                        if mtype == "config":
                            sess.set_lang(d.get("lang", "en"))
                        elif mtype == "answer":
                            for e in sess.on_answer(d.get("value")):
                                await ws.send_json(e)
                        elif mtype == "reset":
                            sess.reset(); await ws.send_json({"type": "state", "state": "listening"})
                        else:
                            await ws.send_json({"type": "warn", "message": f"Unknown type: {mtype}"})
                    except Exception as ex:
                        log.exception("ws msg error: %s", ex)
                        await ws.send_json({"type": "warn", "message": str(ex)})
        finally:
            close = getattr(ext, "close", None)
            if close: close()
            _last_frames.pop(sid, None)

    return app


def app_factory():
    return create_app()
