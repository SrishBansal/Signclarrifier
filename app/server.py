"""FastAPI shell. All logic lives in core/. No app/semantics or app/clarify imports."""
import os, json, asyncio, time, uuid
import numpy as np, cv2
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from core.ontology import get_ontology
from core.nlu import NLUParser
from core.dialogue import DialogueManager
from core.planner import ISLPlanner, get_planner
from core.semantics import NaturalLanguageRealizer
from .session import SignSession

HERE = os.path.dirname(__file__)
MODEL  = os.environ.get("MODEL_PATH",  "models/clarifysign_bilstm.pt")
TASK   = os.environ.get("TASK_PATH",   "models/holistic_landmarker.task")
SIGNS  = os.environ.get("SIGNS_PATH",  "models/signs.json")
PORT   = int(os.environ.get("PORT", "8000"))


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
    # Evict stale
    stale = [k for k, (_, t) in _sessions.items() if now - t > _SESSION_TTL]
    for k in stale:
        del _sessions[k]
    return dm


def create_app(recognizer=None, reason="", extractor_factory=None):
    if recognizer is None and not reason:
        recognizer, reason = load_backend()
    ont = get_ontology()
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
    def understand(text: str = Query(""), lang: str = "en", sid: str = Query("")):
        if not sid:
            sid = str(uuid.uuid4())
        dm = _get_dm(sid)
        lang_names = {"en": "English", "hi": "Hindi", "ta": "Tamil", "hinglish": "Hinglish"}
        language = lang_names.get(lang, "English")
        parsed = nlu.parse(text, language, dm.get_state())
        state = dm.update_from_shopkeeper(parsed)
        plan_items = planner.plan(state)   # list of concept strings
        # Build plan as dicts with sign_id, gloss, kind
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

    @app.websocket("/ws/sign")
    async def ws_sign(ws: WebSocket, sid: str = Query("")):
        await ws.accept()
        if not sid:
            sid = str(uuid.uuid4())
        if recognizer is None:
            await ws.send_json({"type": "error", "message": reason}); await ws.close(); return
        dm = _get_dm(sid)
        sess = SignSession(recognizer, extractor_factory(), dm=dm, decode=dec)
        loop = asyncio.get_running_loop()
        await ws.send_json({"type": "state", "state": "listening"})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    try:
                        img = dec(msg["bytes"])
                        events = await loop.run_in_executor(None, sess.on_frame, msg["bytes"]) if img is not None else []
                        for e in events:
                            await ws.send_json(e)
                    except Exception as ex:
                        import logging; logging.exception("frame error: %s", ex)
                        await ws.send_json({"type": "error", "message": f"Frame error: {ex}"})
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
                            await ws.send_json({"type": "error", "message": f"Unknown type: {mtype}"})
                    except Exception as ex:
                        import logging; logging.exception("ws msg error: %s", ex)
                        await ws.send_json({"type": "error", "message": str(ex)})
        finally:
            close = getattr(sess.ext, "close", None)
            if close: close()

    return app


def app_factory():
    return create_app()
