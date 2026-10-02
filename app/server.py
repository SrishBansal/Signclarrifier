"""Thin FastAPI shell. All logic lives in session.py / clarify.py / semantics.py."""
import os, json, asyncio
import numpy as np, cv2
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import FileResponse, JSONResponse
from .semantics import LANGS, CONCEPTS, EN, realize, match_text
from .session import SignSession

HERE = os.path.dirname(__file__)
MODEL = os.environ.get("MODEL_PATH", "models/clarifysign_bilstm.pt")
TASK = os.environ.get("TASK_PATH", "models/holistic_landmarker.task")
SIGNS = os.environ.get("SIGNS_PATH", "models/signs.json")


def load_backend():
    """Returns (recognizer|None, reason). Never fabricates predictions: no model means no recognition."""
    if not os.path.exists(MODEL):
        return None, f"No trained model at {MODEL}. Run the Colab notebook and copy the model here."
    if not os.path.exists(TASK):
        return None, f"Missing MediaPipe model at {TASK}."
    from clarifysign_ml.predict import Recognizer
    return Recognizer(MODEL), ""


def create_app(recognizer=None, reason="", extractor_factory=None):
    if recognizer is None and not reason:
        recognizer, reason = load_backend()
    if extractor_factory is None:
        from clarifysign_ml.features import LandmarkExtractor
        extractor_factory = lambda: LandmarkExtractor(TASK)
    app = FastAPI(title="ClarifySign")
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "static", "index.html"))

    @app.get("/api/config")
    def config():
        return {"languages": [{"code": k, "name": v[0], "tag": v[1]} for k, v in LANGS.items()],
                "concepts": [{"id": c, "english": EN[c], "kind": k} for c, k in CONCEPTS.items()],
                "recognition_ready": recognizer is not None, "reason": reason,
                "signs_available": os.path.exists(SIGNS)}

    @app.get("/api/understand")
    def understand(text: str = Query(""), lang: str = "hi"):
        if lang not in LANGS:
            return JSONResponse({"error": "unsupported language"}, 400)
        ids = match_text(text, lang)
        return {"concepts": [{"id": c, "english": EN[c], "text": realize(c, lang)} for c in ids], "heard": text}

    @app.get("/api/signs")
    def signs():
        if not os.path.exists(SIGNS):
            return JSONResponse({"error": "signs.json missing. Run the Colab export cell."}, 404)
        return FileResponse(SIGNS, media_type="application/json")

    @app.websocket("/ws/sign")
    async def ws_sign(ws: WebSocket):
        await ws.accept()
        if recognizer is None:
            await ws.send_json({"type": "error", "message": reason}); await ws.close(); return
        sess = SignSession(recognizer, extractor_factory(), decode=dec)
        loop = asyncio.get_running_loop()
        await ws.send_json({"type": "state", "state": "listening"})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    img = dec(msg["bytes"])
                    events = await loop.run_in_executor(None, sess.on_frame, msg["bytes"]) if img is not None else []
                    for e in events:
                        await ws.send_json(e)
                    await ws.send_json({"type": "ack"})
                elif msg.get("text"):
                    d = json.loads(msg["text"])
                    if d.get("type") == "config":
                        sess.set_lang(d["lang"])
                    elif d.get("type") == "answer":
                        for e in sess.on_answer(d["value"]):
                            await ws.send_json(e)
                    elif d.get("type") == "reset":
                        sess.reset(); await ws.send_json({"type": "state", "state": "listening"})
        finally:
            close = getattr(sess.ext, "close", None)
            if close: close()
    return app


def app_factory():
    return create_app()
