"""Thin FastAPI shell. All logic lives in session.py / clarify.py / semantics.py."""
import os, json, asyncio
import numpy as np, cv2
from fastapi import FastAPI, WebSocket, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from .semantics import LANGS, CONCEPTS, EN, realize, match_text
from .session import SignSession
from core.speech import MockASRProvider, MockTTSProvider, ServerWhisperASRProvider, ServerTTSProvider

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
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    # Speech providers & interaction debug logs
    asr_provider = MockASRProvider()
    tts_provider = MockTTSProvider()
    interaction_logs = []

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

    @app.post("/api/asr")
    async def asr_endpoint(request: Request, lang: str = Query("en-IN")):
        body = await request.body()
        if not body:
            return JSONResponse({"error": "Empty audio body"}, 400)
        try:
            res = asr_provider.transcribe(body, language=lang)
            return {
                "text": res.text,
                "confidence": res.confidence,
                "language": res.language,
                "is_fallback": res.is_fallback,
                "provider": res.raw_provider
            }
        except Exception as ex:
            return JSONResponse({"error": str(ex)}, 400)

    @app.post("/api/tts")
    async def tts_endpoint(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        text = data.get("text", "")
        lang = data.get("lang", "en-IN")
        voice = data.get("voice")
        if not text:
            return JSONResponse({"error": "Missing 'text' in payload"}, 400)
        try:
            res = tts_provider.synthesize(text, language=lang, voice=voice)
            return Response(content=res.audio_data, media_type=res.mime_type)
        except Exception as ex:
            return JSONResponse({"error": str(ex)}, 400)

    @app.post("/api/debug/log_interaction")
    async def log_interaction(request: Request):
        try:
            data = await request.json()
            import time
            entry = {
                "timestamp": time.time(),
                "asr_text": data.get("asr_text", ""),
                "confidence": data.get("confidence", 1.0),
                "edited_text": data.get("edited_text", ""),
                "language": data.get("language", ""),
                "downstream_state": data.get("downstream_state", {}),
            }
            interaction_logs.append(entry)
            if len(interaction_logs) > 100:
                interaction_logs.pop(0)
            import logging
            logging.info("SPEECH_DEBUG: ASR='%s' (conf=%.2f) -> EDITED='%s' -> STATE=%s",
                         entry["asr_text"], entry["confidence"], entry["edited_text"], entry["downstream_state"])
            return {"status": "ok", "logged": True}
        except Exception as ex:
            return JSONResponse({"error": str(ex)}, 400)

    @app.get("/api/debug/interactions")
    def get_interaction_logs():
        return {"interactions": interaction_logs[-20:]}

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
                    try:
                        img = dec(msg["bytes"])
                        events = await loop.run_in_executor(None, sess.on_frame, msg["bytes"]) if img is not None else []
                        for e in events:
                            await ws.send_json(e)
                    except Exception as ex:
                        import logging, traceback
                        logging.exception("Error processing frame: %s", ex)
                        await ws.send_json({"type": "error", "message": f"Frame error: {str(ex)}"})
                    await ws.send_json({"type": "ack"})
                elif msg.get("text"):
                    try:
                        d = json.loads(msg["text"])
                        if not isinstance(d, dict):
                            raise ValueError("Message must be a JSON object")
                        mtype = d.get("type")
                        if mtype == "config":
                            sess.set_lang(d.get("lang", "hi"))
                        elif mtype == "answer":
                            for e in sess.on_answer(d.get("value")):
                                await ws.send_json(e)
                        elif mtype == "reset":
                            sess.reset()
                            await ws.send_json({"type": "state", "state": "listening"})
                        else:
                            await ws.send_json({"type": "error", "message": f"Unknown message type: {mtype}"})
                    except (json.JSONDecodeError, ValueError, KeyError) as ex:
                        import logging
                        logging.warning("Invalid WebSocket message: %s", ex)
                        await ws.send_json({"type": "error", "message": f"Invalid message: {str(ex)}"})
                    except Exception as ex:
                        import logging, traceback
                        logging.exception("Error processing WebSocket message: %s", ex)
                        await ws.send_json({"type": "error", "message": f"Server error: {str(ex)}"})
        finally:
            close = getattr(sess.ext, "close", None)
            if close:
                close()
    return app


def app_factory():
    return create_app()
