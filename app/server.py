import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .semantics import parse_to_semantic_sequence
from .session import SignSession
from clarifysign_ml.predict import Recognizer

app = FastAPI()

# 1. Dynamic Initialization: Reads actual trained classes from your model
recognizer = Recognizer("models/clarifysign_bilstm.pt", device="cpu")

@app.get("/api/config")
async def get_config():
    # Feeds the dynamic vocabulary directly to the frontend renderer
    return {"classes": recognizer.classes}

@app.get("/api/understand")
async def understand_speech(text: str = "", lang: str = "en", sid: str = ""):
    # Routes through your semantic NLU pipeline
    try:
        sequence = parse_to_semantic_sequence(text)
    except Exception:
        sequence = []
        
    return {
        "status": "success",
        "action": {
            "kind": "sequence", 
            "signs": sequence
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SignSession(recognizer)
    
    try:
        while True:
            data = await websocket.receive_text()
            feat_dict = json.loads(data)
            events = session.on_frame(feat_dict)
            if events:
                await websocket.send_json(events)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("app/static/index.html")