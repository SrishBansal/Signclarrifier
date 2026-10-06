import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.server import _json_native


def test_json_native_converts_nested_numpy_payload_values():
    event = {
        "type": "diag",
        "research": {
            "fps": np.float32(23.9),
            "frames": np.int64(48),
            "pose": np.bool_(True),
            "top": np.array([np.float32(0.38), np.float32(0.18)]),
        },
    }
    payload = _json_native(event)
    assert payload == {"type": "diag", "research": {"fps": 23.899999618530273,
                                                         "frames": 48, "pose": True,
                                                         "top": [0.3799999952316284, 0.18000000715255737]}}
    json.dumps(payload, allow_nan=False)


def test_send_event_disconnect_catches_websocket_disconnect_and_does_not_retry():
    import asyncio
    from starlette.websockets import WebSocketDisconnect
    from app.server import _send_event

    class DisconnectedWS:
        def __init__(self):
            self.calls = 0

        async def send_json(self, payload):
            self.calls += 1
            raise WebSocketDisconnect(code=1000)

    async def _run():
        ws = DisconnectedWS()
        sid = "sid-test-unit"
        ok1 = await _send_event(ws, {"type": "test1"}, sid=sid, source="test")
        assert ok1 is False
        assert getattr(ws, "_client_disconnected", False) is True
        assert ws.calls == 1

        # Follow-up send should return False immediately without calling send_json
        ok2 = await _send_event(ws, {"type": "test2"}, sid=sid, source="test")
        assert ok2 is False
        assert ws.calls == 1

    asyncio.run(_run())


def test_send_event_catches_connection_closed_ok():
    import asyncio
    from websockets.exceptions import ConnectionClosedOK
    from app.server import _send_event

    class ClosedWS:
        def __init__(self):
            self.calls = 0

        async def send_json(self, payload):
            self.calls += 1
            raise ConnectionClosedOK(None, None)

    async def _run():
        ws = ClosedWS()
        sid = "sid-test-closed"
        ok1 = await _send_event(ws, {"type": "test1"}, sid=sid, source="test")
        assert ok1 is False
        assert getattr(ws, "_client_disconnected", False) is True
        assert ws.calls == 1

    asyncio.run(_run())


def test_midstream_disconnect_logs_single_clean_line_and_no_traceback():
    import asyncio
    import logging
    import cv2
    import uvicorn
    import websockets
    from tests.test_e2e import make_app

    logger = logging.getLogger("clarifysign")
    logger.setLevel(logging.INFO)
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    h = Handler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)

    app = make_app()

    async def _scenario():
        config = uvicorn.Config(app, host="127.0.0.1", port=8995, log_level="critical")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)

        img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        jpeg_data = buf.tobytes()

        sid = "sid-midstream-test"
        ws = await websockets.connect(f"ws://127.0.0.1:8995/ws/sign?sid={sid}")
        await ws.recv()
        await ws.send(jpeg_data)
        ws.transport.abort()

        await asyncio.sleep(0.4)
        server.should_exit = True
        await task

    try:
        asyncio.run(_scenario())
    finally:
        logger.removeHandler(h)

    disconnect_lines = [r for r in records if "disconnected" in r or "closed" in r]
    errors_and_tracebacks = [r for r in records if "ERROR" in r or "Traceback" in r or "Exception" in r]

    assert len(disconnect_lines) == 1, f"Expected 1 disconnect line, got: {disconnect_lines}"
    assert "client disconnected sid=sid-midstream-test" in disconnect_lines[0]
    assert len(errors_and_tracebacks) == 0, f"Expected 0 errors/tracebacks, got: {errors_and_tracebacks}"


