#!/usr/bin/env bash
# 1) put these 4 files in ./models :  clarifysign_bilstm.pt  holistic_landmarker.task  signs.json  (from the Colab download)
# 2) ./run.sh   then open http://localhost:8001 in Chrome
set -e
PORT="${PORT:-8001}"
if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port $PORT is busy and already in use by process:"
    lsof -i :"$PORT"
    echo "Exit clearly without launching."
    exit 1
fi
python3 -m pip install -q -r requirements.txt
python3 tests/test_ml_core.py && python3 tests/test_app_core.py
python3 -m uvicorn app.server:app_factory --factory --host 127.0.0.1 --port "$PORT"
