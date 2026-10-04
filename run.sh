#!/usr/bin/env bash
# Place these 4 files in ./models/ before running:
#   clarifysign_bilstm.pt   signs.json   metrics.json   holistic_landmarker.task
# Then: ./run.sh  ->  http://localhost:8000
set -e
PORT="${PORT:-8000}"
if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port $PORT is already in use."
    exit 1
fi
python3 -m pip install -q -r requirements.txt
python3 -m uvicorn app.server:app_factory --factory --host 127.0.0.1 --port "$PORT"
