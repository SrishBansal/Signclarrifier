#!/usr/bin/env bash
# 1) put these 4 files in ./models :  clarifysign_bilstm.pt  holistic_landmarker.task  signs.json  (from the Colab download)
# 2) ./run.sh   then open http://localhost:8001 in Chrome
set -e
python3 -m pip install -q -r requirements.txt
python3 tests/test_ml_core.py && python3 tests/test_app_core.py
python3 -m uvicorn app.server:app_factory --factory --host 127.0.0.1 --port 8001
