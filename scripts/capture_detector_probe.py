#!/usr/bin/env python3
"""Capture one direct webcam frame and test ClarifySign's live extractor.

This deliberately bypasses the browser, JPEG/WebSocket transport, and FastAPI.
It uses the same LandmarkExtractor, Holistic task, and hand-fallback task as the app.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from clarifysign_ml.features import LandmarkExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "detector_probe.jpg",
                        help="Where to save the exact raw BGR frame passed to LandmarkExtractor")
    parser.add_argument("--task", type=Path, default=ROOT / "models" / "holistic_landmarker.task",
                        help="MediaPipe holistic task model used by the app")
    parser.add_argument("--auto", action="store_true", help="Capture after camera warm-up instead of waiting for Space")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera_index}. On macOS, allow camera access for the terminal, "
            "then retry (or pass --camera-index 1)."
        )
    try:
        # Let auto-exposure and focus settle before allowing capture.
        frame = None
        for _ in range(30):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("OpenCV could not read a frame from the camera")
        if not args.auto:
            print("Show your hand in the preview, then press Space to capture (Esc to cancel).")
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("OpenCV could not read a frame from the camera")
                preview = frame.copy()
                cv2.putText(preview, "SPACE: capture  ESC: cancel", (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.imshow("ClarifySign detector probe", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    break
                if key == 27:
                    print("Capture cancelled.")
                    return 0
        assert frame is not None
    finally:
        capture.release()
        cv2.destroyAllWindows()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise RuntimeError(f"Could not write {args.output}")

    extractor = LandmarkExtractor(str(args.task))
    try:
        features = extractor(frame)
        detected = extractor.last
    finally:
        extractor.close()

    print(f"saved_frame={args.output.resolve()}")
    print(f"shape={frame.shape} dtype={frame.dtype} bgr_mean={frame.mean(axis=(0, 1)).round(1).tolist()}")
    print(f"pose_landmarks={'POPULATED' if detected.get('pose') else 'EMPTY'}")
    print(f"left_hand_landmarks={'POPULATED' if detected.get('lh') else 'EMPTY'}")
    print(f"right_hand_landmarks={'POPULATED' if detected.get('rh') else 'EMPTY'}")
    print(f"hand_sources={detected.get('hand_source', {})}")
    print(f"extractor_aspect={detected.get('aspect')} y_scale={detected.get('y_scale')}")
    print(f"feature_nonzero={int((features != 0).sum())}/{features.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
