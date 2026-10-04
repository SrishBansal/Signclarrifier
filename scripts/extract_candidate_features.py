#!/usr/bin/env python3
"""Extract cloud-hosted videos into candidate feature files and a validated manifest.

Input JSONL is deliberately dataset-neutral.  Each row must provide dataset,
sample_id, group_id, split, label, and video_path.  The researcher supplies the
label mapping; this script never invents one from source-dataset labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarifysign_ml.features import LandmarkExtractor, extract_video
from clarifysign_ml.research import ManifestSample, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL rows with source video paths and reviewed labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="models/holistic_landmarker.task")
    parser.add_argument("--min-pose-rate", type=float, default=0.70)
    args = parser.parse_args()
    if not 0 <= args.min_pose_rate <= 1:
        raise ValueError("--min-pose-rate must be between 0 and 1")

    destination = Path(args.output_dir)
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    extracted = []
    extractor = LandmarkExtractor(args.task)
    try:
        for row in rows:
            required = ("dataset", "sample_id", "group_id", "split", "label", "video_path")
            if not all(row.get(key) for key in required):
                raise ValueError(f"missing required input fields for {row.get('sample_id', '<unknown>')}")
            features, _, detections = extract_video(row["video_path"], extractor)
            if detections["pose"] < args.min_pose_rate:
                raise ValueError(f"{row['sample_id']}: pose detection rate {detections['pose']:.2%} is below threshold")
            output = feature_dir / f"{row['sample_id']}.npy"
            import numpy as np
            np.save(output, features)
            extracted.append(ManifestSample(row["dataset"], row["sample_id"], row["group_id"],
                                            row["split"], str(output), label=row["label"],
                                            signer_id=row.get("signer_id", ""), text=row.get("text", "")))
    finally:
        extractor.close()
    manifest = destination / "manifest.jsonl"
    write_manifest(manifest, extracted)
    print(manifest)


if __name__ == "__main__":
    main()
