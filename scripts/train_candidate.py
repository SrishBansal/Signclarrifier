#!/usr/bin/env python3
"""Cloud/Colab entry point for a candidate model. Does not modify models/."""
import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clarifysign_ml.candidate import train_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL with labelled 225-feature files and group-safe splits")
    parser.add_argument("--output-dir", required=True, help="Cloud/Drive directory for candidate artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", default="models/clarifysign_bilstm.pt")
    parser.add_argument("--policy", default="data/candidate_policy.yaml")
    args = parser.parse_args()
    policy = (yaml.safe_load(Path(args.policy).read_text(encoding="utf-8")) or {})["promotion"]
    report = train_candidate(args.manifest, args.baseline, args.output_dir,
                             policy, args.device, args.epochs, args.patience, args.seed)
    print(f"candidate: {report['model_path']}")
    print(f"report: {report['report_path']}")
    print(f"promotion eligible: {report['promotion']['eligible']}")


if __name__ == "__main__":
    main()
