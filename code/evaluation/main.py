#!/usr/bin/env python3
"""Evaluation entry point for the damage claim verification project."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample evaluation.")
    parser.add_argument("--sample-claims-file", default="dataset/sample_claims.csv")
    parser.add_argument("--evidence-file", default="dataset/evidence_requirements.csv")
    parser.add_argument("--history-file", default="dataset/user_history.csv")
    parser.add_argument("--output-file", default="evaluation/sample_predictions_full.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline = os.path.join(repo_root, "code", "main.py")

    command = [
        sys.executable,
        pipeline,
        "--claims-file",
        args.sample_claims_file,
        "--sample-claims-file",
        args.sample_claims_file,
        "--evidence-file",
        args.evidence_file,
        "--history-file",
        args.history_file,
        "--output-file",
        args.output_file,
        "--run-evaluation",
        "--mock-mode",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
