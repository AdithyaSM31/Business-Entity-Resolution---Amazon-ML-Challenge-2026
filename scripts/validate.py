"""Run the official challenge submission validator for the team's output files."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Chai-Square submission files.")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test")
    args = parser.parse_args()

    validator = Path(__file__).resolve().parents[1] / "utils" / "validate_submission.py"
    if not validator.exists():
        print(f"ERROR: validator not found: {validator}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, str(validator),
        "--matching", str(args.matching),
        "--candidate", str(args.candidate),
        "--test-dir", str(args.test_dir),
    ]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
