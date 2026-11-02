"""Each seed/case starts a fresh process to isolate simulator and config state."""

import argparse
import subprocess
import sys
from pathlib import Path

from .common import CASES, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--terrain", choices=("plane", "competition"), default="plane")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--goal-m", type=float, default=10)
    parser.add_argument("--min-success-rate", type=float)
    parser.add_argument("--trace", action="store_true", help="Record motion traces for diagnosis")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("Model not found.")
    if args.output.exists():
        parser.error("Output exists. Use a new directory.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.cases)) != len(args.cases):
        parser.error("Duplicate seeds/cases would overwrite evidence.")
    args.output.mkdir(parents=True)
    results = []
    for seed in args.seeds:
        for case in args.cases:
            out = args.output / (case + "_seed_" + str(seed))
            command = [sys.executable, "-m", "robot_test.evaluate", "--task=go2", "--headless", "--seed=" + str(seed), "--num_envs=" + str(args.num_envs), "--qa-model", str(args.model.resolve()), "--qa-output", str(out), "--qa-case", case, "--qa-terrain", args.terrain, "--qa-duration", str(args.duration), "--qa-goal-m", str(args.goal_m)]
            if args.min_success_rate is not None:
                command.extend(["--qa-min-success-rate", str(args.min_success_rate)])
            if args.trace:
                command.append("--qa-trace")
            print("RUN", " ".join(command), flush=True)
            result = subprocess.run(command)
            results.append({"seed": seed, "case": case, "returncode": result.returncode, "path": str(out)})
            save_json(args.output / "matrix_status.json", results)
    return 1 if any(r["returncode"] != 0 for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
