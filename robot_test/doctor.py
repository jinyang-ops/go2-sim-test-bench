"""Read-only diagnostics. Does not install software or modify driver settings."""

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

from .common import command_output, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="environment.json")
    parser.add_argument("--deep", action="store_true", help="Also try Isaac Gym then torch in an isolated child process")
    args = parser.parse_args()
    report = {
        "os": dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line) if Path("/etc/os-release").exists() else platform.system(),
        "architecture": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "conda_version": command_output(["conda", "--version"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "nvcc": command_output(["nvcc", "--version"]),
        "ram": command_output(["free", "-h"]),
        "packages": {},
        "notes": ["Ubuntu 22.04 is outside Isaac Gym's published 18.04/20.04 prerequisites; needs a local smoke test.", "ROS 2/Gazebo installation does not establish Isaac Gym compatibility."],
    }
    for name in ("isaacgym", "torch", "legged_gym", "rsl_rl"):
        try:
            spec = importlib.util.find_spec(name)
            report["packages"][name] = spec.origin if spec else None
        except (ImportError, ValueError):
            report["packages"][name] = None
    if args.deep:
        import subprocess
        code = "import isaacgym; import torch; print('Isaac Gym import OK'); print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda)"
        try:
            result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=90)
            report["deep_import"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr[-8000:]}
        except subprocess.TimeoutExpired:
            report["deep_import"] = {"error": "Import did not finish within 90 seconds"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
