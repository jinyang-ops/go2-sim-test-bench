"""Shared utilities; heavy libraries are imported only by simulator entry points."""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


UPSTREAM_URL = "https://github.com/jindadu00/legged_robot_competition"
UPSTREAM_COMMIT = "7ef08a3f8069750213071adf0032c04cf1be1777"
CASES = ("nominal", "robot_friction", "added_mass", "noisy_observation", "push")


def dependency_versions():
    versions = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "matplotlib", "tensorboard", "isaacgym", "legged_gym", "rsl_rl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_output(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_identity(root):
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], root),
        "status": command_output(["git", "status", "--porcelain"], root),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def upstream_arguments(remaining):
    sys.argv = [sys.argv[0]] + list(remaining)
    from legged_gym.utils import get_args
    args = get_args()
    if args.task != "go2":
        raise ValueError("This bench supports --task=go2 only.")
    return args


def configure_case(cfg, case):
    # Nominal is explicitly defined, including robot-shape friction = 1.0.
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.noise.add_noise = case == "noisy_observation"
    cfg.domain_rand.randomize_friction = True
    friction = 0.3 if case == "robot_friction" else 1.0
    cfg.domain_rand.friction_range = [friction, friction]
    cfg.domain_rand.randomize_base_mass = case == "added_mass"
    cfg.domain_rand.added_mass_range = [2.0, 2.0] if case == "added_mass" else [0.0, 0.0]
    cfg.domain_rand.randomize_base_com = False
    cfg.domain_rand.randomize_motor = False
    cfg.domain_rand.push_robots = case == "push"
    cfg.domain_rand.push_interval_s = 5.0
    cfg.domain_rand.max_push_vel_xy = 0.5
