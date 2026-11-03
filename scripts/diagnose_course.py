#!/usr/bin/env python3
"""Observe one GO2 episode without updating the policy or upstream files."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--headless", action="store_true")
    options = parser.parse_args()
    model = options.model.expanduser().resolve()
    if not model.is_file():
        parser.error("Checkpoint not found: {}".format(model))
    if not math.isfinite(options.seconds) or options.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    model_hash = checkpoint_sha256(model)

    # Isaac Gym must be imported before PyTorch.
    import isaacgym  # noqa: F401
    import torch
    import legged_gym.envs  # noqa: F401; registers go2
    from legged_gym.utils import get_args, task_registry

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--task=go2", "--num_envs=1", "--seed=1"]
        if options.headless:
            sys.argv.append("--headless")
        args = get_args()
    finally:
        sys.argv = original_argv

    cfg, train_cfg = task_registry.get_cfgs("go2")
    if cfg.terrain.mesh_type != "competition":
        parser.error("Expected the GO2 competition terrain")
    cfg.env.episode_length_s = options.seconds
    # Match the upstream play.py evaluation overrides.
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = False
    args.resume = False
    env = None
    try:
        env, _ = task_registry.make_env(name="go2", args=args, env_cfg=cfg)
        origin = env.env_origins[0, :2].detach().cpu().tolist()
        expected_origin = [2.0, 0.5 * cfg.terrain.terrain_width]
        if any(abs(a - b) > 1e-4 for a, b in zip(origin, expected_origin)):
            raise RuntimeError("Course-start patch is not active: origin={}".format(origin))
        runner, _ = task_registry.make_alg_runner(
            env=env, args=args, train_cfg=train_cfg, log_root=None
        )
        print("Loading checkpoint: {}".format(model), flush=True)
        runner.load(str(model), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        # Runner construction resets the environment: obtain observations after it.
        obs = env.get_observations()
        start_xyz = env.root_states[0, :3].detach().cpu().tolist()
        peak_x = start_xyz[0]
        at_25s = None
        final_state = None
        next_print = 5.0
        original_check = env.check_termination

        def observe_termination():
            nonlocal peak_x, at_25s, final_state, next_print
            original_check()
            # This hook runs before reset_idx overwrites the terminal state.
            root = env.root_states[0].detach().cpu().tolist()
            elapsed = float(env.episode_length_buf[0].item()) * env.dt
            peak_x = max(peak_x, root[0])
            qx, qy, qz, qw = root[3:7]
            yaw = math.degrees(math.atan2(
                2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)
            ))
            state = {
                "episode_s": round(elapsed, 3),
                "xyz_m": [round(v, 3) for v in root[:3]],
                "max_progress_x_m": round(peak_x - start_xyz[0], 3),
                "yaw_deg": round(yaw, 1),
                "body_vx_mps": round(float(env.base_lin_vel[0, 0].item()), 3),
                "world_vx_mps": round(root[7], 3),
                "command_vx_mps": round(float(env.commands[0, 0].item()), 3),
                "reset_reasons": [],
            }
            done = bool(env.reset_buf[0].item())
            if done:
                contact = torch.norm(
                    env.contact_forces[0, env.termination_contact_indices, :], dim=-1
                )
                if bool(torch.any(contact > 1.0).item()):
                    state["reset_reasons"].append("base_contact")
                if bool(env.time_out_buf[0].item()):
                    state["reset_reasons"].append("time_limit")
                if root[2] < -3.0:
                    state["reset_reasons"].append("below_height_limit")
                if not state["reset_reasons"]:
                    state["reset_reasons"].append("unknown_termination")
                final_state = state
            if at_25s is None and elapsed >= 25.0:
                at_25s = state
            if elapsed >= next_print or done:
                print("DIAG_STEP " + json.dumps(state, allow_nan=False), flush=True)
                next_print = elapsed + 5.0

        env.check_termination = observe_termination
        with torch.inference_mode():
            for _ in range(int(env.max_episode_length) + 2):
                obs, _, _, _, _ = env.step(policy(obs.detach()).detach())
                if final_state is not None:
                    break
        if final_state is None:
            raise RuntimeError("No termination observed within the configured episode")
        result = {
            "model_sha256": model_hash,
            "limit_s": options.seconds,
            "seed": 1,
            "start_xyz_m": [round(v, 3) for v in start_xyz],
            "at_25s": at_25s,
            "before_reset": final_state,
        }
        print("DIAG_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    finally:
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
