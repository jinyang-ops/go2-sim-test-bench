"""Evaluate one non-recurrent GO2 checkpoint; one first episode per environment."""

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import CASES, UPSTREAM_COMMIT, UPSTREAM_URL, command_output, dependency_versions, git_identity, save_json, sha256, upstream_arguments, configure_case
from .instrumentation import SnapshotMixin
from .metrics import Trial, summarize


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-model", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--qa-terrain", choices=("plane", "competition"), default="plane")
    parser.add_argument("--qa-case", choices=CASES, default="nominal")
    parser.add_argument("--qa-duration", type=float, default=20.0)
    parser.add_argument("--qa-goal-m", type=float, default=10.0)
    parser.add_argument("--qa-speed", type=float, default=1.0)
    parser.add_argument("--qa-min-success-rate", type=float)
    qa, remaining = parser.parse_known_args()
    for value in (qa.qa_duration, qa.qa_goal_m, qa.qa_speed):
        if not math.isfinite(value) or value <= 0:
            parser.error("Duration, goal and speed must be positive finite values.")
    if qa.qa_case == "push" and qa.qa_duration <= 5:
        parser.error("Push tests need more than 5 seconds to encounter a scheduled push.")
    if qa.qa_terrain == "competition" and qa.qa_goal_m > 94:
        parser.error("Fixed course spawn is x=2; use a goal <=94 m (x<=96).")
    threshold = qa.qa_min_success_rate
    if threshold is not None and not (math.isfinite(threshold) and 0 <= threshold <= 1):
        parser.error("Success threshold must be in [0, 1].")
    if not qa.qa_model.is_file():
        parser.error("Checkpoint does not exist: " + str(qa.qa_model))
    if qa.qa_output.exists():
        parser.error("Output already exists; select a fresh directory to preserve prior evidence.")

    import isaacgym  # NVIDIA requires this before importing torch.
    import torch
    import legged_gym.envs
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs.go2.go2_robot import Go2Robot
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.modules import ActorCritic

    args = upstream_arguments(remaining)
    if args.num_envs is None:
        args.num_envs = 8
    if args.num_envs < 1:
        parser.error("--num_envs must be positive.")
    env_cfg, train_cfg = task_registry.get_cfgs("go2")
    seed = args.seed if args.seed is not None else 0
    if seed < 0:
        parser.error("Use an explicit nonnegative evaluation seed.")
    env_cfg.seed = train_cfg.seed = seed
    env_cfg.env.episode_length_s = qa.qa_duration
    env_cfg.terrain.mesh_type = qa.qa_terrain
    configure_case(env_cfg, qa.qa_case)
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [qa.qa_speed, qa.qa_speed]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    class TestGo2(SnapshotMixin, Go2Robot):
        def _get_env_origins(self):
            if self.cfg.terrain.mesh_type != "competition":
                return super()._get_env_origins()
            self.custom_origins = False
            self.env_origins = torch.tensor([2.0, 6.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        def _process_rigid_shape_props(self, props, env_id):
            props = super()._process_rigid_shape_props(props, env_id)
            if not hasattr(self, "qa_robot_friction"):
                self.qa_robot_friction = {}
            self.qa_robot_friction[env_id] = [float(p.friction) for p in props]
            return props

        def _process_rigid_body_props(self, props, env_id):
            props = super()._process_rigid_body_props(props, env_id)
            if not hasattr(self, "qa_base_masses"):
                self.qa_base_masses = {}
            self.qa_base_masses[env_id] = float(props[0].mass)
            return props

    task_registry.register("go2_qa", TestGo2, env_cfg, train_cfg)
    env, env_cfg = task_registry.make_env("go2_qa", args=args, env_cfg=env_cfg)
    if train_cfg.runner.policy_class_name != "ActorCritic":
        raise ValueError("Recurrent policies need memory-reset handling and are not supported here.")
    critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
    actor = ActorCritic(env.num_obs, critic_obs, env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
    # The supplied environment recipe uses torch 2.4, which supports weights_only.
    checkpoint = torch.load(str(qa.qa_model), map_location=env.device, weights_only=True)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    obs, _ = env.reset()
    initial_x = env.root_states[:, 0].detach().cpu().tolist()
    trials = [Trial(i, x, qa.qa_goal_m, env.dt, lane=12.0 if qa.qa_terrain == "competition" else None) for i, x in enumerate(initial_x)]
    obs_shape = tuple(obs.shape)
    if obs_shape != (env.num_envs, env.num_obs) or not bool(torch.isfinite(obs).all()):
        raise ValueError("Invalid observation shape or nonfinite initial observations: " + str(obs_shape))
    max_steps = math.ceil(qa.qa_duration / env.dt)
    with torch.no_grad():
        for _ in range(max_steps):
            if not bool(torch.isfinite(obs).all()):
                raise FloatingPointError("Nonfinite observation; run is invalid.")
            actions = actor.act_inference(obs)
            if actions.shape != (env.num_envs, env.num_actions) or not bool(torch.isfinite(actions).all()):
                raise FloatingPointError("Invalid policy actions; run is invalid.")
            obs, _, _, _, _ = env.step(actions)
            # Consume the PRE-RESET state, never env.root_states after env.step().
            snapshot = {key: value.cpu().tolist() for key, value in env.qa_snapshot.items()}
            for i, trial in enumerate(trials):
                trial.update(*(snapshot[key][i] for key in ("state", "velocity", "command", "torques", "contact", "timeout", "done")))
            if all(t.reason is not None for t in trials):
                break

    rows = [t.finish() for t in trials]
    summary = summarize(rows)
    qa.qa_output.mkdir(parents=True, exist_ok=False)
    with (qa.qa_output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_url": UPSTREAM_URL, "inspected_upstream_commit": UPSTREAM_COMMIT,
        "upstream_worktree": git_identity(LEGGED_GYM_ROOT_DIR),
        "test_bench": git_identity(Path(__file__).resolve().parents[1]),
        "checkpoint_path": str(qa.qa_model.resolve()), "checkpoint_sha256": sha256(qa.qa_model),
        "seed": seed, "case": qa.qa_case, "terrain": qa.qa_terrain,
        "duration_limit_s": qa.qa_duration, "goal_progress_m": qa.qa_goal_m,
        "command_vx_mps": qa.qa_speed, "dt_s": env.dt, "observation_shape": list(obs_shape),
        "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "dependencies": dependency_versions(),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "robot_shape_frictions": env.qa_robot_friction, "base_masses_kg": env.qa_base_masses,
        "env_config": class_to_dict(env_cfg), "policy_config": class_to_dict(train_cfg.policy),
        "interpretation": "First-episode fixed-distance milestone; not proof of full-course traversal. Mean torque is not energy consumption. CUDA determinism is not guaranteed.",
    }
    save_json(qa.qa_output / "metadata.json", metadata)
    save_json(qa.qa_output / "summary.json", summary)
    print(summary)
    print("Results:", qa.qa_output)
    if summary["outcomes"].get("invalid_state", 0):
        return 1
    if threshold is not None and summary["success_rate"] < threshold:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
