"""Close-up plane playback with a following camera; separate from benchmark scoring."""

import argparse
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--terrain", choices=("plane", "competition"), default="plane")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--case", choices=("nominal", "added_mass", "robot_friction"), default="nominal")
    parser.add_argument("--heading-mode", choices=("heading", "yaw_rate"), default="heading")
    qa = parser.parse_args()
    if not qa.model.is_file():
        parser.error("Model not found: " + str(qa.model))
    if not math.isfinite(qa.seconds) or qa.seconds <= 0 or not math.isfinite(qa.speed) or qa.speed <= 0:
        parser.error("Seconds and speed must be positive finite numbers.")

    import isaacgym  # Must precede torch.
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.modules import ActorCritic
    from .common import configure_case, upstream_arguments

    args = upstream_arguments(["--task=go2", "--num_envs=1", "--seed=0"])
    cfg, train_cfg = task_registry.get_cfgs("go2")
    cfg.seed = train_cfg.seed = 0
    cfg.terrain.mesh_type = qa.terrain
    cfg.env.episode_length_s = max(25.0, qa.seconds + 1.0)
    configure_case(cfg, qa.case)
    cfg.commands.heading_command = qa.heading_mode == "heading"
    cfg.commands.ranges.lin_vel_x = [qa.speed, qa.speed]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.ranges.heading = [0.0, 0.0]
    cfg.viewer.pos = [2.5, -3.0, 1.8]
    cfg.viewer.lookat = [0.0, 0.0, 0.3]
    env, _ = task_registry.make_env("go2", args=args, env_cfg=cfg)
    try:
        if env.viewer is None:
            raise RuntimeError("No viewer was created; run this command in a graphical desktop session.")
        if train_cfg.runner.policy_class_name != "ActorCritic":
            raise ValueError("This playback supports the non-recurrent GO2 ActorCritic only.")
        critic_obs = env.num_privileged_obs if env.num_privileged_obs is not None else env.num_obs
        actor = ActorCritic(env.num_obs, critic_obs, env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
        checkpoint = torch.load(str(qa.model), map_location=env.device, weights_only=True)
        actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
        actor.eval()
        obs, _ = env.reset()
        print("DEMO: terrain={}, case={}, heading_mode={}, speed={}m/s; not a scored benchmark.".format(qa.terrain, qa.case, qa.heading_mode, qa.speed), flush=True)
        print("Loaded:", qa.model, flush=True)
        print("Following camera. Close the viewer or press Escape/Ctrl+C to finish.", flush=True)
        interval = max(1, round(1.0 / env.dt))
        with torch.no_grad():
            for step in range(math.ceil(qa.seconds / env.dt)):
                if env.gym.query_viewer_has_closed(env.viewer):
                    break
                if not bool(torch.isfinite(obs).all()):
                    raise FloatingPointError("Nonfinite observation.")
                actions = actor.act_inference(obs)
                if not bool(torch.isfinite(actions).all()):
                    raise FloatingPointError("Nonfinite action.")
                obs, _, _, done, _ = env.step(actions)
                position = env.root_states[0, :3].detach().cpu().tolist()
                x, y, z = position
                env.set_camera([x + 2.5, y - 3.0, z + 1.5], [x, y, z])
                if step % interval == 0:
                    print("t={:.1f}s world_xy=({:.2f},{:.2f}) body_vx={:.2f}m/s".format((step + 1) * env.dt, x, y, env.base_lin_vel[0, 0].item()), flush=True)
                if bool(done[0]):
                    print("Environment reset (e.g. termination contact); playback continues.", flush=True)
    except KeyboardInterrupt:
        print("Playback stopped.")
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
            env.viewer = None
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
