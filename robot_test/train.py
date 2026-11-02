"""Training wrapper that synchronizes CLI seed before environment creation."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-terrain", choices=("plane", "competition"), default="plane")
    parser.add_argument("--qa-domain-rand", action="store_true")
    qa, remaining = parser.parse_known_args()

    import isaacgym  # Must precede torch and legged_gym.envs.
    import legged_gym.envs
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from .common import dependency_versions, git_identity, save_json, upstream_arguments

    args = upstream_arguments(remaining)
    env_cfg, train_cfg = task_registry.get_cfgs("go2")
    seed = args.seed if args.seed is not None else train_cfg.seed
    if seed < 0:
        raise ValueError("Use an explicit nonnegative training seed for recorded experiments.")
    train_cfg.seed = env_cfg.seed = seed
    env_cfg.terrain.mesh_type = qa.qa_terrain
    env_cfg.noise.add_noise = qa.qa_domain_rand
    env_cfg.domain_rand.randomize_friction = qa.qa_domain_rand
    env_cfg.domain_rand.friction_range = [0.3, 1.2]
    env_cfg.domain_rand.randomize_base_mass = qa.qa_domain_rand
    env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]
    env_cfg.domain_rand.push_robots = qa.qa_domain_rand
    env_cfg.domain_rand.push_interval_s = 5.0
    env_cfg.domain_rand.max_push_vel_xy = 0.5
    env, env_cfg = task_registry.make_env("go2", args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    save_json(Path(runner.log_dir) / "qa_training.json", {
        "seed": seed, "terrain": qa.qa_terrain, "domain_randomization": qa.qa_domain_rand,
        "upstream": git_identity(LEGGED_GYM_ROOT_DIR),
        "test_bench": git_identity(Path(__file__).resolve().parents[1]),
        "dependencies": dependency_versions(),
        "env_cfg": class_to_dict(env_cfg), "train_cfg": class_to_dict(train_cfg),
    })
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
