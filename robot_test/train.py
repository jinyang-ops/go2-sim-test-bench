"""Training wrapper that synchronizes CLI seed before environment creation."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-terrain", choices=("plane", "competition"), default="plane")
    randomization = parser.add_mutually_exclusive_group()
    randomization.add_argument("--qa-domain-rand", action="store_true")
    randomization.add_argument("--qa-mass-rand", action="store_true")
    qa, remaining = parser.parse_known_args()

    import isaacgym  # Must precede torch and legged_gym.envs.
    import legged_gym.envs
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from .common import configure_training_randomization, dependency_versions, git_identity, save_json, upstream_arguments

    args = upstream_arguments(remaining)
    env_cfg, train_cfg = task_registry.get_cfgs("go2")
    seed = args.seed if args.seed is not None else train_cfg.seed
    if seed < 0:
        raise ValueError("Use an explicit nonnegative training seed for recorded experiments.")
    train_cfg.seed = env_cfg.seed = seed
    env_cfg.terrain.mesh_type = qa.qa_terrain
    profile = "full" if qa.qa_domain_rand else "mass_only" if qa.qa_mass_rand else "nominal"
    configure_training_randomization(env_cfg, profile)
    env, env_cfg = task_registry.make_env("go2", args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    save_json(Path(runner.log_dir) / "qa_training.json", {
        "seed": seed, "terrain": qa.qa_terrain, "domain_randomization": profile != "nominal",
        "randomization_profile": profile,
        "upstream": git_identity(LEGGED_GYM_ROOT_DIR),
        "test_bench": git_identity(Path(__file__).resolve().parents[1]),
        "dependencies": dependency_versions(),
        "env_cfg": class_to_dict(env_cfg), "train_cfg": class_to_dict(train_cfg),
    })
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
