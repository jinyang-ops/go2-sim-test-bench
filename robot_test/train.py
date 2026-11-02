"""Training wrapper that synchronizes CLI seed before environment creation."""

import argparse
from dataclasses import asdict
from pathlib import Path

from .stages import STAGES, CourseStageMixin, resolve_stage
from .optimization import configure_optimizer_overrides, validate_optimizer_overrides


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-terrain", choices=("plane", "competition"), default="plane")
    parser.add_argument("--qa-init-model", type=Path, help="Initialize network weights with fresh optimizer/iteration state")
    parser.add_argument("--qa-course-stage", choices=tuple(STAGES), help="Restrict competition training to a short stage")
    parser.add_argument("--qa-learning-rate", type=float, help="Initial PPO learning rate; use a fixed schedule for a constant rate")
    parser.add_argument("--qa-lr-schedule", choices=("fixed", "adaptive"))
    randomization = parser.add_mutually_exclusive_group()
    randomization.add_argument("--qa-domain-rand", action="store_true")
    randomization.add_argument("--qa-mass-rand", action="store_true")
    qa, remaining = parser.parse_known_args()
    try:
        validate_optimizer_overrides(qa.qa_learning_rate, qa.qa_lr_schedule, resume="--resume" in remaining)
    except ValueError as exc:
        parser.error(str(exc))
    if qa.qa_course_stage is not None and qa.qa_terrain != "competition":
        parser.error("--qa-course-stage requires --qa-terrain competition")
    if qa.qa_init_model is not None and not qa.qa_init_model.is_file():
        parser.error("Initialization checkpoint does not exist: " + str(qa.qa_init_model))

    import isaacgym  # Must precede torch and legged_gym.envs.
    import torch
    import legged_gym.envs
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from .common import configure_training_randomization, dependency_versions, git_identity, save_json, upstream_arguments
    from .checkpoints import initialize_network_weights, save_pre_update_checkpoint, validate_initialization

    args = upstream_arguments(remaining)
    env_cfg, train_cfg = task_registry.get_cfgs("go2")
    validate_initialization(qa.qa_init_model, resume=args.resume or train_cfg.runner.resume)
    configure_optimizer_overrides(train_cfg, qa.qa_learning_rate, qa.qa_lr_schedule,
                                  resume=args.resume or train_cfg.runner.resume)
    seed = args.seed if args.seed is not None else train_cfg.seed
    if seed < 0:
        raise ValueError("Use an explicit nonnegative training seed for recorded experiments.")
    train_cfg.seed = env_cfg.seed = seed
    env_cfg.terrain.mesh_type = qa.qa_terrain
    profile = "full" if qa.qa_domain_rand else "mass_only" if qa.qa_mass_rand else "nominal"
    configure_training_randomization(env_cfg, profile)
    stage = resolve_stage(qa.qa_course_stage, env_cfg) if qa.qa_course_stage else None
    env_name = "go2"
    if stage is not None:
        class StagedGo2(CourseStageMixin, task_registry.get_task_class("go2")):
            stage_spec = stage
        env_name = "go2_course_" + stage.name
        task_registry.register(env_name, StagedGo2, env_cfg, train_cfg)
        print("Course stage:", asdict(stage), flush=True)
    env, env_cfg = task_registry.make_env(env_name, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    initialization = None
    pre_update_checkpoint = None
    if qa.qa_init_model is not None:
        initialization = initialize_network_weights(runner.alg.actor_critic, qa.qa_init_model, torch.load, args.rl_device)
        print("Initialized network weights from:", qa.qa_init_model, flush=True)
        print("Optimizer and training iteration state are fresh.", flush=True)
        pre_update_checkpoint = save_pre_update_checkpoint(runner)
        print("Saved pre-update checkpoint:", Path(runner.log_dir) / pre_update_checkpoint["file"], flush=True)
    save_json(Path(runner.log_dir) / "qa_training.json", {
        "seed": seed, "terrain": qa.qa_terrain, "domain_randomization": profile != "nominal",
        "randomization_profile": profile,
        "initialization": initialization,
        "pre_update_checkpoint": pre_update_checkpoint,
        "optimizer_at_start": {"algorithm_learning_rate": runner.alg.learning_rate,
                               "schedule": runner.alg.schedule,
                               "param_group_learning_rates": [group["lr"] for group in runner.alg.optimizer.param_groups]},
        "course_stage": asdict(stage) if stage is not None else None,
        "upstream": git_identity(LEGGED_GYM_ROOT_DIR),
        "test_bench": git_identity(Path(__file__).resolve().parents[1]),
        "dependencies": dependency_versions(),
        "env_cfg": class_to_dict(env_cfg), "train_cfg": class_to_dict(train_cfg),
    })
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
