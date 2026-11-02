"""Explicit optimizer configuration for controlled training comparisons."""

import math


def validate_optimizer_overrides(learning_rate=None, schedule=None, resume=False):
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("--qa-learning-rate must be positive and finite.")
    if schedule is not None and schedule not in ("fixed", "adaptive"):
        raise ValueError("Unknown learning-rate schedule: " + str(schedule))
    if resume and (learning_rate is not None or schedule is not None):
        raise ValueError("Optimizer overrides cannot be combined with --resume; use a fresh experiment.")


def configure_optimizer_overrides(cfg, learning_rate=None, schedule=None, resume=False):
    validate_optimizer_overrides(learning_rate, schedule, resume)
    if learning_rate is not None:
        cfg.algorithm.learning_rate = learning_rate
    if schedule is not None:
        cfg.algorithm.schedule = schedule
