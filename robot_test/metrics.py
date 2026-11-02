"""Simulator-independent first-episode metrics, suitable for CPU-only CI."""

import math
from collections import Counter


class Trial:
    def __init__(self, env_id, start_x, goal_m, dt, lane=None):
        self.env_id = env_id
        self.start_x = start_x
        self.goal_m = goal_m
        self.dt = dt
        self.lane = lane
        self.steps = 0
        self.progress = 0.0
        self.max_progress = 0.0
        self.squared_error = 0.0
        self.abs_torque = 0.0
        self.reason = None

    def update(self, state, velocity, command, torques, contact, timeout, done):
        if self.reason is not None:
            return
        values = list(state) + list(velocity) + list(command) + list(torques)
        if not values or not all(math.isfinite(float(v)) for v in values):
            self.reason = "invalid_state"
            return
        self.steps += 1
        self.progress = float(state[0]) - self.start_x
        self.max_progress = max(self.max_progress, self.progress)
        # RMS of planar velocity-vector error; units m/s.
        self.squared_error += sum((float(velocity[k]) - float(command[k])) ** 2 for k in (0, 1))
        self.abs_torque += sum(abs(float(x)) for x in torques) / len(torques)
        # Physical failure wins over a goal or timeout on the same sample.
        if contact or state[2] < -3 or (done and not timeout):
            self.reason = "fall_or_base_contact"
        elif self.lane is not None and not (0 < state[1] < self.lane):
            self.reason = "off_course"
        elif self.progress >= self.goal_m:
            self.reason = "goal_reached"
        elif timeout:
            self.reason = "time_limit"

    def finish(self):
        if self.reason is None:
            self.reason = "time_limit"
        return {
            "env_id": self.env_id,
            "outcome": self.reason,
            "success": int(self.reason == "goal_reached"),
            "valid_steps": self.steps,
            "duration_s": self.steps * self.dt,
            "progress_m": self.progress,
            "max_progress_m": self.max_progress,
            "velocity_vector_rmse_mps": math.sqrt(self.squared_error / self.steps) if self.steps else None,
            "mean_abs_joint_torque_nm": self.abs_torque / self.steps if self.steps else None,
        }


def summarize(rows):
    if not rows:
        raise ValueError("No trials: empty results cannot pass.")
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None
    return {
        "trials": len(rows),
        "success_rate": mean("success"),
        "mean_max_progress_m": mean("max_progress_m"),
        "mean_velocity_vector_rmse_mps": mean("velocity_vector_rmse_mps"),
        "mean_abs_joint_torque_nm": mean("mean_abs_joint_torque_nm"),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
    }
