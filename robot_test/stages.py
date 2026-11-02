"""Small competition-map training stages with safe flat-ground starts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CourseStage:
    name: str
    start_x: float
    start_y: float
    goal_x: float


STAGES = {
    "flat": CourseStage("flat", 2.0, 6.0, 10.0),
    "front": CourseStage("front", 8.0, 6.0, 24.0),
}


def resolve_stage(name, cfg):
    if name not in STAGES:
        raise ValueError("Unknown course stage: " + str(name))
    expected = {"mesh_type": "competition", "terrain_length": 12.0,
                "terrain_width": 12.0, "num_rows": 9, "num_cols": 1}
    if any(getattr(cfg.terrain, key, None) != value for key, value in expected.items()):
        raise ValueError("Stages require the reference 12m x 108m competition layout.")
    if list(cfg.init_state.pos[:2]) != [0.0, 0.0]:
        raise ValueError("Stage origins require zero initial XY position offsets.")
    return STAGES[name]


def stage_masks(x, y, native_done, goal_x, width):
    """Operators work with scalar booleans in CPU tests or torch tensors at runtime."""
    outside = (y <= 0) | (y >= width)
    reached = (x >= goal_x) & (native_done == 0) & (outside == 0)
    return outside, reached


class CourseStageMixin:
    # The concrete subclass supplies stage_spec after reference-layout validation.
    def _get_env_origins(self):
        import torch
        stage = self.stage_spec
        self.custom_origins = False
        self.env_origins = torch.tensor([stage.start_x, stage.start_y, 0.0],
                                        device=self.device).repeat(self.num_envs, 1)

    def check_termination(self):
        super().check_termination()
        outside, reached = stage_masks(self.root_states[:, 0], self.root_states[:, 1],
                                       self.reset_buf, self.stage_spec.goal_x,
                                       self.cfg.terrain.terrain_width)
        self._qa_stage_reached = reached.detach().clone()
        self.reset_buf |= outside | reached
        # A completed stage is an artificial episode boundary, not a fall.
        self.time_out_buf |= reached

    def reset_idx(self, env_ids):
        fraction = None
        if len(env_ids) and hasattr(self, "_qa_stage_reached"):
            fraction = self._qa_stage_reached[env_ids].float().mean().item()
        super().reset_idx(env_ids)
        if fraction is not None:
            self.extras.setdefault("episode", {})["stage_goal_fraction"] = fraction
            self._qa_stage_reached[env_ids] = False
