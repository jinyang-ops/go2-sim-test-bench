import subprocess
import sys
import unittest
from types import SimpleNamespace

from robot_test.stages import STAGES, resolve_stage, stage_masks


def config():
    return SimpleNamespace(terrain=SimpleNamespace(mesh_type="competition", terrain_length=12.,
                           terrain_width=12., num_rows=9, num_cols=1),
                           init_state=SimpleNamespace(pos=[0., 0., .42]))


class StageTests(unittest.TestCase):
    def test_reference_starts_are_inside_flat_segment(self):
        for name in STAGES:
            stage = resolve_stage(name, config())
            self.assertGreater(stage.start_x, 0)
            self.assertLess(stage.start_x, 12)
            self.assertLess(stage.start_x, stage.goal_x)
            self.assertEqual(stage.start_y, 6)

    def test_stage_cannot_claim_success_on_physical_termination(self):
        outside, reached = stage_masks(24., 6., True, 24., 12.)
        self.assertFalse(outside)
        self.assertFalse(reached)

    def test_out_of_lane_is_not_success_even_at_goal(self):
        outside, reached = stage_masks(24., 13., False, 24., 12.)
        self.assertTrue(outside)
        self.assertFalse(reached)

    def test_stage_end_and_continuing_episode_are_distinct(self):
        self.assertEqual(stage_masks(10., 6., False, 10., 12.), (False, True))
        self.assertEqual(stage_masks(9.9, 6., False, 10., 12.), (False, False))

    def test_changed_map_geometry_requires_an_explicit_new_stage(self):
        cfg = config()
        cfg.terrain.terrain_length = 10
        with self.assertRaises(ValueError):
            resolve_stage("flat", cfg)

    def test_nonzero_initial_position_offset_is_rejected(self):
        cfg = config()
        cfg.init_state.pos[0] = 2
        with self.assertRaises(ValueError):
            resolve_stage("flat", cfg)

    def test_plane_stage_combination_is_rejected_before_gpu_import(self):
        result = subprocess.run([sys.executable, "-m", "robot_test.train", "--qa-course-stage", "flat"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --qa-terrain competition", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
