import math
import unittest
from types import SimpleNamespace

from robot_test.common import configure_case
from robot_test.metrics import Trial, summarize


def sample(trial, x, z=0.4, contact=False, timeout=False, done=False, y=6.0, vx=1.0):
    trial.update([x, y, z], [vx, 0, 0], [1, 0, 0], [2, -2], contact, timeout, done)


class MetricsTests(unittest.TestCase):
    def test_pre_reset_terminal_position_is_retained(self):
        trial = Trial(0, start_x=2, goal_m=10, dt=0.02)
        sample(trial, 7, contact=True, done=True)
        # The next upstream state would be at spawn x=2; finished trials ignore it.
        sample(trial, 2)
        row = trial.finish()
        self.assertEqual(row["progress_m"], 5)
        self.assertEqual(row["valid_steps"], 1)
        self.assertEqual(row["outcome"], "fall_or_base_contact")

    def test_contact_has_priority_over_goal_and_timeout(self):
        trial = Trial(0, 0, 10, 0.02)
        sample(trial, 10, contact=True, timeout=True, done=True)
        self.assertEqual(trial.finish()["success"], 0)

    def test_goal_is_a_milestone_not_a_timeout_success(self):
        trial = Trial(0, 2, 10, 0.02)
        sample(trial, 3, timeout=True, done=True)
        self.assertEqual(trial.finish()["outcome"], "time_limit")

    def test_goal_reached(self):
        trial = Trial(0, 2, 10, 0.02)
        sample(trial, 12)
        self.assertEqual(trial.finish()["success"], 1)

    def test_horizon_closes_unfinished_trial_as_failure(self):
        trial = Trial(0, 2, 10, 0.02)
        sample(trial, 5)
        self.assertEqual(trial.finish()["outcome"], "time_limit")

    def test_nonfinite_physics_is_not_scored_as_success(self):
        trial = Trial(0, 2, 10, 0.02)
        sample(trial, float("nan"))
        self.assertEqual(trial.finish()["outcome"], "invalid_state")
        self.assertIsNone(trial.finish()["velocity_vector_rmse_mps"])

    def test_rmse_and_torque_units(self):
        trial = Trial(0, 2, 10, 0.5)
        sample(trial, 3, vx=0)
        sample(trial, 4, vx=1)
        row = trial.finish()
        self.assertAlmostEqual(row["velocity_vector_rmse_mps"], math.sqrt(0.5))
        self.assertEqual(row["mean_abs_joint_torque_nm"], 2)
        self.assertEqual(row["duration_s"], 1)

    def test_out_of_lane_is_failure(self):
        trial = Trial(0, 2, 10, 0.02, lane=12)
        sample(trial, 12, y=13)
        self.assertEqual(trial.finish()["outcome"], "off_course")

    def test_drop_below_map_is_failure(self):
        trial = Trial(0, 2, 10, 0.02)
        sample(trial, 4, z=-4)
        self.assertEqual(trial.finish()["outcome"], "fall_or_base_contact")

    def test_empty_results_do_not_pass(self):
        with self.assertRaises(ValueError):
            summarize([])

    def test_success_rate_counts_time_limit_as_failure(self):
        one, two = Trial(0, 0, 1, 0.02), Trial(1, 0, 1, 0.02)
        sample(one, 1)
        sample(two, 0)
        self.assertEqual(summarize([one.finish(), two.finish()])["success_rate"], 0.5)

    def test_scenarios_set_real_upstream_config_fields(self):
        cfg = SimpleNamespace(terrain=SimpleNamespace(), commands=SimpleNamespace(), noise=SimpleNamespace(), domain_rand=SimpleNamespace())
        configure_case(cfg, "robot_friction")
        self.assertTrue(cfg.domain_rand.randomize_friction)
        self.assertEqual(cfg.domain_rand.friction_range, [0.3, 0.3])
        configure_case(cfg, "added_mass")
        self.assertTrue(cfg.domain_rand.randomize_base_mass)
        self.assertEqual(cfg.domain_rand.added_mass_range, [2, 2])
        configure_case(cfg, "nominal")
        self.assertFalse(cfg.domain_rand.randomize_base_mass)
        self.assertFalse(cfg.domain_rand.push_robots)


if __name__ == "__main__":
    unittest.main()
