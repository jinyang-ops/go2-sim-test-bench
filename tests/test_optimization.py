import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from robot_test.checkpoints import save_pre_update_checkpoint
from robot_test.optimization import configure_optimizer_overrides


class OptimizationTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(algorithm=SimpleNamespace(learning_rate=.001, schedule="adaptive", entropy_coef=.01),
                               policy=SimpleNamespace(init_noise_std=1.))

    def test_fixed_low_rate_preserves_exploration_and_other_algorithm_fields(self):
        cfg = self.config()
        configure_optimizer_overrides(cfg, 1e-5, "fixed")
        self.assertEqual(vars(cfg.algorithm), {"learning_rate": 1e-5, "schedule": "fixed", "entropy_coef": .01})
        self.assertEqual(cfg.policy.init_noise_std, 1.)

    def test_no_override_preserves_existing_configuration(self):
        cfg = self.config()
        before = copy.deepcopy(cfg)
        configure_optimizer_overrides(cfg)
        self.assertEqual(cfg, before)

    def test_invalid_or_resumed_override_is_rejected_without_mutating_configuration(self):
        for rate, schedule, resume in [(0, "fixed", False), (-1, None, False),
                                      (float("nan"), None, False), (float("inf"), None, False),
                                      (1e-5, "other", False), (1e-5, "fixed", True)]:
            cfg = self.config()
            before = copy.deepcopy(cfg)
            with self.assertRaises(ValueError):
                configure_optimizer_overrides(cfg, rate, schedule, resume)
            self.assertEqual(cfg, before)

    def test_cli_rejects_invalid_rate_or_resume_before_simulator_import(self):
        for args in [("--qa-learning-rate", "nan"), ("--qa-learning-rate", "0"),
                     ("--qa-lr-schedule", "fixed", "--resume")]:
            result = subprocess.run([sys.executable, "-m", "robot_test.train", *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_initial_snapshot_is_saved_with_explicit_pre_update_role(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Mock(log_dir=str(Path(root) / "run"), current_learning_iteration=0)
            runner.save.side_effect = lambda path, infos: Path(path).write_bytes(b"synthetic pre-update checkpoint")
            meta = save_pre_update_checkpoint(runner)
            runner.save.assert_called_once_with(str(Path(runner.log_dir) / "qa_initial.pt"),
                                               infos={"qa_checkpoint_role": "pre_update"})
            self.assertEqual(meta["completed_updates"], 0)
            self.assertNotIn("model", meta["file"])
            self.assertEqual(len(meta["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                save_pre_update_checkpoint(runner)
            self.assertEqual(runner.save.call_count, 1)

    def test_initial_snapshot_rejects_an_already_advanced_runner(self):
        runner = Mock(current_learning_iteration=1)
        with self.assertRaises(ValueError):
            save_pre_update_checkpoint(runner)
        runner.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
