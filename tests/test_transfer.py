import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from robot_test.checkpoints import initialize_network_weights, validate_initialization
from robot_test.common import configure_motion_commands
from robot_test.course import segment_at
from robot_test.metrics import Trial


class TransferTests(unittest.TestCase):
    def test_only_network_state_is_restored_with_strict_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"synthetic checkpoint")
            network = Mock()
            weights = {"actor.weight": "synthetic"}
            loader = Mock(return_value={"model_state_dict": weights, "optimizer_state_dict": {"old": True}, "iter": 1000})
            meta = initialize_network_weights(network, path, loader, "cpu")
            loader.assert_called_once_with(str(path), map_location="cpu", weights_only=True)
            network.load_state_dict.assert_called_once_with(weights, strict=True)
            self.assertEqual(meta["source_iteration"], 1000)
            self.assertFalse(meta["optimizer_restored"])
            self.assertFalse(meta["training_iteration_restored"])
            self.assertEqual(len(meta["sha256"]), 64)

    def test_resume_conflicts_with_initialization(self):
        with self.assertRaises(ValueError):
            validate_initialization("model.pt", resume=True)
        validate_initialization(None, resume=True)

    def test_missing_initialization_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_initialization(Path(directory) / "missing.pt")

    def test_missing_network_state_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"fixture")
            network = Mock()
            with self.assertRaises(ValueError):
                initialize_network_weights(network, path, lambda *a, **k: {"iter": 1000}, "cpu")
            network.load_state_dict.assert_not_called()

    def test_heading_protocol_configures_heading_feedback(self):
        cfg = SimpleNamespace(commands=SimpleNamespace(ranges=SimpleNamespace()))
        protocol = configure_motion_commands(cfg, 1.0, "heading")
        self.assertTrue(cfg.commands.heading_command)
        self.assertEqual(cfg.commands.ranges.heading, [0, 0])
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [1, 1])
        self.assertEqual(protocol, "world_x_heading_v2")

    def test_yaw_rate_protocol_preserves_legacy_command(self):
        cfg = SimpleNamespace(commands=SimpleNamespace(ranges=SimpleNamespace()))
        protocol = configure_motion_commands(cfg, 1.0, "yaw_rate")
        self.assertFalse(cfg.commands.heading_command)
        self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [0, 0])
        self.assertEqual(protocol, "world_x_yaw_rate_v1")

    def test_reference_map_boundaries(self):
        self.assertEqual(segment_at(0), "flat")
        self.assertEqual(segment_at(12), "pyramid_slope")
        self.assertEqual(segment_at(84), "stepping_stones")
        self.assertEqual(segment_at(96), "tail_flat")
        self.assertEqual(segment_at(108), "outside_map")
        self.assertEqual(segment_at(-1), "outside_map")
        self.assertEqual(segment_at(float("nan")), "unknown")

    def test_terminal_world_position_survives_reset_sample(self):
        trial = Trial(0, 2, 94, .02)
        trial.update([64, 6, 1], [1, 0, 0], [1, 0, 0], [1], True, False, True)
        trial.update([2, 6, .4], [0, 0, 0], [1, 0, 0], [0], False, False, False)
        row = trial.finish()
        self.assertEqual(row["end_x_m"], 64)
        self.assertEqual(segment_at(row["end_x_m"]), "stairs_up")


if __name__ == "__main__":
    unittest.main()
