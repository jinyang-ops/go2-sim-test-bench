import unittest
import json
import tempfile
from pathlib import Path

from robot_test.common import sha256
from robot_test.inspect_training import checkpoint_info, compare_weights, training_info, validate_probe_baseline, vector_delta


class TensorFixture:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *args):
        return self

    def tolist(self):
        return list(self.values)


class InspectionTests(unittest.TestCase):
    def make_baseline(self, root):
        source = root / "source.pt"
        source.write_bytes(b"synthetic checkpoint")
        metadata = {"seed": 0, "case": "nominal", "terrain": "competition", "heading_mode": "heading",
                    "evaluation_protocol": "world_x_heading_v2", "duration_limit_s": 5.,
                    "goal_progress_m": 1., "command_vx_mps": 1., "observation_shape": [1, 235],
                    "checkpoint_sha256": sha256(source)}
        (root / "metadata.json").write_text(json.dumps(metadata))
        (root / "summary.json").write_text(json.dumps({"trials": 1, "success_rate": 1.0}))
        return source, metadata

    def test_probe_requires_a_passing_source_under_the_exact_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.make_baseline(root)
            self.assertEqual(validate_probe_baseline(source, root)["success_rate"], 1.)
            (root / "summary.json").write_text(json.dumps({"trials": 1, "success_rate": 0.}))
            with self.assertRaises(ValueError):
                validate_probe_baseline(source, root)

    def test_probe_rejects_a_different_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self.make_baseline(root)
            source.write_bytes(b"different checkpoint")
            with self.assertRaises(ValueError):
                validate_probe_baseline(source, root)

    def test_probe_rejects_a_changed_protocol_even_when_it_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata = self.make_baseline(root)
            for key, value in [("heading_mode", "yaw_rate"), ("duration_limit_s", 20.), ("seed", 1)]:
                altered = dict(metadata, **{key: value})
                (root / "metadata.json").write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    validate_probe_baseline(source, root)

    def test_known_parameter_change(self):
        result = vector_delta([3., 4.], [6., 8.])
        self.assertEqual(result["source_squared_l2"], 25)
        self.assertEqual(result["delta_squared_l2"], 25)
        self.assertEqual(result["changed_elements"], 2)

    def test_incomplete_or_nonfinite_vector_comparison_is_rejected(self):
        for left, right in [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])]:
            with self.assertRaises(ValueError):
                vector_delta(left, right)

    def test_actor_critic_and_noise_are_reported_separately(self):
        source = {"actor.weight": TensorFixture([3, 4]), "critic.bias": TensorFixture([0]), "std": TensorFixture([1])}
        candidate = {"actor.weight": TensorFixture([6, 8]), "critic.bias": TensorFixture([2]), "std": TensorFixture([1])}
        result = compare_weights(source, candidate)
        self.assertEqual(result["actor"]["relative_delta_l2"], 1)
        self.assertEqual(result["critic"]["delta_l2"], 2)
        self.assertIsNone(result["critic"]["relative_delta_l2"])
        self.assertEqual(result["std"]["changed_elements"], 0)

    def test_changed_architecture_is_not_silently_partially_compared(self):
        source = {"actor.weight": TensorFixture([1])}
        with self.assertRaises(ValueError):
            compare_weights(source, {})
        with self.assertRaises(ValueError):
            compare_weights(source, {"actor.weight": TensorFixture([1, 2])})

    def test_metadata_does_not_export_initialization_path(self):
        result = training_info({"initialization": {"path": "/local/private/model.pt", "sha256": "abc"}})
        self.assertNotIn("path", result["initialization"])
        self.assertEqual(result["initialization"]["sha256"], "abc")

    def test_invalid_action_std_is_rejected(self):
        for value in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                checkpoint_info({"model_state_dict": {"std": TensorFixture([value])}})

    def test_optimizer_lr_and_stored_iter_are_not_confused_with_update_count(self):
        info = checkpoint_info({"model_state_dict": {"std": TensorFixture([1, 2])}, "iter": 0,
                                "optimizer_state_dict": {"param_groups": [{"lr": .002}]}})
        self.assertEqual(info["stored_iter"], 0)
        self.assertEqual(info["saved_optimizer_lr"], [.002])
        self.assertEqual(info["mean_action_std"], 1.5)


if __name__ == "__main__":
    unittest.main()
