"""Synthetic test fixtures below verify reporting, not robot performance."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReportingTests(unittest.TestCase):
    def test_actual_output_protocols_are_kept_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, goal in enumerate((10, 20)):
                run = root / "inputs" / str(i)
                run.mkdir(parents=True)
                meta = {"checkpoint_sha256": "a" * 64, "terrain": "plane", "case": "nominal", "seed": 0, "goal_progress_m": goal, "duration_limit_s": 20, "command_vx_mps": 1}
                summary = {"trials": 2, "success_rate": 0.5, "mean_max_progress_m": 5, "mean_velocity_vector_rmse_mps": 0.1}
                (run / "metadata.json").write_text(json.dumps(meta))
                (run / "summary.json").write_text(json.dumps(summary))
            subprocess.run([sys.executable, "-m", "robot_test.report", str(root / "inputs"), "--output", str(root / "out")], check=True, capture_output=True)
            data = json.loads((root / "out" / "report_data.json").read_text())
            self.assertEqual(len(data["groups"]), 2)

    def test_no_results_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "robot_test.report", directory, "--output", str(Path(directory) / "out")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
