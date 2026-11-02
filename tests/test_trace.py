"""Synthetic motion examples validate frame interpretation, not robot performance."""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robot_test.trace import TRACE_FIELDS, make_trace_row, summarize_environment


class TraceTests(unittest.TestCase):
    def test_body_forward_can_be_world_lateral_after_turn(self):
        half = math.sqrt(0.5)
        state = [2, 4, 0.4, 0, 0, half, half, 0, 1, 0, 0, 0, 0]
        row = make_trace_row(0, 10, 0.02, (2, 3), state, [1, 0, 0], [1, 0, 0], [2, -2])
        self.assertAlmostEqual(row["yaw_deg"], 90)
        self.assertEqual(row["body_vx_mps"], 1)
        self.assertEqual(row["world_vx_mps"], 0)
        self.assertEqual(row["progress_m"], 0)
        self.assertEqual(row["lateral_displacement_m"], 1)
        self.assertAlmostEqual(row["time_s"], 0.2)
        self.assertEqual(set(row), set(TRACE_FIELDS))

    def test_invalid_state_is_not_emitted_as_valid_trajectory(self):
        state = [0] * 13
        state[0] = float("nan")
        self.assertIsNone(make_trace_row(0, 1, .02, (0, 0), state, [0, 0, 0], [1, 0, 0], [1]))

    def test_trace_summary_preserves_episode_outcome_and_frame_difference(self):
        state = [2, 4, .4, 0, 0, 0, 1, .2, 0, 0, 0, 0, 0]
        row = make_trace_row(0, 1, .02, (0, 0), state, [.9, 0, 0], [1, 0, 0], [1])
        episode = {"env_id": "0", "outcome": "time_limit", "duration_s": "20.02", "progress_m": "2"}
        summary = summarize_environment([row], episode)
        self.assertEqual(summary["outcome"], "time_limit")
        self.assertEqual(summary["mean_body_vx_mps"], .9)
        self.assertEqual(summary["mean_world_vx_mps"], .2)
        self.assertEqual(summary["duration_s"], 20.02)
        self.assertEqual(summary["max_abs_lateral_displacement_m"], 4)

    def test_trace_cli_joins_real_file_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "input" / "robot_friction_seed_0"
            run.mkdir(parents=True)
            state = [0, 0, .4, 0, 0, 0, 1, .5, 0, 0, 0, 0, 0]
            row = make_trace_row(0, 1, .02, (0, 0), state, [.5, 0, 0], [1, 0, 0], [1])
            with (run / "motion_trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            (run / "metadata.json").write_text(json.dumps({"seed": 0, "case": "robot_friction", "checkpoint_sha256": "a" * 64}))
            (run / "episodes.csv").write_text("env_id,outcome,duration_s,progress_m\n0,time_limit,20.02,2\n")
            subprocess.run([sys.executable, "-m", "robot_test.trace", str(root / "input"), "--output", str(root / "output")], check=True, capture_output=True)
            result = json.loads((root / "output" / "trace_report.json").read_text())
            self.assertEqual(result[0]["outcome"], "time_limit")
            self.assertEqual(result[0]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
