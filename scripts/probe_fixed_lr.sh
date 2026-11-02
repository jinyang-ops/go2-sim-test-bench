#!/usr/bin/env bash
# One controlled PPO iteration, then identical pre-/post-update evaluations.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/probe_fixed_lr.sh SOURCE_MODEL PASSING_SOURCE_EVAL_DIR [NEW_OUTPUT_DIR]" >&2
  exit 2
fi

source_model="$1"
baseline_dir="$2"
probe_name="fixed1e5_seed42_$(date +%Y%m%d_%H%M%S)_$$"
probe_output="${3:-reports/$probe_name}"

# This check uses only existing files and runs before starting any simulation.
python - "$source_model" "$baseline_dir" <<'PY'
import json
import sys
from pathlib import Path
from robot_test.inspect_training import validate_probe_baseline
print("Source baseline:", json.dumps(validate_probe_baseline(Path(sys.argv[1]), Path(sys.argv[2]))))
PY

mkdir -p "$(dirname "$probe_output")"
mkdir "$probe_output"  # Refuse to overwrite an existing probe directory.
cp "$baseline_dir/summary.json" "$probe_output/source_summary.json"
cp "$baseline_dir/metadata.json" "$probe_output/source_metadata.json"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MAX_JOBS="${MAX_JOBS:-2}"

python -u -m robot_test.train \
  --task=go2 --qa-terrain competition --qa-course-stage flat \
  --qa-init-model "$source_model" \
  --qa-learning-rate 1e-5 --qa-lr-schedule fixed \
  --num_envs=128 --headless --max_iterations=1 --seed=42 \
  --experiment_name=qa_course_flat_lr_probe --run_name="$probe_name" \
  2>&1 | tee "$probe_output/train.log"

# Read the actual runner location printed by the training entry point.
run_dir="$(python - "$probe_output/train.log" <<'PY'
import sys
from pathlib import Path
prefix = "Saved pre-update checkpoint: "
matches = [line[len(prefix):] for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith(prefix)]
if len(matches) != 1:
    raise SystemExit("Expected exactly one pre-update checkpoint path in the training log.")
path = Path(matches[0])
if not path.is_file():
    raise SystemExit("Recorded pre-update checkpoint is missing.")
print(path.parent)
PY
)"

for checkpoint in qa_initial model_1; do
  python -m robot_test.evaluate \
    --task=go2 --headless --num_envs=1 --seed=0 \
    --qa-model "$run_dir/$checkpoint.pt" \
    --qa-output "$probe_output/$checkpoint" \
    --qa-terrain competition --qa-case nominal --qa-heading-mode heading \
    --qa-speed 1.0 --qa-duration 5 --qa-goal-m 1 --qa-trace \
    2>&1 | tee "$probe_output/${checkpoint}_evaluate.log"
done

echo "Probe results: $probe_output"
echo "Normal script completion means the probe ran; inspect both summaries for performance."
