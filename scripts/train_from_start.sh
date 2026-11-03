#!/usr/bin/env bash
set -euo pipefail

bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="${1:-$HOME/legged_robot_competition}"
patch_file="$bench_dir/patches/go2-course-start.patch"

: "${CONDA_PREFIX:?Activate the go2-test conda environment first.}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-2}"

if git -C "$upstream_dir" apply --check "$patch_file" 2>/dev/null; then
    git -C "$upstream_dir" apply "$patch_file"
    printf 'Applied the course-start patch.\n'
elif git -C "$upstream_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    printf 'The course-start patch is already applied.\n'
else
    printf 'Patch does not match the local source. No training started.\n' >&2
    exit 1
fi

run_tag="from_start_$(date +%Y%m%d_%H%M%S)_$$"
log_dir="$bench_dir/reports/$run_tag"
mkdir -p "$log_dir"
git -C "$upstream_dir" rev-parse HEAD > "$log_dir/upstream_commit.txt"
git -C "$upstream_dir" diff -- legged_gym/legged_gym/envs/go2/go2_robot.py legged_gym/legged_gym/envs/go2/go2_config.py > "$log_dir/source_changes.patch"

python - "$upstream_dir" <<'PY' 2>&1 | tee "$log_dir/start_check.log"
from pathlib import Path
import sys

upstream = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], '--task=go2', '--num_envs=4', '--headless', '--seed=1']

import isaacgym
from isaacgym import gymtorch
import torch
import legged_gym
import legged_gym.envs
import rsl_rl
from legged_gym.utils import get_args, task_registry

for module, relative in (
    (isaacgym, 'isaacgym/python/isaacgym/__init__.py'),
    (legged_gym, 'legged_gym/legged_gym/__init__.py'),
    (rsl_rl, 'rsl_rl/rsl_rl/__init__.py'),
):
    actual = Path(module.__file__).resolve()
    assert actual == upstream / relative, 'Unexpected import path: ' + str(actual)

assert torch.cuda.is_available(), 'CUDA is unavailable; training was not started.'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
args = get_args()
env, cfg = task_registry.make_env(name='go2', args=args)
try:
    assert cfg.terrain.mesh_type == 'competition'
    expected = torch.tensor([2.0, cfg.terrain.terrain_width / 2.0, 0.42], device=env.device)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    torch.testing.assert_close(env.root_states[:, :3], expected.expand_as(env.root_states[:, :3]), rtol=0, atol=1e-5)
    ids = torch.arange(env.num_envs, device=env.device)
    for _ in range(2):
        env.reset_idx(ids)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        torch.testing.assert_close(env.root_states[:, :3], expected.expand_as(env.root_states[:, :3]), rtol=0, atol=1e-5)
    print('Start check passed: creation and resets use xyz =', expected.cpu().tolist(), flush=True)
    print('Episode time limit (s):', cfg.env.episode_length_s, flush=True)
finally:
    env.gym.destroy_sim(env.sim)
PY

printf '\nStarting 6000 iterations from scratch. Logs: %s\n' "$log_dir"
cd "$upstream_dir/legged_gym/legged_gym/scripts"
python -u train.py \
    --task=go2 \
    --headless \
    --num_envs=256 \
    --seed=1 \
    --max_iterations=6000 \
    --experiment_name=course_start_6000 \
    --run_name="$run_tag" \
    2>&1 | tee "$log_dir/train.log"

printf '\nTraining completed. Model location:\n'
find "$upstream_dir/legged_gym/logs/course_start_6000" \
    -type f -path "*_${run_tag}/model_6000.pt"
