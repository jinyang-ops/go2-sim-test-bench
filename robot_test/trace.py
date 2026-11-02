"""Optional pre-reset motion traces and CPU-only summaries; no automatic diagnosis."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .common import save_json


TRACE_FIELDS = ("env_id", "step", "time_s", "x_m", "y_m", "z_m", "progress_m",
                "lateral_displacement_m", "world_vx_mps", "world_vy_mps",
                "body_vx_mps", "body_vy_mps", "command_vx_mps", "command_vy_mps",
                "command_yaw_radps", "yaw_deg", "mean_abs_joint_torque_nm")


def make_trace_row(env_id, step, dt, origin_xy, state, velocity, command, torques):
    values = list(state) + list(velocity) + list(command) + list(torques)
    if len(state) < 13 or len(velocity) < 2 or len(command) < 3 or not torques:
        return None
    if not all(math.isfinite(float(v)) for v in values):
        return None
    # Isaac Gym root-state quaternion is x,y,z,w, followed by WORLD velocity.
    x, y, z, w = state[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return dict(zip(TRACE_FIELDS, (
        env_id, step, step * dt, *state[:3], state[0] - origin_xy[0],
        state[1] - origin_xy[1], state[7], state[8], velocity[0], velocity[1],
        command[0], command[1], command[2], math.degrees(yaw),
        sum(abs(t) for t in torques) / len(torques),
    )))


def summarize_environment(samples, episode):
    if not samples:
        return {"env_id": int(episode["env_id"]), "outcome": episode["outcome"], "samples": 0}
    def mean(key):
        return sum(float(r[key]) for r in samples) / len(samples)
    return {
        "env_id": int(episode["env_id"]), "outcome": episode["outcome"],
        "samples": len(samples), "duration_s": float(episode["duration_s"]),
        "progress_m": float(episode["progress_m"]),
        "mean_body_vx_mps": mean("body_vx_mps"),
        "mean_world_vx_mps": mean("world_vx_mps"),
        "mean_command_vx_mps": mean("command_vx_mps"),
        "max_abs_yaw_deg": max(abs(float(r["yaw_deg"])) for r in samples),
        "max_abs_lateral_displacement_m": max(abs(float(r["lateral_displacement_m"])) for r in samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("Input does not exist.")
    paths = sorted(args.input.rglob("motion_trace.csv"))
    if not paths:
        parser.error("No motion_trace.csv found; evaluate with --qa-trace or matrix with --trace.")
    rows = []
    for path in paths:
        meta = json.loads(path.with_name("metadata.json").read_text())
        by_env = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as stream:
            for sample in csv.DictReader(stream):
                by_env[int(sample["env_id"])].append(sample)
        with path.with_name("episodes.csv").open(newline="", encoding="utf-8") as stream:
            for episode in csv.DictReader(stream):
                rows.append({"seed": meta["seed"], "case": meta["case"],
                             "checkpoint_sha256": meta["checkpoint_sha256"],
                             **summarize_environment(by_env[int(episode["env_id"])], episode)})
    lines = ["# 运动轨迹诊断", "", "数值来自新执行的轨迹记录；采样在自动重置前，只包含各环境第一个计分回合。机体坐标速度与世界坐标 X 前进距离不是同一指标。", "", "|seed|场景|环境|结果|时长s|前进m|均值机体vx|均值世界vx|最大绝对偏航°|最大横向位移m|", "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if not row["samples"]:
            lines.append("|{}|{}|{}|{}|NA|NA|NA|NA|NA|NA|".format(row["seed"], row["case"], row["env_id"], row["outcome"]))
            continue
        lines.append("|{seed}|{case}|{env_id}|{outcome}|{duration_s:.3f}|{progress_m:.3f}|{mean_body_vx_mps:.3f}|{mean_world_vx_mps:.3f}|{max_abs_yaw_deg:.1f}|{max_abs_lateral_displacement_m:.3f}|".format(**row))
    lines += ["", "偏航为相对世界 X 正方向的 yaw，由状态中的 xyzw 四元数计算。机体向前速度较高而世界 X 速度低、偏航/横向位移大时，朝向偏离是候选解释；两种前进速度都低时再检查速度跟踪。单靠这些统计不能认定脚打滑或控制器根因。", "", "追踪记录不修改命令、地形、奖励、失败或达标判定；GPU 重跑仍不保证位级一致。"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trace_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(args.output / "trace_report.json", rows)
    print(args.output / "trace_report.md")


if __name__ == "__main__":
    main()
