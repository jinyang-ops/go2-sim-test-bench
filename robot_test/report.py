"""Aggregate actual evaluator outputs; never fills missing runs with zeroes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    seen = set()
    missing = []
    for root in args.inputs:
        if not root.exists():
            parser.error("Input not found: " + str(root))
        for status_file in root.rglob("matrix_status.json"):
            for run in json.loads(status_file.read_text()):
                if run["returncode"] != 0:
                    missing.append(run)
        for meta_file in sorted(root.rglob("metadata.json")):
            if meta_file.resolve() in seen:
                continue
            seen.add(meta_file.resolve())
            meta = json.loads(meta_file.read_text())
            summary = json.loads(meta_file.with_name("summary.json").read_text())
            # Keep different protocols separate; do not average incompatible tests.
            key = (meta["checkpoint_sha256"], meta["terrain"], meta["case"], meta["goal_progress_m"], meta["duration_limit_s"], meta["command_vx_mps"])
            groups[key].append({"seed": meta["seed"], **summary})
    if not groups:
        parser.error("No actual metadata.json + summary.json pairs found.")
    lines = ["# GO2 仿真测试报告", "", "数据来自本地实际评估输出。成功=限定时间内首次达到指定前进距离且该帧无失败；不等同于完成全赛道。", "", "|模型 SHA256 前12位|地形|场景|目标m/时限s/速度m·s⁻¹|回合数|成功率|均值最大前进m|均值速度误差m/s|", "|---|---|---|---|---:|---:|---:|---:|"]
    serializable = []
    for key, runs in sorted(groups.items()):
        count = sum(r["trials"] for r in runs)
        rate = sum(r["success_rate"] * r["trials"] for r in runs) / count
        progress = sum(r["mean_max_progress_m"] * r["trials"] for r in runs) / count
        valid_rmse = [r for r in runs if r["mean_velocity_vector_rmse_mps"] is not None]
        error = sum(r["mean_velocity_vector_rmse_mps"] * r["trials"] for r in valid_rmse) / sum(r["trials"] for r in valid_rmse) if valid_rmse else None
        error_text = "NA" if error is None else "{:.3f}".format(error)
        lines.append("|{}|{}|{}|{}/{}/{}|{}|{:.1%}|{:.3f}|{}|".format(key[0][:12], key[1], key[2], key[3], key[4], key[5], count, rate, progress, error_text))
        serializable.append({"protocol": list(key), "runs": runs})
    lines += ["", "每个模型应使用相同协议和评估种子。批内回合共享部分扰动；不能把回合数当成独立训练次数。另查看每个 summary.json 的失败原因及每个 seed 的结果。", "", "运行异常或门槛未通过的记录：{}。缺失结果不计作通过。".format(len(missing)), ""]
    for run in missing:
        lines.append("- {}: returncode={}".format(run["path"], run["returncode"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(args.output / "report_data.json", {"groups": serializable, "failed_runs": missing})
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
