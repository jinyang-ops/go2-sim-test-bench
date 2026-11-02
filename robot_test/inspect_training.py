"""Read existing checkpoints on CPU; never run simulation or update weights."""

import argparse
import csv
import json
import math
from pathlib import Path

from .common import save_json, sha256


def vector_delta(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Parameter vectors must have the same nonzero length.")
    if any(not math.isfinite(x) for x in reference + candidate):
        raise ValueError("Nonfinite checkpoint parameter.")
    source_sq = math.fsum(x * x for x in reference)
    delta_sq = math.fsum((y - x) ** 2 for x, y in zip(reference, candidate))
    return {"elements": len(reference), "changed_elements": sum(x != y for x, y in zip(reference, candidate)),
            "source_squared_l2": source_sq, "delta_squared_l2": delta_sq}


def compare_weights(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("Checkpoint parameter names differ; comparison would be incomplete.")
    groups = {}
    for name, before in reference.items():
        after = candidate[name]
        if tuple(before.shape) != tuple(after.shape):
            raise ValueError("Checkpoint shape mismatch: " + name)
        stats = vector_delta(before.detach().cpu().reshape(-1).tolist(),
                             after.detach().cpu().reshape(-1).tolist())
        group = name.split(".", 1)[0]
        total = groups.setdefault(group, {key: 0 for key in stats})
        for key, value in stats.items():
            total[key] += value
    for total in groups.values():
        source = math.sqrt(total.pop("source_squared_l2"))
        delta = math.sqrt(total.pop("delta_squared_l2"))
        total.update(delta_l2=delta, source_l2=source,
                     relative_delta_l2=delta / source if source else None)
    return groups


def checkpoint_info(checkpoint):
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict) or "std" not in weights:
        raise ValueError("Expected an rsl_rl ActorCritic checkpoint with a std parameter.")
    std = weights["std"].detach().cpu().reshape(-1).tolist()
    if not std or any(not math.isfinite(x) or x <= 0 for x in std):
        raise ValueError("Action standard deviations must be finite and positive.")
    return {"stored_iter": checkpoint.get("iter"), "action_std": std,
            "mean_action_std": math.fsum(std) / len(std),
            "saved_optimizer_lr": [group.get("lr") for group in
                                   checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_probe_baseline(source, directory):
    """Require a passing source result under the exact short-motion protocol."""
    meta = read_json(directory / "metadata.json")
    summary = read_json(directory / "summary.json")
    expected = {"seed": 0, "case": "nominal", "terrain": "competition", "heading_mode": "heading",
                "evaluation_protocol": "world_x_heading_v2", "duration_limit_s": 5.0,
                "goal_progress_m": 1.0, "command_vx_mps": 1.0, "observation_shape": [1, 235]}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Source baseline protocol differs from the 5s/1m heading probe.")
    if meta.get("checkpoint_sha256") != sha256(source):
        raise ValueError("Source baseline checkpoint SHA256 does not match the supplied source.")
    if summary.get("trials") != 1 or summary.get("success_rate") != 1.0:
        raise ValueError("Source baseline did not pass. Stop the update probe and inspect its results first.")
    return summary


def training_info(data):
    env = data.get("env_cfg", {})
    train = data.get("train_cfg", {})
    init = data.get("initialization") or {}
    return {"seed": data.get("seed"), "terrain": data.get("terrain"),
            "profile": data.get("randomization_profile"), "course_stage": data.get("course_stage"),
            "initialization": {key: init.get(key) for key in
                               ("sha256", "source_iteration", "optimizer_restored", "training_iteration_restored")},
            "env": env.get("env"), "commands": env.get("commands"),
            "control": env.get("control"), "sim_dt": env.get("sim", {}).get("dt"),
            "normalization": env.get("normalization"), "domain_rand": env.get("domain_rand"),
            "noise": env.get("noise"), "algorithm": train.get("algorithm"),
            "runner": train.get("runner"), "upstream": data.get("upstream"),
            "test_bench": data.get("test_bench")}


def evaluation_info(directory):
    meta = read_json(directory / "metadata.json")
    fields = ("env_id", "success", "outcome", "duration_s", "max_progress_m",
              "start_x_m", "end_x_m", "end_y_m", "end_z_m", "terminal_segment")
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: row[key] for key in fields if key in row} for row in csv.DictReader(stream)]
    return {"summary": read_json(directory / "summary.json"), "episodes": rows,
            "protocol": {key: meta.get(key) for key in
                         ("checkpoint_sha256", "seed", "case", "terrain", "heading_mode",
                          "evaluation_protocol", "duration_limit_s", "goal_progress_m",
                          "command_vx_mps", "dt_s", "observation_shape")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.source.is_file() or not (args.run_dir / "qa_training.json").is_file():
        parser.error("Source checkpoint and run-dir/qa_training.json must exist.")
    if args.output.exists():
        parser.error("Output already exists; use a fresh JSON filename.")
    if any(i < 0 for i in args.iterations) or len(set(args.iterations)) != len(args.iterations):
        parser.error("Iterations must be unique and nonnegative.")

    import torch  # CPU checkpoint inspection; Isaac Gym is not imported.

    source_hash = sha256(args.source)
    source = torch.load(str(args.source), map_location="cpu", weights_only=True)
    data = read_json(args.run_dir / "qa_training.json")
    report = {"schema_version": 1, "source_sha256": source_hash,
              "source_checkpoint": checkpoint_info(source), "training": training_info(data),
              "recorded_initialization_matches_source":
                  (data.get("initialization") or {}).get("sha256") == source_hash,
              "checkpoints": [], "missing_checkpoints": []}
    source_record = args.source.parent / "qa_training.json"
    if source_record.is_file():
        report["source_training"] = training_info(read_json(source_record))
    for iteration in args.iterations:
        path = args.run_dir / ("model_" + str(iteration) + ".pt")
        if not path.is_file():
            report["missing_checkpoints"].append(path.name)
            continue
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        entry = {"file": path.name, "sha256": sha256(path), **checkpoint_info(checkpoint)}
        entry["parameter_changes_from_source"] = compare_weights(source["model_state_dict"], checkpoint["model_state_dict"])
        report["checkpoints"].append(entry)
    if not report["checkpoints"]:
        parser.error("None of the requested candidate checkpoints exist.")
    if args.evaluation:
        report["evaluation"] = evaluation_info(args.evaluation)
        evaluated_hash = report["evaluation"]["protocol"]["checkpoint_sha256"]
        report["evaluation_matches_inspected_checkpoint"] = evaluated_hash in {
            entry["sha256"] for entry in report["checkpoints"]}
    report["interpretation"] = (
        "Read-only CPU inspection. model_0 is after the first update, not an untouched initialization. "
        "Intermediate stored_iter may remain zero in this upstream runner; use filenames for requested milestones. "
        "Saved optimizer LR is an endpoint snapshot, not its full trajectory. Parameter change and action std "
        "alone do not prove a failure cause or policy performance. No training or evaluation was executed.")
    save_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
