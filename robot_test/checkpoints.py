"""Network-only initialization for transfer training; optimizer state stays fresh."""

from pathlib import Path

from .common import sha256


def validate_initialization(path, resume=False):
    if path is None:
        return
    if resume:
        raise ValueError("--qa-init-model cannot be combined with checkpoint resume.")
    if not Path(path).is_file():
        raise FileNotFoundError("Initialization checkpoint not found: " + str(path))


def initialize_network_weights(network, path, load_checkpoint, device):
    validate_initialization(path)
    checkpoint = load_checkpoint(str(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected an rsl_rl checkpoint containing model_state_dict.")
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "source_iteration": checkpoint.get("iter"), "optimizer_restored": False,
            "training_iteration_restored": False}


def save_pre_update_checkpoint(runner):
    """Snapshot a fresh runner after initialization and before any PPO update."""
    if runner.current_learning_iteration != 0:
        raise ValueError("Pre-update checkpoint requires a fresh training iteration counter.")
    # Native get_load_path scans every name containing 'model' for --checkpoint=-1.
    path = Path(runner.log_dir) / "qa_initial.pt"
    if path.exists():
        raise FileExistsError("Pre-update checkpoint already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(str(path), infos={"qa_checkpoint_role": "pre_update"})
    return {"file": path.name, "sha256": sha256(path), "role": "pre_update",
            "completed_updates": 0}
