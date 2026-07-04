"""Orchestrator helpers: integrity hashing, worktree build+run, tree updates.

Keeps node identity as a SHA-256 Merkle chain: node sha = SHA-256(config_hash
+ parent_sha), parent_sha = 'root' for the baseline. config_hash is the
SHA-256 of the canonical config serialization.
"""

import hashlib
import json
import os
import subprocess

import make_worktree

TRAINER_DIR = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis/.auto-trainer"
ROOT = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis"
TREE_PATH = os.path.join(TRAINER_DIR, "experiment-tree.json")
WORKTREES = os.path.join(TRAINER_DIR, "worktrees")


def integrity():
    section = {
        "dataset": {"train": "./train.csv", "test": "./test.csv"},
        "target_column": "Status",
        "competition": {"metric": "log_loss", "metric_direction": "minimize"},
        "submission_format": {"id_column": "id",
                              "prediction_column": "Status_C,Status_CL,Status_D"},
    }
    integrity_json = json.dumps(section, sort_keys=True)
    h = hashlib.sha256(integrity_json.encode()).hexdigest()
    return integrity_json, h


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def node_sha(cfg_hash, parent_sha):
    return hashlib.sha256(f"{cfg_hash}+{parent_sha}".encode()).hexdigest()


def load_tree():
    with open(TREE_PATH) as f:
        return json.load(f)


def save_tree(tree):
    with open(TREE_PATH, "w") as f:
        json.dump(tree, f, indent=2)


def resolve_config(spec):
    return {
        "architecture_class": spec["architecture_class"],
        "model_type": spec["model_type"],
        "hyperparameters": spec.get("hyperparameters", {}),
        "use_original": spec.get("use_original", False),
        "cv_folds": spec.get("cv_folds", 5),
        "random_seed": spec.get("random_seed", 42),
        "target_column": "Status",
        "metric": "log_loss",
        "metric_direction": "minimize",
    }


def build_and_run(spec):
    exp_id = spec["exp_id"]
    worktree = os.path.join(WORKTREES, exp_id)
    os.makedirs(worktree, exist_ok=True)
    config = resolve_config(spec)
    integrity_json, c_hash_objective = integrity()

    present = make_worktree.build_worktree(
        exp_id=exp_id, worktree=worktree, config=config,
        constraints_hash=c_hash_objective, integrity_json=integrity_json,
        hypothesis=spec["hypothesis"], parent_id=spec.get("parent_id"),
        parent_desc=spec.get("parent_desc", ""))
    missing = [m for m in make_worktree.REQUIRED_MODULES if m not in present]
    if missing:
        raise RuntimeError(f"{exp_id} missing modules: {missing}")

    proc = subprocess.run(["bash", os.path.join(worktree, "run.sh")],
                          capture_output=True, text=True, cwd=worktree)
    if proc.returncode != 0:
        raise RuntimeError(f"{exp_id} run.sh failed:\nSTDOUT{proc.stdout[-2000:]}\n"
                           f"STDERR{proc.stderr[-3000:]}")

    with open(os.path.join(worktree, "metrics.json")) as f:
        metrics = json.load(f)
    make_worktree.patch_trainable_params(worktree, metrics["trainable_params"])

    for art in ["val_predictions.npy", "val_labels.npy", "metrics.json",
                "submission.csv", "EVALUATION.json"][:4]:
        assert os.path.exists(os.path.join(worktree, art)), f"missing artifact {art}"

    return config, metrics, worktree, proc.stdout


def add_node(spec, config, metrics, worktree):
    tree = load_tree()
    cfg_hash = config_hash(config)
    parent_id = spec.get("parent_id")
    if parent_id is None:
        parent_sha = "root"
        depth = 0
    else:
        parent_sha = tree["nodes"][parent_id]["sha"]
        depth = tree["nodes"][parent_id]["depth"] + 1
    sha = node_sha(cfg_hash, parent_sha)
    rel_worktree = os.path.relpath(worktree, ROOT)
    tree["nodes"][spec["exp_id"]] = {
        "exp_id": spec["exp_id"],
        "parent": parent_id,
        "depth": depth,
        "architecture_class": spec["architecture_class"],
        "model": spec["model_type"],
        "config_hash": cfg_hash,
        "sha": sha,
        "status": "DONE",
        "metrics": {"log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                    "train_log_loss": metrics["train_log_loss"]},
        "trainable_params": int(metrics["trainable_params"]),
        "hypothesis": spec["hypothesis"],
        "worktree_path": rel_worktree + "/",
        "fold_log_losses": metrics["fold_log_losses"],
        "config_delta_key": spec.get("config_delta_key"),
        "cv_folds": spec.get("cv_folds", 5),
    }
    save_tree(tree)
    return sha, depth
