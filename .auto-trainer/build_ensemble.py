"""Materialize the exp_ensemble node from caruana's selected weights.

Blends member out-of-fold and test probability matrices with the selected
weights, recomputes OOF log_loss and per-fold trajectory on the same
StratifiedKFold split the members used, runs the 4-layer evaluate, writes the
blended submission, and chains the node to the best single model.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import orch
import eval_review

ROOT = orch.ROOT
ENS = os.path.join(orch.WORKTREES, "exp_ensemble")

tree = orch.load_tree()
cfg = json.load(open(os.path.join(ENS, "ensemble_config.json")))
members = cfg["selected_models"]

labels = None
blend_oof = None
blend_test = None
for m in members:
    node = tree["nodes"][m["exp_id"]]
    wt = os.path.join(ROOT, node["worktree_path"])
    oof = np.load(os.path.join(wt, "val_predictions.npy"))
    test = np.load(os.path.join(wt, "test_predictions.npy"))
    y = np.load(os.path.join(wt, "val_labels.npy"))
    if labels is None:
        labels = y
        blend_oof = np.zeros_like(oof)
        blend_test = np.zeros_like(test)
    assert np.array_equal(y, labels), f"label mismatch for {m['exp_id']}"
    blend_oof += m["weight"] * oof
    blend_test += m["weight"] * test

total_w = sum(m["weight"] for m in members)
blend_oof /= total_w
blend_test /= total_w

oof_loss = float(log_loss(labels, blend_oof, labels=[0, 1, 2]))
oof_acc = float(accuracy_score(labels, np.argmax(blend_oof, axis=1)))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []
X_dummy = np.zeros(len(labels))
for _, va in skf.split(X_dummy, labels):
    fold_losses.append(float(log_loss(labels[va], blend_oof[va], labels=[0, 1, 2])))

np.save(os.path.join(ENS, "val_predictions.npy"), blend_oof)
np.save(os.path.join(ENS, "val_labels.npy"), labels)
np.save(os.path.join(ENS, "test_predictions.npy"), blend_test)

test_ids = pd.read_csv(os.path.join(ROOT, "test.csv"))["id"].to_numpy()
sub = pd.DataFrame({"id": test_ids,
                    "Status_C": blend_test[:, 0],
                    "Status_CL": blend_test[:, 1],
                    "Status_D": blend_test[:, 2]})
sub.to_csv(os.path.join(ENS, "submission.csv"), index=False)

ens_params = sum(tree["nodes"][m["exp_id"]]["trainable_params"] for m in members)
metrics = {
    "log_loss": oof_loss,
    "accuracy": oof_acc,
    "train_log_loss": oof_loss,
    "fold_log_losses": fold_losses,
    "n_features": tree["nodes"]["exp_009"].get("n_features", 0),
    "trainable_params": int(ens_params),
}
json.dump(metrics, open(os.path.join(ENS, "metrics.json"), "w"), indent=2)

best_single = "exp_009"
parent_sha = tree["nodes"][best_single]["sha"]
config_payload = {"ensemble": "caruana", "members": members,
                  "metric": "log_loss", "metric_direction": "minimize"}
cfg_hash = orch.config_hash(config_payload)
sha = orch.node_sha(cfg_hash, parent_sha)
node = {
    "exp_id": "exp_ensemble", "parent": best_single,
    "depth": tree["nodes"][best_single]["depth"] + 1,
    "architecture_class": "ensemble", "model": "caruana_blend",
    "config_hash": cfg_hash, "sha": sha, "status": "DONE",
    "metrics": {"log_loss": oof_loss, "accuracy": oof_acc, "train_log_loss": oof_loss},
    "trainable_params": int(ens_params),
    "hypothesis": "Caruana greedy blend of accepted models for log_loss reduction.",
    "worktree_path": os.path.relpath(ENS, ROOT) + "/",
    "fold_log_losses": fold_losses,
    "config_delta_key": "ensemble", "cv_folds": 5,
    "members": members,
}
tree["nodes"]["exp_ensemble"] = node
orch.save_tree(tree)

baseline = tree["nodes"]["exp_000"]["metrics"]
ev = eval_review.evaluate(ENS, node, baseline)
tree = orch.load_tree()
tree["nodes"]["exp_ensemble"]["evaluate_verdict"] = ev["verdict"]

best_single_score = tree["nodes"][best_single]["metrics"]["log_loss"]
if oof_loss < best_single_score:
    outcome = "DONE"
    tree["winner"] = "exp_ensemble"
elif abs(oof_loss - best_single_score) < 1e-9:
    outcome = "DONE_WITH_CONCERNS"
    tree["winner"] = best_single
else:
    outcome = "DONE_WITH_CONCERNS"
    tree["winner"] = best_single
orch.save_tree(tree)

print(json.dumps({
    "ensemble_oof_log_loss": oof_loss, "ensemble_acc": oof_acc,
    "best_single": best_single, "best_single_log_loss": best_single_score,
    "beats_single": oof_loss < best_single_score,
    "evaluate_verdict": ev["verdict"], "outcome": outcome,
    "winner": tree["winner"], "fold_losses": [round(x, 4) for x in fold_losses],
}, indent=2))
