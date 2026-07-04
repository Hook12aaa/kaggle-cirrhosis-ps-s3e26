"""Executed evaluate (4-layer) and review-strategy logic for one variant.

Every comparison is computed here as code, never eyeballed. evaluate writes
EVALUATION.json; review writes REVIEW.json. Thresholds are calibrated for a
synthetic Playground-Series domain (overfitting 0.10, significance 0.02).
"""

import json
import os

import numpy as np

OVERFIT_THRESHOLD = 0.10
SIG_THRESHOLD = 0.02
HP_KEYS = {"learning_rate", "batch_size", "epochs", "dropout", "regularization",
           "weight_decay", "C", "max_iter", "n_estimators", "num_leaves",
           "max_depth", "min_samples_leaf", "iterations", "depth", "l2_leaf_reg",
           "max_iter", "max_leaf_nodes", "l2_regularization", "subsample",
           "colsample_bytree", "reg_lambda", "n_neighbors", "weights", "gamma",
           "kernel", "alpha", "learning_rate_init", "hidden_layer_sizes"}


def evaluate(worktree, node, baseline):
    m = json.load(open(os.path.join(worktree, "metrics.json")))
    val_ll = m["log_loss"]
    train_ll = m["train_log_loss"]
    folds = m["fold_log_losses"]
    base_val = baseline["log_loss"]
    base_train = baseline["train_log_loss"]

    layers = {}
    # Layer 1: data validation
    l1_fail = []
    for key in ("log_loss", "accuracy", "train_log_loss"):
        if key not in m:
            l1_fail.append(f"missing {key}")
    vals = [val_ll, train_ll, m["accuracy"]] + folds
    if any((v is None or np.isnan(v) or np.isinf(v)) for v in vals):
        l1_fail.append("nan_or_inf")
    if abs(len(folds) - node.get("cv_folds", 5)) > 1:
        l1_fail.append("fold_count_mismatch")
    layers["data_validation"] = {"passed": len(l1_fail) == 0, "failures": l1_fail}

    # Layer 2: overfitting. Lower log_loss is better, so a positive gap means
    # the held-out fold is worse than train. Low-bias boosted models always show
    # a large train-vs-OOF gap by construction, so a gap alone is not overfitting:
    # flag only when the gap is large AND the honest OOF fails to beat baseline.
    gap_ratio = (val_ll - train_ll) / val_ll if val_ll else 0.0
    base_gap = (base_val - base_train) / base_val if base_val else 0.0
    oof_beats_baseline = val_ll < base_val * (1 - SIG_THRESHOLD)
    overfit_flag = (gap_ratio > max(2 * base_gap, OVERFIT_THRESHOLD)) and not oof_beats_baseline
    layers["overfitting"] = {"passed": not overfit_flag, "gap_ratio": gap_ratio,
                             "baseline_gap_ratio": base_gap,
                             "oof_beats_baseline": oof_beats_baseline}

    # Layer 3: statistical significance vs baseline (relative log_loss reduction)
    rel_improvement = (base_val - val_ll) / base_val if base_val else 0.0
    significant = rel_improvement >= SIG_THRESHOLD
    layers["statistical_significance"] = {"passed": significant,
                                          "relative_improvement": rel_improvement,
                                          "threshold": SIG_THRESHOLD}

    # Layer 4: forensics on the per-fold trajectory
    arr = np.array(folds, dtype=float)
    nan_epochs = [i for i, v in enumerate(arr) if np.isnan(v) or np.isinf(v)]
    diffs = np.abs(np.diff(arr)) if len(arr) > 1 else np.array([0.0])
    med = np.median(diffs) if len(diffs) else 0.0
    spikes = [int(i) for i, d in enumerate(diffs) if med > 0 and d > 3 * med]
    tail = arr[max(0, int(len(arr) * 0.75)):]
    mode_collapse = bool(len(tail) > 1 and np.std(tail) < 1e-6)
    f_pass = len(nan_epochs) == 0 and not mode_collapse
    layers["forensics"] = {"passed": f_pass, "nan_epochs": nan_epochs,
                           "spikes": spikes, "mode_collapse": mode_collapse}

    if not layers["data_validation"]["passed"]:
        verdict = "REJECT"
    elif not layers["overfitting"]["passed"]:
        verdict = "REJECT"
    elif not layers["forensics"]["passed"]:
        verdict = "REJECT"
    elif not layers["statistical_significance"]["passed"]:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "ACCEPT"

    result = {
        "node_sha": node["sha"], "exp_id": node["exp_id"], "verdict": verdict,
        "primary_metric": {"name": "log_loss", "value": val_ll,
                           "baseline_value": base_val,
                           "relative_improvement": rel_improvement},
        "layers": layers,
    }
    json.dump(result, open(os.path.join(worktree, "EVALUATION.json"), "w"), indent=2)
    return result


def _lineage_keys(tree, exp_id):
    keys = []
    node = tree["nodes"][exp_id]
    while node["parent"] is not None:
        parent = tree["nodes"][node["parent"]]
        ch = node.get("config_delta_key")
        keys.append(ch)
        node = parent
    return keys


def review(worktree, tree, node, baseline_params, eval_result, delta_key):
    var_params = node["trainable_params"]
    param_ratio = var_params / baseline_params if baseline_params else float("inf")
    rel_improvement = eval_result["primary_metric"]["relative_improvement"]
    efficiency = rel_improvement / param_ratio if param_ratio else 0.0

    if param_ratio < 1.0 and rel_improvement >= -SIG_THRESHOLD:
        complexity = "Worth it"
    elif efficiency > 0.5:
        complexity = "Worth it"
    elif 0.1 <= efficiency <= 0.5 and param_ratio < 5.0:
        complexity = "Worth it"
    elif 0.1 <= efficiency <= 0.5:
        complexity = "Diminishing"
    else:
        complexity = "Not worth it"

    # consecutive HP-only changes back through lineage
    consecutive_hp = 0
    cur = node
    while cur["parent"] is not None:
        dk = cur.get("config_delta_key")
        if dk in HP_KEYS:
            consecutive_hp += 1
            cur = tree["nodes"][cur["parent"]]
        else:
            break
    stale = consecutive_hp > 3

    on_front = node["exp_id"] in tree.get("pareto_front", [])
    pareto_status = "dominant" if on_front else "dominated"

    if complexity == "Not worth it":
        verdict = "DISCARD"
    elif complexity == "Worth it" and not stale and pareto_status != "dominated":
        verdict = "KEEP"
    elif complexity == "Worth it" and not stale:
        verdict = "KEEP_WITH_CONCERNS"
    elif complexity == "Worth it" and stale and pareto_status == "dominant":
        verdict = "KEEP_WITH_CONCERNS"
    elif complexity == "Diminishing" and not stale and pareto_status == "dominant":
        verdict = "KEEP_WITH_CONCERNS"
    else:
        verdict = "DISCARD"

    result = {
        "node_sha": node["sha"], "exp_id": node["exp_id"], "verdict": verdict,
        "complexity": {"variant_params": var_params, "baseline_params": baseline_params,
                       "param_ratio": param_ratio, "relative_improvement": rel_improvement,
                       "efficiency": efficiency, "complexity_verdict": complexity},
        "hp_balance": {"consecutive_hp_only": consecutive_hp, "stale": stale},
        "pareto": {"status": pareto_status},
    }
    json.dump(result, open(os.path.join(worktree, "REVIEW.json"), "w"), indent=2)
    return result
