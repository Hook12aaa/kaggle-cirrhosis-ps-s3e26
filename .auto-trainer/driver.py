"""Run one exploration round: build+run+evaluate+review each spec, then the
convergence scripts. Invoked as: python driver.py <round_specs.json>.

Each spec drives a worktree; baseline metrics anchor evaluation. After
aggregating verdicts the three convergence scripts run in order and the
resulting global_status is printed.
"""

import json
import os
import subprocess
import sys

import orch
import eval_review

TRAINER = orch.TRAINER_DIR
TREE = orch.TREE_PATH
CONFIG = os.path.join(TRAINER, "convergence-config.json")
SCRIPTS = os.path.join(TRAINER, "scripts")
PY = sys.executable


def run_round(specs):
    summaries = []
    for spec in specs:
        config, metrics, wt, _ = orch.build_and_run(spec)
        sha, depth = orch.add_node(spec, config, metrics, wt)
        tree = orch.load_tree()
        node = tree["nodes"][spec["exp_id"]]
        baseline = tree["nodes"]["exp_000"]["metrics"]
        base_params = tree["nodes"]["exp_000"]["trainable_params"]
        ev = eval_review.evaluate(wt, node, baseline)
        rv = eval_review.review(wt, tree, node, base_params, ev,
                                spec.get("config_delta_key"))
        tree = orch.load_tree()
        tree["nodes"][spec["exp_id"]]["evaluate_verdict"] = ev["verdict"]
        tree["nodes"][spec["exp_id"]]["review_verdict"] = rv["verdict"]
        orch.save_tree(tree)
        summaries.append({
            "exp_id": spec["exp_id"], "class": spec["architecture_class"],
            "model": spec["model_type"], "depth": depth,
            "log_loss": round(metrics["log_loss"], 5),
            "acc": round(metrics["accuracy"], 4),
            "params": metrics["trainable_params"],
            "eval": ev["verdict"], "review": rv["verdict"],
            "rel_impr": round(ev["primary_metric"]["relative_improvement"], 4),
        })
    return summaries


def run_convergence():
    subprocess.run([PY, os.path.join(SCRIPTS, "compute_pareto.py"), TREE], check=True,
                   capture_output=True, text=True)
    subprocess.run([PY, os.path.join(SCRIPTS, "check_class_exhaustion.py"), TREE],
                   check=True, capture_output=True, text=True)
    cov = subprocess.run([PY, os.path.join(SCRIPTS, "check_cross_class_coverage.py"),
                          TREE, CONFIG], check=True, capture_output=True, text=True)
    return json.loads(cov.stdout)


if __name__ == "__main__":
    specs = json.load(open(sys.argv[1]))
    summaries = run_round(specs)
    for s in summaries:
        print(json.dumps(s))
    cov = run_convergence()
    tree = orch.load_tree()
    print("PARETO_FRONT", sorted(tree["pareto_front"]))
    print("CLASS_STATUS", json.dumps(tree["class_status"]))
    print("CONVERGENCE", json.dumps(cov))
