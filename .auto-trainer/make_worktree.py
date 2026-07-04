"""Generate the 10-module worktree for one experiment variant.

The orchestrator resolves the full config and calls build_worktree(); each
generated module is a thin, deterministic wrapper over the shared harness so
every variant produces an identical module contract that actually runs.
"""

import json
import os
import stat

TRAINER_DIR = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis/.auto-trainer"
VENV_PY = "/Users/hook/Documents/coding/python/kaggle/ps-s3e26-cirrhosis/.venv/bin/python"

REQUIRED_MODULES = [
    "config.py", "data.py", "model.py", "train.py", "eval.py", "preflight.py",
    "run.sh", "metrics_manifest.json", "constraints.lock", "BUILD_REPORT.md",
]


def build_worktree(exp_id, worktree, config, constraints_hash, integrity_json,
                   hypothesis, parent_id, parent_desc):
    os.makedirs(worktree, exist_ok=True)

    import pprint
    with open(os.path.join(worktree, "config.py"), "w") as f:
        f.write("CONFIG = " + pprint.pformat(config, indent=4, sort_dicts=True) + "\n")
        f.write("WORKTREE = " + repr(worktree) + "\n")

    header = f'import sys\nsys.path.insert(0, {json.dumps(TRAINER_DIR)})\n'

    with open(os.path.join(worktree, "data.py"), "w") as f:
        f.write(header)
        f.write("from harness import load_data\n")
        f.write("from config import CONFIG\n\n\n")
        f.write("def get_data():\n    return load_data(CONFIG)\n")

    with open(os.path.join(worktree, "model.py"), "w") as f:
        f.write(header)
        f.write("from harness import build_estimator, load_data\n")
        f.write("from config import CONFIG\n\n\n")
        f.write("def build():\n")
        f.write("    _, _, _, _, _, _, meta = load_data(CONFIG)\n")
        f.write("    return build_estimator(CONFIG, meta)\n")

    with open(os.path.join(worktree, "train.py"), "w") as f:
        f.write(header)
        f.write("import json\nimport harness\nfrom config import CONFIG, WORKTREE\n\n")
        f.write("metrics = harness.run_experiment(CONFIG, WORKTREE)\n")
        f.write("for i, fl in enumerate(metrics['fold_log_losses']):\n")
        f.write("    print(json.dumps({'fold': i, 'val_log_loss': fl}))\n")
        f.write("print(json.dumps({'oof_log_loss': metrics['log_loss'], "
                "'train_log_loss': metrics['train_log_loss'], "
                "'trainable_params': metrics['trainable_params']}))\n")

    with open(os.path.join(worktree, "eval.py"), "w") as f:
        f.write(header)
        f.write("import json\nimport os\nimport numpy as np\n")
        f.write("from sklearn.metrics import log_loss\n\n")
        f.write("here = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("oof = np.load(os.path.join(here, 'val_predictions.npy'))\n")
        f.write("y = np.load(os.path.join(here, 'val_labels.npy'))\n")
        f.write("recomputed = float(log_loss(y, oof, labels=[0, 1, 2]))\n")
        f.write("m = json.load(open(os.path.join(here, 'metrics.json')))\n")
        f.write("assert abs(recomputed - m['log_loss']) < 1e-9, "
                "(recomputed, m['log_loss'])\n")
        f.write("print(json.dumps({'log_loss': m['log_loss'], "
                "'accuracy': m['accuracy'], 'recomputed_log_loss': recomputed}))\n")

    with open(os.path.join(worktree, "preflight.py"), "w") as f:
        f.write(header)
        f.write("import os\n\n")
        f.write(f"EXPECTED_HASH = {json.dumps(constraints_hash)}\n")
        f.write("here = os.path.dirname(os.path.abspath(__file__))\n")
        f.write("lock = open(os.path.join(here, 'constraints.lock')).read()\n")
        f.write("assert EXPECTED_HASH in lock, 'constraints hash mismatch'\n")
        f.write("import harness  # noqa: F401\n")
        f.write("assert os.path.exists(os.path.join("
                f"{json.dumps(TRAINER_DIR)}, '..', 'train.csv'))\n")
        f.write("print('preflight ok')\n")

    run_sh = os.path.join(worktree, "run.sh")
    with open(run_sh, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write('cd "$(dirname "$0")"\n')
        f.write(f"{VENV_PY} preflight.py\n")
        f.write(f"{VENV_PY} train.py\n")
        f.write(f"{VENV_PY} eval.py\n")
    os.chmod(run_sh, os.stat(run_sh).st_mode | stat.S_IEXEC)

    manifest = {
        "metrics": [
            {"name": "log_loss", "type": "lower_better",
             "extract": "python3 -c \"import json;print(json.load(open('metrics.json'))['log_loss'])\""},
            {"name": "accuracy", "type": "higher_better",
             "extract": "python3 -c \"import json;print(json.load(open('metrics.json'))['accuracy'])\""},
        ],
        "primary": "log_loss",
        "metrics_file": "metrics.json",
    }
    with open(os.path.join(worktree, "metrics_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(worktree, "constraints.lock"), "w") as f:
        f.write(integrity_json + "\n")
        f.write(f"SHA256={constraints_hash}\n")
        f.write("max_seconds=1800\n")

    with open(os.path.join(worktree, "BUILD_REPORT.md"), "w") as f:
        f.write(f"# {exp_id}\n\n")
        f.write(f"**Hypothesis:** {hypothesis}\n\n")
        f.write(f"**Architecture class:** {config['architecture_class']}\n\n")
        f.write(f"**Model type:** {config['model_type']}\n\n")
        f.write(f"**Parent:** {parent_id} ({parent_desc})\n\n")
        f.write("**Resolved config:**\n\n```json\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n")
        f.write("trainable_params: PENDING\n")

    present = [m for m in REQUIRED_MODULES if os.path.exists(os.path.join(worktree, m))]
    return present


def patch_trainable_params(worktree, value):
    path = os.path.join(worktree, "BUILD_REPORT.md")
    text = open(path).read().replace("trainable_params: PENDING",
                                     f"trainable_params: {int(value)}")
    open(path, "w").write(text)
