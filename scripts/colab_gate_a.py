#!/usr/bin/env python
"""Stage 2 Gate A — Colab handoff recipe (documentation module).

Purpose: Stage 2 (10 seeds x 30k timesteps x 8 scenarios) is a ~20h wall job
even under the W-38 / W-39 fixes — too long for a laptop overnight session.
This file's docstring is the paste-into-Colab recipe. Running Stage 2 on Colab
must use the same frozen protocol as local; the recipe below wraps the local
resumable runner (`run_experiment.py --stage 2 --resume`) unchanged.

Supersedes the Aug-30 per-seed subprocess driver, which pre-dated
run_experiment.py's atomic ledger + resume-from-COMPLETED and did not use
the W-36 `--per-seed-policies` flag or the W-32..W-40 fix chain.

The scientific protocol is unchanged from local execution — no config edits,
no seed cherry-picking, no post-hoc parameter tuning. Colab is *only* a
compute substrate. Any deviation from the frozen protocol invalidates the
pre-registration (docs/gate_a_preregistration.md Part 3 rule #3).

## Paste-into-Colab recipe

Create a fresh Colab notebook with a **T4 or A100 GPU** runtime. Paste the
following cells verbatim; do not modify.

### Cell 1 — clone the repo at the frozen commit

```python
import subprocess, os
REPO_URL = "https://github.com/<your-user>/<cadence-repo>.git"
# Replace with the exact git SHA that produced R-Gate-A-stage1-pass locally
# (find it via `git log --oneline` after Stage 1 passes; do NOT use HEAD if
# HEAD has moved past the frozen protocol).
FROZEN_COMMIT = "REPLACE_WITH_SHA_FROM_STAGE1_PASS"
subprocess.check_call(["git", "clone", "--quiet", REPO_URL, "/content/cadence"])
os.chdir("/content/cadence")
subprocess.check_call(["git", "checkout", FROZEN_COMMIT])
subprocess.check_call(["git", "rev-parse", "HEAD"])
```

### Cell 2 — install deps

```python
!pip install -q -e /content/cadence
!pip install -q fastapi httpx  # for API tests if you want to sanity-check
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Cell 3 — fetch the Credit Card Fraud dataset

Dataset is Kaggle-licensed; do NOT commit it to the repo. Use Kaggle
credentials from Colab Secrets (Add secret `KAGGLE_JSON` with the raw JSON):

```python
from google.colab import userdata
import os, pathlib
kaggle_json = userdata.get("KAGGLE_JSON")
pathlib.Path("/root/.kaggle").mkdir(exist_ok=True)
pathlib.Path("/root/.kaggle/kaggle.json").write_text(kaggle_json)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
!pip install -q kaggle
!kaggle datasets download -d mlg-ulb/creditcardfraud -p /content/cadence/Dataset/ --unzip
!mkdir -p "/content/cadence/Dataset/Credit Card Fraud Detection"
!mv "/content/cadence/Dataset/creditcard.csv" "/content/cadence/Dataset/Credit Card Fraud Detection/creditcard.csv"
!ls -la "/content/cadence/Dataset/Credit Card Fraud Detection/"
```

### Cell 4 — sanity check (must PASS before Stage 2 starts)

```python
!cd /content/cadence && python -m pytest \
    tests/unit/test_rso_audit.py \
    tests/unit/test_w38_per_seed_training.py \
    tests/unit/test_w39_dual_lr_calmed.py \
    tests/unit/test_w40_reward_component_callback.py \
    tests/unit/test_run_experiment.py -q
```

If any test fails: STOP. The frozen protocol is not intact on this Colab
instance; do not run Stage 2. Diagnose (usually a dep-version drift).

### Cell 5 — run Stage 2 with resumable ledger

```python
# Runs 10 seeds x 30k timesteps x 8 scenarios per pre-reg Part 2. Expect
# multi-hour wall on a T4 (probably 15-25h; A100 will be faster).
# Colab may disconnect mid-run — the runner's atomic ledger + per-seed
# checkpoints survive it. Re-running the same command resumes.
!cd /content/cadence && python run_experiment.py --config configs/gate_a.yaml --stage 2 --resume
```

### Cell 6 — periodic checkpoint (recommended: run in a separate cell every ~2h)

```python
!cd /content/cadence && python scripts/evaluate_stage1_gates.py --log logs/stage2.log 2>/dev/null || true
!ls -la /content/cadence/experiments/gate_a_stage2/ 2>/dev/null | head -20
!cat /content/cadence/experiments/gate_a_ledger.json | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps([{'seed':s['seed'],'status':s['status'],'wall_s':s.get('wall_seconds')} for s in d['stages']['2']['seeds']], indent=2))"
```

### Cell 7 — bring results back

After Stage 2 completes (all 10 seeds `COMPLETED` in the ledger):

```python
# Bundle just the results, not the entire runtime state.
!cd /content/cadence && tar czf /content/gate_a_stage2_results.tar.gz \
    experiments/gate_a_ledger.json \
    experiments/gate_a_stage2/ \
    experiments/rso_ppo_phase_a_seed*.zip \
    experiments/rso_ppo_phase_a_seed*.vecnorm.pkl \
    experiments/mlflow.db \
    logs/stage2.log

from google.colab import files
files.download("/content/gate_a_stage2_results.tar.gz")
```

Then on the local machine:

```bash
mv experiments/mlflow.db experiments/mlflow.db.bak-pre-colab-import
tar xzf ~/Downloads/gate_a_stage2_results.tar.gz -C .
python scripts/evaluate_stage1_gates.py     # sanity-check imported state
# Compile R-Gate-A-final by hand from experiments/gate_a_stage2/seed_*.json
# aggregates + the four pre-registered verdict criteria.
```

## What is NOT allowed on Colab (per pre-registration)

- Modifying configs/gate_a.yaml
- Modifying cadence/rso/*, benchmarks/phase_a_run.py, or run_experiment.py
  from the frozen commit
- Adding/removing seeds
- Running only a subset of scenarios
- Selecting a checkpoint by test-set performance
- Any change that would produce numbers differently from local execution

If Colab's PyTorch/SB3 pins differ from local, verify the sanity-check cell
above passes first; if it doesn't, `pip install` to match local `.venv`
exact versions before running Stage 2.

## Cost estimate

- T4 (Colab free): ~15-25h wall for Stage 2. Colab free-tier 12h session
  limit; expect at least one disconnect. Resume with the same Cell 5 command.
- A100 (Colab Pro): ~4-10h wall.
- Kaggle T4x2: alternative, 30h/week free quota; adapt the git-clone cell.

## Provenance record

After Stage 2 completes, write R-Gate-A-final to docs/results.md with:
  * The exact FROZEN_COMMIT git SHA used for Colab.
  * The Colab instance type (T4 / A100 / etc.).
  * Total wall-time per seed and aggregate.
  * The full ledger and MLflow DB imported back to local.
"""
from __future__ import annotations

import sys


def main() -> int:
    """This module is documentation, not a runnable driver. Print help + exit."""
    print(__doc__)
    print("\n--- This file is a documentation module. Do not run it directly. ---")
    print("Paste the Cell blocks above into a Google Colab notebook.")
    print("For local Stage 2 (not recommended, ~20h wall), use:")
    print("  python run_experiment.py --config configs/gate_a.yaml --stage 2 --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
