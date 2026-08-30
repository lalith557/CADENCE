# Colab / Kaggle GPU quickstart — paper-scale Gate A

The local RTX 4070 Laptop covers development and every gate at smoke
scale, but the **full paper protocol for H2** (10 seeds × 5+ scenarios
× 30 000 PPO timesteps × full-fidelity FraudNet retrains) is a
multi-day CPU-bound job on this machine (each 800-timestep single-seed
run already takes ~2.75 h — see R-Gate-A-w28). This document is how to
run it on a Colab T4 / A100 or a Kaggle P100 in an afternoon.

## Colab

```
!git clone <this repo>
%cd Capstone-Project
!pip install -q -e ".[ml,gnn,drift,dev]"

# Confirm GPU is visible
!python -m cadence.cli gpu-check

# Kick off the full sweep (10 seeds, checkpointed per seed — safe to
# refresh the tab / disconnect / reconnect):
!python scripts/colab_gate_a.py --seeds 10 --train-timesteps 30000
```

The script writes one `experiments/gate_a_full/seed{n}.json` per
finished seed and skips already-completed seeds on resume, so a
disconnected Colab kernel loses at most one seed of work.

## Kaggle

Same script; add:

```python
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
```

before the pip install (Kaggle's mlflow may be older).

## Pulling results back

The `experiments/gate_a_full/` directory is what should end up in the
next commit — either `git add` from Colab or download+push locally.
The next `R-Gate-A-full` entry in `docs/results.md` writes from that
aggregate JSON.

## Config notes

- `--train-timesteps 30000` matches the plan's paper-scale threshold.
  Below that PPO under-converges; above ~50k the marginal value drops.
- `--seeds 10` gets you the ≥10 required for paired Wilcoxon at
  α=0.05 with reasonable power.
- Default `--n-windows 15 --window-size 2048` matches the R-2/R-4
  Fraud protocol so numbers are directly comparable.
- SLA=0.65 matches R-4; raise to 0.75 for a variant that contests SLA
  on more scenarios (per W-24 fix).
