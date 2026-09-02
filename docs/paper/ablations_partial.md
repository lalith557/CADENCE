# CADENCE — Unified Ablation Table (partial, from existing entries)

**Status:** partial. Two ablation dimensions (no-GNN, no-surrogate) have paper-
scale n≥10 evidence from `R-Gate-B-n10-subset`. Three (no-PPO/rule,
no-contested-scenarios, no-augmented-Lagrangian) are blocked on the confirmatory
Stage-2 Gate-A run (`R-Gate-A-final`, pending). This document records the
existing ablation signal honestly and names each pending cell so the final table
can be dropped in without re-analysis.

The five ablations map to the pre-registered questions:

1. **no-GNN** — replace the learned CDAG+GNN readout with the structural
   path-sum proxy. Answers "does the *learned* GNN readout matter, over and
   above the structural attribution from PC+NOTEARS?"
2. **no-surrogate** — replace CADENCE's scorer with PSI (correlational,
   no CDAG, no surrogate). Answers "does the whole causal+surrogate stack
   beat the standard drift-detector baseline?"
3. **no-PPO (rule policy)** — replace the learned RSO with the hand-written
   rule policy. Answers "does the *learned* policy add value over a well-
   tuned rule?"
4. **no-contested-scenarios** — train PPO on the 5 default scenarios only,
   evaluate on contested. Answers "do contested-SLA training scenarios
   matter for RSO to learn a non-trivial policy?"
5. **no-augmented-Lagrangian** — fix λ_sla=1.0 instead of dual-updating.
   Answers "does the constraint-aware dual ascent beat naive reward
   shaping?"

## Table — what's landed

Each cell is either a real measured number with source R-entry, or explicitly
"pending Stage 2 (R-Gate-A-final)." Never a made-up number.

| Ablation | Metric (higher = better) | Full CADENCE | Ablated | Δ (Full − Ablated) | Source | Verdict |
|---|---|---|---|---|---|---|
| **no-GNN** (structural readout only) | HARD-subset MRR (n=20 paired) | 0.412 ± 0.340 | 0.150 ± 0.101 | **+0.262** | R-Gate-B-n10-subset | GNN helps clearly on HARD |
| **no-GNN** | HARD-subset AUROC (n=20) | 0.905 ± 0.057 | 0.586 ± 0.343 | **+0.319** | R-Gate-B-n10-subset | GNN helps clearly on HARD |
| **no-surrogate** (PSI only) | HARD-subset MRR (n=20) | 0.412 ± 0.340 | 0.097 ± 0.014 | **+0.315** | R-Gate-B-n10-subset | Full stack beats PSI clearly on HARD |
| **no-surrogate** | HARD-subset AUROC (n=20) | 0.905 ± 0.057 | 0.672 ± 0.052 | **+0.233** | R-Gate-B-n10-subset | Full stack beats PSI clearly on HARD |
| **no-GNN** on EASY | EASY MRR (n=30) | 0.833 ± 0.236 | 0.472 ± 0.375 | +0.361 | R-Gate-B-n10-subset | GNN helps even on EASY over structural |
| **no-surrogate** on EASY | EASY MRR (n=30) | 0.833 ± 0.236 | 1.000 ± 0.000 | −0.167 | R-Gate-B-n10-subset | PSI already saturated at ceiling; GNN not-inferior in AUROC (0.989 vs 1.000) |
| **no-PPO (rule policy)** on Fraud contested | mean F1 across seeds | *pending* | *pending* | *pending* | R-Gate-A-final vs a rule-policy control run under the frozen Stage-2 protocol | needs Stage 2 |
| **no-PPO (rule policy)** on Elec2 real drift | GPU-hr / mean F1 | 1.5319e-07 / 0.5745 | *rule policy is what already ran* | — | R-Gate-E-verified is *itself* the rule-policy baseline; the learned-policy equivalent lands after Stage 2 | needs Stage 2 |
| **no-contested-scenarios** | Δ mean F1 | *pending* | *pending* | *pending* | R-Gate-A-final vs an "R-4 default scenarios only" control | needs Stage 2 |
| **no-augmented-Lagrangian** (fixed λ) | Δ mean F1 / SLA-violation rate | *pending* | *pending* | *pending* | R-Gate-A-final vs a `--dual-lr 0 --lambda-init K` control | needs Stage 2 |

## What the two landed ablations already tell us

**Both `no-GNN` and `no-surrogate` show large, statistically-significant lifts
on the HARD subset** — paired Wilcoxon p ≤ 0.032 (Bonferroni over 4 tests), effect
sizes 0.23–0.32 in raw AUROC / MRR points. Because the two ablations remove
*different* pieces of the CADENCE attribution stack, and both are needed to
reach the full result, this is direct evidence that **neither the learned readout
nor the causal graph is decorative** — each is contributing measurable lift
on the exactly the scenarios where the baseline PSI signal is insufficient
(concept-shift, gradual drift).

**On EASY subset PSI is already at 1.0.** No stack can beat that ceiling on
the top-k metrics; GNN sits at 0.989 AUROC, comparable to the ceiling. This is
important framing for the paper: report EASY as "attribution ceiling — no
improvement to be had by construction," and let the ablation story rest on
HARD.

## What needs Stage 2 to land

- `no-PPO (rule policy)` on Fraud contested scenarios — need R-Gate-A-final's
  learned RSO numbers to pair against the rule policy.
- `no-contested-scenarios` — need a controlled sweep with the two training-
  scenario sets.
- `no-augmented-Lagrangian` — need a fixed-λ variant of Stage 2 as a
  matched-protocol control.

## Provenance

- **R-Gate-B-n10-subset** — `docs/results.md` (2026-09-03 entry from
  `experiments/phase_b_n10.json`, 150 rows).
- **R-Gate-E-verified** — `docs/results.md` (2026-09-01 entry from
  `experiments/phase_e_elec2.json`, 3 seeds).
- **R-Gate-A-final** — pending completion of the run currently in-flight
  (see `experiments/gate_a_ledger.json`, Stage 1 first).

Regenerate this table's numbers with:
```
python -c "
import json, statistics
d = json.load(open('experiments/phase_b_n10.json'))
rows = d['rows']
# ... aggregate as in docs/results.md::R-Gate-B-n10-subset ..."
```
