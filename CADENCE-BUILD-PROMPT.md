# CADENCE — Master Build Prompt for Claude Code

> Paste everything below the line into a fresh Claude Code session opened in
> `C:\Users\Lalith Gona\HTML\Capstone Project`. It is written to be self-contained.

---

## 0. Who you are and what you're building

You are the lead engineer + research engineer building **CADENCE — Causal
Attribution-Driven Efficient Continual Retraining for Production ML Systems**.

This is not a toy. It has two simultaneous bars it must clear, and you must never
sacrifice one for the other:

1. **Research bar** — the work must be strong enough that a paper written from it
   could be submitted to a **main track of a well-known international conference**
   (e.g., IEEE Big Data, ACM SoCC, EuroMLSys, or a NeurIPS/ICML-adjacent systems
   venue). That means real experiments, real baselines, real ablations, honest
   statistics, and full reproducibility.
2. **Product bar** — after the core is working, CADENCE must become a **launchable
   software product** with genuine market value (an MLOps drift-attribution +
   cost-aware retraining tool). You will repeatedly hunt for weaknesses and fix
   them until it is something that could ship.

**The single most important rule:** every quantitative claim you record must come
from a real run you actually executed. Never fabricate, extrapolate, or copy the
illustrative numbers from the reference docs into results as if they were measured.
If reality diverges from the reference's illustrative example, that divergence is a
*finding*, not a failure — report it truthfully.

---

## 1. Source of truth

Two documents in this folder define the project. Read them fully before writing any
code, and re-read the relevant section before building each component:

- **`CADENCE-master-reference.md`** — the full spec. Part I = the 20-section
  proposal (architecture, algorithms, novelty claims A/B/C, evaluation plan,
  patent claims). Part II = the from-zero walkthrough (the FraudNet example, the
  46-question mechanics, per-model tables A–F, the exact partial-vs-full retraining
  code sketch, and **Part 6's Honest Gap Analysis** listing everything the proposal
  left unspecified). Part III = the dataset reference.
- **`CADENCE-datasets-verified.md`** — the dataset/link list and the "where to start
  given a GTX 1650" guidance.

Treat Part 6's "Honest Gap Analysis" and the recommended defaults as your decision
backlog: every gap listed there is a decision *you* must make explicitly and record
in `decisions.md`.

**Hardware reality:** the target dev machine is a GTX 1650 (4 GB VRAM). Keep the
default pipeline CPU/GPU-light and runnable end-to-end locally. Anything heavy
(CIFAR-10 full runs, Criteo/Avazu at scale) must be optional and clearly flagged as
"run on Colab/Kaggle," never on the critical path.

---

## 2. The three living documents (create these first, update them continuously)

These are not end-of-project reports. They are **living documents** you update in the
same commit as the code they describe. Never batch them to the end.

### 2a. `docs/decisions.md` — the active decision log
Every time you make a real choice, append an entry. Format per entry:

```
### D-<n>: <short title>   (<date>)
- **Context:** what forced a decision (link the gap in Part 6 if it maps to one).
- **Decision:** what you chose, concretely (numbers, method, config).
- **Why:** the reasoning, in your own words.
- **Alternatives considered:** what else was on the table and why you rejected each.
- **Reversibility:** cheap / costly to change later, and what would make you revisit.
- **Affects:** which claim (A/B/C), which metric, which module.
```
Cover at minimum every open item in Part 6: production model architecture, activation
→ node clustering method + cluster counts + tapped layers, CDAG node granularity, GNN
featurization/embedding dim/loss, GNN+surrogate joint-vs-separate training, the
surrogate's offline label-generation procedure, PPO state vector / action set /
reward weights, EWC scope + Fisher computation + replay-buffer ratio, shadow/canary
promotion threshold, CADENCE's own drift trigger, all rolling-window sizes, and the
multi-node/diffuse-responsibility fallback rule.

### 2b. `docs/tech.md` — the technology-stack rationale
For **every** library, framework, model family, and infra choice, one entry:

```
### <tech name> — used for <what>
- **Why this:** the concrete reason it fits CADENCE.
- **Alternatives:** the realistic options you compared.
- **Why not the alternatives:** specific, not hand-wavy.
- **Version pinned:** exact version, and why (if it matters).
- **Risk / lock-in:** how hard to swap out later.
```
Cover the whole stack from the reference Section 12 (PyTorch, PyTorch Geometric,
scikit-learn/LightGBM, Stable-Baselines3 or RLlib, causal-learn/gcastle/NOTEARS,
river, Avalanche, FastAPI/Triton, Docker, storage, MLflow, CodeCarbon/Electricity
Maps, dashboard, CI) — plus anything you actually add. If you choose differently from
the reference, that's allowed; just record why in both `tech.md` and `decisions.md`.

### 2c. `docs/results.md` — the measured-results ledger
Every experiment you run appends a dated, reproducible entry. **Measured numbers
only.** Format:

```
### R-<n>: <experiment name>   (<date>)
- **Hypothesis / question:** (map to H1/H2/H3 where relevant).
- **Setup:** dataset, drift type, baseline(s), seeds, config hash / MLflow run id.
- **Command to reproduce:** exact CLI line.
- **Result:** the actual numbers, with mean ± std / CI over seeds.
- **Statistical test:** paired t-test or Wilcoxon vs. baseline, p-value, effect size.
- **Interpretation:** what it means, honestly — including negative results.
- **Threats to validity:** what could make this misleading.
```
The three hypotheses to drive experiments toward (from Section 9):
- **H1** — CDAG attribution finds the true root cause with higher precision than
  correlational PSI/KS baselines.
- **H2** — the RSO reduces total retraining compute/carbon vs. periodic-retrain and
  reactive-full-retrain baselines, at equal-or-better SLA recovery.
- **H3** — causal-targeted retraining reduces catastrophic forgetting vs. naive full
  retrains on unrelated held-out segments.

If a hypothesis fails to hold, say so plainly in `results.md` and open a weakness
item — a paper with one honest negative result is publishable; a paper with faked
positives is not.

---

## 3. Definition of "done / desired output"

"Perfect" is not "matches the walkthrough numbers." Perfect is:

1. **End-to-end loop runs** on the Credit Card Fraud dataset: train FraudNet → inject a
   known `river` drift → PSI trigger fires → CDAG built (PC+NOTEARS) → GNN+surrogate
   produce ranked responsibility scores → PPO picks a scope → EWC partial / full
   retrain executes → shadow/canary validates → promote or roll back → resume
   monitoring. One command runs the whole thing and writes an MLflow run + a
   `results.md` entry.
2. **The three hypotheses are actually tested** with the baselines and ablations from
   Sections 9 & 14, across multiple seeds, with paired significance tests and CIs.
3. **Generality is demonstrated** on at least one non-neural adapter (tree-ensemble on
   Give Me Some Credit) and the forgetting benchmark (Avalanche + MNIST). This is the
   experiment that lets the paper claim CADENCE isn't hardcoded to neural nets.
4. **Attribution is validated against ground truth** on synthetic drift where you
   injected the true cause, reporting precision/recall/AUROC of "is this node
   responsible" and Structural Hamming Distance for the recovered graph.
5. **Reproducibility is real:** fixed seeds, versioned synthetic-drift generator,
   Docker image, configs tracked in MLflow, and a `make reproduce` (or equivalent)
   that regenerates the headline results.
6. **The three living docs are current** and the weakness register (Section 6) is
   converging, not growing.

---

## 4. Research-grade requirements (non-negotiable for the paper)

- **Baselines, really implemented:** PSI/KS-test monitoring + full retrain; fixed-
  schedule (e.g., weekly) retrain; EWC-only continual learning without causal
  targeting. Not strawmen — tuned enough to be fair.
- **Ablations, really run:** remove the GNN (raw feature stats only); remove the
  surrogate (naive correlation scoring); replace PPO with a fixed-threshold rule.
  Each ablation is a `results.md` entry.
- **Statistics:** ≥5 seeds per condition; paired t-test / Wilcoxon signed-rank across
  seeds and drift scenarios; report means, std, and 95% CIs. No single-run claims.
- **Metrics** exactly as Section 13: attribution precision/recall/AUROC + calibration;
  compute (FLOPs / GPU-hours) and carbon (kg CO₂e via CodeCarbon) saved vs. baselines;
  SLA recovery latency; post-retrain F1/AUROC; forgetting rate on unrelated segments;
  robustness under noisy/partial telemetry; false-alarm rate.
- **Honesty about scope:** where you used an illustrative default (e.g., FraudNet MLP,
  k-means clusters), label it as a design choice in the paper's method section, not a
  discovery.
- **A `docs/paper/` skeleton** (abstract, related work, method, experiments, results
  tables auto-populated from `results.md`, limitations, reproducibility appendix) that
  grows as results land — so the paper is never a from-scratch scramble at the end.

---

## 5. Product-grade requirements (what makes it launchable, not just a repo)

Maintain and *increase* market value as you build. Concretely:

- **Clean architecture matching Section 15's repo layout**, with a real Model Adapter
  interface (neural / tree / linear) so CADENCE is not fraud-specific — this
  extensibility *is* the product moat.
- **A working demo:** a dashboard (React or Grafana+Prometheus) that shows live drift,
  the CDAG, responsibility scores, the retraining decision, and cost/carbon saved.
  A user should *see* "why the model broke" — that's the differentiator vs. Evidently/
  WhyLabs/Arize/Fiddler.
- **Packaging & DX:** `pip install -e .`, a CLI (`cadence run …`), a `docker/` image,
  a `docker-compose` for the streaming+storage+dashboard stack, a 5-minute quickstart
  in the README that actually works from a clean clone.
- **Engineering hygiene:** typed Python, unit + integration tests, GitHub Actions CI
  (lint + tests + a tiny benchmark-regression check), config via files not hardcoding,
  structured logging, graceful failure and rollback, and an audit log for the
  retraining decisions (compliance is a selling point in fintech/health).
- **Privacy/security posture:** PII-safe feature hashing and activation sampling as the
  reference describes; document the threat model. Never commit real secrets or API
  keys — Electricity Maps key via env var, with CodeCarbon as the offline fallback.
- **A one-page `docs/product.md`:** who it's for, the ROI pitch ("pay a % of GPU-hours
  saved"), how it differs from incumbents, and what the launchable MVP scope is.

---

## 6. The weakness-audit loop (do this repeatedly, not once)

Maintain **`docs/weaknesses.md`** as a living register. After every phase — and again
before you call the project done — run an explicit self-critique pass and act on it:

1. **Audit** across five lenses and write findings into `weaknesses.md`:
   - *Correctness* — does each component do what the spec says? Edge cases?
   - *Research validity* — would a hostile reviewer believe these numbers? Missing
     baseline/ablation/seed/stat test? Confounds? Overclaiming?
   - *Product readiness* — install-from-clean-clone, error handling, docs, tests, demo.
   - *Performance/cost* — does it stay within GTX-1650 reality? Any silent slowness?
   - *Security/privacy* — leaks, secrets, PII, unsafe defaults.
2. **Rank** each finding: severity (blocker / major / minor) × effort.
3. **Fix** the blockers and majors now; log minors with a plan.
4. **Re-verify** with a test or a rerun, and record the outcome.
5. Repeat until a full audit pass produces no new blocker/major findings **and** the
   register is shrinking. Then do one final adversarial pass specifically asking:
   "If I were reviewer 2 rejecting this paper, or a CTO declining to buy this product,
   what would my reason be?" — and close those gaps.

Every weakness closed should make either the paper stronger or the product more
launchable (ideally both). If a fix would help the paper but hurt the product, or vice
versa, flag the tradeoff in `decisions.md` and pick deliberately.

---

## 7. Phased build plan (with verification gates)

Follow the reference's own priority order (Section 10B): **Claim C first** (strongest,
most self-contained), then **Claim A**, then **Claim B** narrowly. Do not advance a
phase until its gate passes. Commit at each gate; keep the three docs current.

- **Phase 0 — Scaffold.** Repo skeleton (Section 15), env + deps, MLflow, CI, test
  harness, config system, seed control, Docker stub. Load the local datasets (paths in
  Section 9). Write the initial `decisions.md` / `tech.md`.
  *Gate:* clean clone → `make setup` → tests pass in CI.
- **Phase 1 — FraudNet + drift + trigger + baselines (Claim C foundation).** Train
  FraudNet on Credit Card Fraud; build the `river`-based synthetic drift injector with
  logged ground-truth causes; implement the PSI/KS trigger and the three baselines;
  build the sandbox/Gym environment with cost + CodeCarbon carbon models.
  *Gate:* drift injected → trigger fires → baselines produce measured F1/cost numbers
  in `results.md`.
- **Phase 2 — RSO / PPO (Claim C core).** Implement the PPO scope optimizer with the
  Lagrangian-constrained reward; train in the sandbox; compare against the fixed-
  schedule and reactive-full-retrain baselines for compute/carbon vs. SLA recovery.
  *Gate:* H2 tested with ≥5 seeds + paired stats; RSO beats or matches baselines on
  cost at equal SLA recovery, recorded honestly.
- **Phase 3 — CDAG (Claim A).** Activation tapping + k-means cluster nodes + feature
  nodes; hybrid PC (skeleton) → NOTEARS (weights on the skeleton); the GNN encoder.
  Validate graph recovery (SHD) and attribution precision/recall vs. PSI on synthetic
  ground truth.
  *Gate:* H1 tested; CDAG attribution beats correlational baseline, with stats.
- **Phase 4 — Surrogate (Claim B, narrow).** Offline sandbox procedure that generates
  labeled (node, intervention, measured-outcome) triples; train GNN+surrogate jointly;
  produce responsibility scores; feed them into the RSO to close the loop.
  *Gate:* full closed loop runs on one command; surrogate calibration reported.
- **Phase 5 — EWC executor + shadow/canary.** Partial retrain per Part 3B mechanics
  (freeze, Fisher, replay buffer, EWC penalty), full retrain, shadow/canary validation,
  promote/rollback, and the rollback→surrogate-feedback learning loop.
  *Gate:* H3 tested; forgetting reduced vs. naive full retrain, with stats.
- **Phase 6 — Generality + robustness.** Tree-ensemble adapter on Give Me Some Credit;
  Avalanche+MNIST forgetting; noisy/partial-telemetry robustness; false-alarm rate.
  *Gate:* generality claim supported by measured cross-model results.
- **Phase 7 — Productization + paper.** Dashboard, packaging, docker-compose, docs,
  `product.md`, and the auto-populated paper skeleton. Full weakness-audit loop until
  convergence.
  *Gate:* clean-clone quickstart works; paper skeleton has real results tables; final
  adversarial audit finds no blocker/major.

At the start of each phase, restate the plan for that phase (a short todo list is fine)
so progress is visible.

---

## 8. Datasets available locally (already downloaded — use these paths)

Under `Dataset/` in this folder:
- **Credit Card Fraud** → `Dataset/Credit Card Fraud Detection/creditcard.csv` (primary
  neural adapter).
- **Give Me Some Credit** → `Dataset/Give Me Some Credit/cs-training.csv` (tree adapter,
  key generality experiment).
- **Telco Customer Churn** → `Dataset/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`
  (linear adapter).
- **Adult / Census** → `Dataset/adult/` (linear adapter alt).
- **Airlines** → `Dataset/Airlines1.arff` etc. (real drift).
- **MNIST** → `Dataset/MNIST/` and **CIFAR-10** → `Dataset/cifar-10/` (forgetting).
- **Criteo** → `Dataset/display advertising challenge dataset/` and **Avazu** →
  `Dataset/Click-Through Rate Prediction/` (optional, large — Colab/Kaggle only).
- **Yelp** → `Dataset/Yelp JSON Dataset1/` (optional text drift).
- **Elec2** is not on disk — load it via `river.datasets.Elec2()`.
- `river` synthetic generators and `Avalanche` splits are created in code, not files.

Verify each file loads before depending on it; if a file is compressed (`.7z`, `.gz`,
`.tar`, `.arff`), write the loader/extraction step and note it in `tech.md`.

---

## 9. Working agreement (how you operate)

- **Real runs only.** No invented metrics. If you haven't run it, it doesn't go in
  `results.md`. Mark anything provisional as provisional.
- **Determinism.** Set and log all seeds; every result must carry the config needed to
  reproduce it.
- **Small commits, current docs.** Each meaningful change is a commit that also updates
  the relevant living doc. Never leave the three docs stale.
- **Stay in scope but flag gold.** If you spot a genuine research or product upgrade
  beyond the current phase, note it in `weaknesses.md` (or a `docs/ideas.md`) instead
  of derailing — then return to the plan.
- **Ask when genuinely blocked** on a decision only I can make (e.g., a dataset license
  concern, a compute budget call, or a research-framing choice that changes the paper's
  main claim). Otherwise pick the reference's recommended default, record it in
  `decisions.md`, and keep moving.
- **Report honestly.** When something fails, say so with the evidence. When a
  hypothesis doesn't hold, treat it as a result, not an embarrassment.

---

## 10. Start here

1. Read both reference docs in full.
2. Confirm the local datasets load.
3. Draft the Phase 0 plan + the initial `decisions.md` and `tech.md` entries, then
   scaffold the repo.
4. Proceed phase by phase, gating as specified, running the weakness-audit loop after
   each phase, and keeping `decisions.md` / `tech.md` / `results.md` / `weaknesses.md`
   alive the whole way.

Build it so that, at the end, I can (a) write a main-track conference paper straight
from `results.md` + the paper skeleton, and (b) launch it as a real product. Both, not
either.
