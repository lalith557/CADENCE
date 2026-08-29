# CADENCE — Completion & Course-Correction Prompt for Claude Code

> Paste everything below the line into a fresh Claude Code session opened in
> `C:\Users\Lalith Gona\HTML\Capstone Project`. This **continues** an in-progress
> build — it does not restart it. Read the current state before changing anything.

---

## 0. Source of truth — read before touching code

- `CADENCE-master-reference.md` (spec), `CADENCE-datasets-verified.md` (data),
  `CADENCE-BUILD-PROMPT.md` (the original operating contract — **still in force**),
  and the current `docs/` (`decisions.md`, `tech.md`, `results.md`, `weaknesses.md`,
  `phase-plan.md`). The build is ~2.5 of 8 phases done; Phases 0–1 complete, Phase 2
  (RSO) built but its headline result is negative, Phase 3 (CDAG) partial with the GNN
  deferred, Phases 4–7 not started.
- The four living docs stay **live**: every change updates the relevant doc in the
  **same commit** as the code. Never batch docs to the end.

## Mission — what "done" means now (four things, none optional)

1. **Build every component fully** — GNN, surrogate, executor. No more deferrals.
   These are the novelty; a "structural proxy" or empty module is not the thing.
2. **Prove the anchor claim (the RSO) where the decision actually matters** — on
   scenarios where "do nothing" is provably wrong, not only easy ones where it wins.
3. **Validate on REAL drift by outcome, while KEEPING synthetic drift for attribution
   ground-truth.** Both, for two different claims (see §2). Never drop either.
4. **Every fallback mechanism implemented and tested, and the GPU genuinely used.**

**Guardrails (unchanged, non-negotiable):** real runs only — no fabricated or
extrapolated numbers; keep both synthetic and real validation; report negative results
honestly; commit at every gate; keep the four docs current.

---

## 1. IMMEDIATE — do this before any new feature

### 1a. Commit the existing work
~5,100 lines of code + every doc are **uncommitted** on top of a single
`Initial commit`. One accident loses all of it. First: add a `.gitignore` (exclude
`Dataset/`, `.venv/`, `mlruns/`, `__pycache__/`, `*.egg-info/`, experiment binaries
like `*.zip`), then commit the current tree in labelled commits (`scaffold`, `phase1`,
`phase2`, `phase3-partial`, `docs`). Do not proceed until there is real history.

### 1b. GPU — the accurate situation and the actual fixes
CUDA is **already working**: `torch 2.5.1+cu121`, `torch.cuda.is_available()==True`,
sees `NVIDIA GeForce RTX 4070 Laptop GPU` (8 GB). `cadence/adapters/neural.py` already
moves FraudNet to cuda. So the task is **not** "install CUDA." The real problems:
- The GPU-hungry components (**GNN, surrogate**) don't exist yet — so today the card is
  barely exercised and the CPU carries causal discovery + the data pipeline.
- PPO (SB3 `MlpPolicy`) runs on CPU — that's SB3's own recommendation for tiny MLP
  policies, but make the device **explicit and configurable**, and re-benchmark GPU vs
  CPU once the state/policy grows.
- **BUG:** `cadence/carbon/model.py` defaults `device_name="gtx-1650"` — wrong
  hardware, and it feeds the FLOPs→GPU-seconds→CO₂ model that drives the **RSO reward**.
  Fix to the actually-detected GPU, make it configurable, and re-run any result that
  depended on it.

Do all of the following:
1. Add `cadence/common/device.py` with one `get_device()` (config override → cuda if
   available → cpu) and a helper that logs device + VRAM. Route **all** new tensor code
   (GNN, surrogate, any batched scoring) through it. No new `nn.Module`/tensor is left
   on CPU by default.
2. Build the **GNN and surrogate on GPU from day one** (PyTorch Geometric matching
   cu121). These are what will actually use the card.
3. Add a `cadence gpu-check` CLI command: prints torch build, cuda availability, device
   name, free/total VRAM, and runs a 1-second cuda matmul to prove real execution.
4. **Prove utilization**, don't assume it: during any training run, assert the model's
   params are on cuda and log `torch.cuda.max_memory_allocated()` to MLflow **and** the
   `results.md` entry. A run that claims GPU must show bytes moved onto the card.
5. **8 GB VRAM guard:** cap batch sizes, call `torch.cuda.empty_cache()` +
   `gc.collect()` between sweep cells (fixes the W-26 mid-run crash), and if a
   GPU-configured run OOMs, **fail loudly** — never silently fall back to CPU.
6. **Profile the CPU hotspots** (causal discovery, PSI, windowing). Vectorize with
   numpy/torch; move NOTEARS's continuous optimization onto torch-GPU if it's a
   bottleneck. Record before/after wall-clock in `results.md`.

**Gate 1:** `cadence gpu-check` green; a FraudNet+GNN smoke run logs non-zero cuda
memory; carbon model uses the real GPU; everything committed.

---

## 2. Validation architecture — synthetic vs real (read carefully, don't conflate)

Two kinds of drift, two kinds of proof. Keep both.

- **SYNTHETIC drift (you inject a known cause)** → the ONLY place with an answer key →
  validates that attribution is **correct** (H1: root-cause precision/recall/AUROC, and
  Structural Hamming Distance of the recovered graph vs. the true graph). This is the
  novelty proof.
- **REAL drift (cause unknown)** → validates that the **system helps** (H2: performance
  recovery + compute/carbon saved; H3: forgetting). Graded by **outcome** — did the fix
  recover performance? — not by known cause.

CADENCE still *estimates* the cause on real models exactly as on synthetic; synthetic is
just where you can *check its homework*. On real data the outcome loop (did the targeted
retrain recover performance?) provides the check instead.

**Dataset → claim map** (all present in `Dataset/` unless noted):
- Synthetic injector on Credit Card Fraud / Give Me Some Credit → **H1** ground-truth.
- **Elec2** (`river.datasets.Elec2()`) and **Airlines** (`Dataset/Airlines*`) → **H2**
  real end-to-end recovery + cost. These are the canonical real concept-drift benchmarks.
- **Amazon Reviews 2023 (timestamped) / Yelp** (`Dataset/Yelp*`) → real vocabulary drift
  for a text-classifier generality experiment.
- **MNIST** (`Dataset/MNIST`) via **Avalanche** → **H3** forgetting.
- **Give Me Some Credit** → tree-adapter generality.

**"A real production model that drifts":** train the model on an EARLIER time slice, then
stream the LATER slice — it degrades on genuine distribution shift with no injected
cause. Run the full CADENCE loop on that and measure recovery + cost.

---

## 3. Build & fix sequence — each step ends in a real `R-<n>` and a commit

Order de-risks the anchor claim first, but completes every component. Run the
weakness-audit loop after each step.

**STEP A — Claim C rescue: make the RSO earn its keep** (fixes W-22/W-23/W-24/W-27)
- Add **contested-SLA scenarios**: post-drift F1 sits just BELOW the SLA so "do nothing"
  is provably wrong. Include one recoverable-by-partial, one that needs a full retrain,
  and one unrecoverable concept-flip — so each of the three actions is the correct choice
  somewhere.
- Put **EWC inside the sandbox "partial" action** (today it's a plain fine-tune, W-23).
- **Augmented-Lagrangian** dual-λ update (W-22) instead of fixed λ.
- Train PPO long enough to escape bang-bang (30k+ steps); enrich the state (drift
  duration, attribution concentration, SLA margin, time since last retrain).
- *Gate A (H2):* on contested scenarios, RSO beats periodic-retrain and reactive-full-
  retrain on compute/carbon at equal-or-better SLA recovery — ≥10 seeds × ≥5 scenarios,
  paired Wilcoxon p<0.05. If it still can't, say so honestly and we re-frame — but this
  is the step where the core claim must be won.

**STEP B — GNN (Claim A)**
- Build the real GraphSAGE/GAT encoder over the CDAG (PyTorch Geometric, GPU); replace
  the Phase-3 structural proxy. Measure SHD and attribution precision vs. PSI on
  synthetic ground-truth.
- *Gate B (H1):* CDAG+GNN attribution beats the correlational PSI baseline, with stats.

**STEP C — Surrogate + close the loop (Claim B)**
- Offline pipeline generating `(graph, node, intervention, measured post-fix F1)` triples;
  train GNN+surrogate **jointly on GPU**; produce responsibility scores; feed the RSO.
  Report calibration (predicted vs. measured F1).
- *Gate C:* one command runs CDAG → GNN → surrogate → RSO → executor → validation
  end-to-end; MLflow captures every stage.

**STEP D — Executor + every fallback (Claim C completion, H3)**
- Implement the Part-3B EWC executor and **all** fallbacks in §4. Redefine the "unrelated"
  segment as a genuinely disjoint sub-population (fixes W-21) for an honest forgetting test.
- *Gate D (H3):* partial retrains show less forgetting than naive full retrain on the
  disjoint segment — ≥10 seeds, p<0.05.

**STEP E — REAL-drift end-to-end (the thing you asked for)**
- Train real models on early slices of Elec2 / Airlines; stream the later slices; run the
  full loop; measure recovery + compute/carbon saved vs. baselines. Graded by outcome.
- *Gate E:* on ≥2 real-drift datasets, CADENCE recovers performance at lower total cost
  than periodic/reactive-full baselines — reported honestly, including where it doesn't.

**STEP F — Generality + robustness**
- Tree adapter (Give Me Some Credit); text drift (Amazon/Yelp); MNIST forgetting;
  noisy/partial-telemetry robustness; false-alarm rate.
- *Gate F:* generality supported by measured cross-model results.

**STEP G — Product + paper**
- Dashboard (Streamlit/React): live drift, the CDAG, responsibility scores, the decision,
  and cost/carbon saved — the "why it broke" view that differentiates from Evidently/
  WhyLabs. `docker-compose` (API + dashboard + MLflow); finalize `docs/product.md`;
  auto-populate the paper skeleton from `results.md`.
- *Gate G:* clean-clone quickstart works; paper skeleton has real results tables; final
  adversarial audit finds no blocker/major weakness.

---

## 4. Fallback mechanisms — implement AND test every one (Step D unless noted)

Each needs a test that forces the failure and checks the fallback fires:
1. **Shadow/canary validation** before promotion — score held-out recent traffic; promote
   only if F1 ≥ SLA target.
2. **Rollback on failure** — keep the old model serving; log; feed (predicted vs. actual)
   back to recalibrate the surrogate.
3. **Escalation ladder** — partial fails → try full retrain → still fails → flag a human;
   never silently serve a bad model.
4. **Diffuse-responsibility fallback** — when no single node dominates (responsibility
   spread above threshold τ), do a full retrain instead of a wrong targeted one (the open
   gap flagged in Part 3B).
5. **Unrecoverable-drift guard** — if repeated retrains don't recover (concept flip like
   v14), stop burning compute, flag "needs new labels / human," escalate. Turns the R-4
   v14 finding into a feature.
6. **Trigger false-alarm handling** — PSI fires but delayed-label F1 is fine → no-op +
   log; don't retrain on noise.
7. **Resource fallback** — OOM/timeout during retrain → abort cleanly, keep serving the
   old model, log; never crash the loop (fixes W-26).
8. **Cold-start / missing telemetry** — degrade to feature-only attribution if activations
   are unavailable; never hard-fail.

---

## 5. GPU acceptance criteria (explicit)
- `cadence gpu-check` confirms cuda + RTX 4070 + a real cuda matmul.
- GNN, surrogate, and FraudNet training run on cuda; each run logs
  `torch.cuda.max_memory_allocated() > 0` to MLflow and its `results.md` entry.
- `carbon/model.py` uses the detected GPU, not the `gtx-1650` default; every affected
  result re-run.
- A GPU-configured run that can't use the GPU **fails loudly** — never silently CPU.
- CPU hotspots profiled; before/after wall-clock recorded.

---

## 6. Definition of done
Every component built (no proxies/stubs left); **H1** on synthetic, **H2** on both
contested-synthetic and **real** drift, **H3** on a disjoint segment — all with ≥10 seeds
and paired stats; every fallback in §4 tested; GPU genuinely used and **proven** per §5;
dashboard + docker + paper skeleton live; the four living docs current; the weakness
register converged. End state: I can write a main-track-aimed paper straight from
`results.md`, and launch the product from the repo.

## Start here
1. Read the reference docs + the current `docs/` state.
2. Do §1 (commit + GPU) first; land Gate 1.
3. Then Steps A→G in order, gating each, running the weakness loop after each, keeping the
   four docs live and committing at every gate.
4. Never drop synthetic (attribution proof) or real (outcome proof). Never fabricate a
   number. When something fails, report it — a failure honestly recorded is progress.
