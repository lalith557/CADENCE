# CADENCE — Technology Stack Rationale

Every library, framework, model family, and infra choice gets one entry.

Format:
```
### <tech> — used for <what>
- **Why this:** concrete fit for CADENCE
- **Alternatives:** realistic options compared
- **Why not the alternatives:** specific, not hand-wavy
- **Version pinned:** exact version, and why (if it matters)
- **Risk / lock-in:** how hard to swap later
```

---

### Python 3.10 — the base runtime
- **Why this:** Broadest wheel availability for the ML stack we need (torch, torch-geometric, causal-learn, stable-baselines3, avalanche-lib). Locally available on the dev machine (`C:\Users\Lalith Gona\AppData\Local\Programs\Python\Python310\python.exe`).
- **Alternatives:** Python 3.13 (default install on this box), Python 3.11.
- **Why not the alternatives:** 3.13 has spotty wheels for torch-geometric, gcastle, and older causal-learn releases. 3.11 works but is not what the dev box has pre-configured; adds one more thing to install.
- **Version pinned:** 3.10.11 for local dev; CI matrix pins python-version: "3.10".
- **Risk / lock-in:** Moderate. Upgrading later means re-locking all deps.

### PyTorch 2.x — deep-learning framework
- **Why this:** FraudNet + GNN + PPO policy net all use it. First-class GPU support on the RTX 4070 (CUDA 12.1 wheels). Native support in PyTorch Geometric, Stable-Baselines3, Avalanche, CodeCarbon.
- **Alternatives:** JAX, TensorFlow.
- **Why not the alternatives:** JAX would require rewriting PyG, Avalanche, and SB3 usage — none have native JAX equivalents that ship the exact algorithms CADENCE uses. TF2 is the same story with older tooling.
- **Version pinned:** torch>=2.2,<2.6 (avoid the very-latest ABI churn that PyG lags behind).
- **Risk / lock-in:** High-value lock-in; PyTorch is central to the whole system.

### PyTorch Geometric — GNN over the CDAG
- **Why this:** Exactly the library the reference names. Ships GraphSAGE, GAT, GCN with matching APIs so ablations are one-line config swaps.
- **Alternatives:** DGL (Deep Graph Library), Spektral (TF-based).
- **Why not the alternatives:** DGL is fine but its API is farther from the reference; Spektral needs TF. PyG is the standard for causal-graph work in the papers we're competing with.
- **Version pinned:** torch-geometric>=2.4,<2.7 with matched torch-scatter/torch-sparse wheels.
- **Risk / lock-in:** Moderate — PyG APIs occasionally break.

### scikit-learn + LightGBM — non-neural surrogate + tree-adapter
- **Why this:** scikit-learn for k-means (D-2), PSI baseline, KS test, standard preprocessing. LightGBM for the tree-ensemble adapter (Give Me Some Credit) — the generality experiment that lets us claim CADENCE isn't neural-only.
- **Alternatives:** XGBoost, CatBoost.
- **Why not the alternatives:** LightGBM is faster on CPU-bound tabular and has cleaner sklearn-compatible API; XGBoost is a legitimate alternative and kept as a swap target if needed. CatBoost brings its own quirks we don't need.
- **Version pinned:** scikit-learn>=1.4,<1.6; lightgbm>=4.1,<4.6.
- **Risk / lock-in:** Low — trivially swappable.

### Stable-Baselines3 — PPO implementation for the RSO
- **Why this:** Battle-tested PPO, easy custom `gym.Env` integration, MLflow-compatible logging, actively maintained.
- **Alternatives:** RLlib (Ray), CleanRL, PPO from scratch.
- **Why not the alternatives:** RLlib pulls in Ray as a huge dependency for what is a single-node, single-agent problem. CleanRL is educational; SB3 has better logging out-of-the-box. Writing PPO from scratch = time sink with no research payoff.
- **Version pinned:** stable-baselines3>=2.2,<2.5.
- **Risk / lock-in:** Low; the RSO's `gym.Env` interface is portable to RLlib later if we scale.

### causal-learn + gcastle + our own NOTEARS wrapper — hybrid PC+NOTEARS
- **Why this:** causal-learn ships a battle-tested PC implementation; gcastle ships a NOTEARS implementation. Both actively maintained. Using both lets us pick the better implementation per stage.
- **Alternatives:** dowhy (missing PC/NOTEARS), econml (effect estimation, not discovery), pycausal (JVM wrapper, dead-slow install on Windows).
- **Why not the alternatives:** dowhy/econml are effect-estimation, not structure learning. pycausal is a maintenance liability on Windows.
- **Version pinned:** causal-learn>=0.1.3; gcastle>=1.0.4.
- **Risk / lock-in:** Low — our CDAG builder wraps them behind a small interface.

### river — synthetic drift generators + Elec2 loader
- **Why this:** The reference names it explicitly. Ships SEA, STAGGER, Hyperplane, RandomRBF, Agrawal generators AND `river.datasets.Elec2()` so we don't have to hand-download Elec2.
- **Alternatives:** scikit-multiflow (older, more or less abandoned).
- **Why not the alternatives:** river is the modern successor and shares maintainers.
- **Version pinned:** river>=0.21,<0.24.
- **Risk / lock-in:** Low.

### avalanche-lib — Split-MNIST / continual-learning benchmarks
- **Why this:** Standard library for CL benchmark construction. Needed for H3's forgetting experiments (Phase 6).
- **Alternatives:** continuum, mammoth.
- **Why not the alternatives:** Avalanche is the most active and has the exact `SplitMNIST` / `PermutedMNIST` benchmarks Section 14 calls out.
- **Version pinned:** avalanche-lib>=0.5,<0.7.
- **Risk / lock-in:** Low — used only in the forgetting experiments.

### MLflow — experiment tracking
- **Why this:** Reference names it. Local file backend needs zero infra. Handles config hashes, git SHA, artifacts, metrics.
- **Alternatives:** Weights & Biases, Aim.
- **Why not the alternatives:** W&B requires an account + cloud dependency, uncomfortable for a project with a patent story attached. Aim is nice but smaller ecosystem.
- **Version pinned:** mlflow>=2.10,<2.20.
- **Risk / lock-in:** Low — the tracking wrapper isolates us from MLflow API drift.

### FastAPI + uvicorn — inference/API surface
- **Why this:** Simple to serve the demo dashboard's backing endpoints; ships JSON schemas + OpenAPI for free; Docker-friendly.
- **Alternatives:** Flask, Litestar, gRPC.
- **Why not the alternatives:** Flask lacks native async. Litestar is fine but less mainstream — not worth the mindshare cost. gRPC is overkill.
- **Version pinned:** fastapi>=0.110,<0.116.
- **Risk / lock-in:** Low.

### CodeCarbon — carbon estimation for the RL reward
- **Why this:** Reference names it. Runs offline, no API key. Reports kg CO2e per tracked block.
- **Alternatives:** Electricity Maps API, `carbontracker`.
- **Why not the alternatives:** Electricity Maps requires an API key and network access — kept as a live-mode fallback but not the default. carbontracker is close but less integrated with PyTorch.
- **Version pinned:** codecarbon>=2.3,<2.9.
- **Risk / lock-in:** Low.

### structlog — logging
- **Why this:** Structured JSON logs pair with MLflow + the audit-log-for-compliance story from the product doc.
- **Alternatives:** stdlib logging, loguru.
- **Why not the alternatives:** stdlib is fine but JSON handling is fiddly. loguru is nice but its API is idiosyncratic and hard to shim later.
- **Version pinned:** structlog>=24.1,<25.
- **Risk / lock-in:** Low.

### pydantic v2 + pydantic-settings — config models
- **Why this:** Typed, validated config from YAML/env, plays well with FastAPI and CI.
- **Alternatives:** OmegaConf/Hydra, dataclasses, attrs.
- **Why not the alternatives:** Hydra is powerful but heavy and its CLI muddies our own CLI. Dataclasses lack validation.
- **Version pinned:** pydantic>=2.6,<3; pydantic-settings>=2.2,<3.
- **Risk / lock-in:** Low.

### pytest + pytest-cov — testing
- **Why this:** Standard. Fixtures fit our data-loader + sandbox-env style.
- **Alternatives:** unittest.
- **Why not the alternatives:** Fewer plugins, uglier fixtures.
- **Version pinned:** pytest>=8,<9.
- **Risk / lock-in:** None.

### ruff + mypy — lint + type-check
- **Why this:** ruff is 10-100× faster than flake8/black together, handles both lint and format. mypy for static types on module boundaries.
- **Alternatives:** flake8, pylint, black, isort.
- **Why not the alternatives:** ruff subsumes all of them.
- **Version pinned:** ruff>=0.4,<0.9; mypy>=1.9,<2.
- **Risk / lock-in:** Low.

### Docker (python:3.10-slim base) — reproducibility
- **Why this:** Section 15 demands it for the paper's reproducibility appendix.
- **Alternatives:** Conda envs only, Nix.
- **Why not the alternatives:** Conda alone doesn't pin OS-level libs. Nix is beautiful but a steep learning curve for a solo build.
- **Version pinned:** python:3.10-slim-bookworm.
- **Risk / lock-in:** Low.

### GitHub Actions — CI
- **Why this:** Free for public repos, matrix over Python versions, native to the GitHub PR flow.
- **Alternatives:** CircleCI, GitLab CI.
- **Why not the alternatives:** Neither is worth the config overhead for a single repo.
- **Version pinned:** ubuntu-latest runners.
- **Risk / lock-in:** Low.

### Streamlit (default) / React + Recharts (upgrade path) — dashboard
- **Why this:** Streamlit ships a working dashboard in one Python file — good enough for the demo and for reviewers to run. React is the upgrade path once a product-quality UI matters (Phase 7).
- **Alternatives:** Grafana+Prometheus, plain matplotlib+Flask.
- **Why not the alternatives:** Grafana requires standing up Prometheus; nice for production, over-scope for the demo. Matplotlib alone doesn't feel like a product.
- **Version pinned:** streamlit>=1.32,<1.42.
- **Risk / lock-in:** Low — dashboard is decoupled.
