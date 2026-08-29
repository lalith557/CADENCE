# CADENCE — Complete Dataset List (Verified Links)

Organized by what each dataset validates, since CADENCE is designed to detect drift across *different types* of production models, not just one. All links re-verified via search before this file was written.

---

## 1. Production model datasets (one per model type CADENCE needs to support)

CADENCE's Model Adapter layer needs to be validated against more than one kind of production model to credibly claim "works across model types." Use one dataset per adapter below.

**Neural network classifier adapter**
**Credit Card Fraud Detection Dataset** — 284,807 European credit card transactions, 492 labeled fraud (~0.17%), features `Time`, `Amount`, PCA-transformed `V1`–`V28`.
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Note:** free Kaggle account required to download. Published by the Machine Learning Group at Université Libre de Bruxelles (ULB).

**Tree ensemble adapter (XGBoost/LightGBM)**
**Give Me Some Credit** — ~150,000 borrower records; predicts probability of serious financial delinquency within two years. A standard benchmark for tree-based credit-risk models.
- **Link:** https://www.kaggle.com/c/GiveMeSomeCredit

**Linear/logistic regression adapter**
**Telco Customer Churn** — 7,043 customer records; predicts churn from account/service/demographic features. Commonly used for interpretable linear/logistic churn models.
- **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Linear/logistic regression adapter (alternative)**
**UCI Adult (Census Income)** — ~48,842 records; predicts whether income exceeds $50K/year from demographic/employment features. A long-standing standard benchmark for linear classifiers.
- **Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Note:** easiest loaded via the official `ucimlrepo` Python package (`pip install ucimlrepo`), which fetches this exact dataset by ID.

---

## 2. Synthetic drift generators (for controlled, ground-truth-known drift experiments)

**`river`** — Python library with built-in synthetic drift generators (SEA, STAGGER, Hyperplane, RandomRBF, Agrawal).
- **Main site:** https://riverml.xyz
- **Synthetic datasets module docs:** https://riverml.xyz/latest/api/datasets/synth/   (link isnt valid)
- **GitHub:** https://github.com/online-ml/river

**`scikit-multiflow`** — older alternative library with similar generators.
- **Link:** https://scikit-multiflow.github.io/
- **GitHub:** https://github.com/scikit-multiflow/scikit-multiflow         (in river found)

---

## 3. Real-world datasets with known genuine drift

**Electricity (Elec2)** — the standard benchmark dataset in the concept-drift research literature.
- **Link (OpenML):** https://www.openml.org/search?type=data&status=active&id=151
- **Note:** also loadable directly via `river.datasets.Elec2()` without a manual download.

**Airlines delay dataset** — flight delay records with genuine temporal drift.
- **Link (OpenML):** https://www.openml.org/search?type=data&status=active&id=1169

**Criteo Display Advertising dataset** (optional, large-scale)
- **Official source:** https://ailab.criteo.com/ressources/      (4 option)            (dataset is available but cant figure out which one is it exactly)
- **Kaggle (smaller, more manageable subset):** https://www.kaggle.com/c/criteo-display-ad-challenge/data

**Avazu CTR Prediction dataset** (optional, alternative to Criteo)
- **Link:** https://www.kaggle.com/c/avazu-ctr-prediction/data

---

## 4. Continual-learning / forgetting benchmarks

**MNIST** — official source for building Split-MNIST/Rotated-MNIST splits.
- **Official link:** http://yann.lecun.com/exdb/mnist/                         (dataset isnt available)
- **Note:** also directly downloadable via `torchvision.datasets.MNIST`.

**CIFAR-10** — official source for building Split-CIFAR-10.
- **Official link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Note:** also directly downloadable via `torchvision.datasets.CIFAR10`.

**Avalanche** — the standard library for building CL benchmark splits from the raw datasets above.
- **Link:** https://avalanche.continualai.org/                                 (dataset isnt properly mentioned anywhere)
- **GitHub:** https://github.com/ContinualAI/avalanche

---

## 5. Text/NLP generality dataset (optional)

**20 Newsgroups** — built directly into `scikit-learn`, no separate download needed.
- **Docs/link:** https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset       (present in skicit-learn)

**Amazon Reviews (2023 release, with timestamps)** — for topic/vocabulary-shift-over-time experiments.
- **Link:** https://amazon-reviews-2023.github.io/

**Yelp Open Dataset** — alternative for text-drift experiments.
- **Link:** https://www.yelp.com/dataset

---

## 6. Real incident case studies (qualitative, for replayed real-world incidents)

**AI Incident Database** — searchable, maintained database of real AI/ML failure incidents.
- **Link:** https://incidentdatabase.ai/

---

## 7. Carbon/cost estimation (for the RL reward's carbon term)

**Electricity Maps API** — real-time grid carbon-intensity data by region.
- **Link:** https://www.electricitymaps.com/
- **Note:** free developer tier available; requires sign-up for an API key.

**CodeCarbon** — Python package estimating carbon footprint from your own local compute usage.
- **Link:** https://codecarbon.io/
- **GitHub:** https://github.com/mlco2/codecarbon

---

## Quick-reference table

| # | Dataset/Tool | Validates | Link |
|---|---|---|---|
| 1 | Credit Card Fraud Detection | Neural network adapter | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| 2 | Give Me Some Credit | Tree ensemble adapter | https://www.kaggle.com/c/GiveMeSomeCredit |
| 3 | Telco Customer Churn | Linear/logistic adapter | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| 4 | UCI Adult (Census Income) | Linear/logistic adapter (alt.) | https://archive.ics.uci.edu/dataset/2/adult |
| 5 | `river` (synthetic generators) | Ground-truth drift injection | https://riverml.xyz/latest/api/datasets/synth/ |
| 6 | `scikit-multiflow` | Alternative synthetic generators | https://scikit-multiflow.github.io/ |
| 7 | Elec2 | Real known-drift benchmark | https://www.openml.org/search?type=data&status=active&id=151 |
| 8 | Airlines | Real known-drift benchmark | https://www.openml.org/search?type=data&status=active&id=1169 |
| 9 | Criteo Display Ads | Large-scale real CTR data (optional) | https://ailab.criteo.com/ressources/ |
| 10 | Avazu CTR | Large-scale real CTR data (optional) | https://www.kaggle.com/c/avazu-ctr-prediction/data |
| 11 | MNIST | Base data for CL/forgetting tests | http://yann.lecun.com/exdb/mnist/ |
| 12 | CIFAR-10 | Base data for CL/forgetting tests | https://www.cs.toronto.edu/~kriz/cifar.html |
| 13 | Avalanche | Builds CL benchmark splits | https://avalanche.continualai.org/ |
| 14 | 20 Newsgroups | Text-drift generality (optional) | https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset |
| 15 | Amazon Reviews 2023 | Text-drift generality (optional) | https://amazon-reviews-2023.github.io/ |
| 16 | Yelp Open Dataset | Text-drift generality (optional) | https://www.yelp.com/dataset |
| 17 | AI Incident Database | Real incident case studies | https://incidentdatabase.ai/ |
| 18 | Electricity Maps | Carbon-cost estimation | https://www.electricitymaps.com/ |
| 19 | CodeCarbon | Local carbon-footprint estimation | https://codecarbon.io/ |

---

## Where to start, given your hardware (GTX 1650)

Start with **#1 (Credit Card Fraud)** for the neural-network adapter, **#5 (`river`)** for controlled drift injection, and **#7 (Elec2)** for a real-world validation set — all three are small and CPU/GPU-light. Once the core pipeline works end-to-end on the neural adapter, add **#2 (Give Me Some Credit)** to build and prove out the tree-ensemble adapter — this is the single most important "generality" experiment for the paper, since it's what lets you credibly claim CADENCE isn't hardcoded to neural networks. Add **#13 (Avalanche + MNIST)** for the forgetting experiments once that's stable. Save #9/#10 (Criteo/Avazu) and full-size CIFAR-10 runs for free Colab/Kaggle GPU sessions — optional strengthening, not required for the core claims.
