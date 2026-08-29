# Dataset/

Raw datasets are **not committed** — they're pulled per user from the sources listed in
`CADENCE-datasets-verified.md`. `.gitignore` explicitly excludes all data extensions here
so nothing leaks into the repo by accident.

Expected files after you set up the datasets:

| Path | Source |
|---|---|
| `Credit Card Fraud Detection/creditcard.csv` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| `Give Me Some Credit/cs-training.csv` | https://www.kaggle.com/c/GiveMeSomeCredit |
| `Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv` | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| `adult/adult.data` | https://archive.ics.uci.edu/dataset/2/adult |
| `MNIST/*` (or via `torchvision.datasets.MNIST` download) | http://yann.lecun.com/exdb/mnist/ |

Run `cadence datasets-check` to verify.
