# CADENCE — Secrets & Configuration

CADENCE reads no secret at import time; every integration is opt-in via
environment variables. See `.env.example` for the full list.

## What the code actually needs

| Variable | Used by | Effect if unset |
|---|---|---|
| `ELECTRICITY_MAPS_API_KEY` | `cadence.carbon.tracker` | Falls back to closed-form global-average grid intensity from `cadence.carbon.model.GridProfile`. |
| `CODECARBON_OUTPUT_DIR` | `cadence.carbon.tracker` | CodeCarbon writes `emissions.csv` in the current working directory. |
| `MLFLOW_TRACKING_URI` | `cadence.common.tracking` | Uses whatever `configs/*.yaml` says. Default is `sqlite:///experiments/mlflow.db`. |
| `MLFLOW_ALLOW_FILE_STORE` | `cadence.common.tracking` (import-time) | Set by the module itself to preserve any historical file-store URI. Only override if you know why. |
| `CADENCE_API_ADMIN_TOKEN` | `cadence.api.server` (`/admin/*` routes) | Admin routes 401. |
| `TF_ENABLE_ONEDNN_OPTS` | tensorflow (dep of mlflow) | oneDNN warnings on Windows. Cosmetic. |
| `CUDA_VISIBLE_DEVICES` | torch | Empty string forces CPU (useful for CI runners without GPU). |

## Local dev

```bash
cp .env.example .env
# edit .env with your actual keys
# dotenv-loading is opt-in; either export by hand or use `dotenv run -- python -m ...`
```

## Docker

`docker-compose.yml` reads no secret. If you need one, add it via
`env_file: .env` or `environment:` on the service.

## CI

The GitHub Actions workflow at `.github/workflows/ci.yml` runs the unit
tests only. It needs no secret; the paid API keys are for the runtime
integrations, not the test suite.

## Rotation policy

- `ELECTRICITY_MAPS_API_KEY`: rotate whenever Electricity Maps rotates
  yours; the free tier is per-org.
- `CADENCE_API_ADMIN_TOKEN`: quarterly. Treat as a production credential.

## What CADENCE will never do

- Log any of these values to MLflow tags/params.
- Include them in `experiments/*.json` artifacts.
- Ship them in `docker/Dockerfile*` layers.

If you catch a secret leaking anywhere in the repo, that's a P0 — file a
weakness entry (W-N) and rotate.
