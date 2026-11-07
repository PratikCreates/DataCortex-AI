# DataCortex-AI

DataCortex-AI is a Python prototype for an automated data-science workflow. It ingests tabular data, computes a data fingerprint, records phase artifacts, trains deterministic scikit-learn style model candidates, and exposes the workflow through a FastAPI gateway and a static frontend.

The repository currently provides local laboratory validation. It should not be described as production-ready, enterprise-grade, or state-of-the-art without additional deployment, security, and benchmark evidence.

## What Works

- CSV, TSV, JSON, Excel, and Parquet ingestion through `IngestionAgent`.
- Schema and data-quality artifact registration.
- Data fingerprinting for sparsity, type mix, correlation, target type, class balance, outliers, and suggested model families.
- Local model training fallback using scikit-learn models when remote sandbox execution is unavailable.
- Static AST validation for generated Python code before execution.
- Checkpoint and artifact serialization for pipeline resumability.
- FastAPI routes for analysis, status, reports, artifacts, events, approvals, and WebSocket event replay.
- Reproducible local validation via pytest, `scripts/validate_claims.py`, and `scripts/run_comparative_benchmarks.py`.

## Installation

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional packages such as XGBoost, LightGBM, CatBoost, Blaxel SDK, and Kaggle tooling are used only when installed or configured.

## Run Tests

```bash
pytest -q
```

Current validation on this workspace:

```text
74 passed
```

## Run the Demo

```bash
python scripts\run_demo.py
```

This runs the real planner pipeline against `tests/fixtures/iris.csv` with remote LLM and sandbox calls disabled by default for repeatability. It writes:

- `demo_outputs/demo_report.md`
- `demo_outputs/demo_events.json`
- `demo_outputs/demo_summary.json`

The last local demo completed all five phases: ingestion, profiling, EDA, modeling, and reporting.

Demo-facing follow-up work is tracked in `DEMO_TODO.md`, including multi-file dataset visualization, plan modes, market-basket analysis, and accessible live training progress.

## Run Local Validation

```bash
python scripts\validate_claims.py
```

The script runs local datasets from `tests/fixtures` plus any available CSVs under `benchmarks/kaggle`, then rewrites `VALIDATION_RESULTS.md`.

Latest local run completed 7 datasets and skipped HR Analytics because the configured target column was not present in the discovered CSV. Summary:

| Dataset | Best Model | Score | Metric |
|---------|------------|-------|--------|
| Iris | RandomForestClassifier | 0.9667 | accuracy |
| Churn | LGBMClassifier | 0.9140 | accuracy |
| Credit Risk | RandomForestClassifier | 0.9187 | accuracy |
| Housing | LinearRegression | 1.0000 | r2 |
| Hard Benchmark | XGBClassifier | 0.7300 | accuracy |
| Titanic | RandomForestClassifier | 0.8384 | accuracy |
| Airline Satisfaction | LGBMClassifier | 0.9600 | accuracy |

These are local benchmark results, not external leaderboard claims.

## Run Executable Comparisons

```bash
python scripts\run_comparative_benchmarks.py
```

This benchmark compares dummy, linear, fixed tree, fixed boosting, DataCortex fingerprint-selected candidates, and a broad local sklearn-style model pool on the same datasets, folds, preprocessing, and metrics. Current summary:

| Metric | Result |
|--------|--------|
| Datasets compared | 6 |
| Mean lift over dummy baselines | 0.3857 |
| Mean gap to broad local pool | 0.0042 |
| Mean DataCortex search-budget ratio | 0.672 |

Outputs:

- `COMPARATIVE_BENCHMARKS.md`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_summary.csv`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_long.csv`
- `disclosure_artifacts/figures/figure_8_executable_model_comparison.png`

## Run the API

```bash
python -m src.api.gateway
```

Then open `http://127.0.0.1:8000`.

Main endpoints:

- `GET /health`
- `POST /analyze`
- `GET /status/{session_id}`
- `GET /report/{session_id}`
- `GET /artifacts/{session_id}`
- `GET /events/{session_id}`
- `WS /ws/{session_id}`

## Configuration

Environment variables are loaded from `.env` when present.

- `NVIDIA_API_KEY`
- `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_2`, ...
- `BLAXEL_ACCOUNT_<n>_API_KEY`
- `BLAXEL_ACCOUNT_<n>_WORKSPACE_NAME`
- `BLAXEL_ACCOUNT_<n>_BUDGET`

Remote providers are opt-in for demos and tests:

- `DATACORTEX_ENABLE_REMOTE_LLM=1` enables remote LLM narration and adaptive model mutation.
- `DATACORTEX_ENABLE_REMOTE_SANDBOX=1` enables remote sandbox execution before local fallback.

With both unset, the project runs deterministic local analysis suitable for demos.

## Validation Scope and Limits

- Static code validation blocks known dangerous imports, builtins, dunder traversal, and direct filesystem mutation calls, but it is not a complete sandbox by itself.
- Local validation checks failure paths such as malformed code, suspicious data mutation, single-class targets, and insufficient modeling inputs.
- Benchmark fixtures are useful regression tests, but they do not prove general real-world reliability.
- Comparative benchmarks use local sklearn-style baselines and a broad local model pool, not installed TPOT, AutoGluon, or auto-sklearn runs.
- The FastAPI service has local workflow coverage, but production deployment hardening remains out of scope for the current evidence.

## Change Log

- Fixed numeric target detection for small binary and multiclass datasets.
- Hardened correlation fingerprinting against all-NaN correlation matrices.
- Hardened generated-code validation against `pathlib`, dunder traversal, and filesystem mutation methods.
- Fixed an untested `ModelingAgent` audit-log runtime bug.
- Added local-modeling regression tests for normal training and single-class failure handling.
- Fixed Windows console encoding for validation runs.
- Rewrote validation and README language to reflect reproducible local evidence instead of unsupported marketing claims.
- Added `scripts/run_demo.py` and a planner-level demo regression test.
- Made remote LLM and sandbox execution opt-in for reliable local demos.
- Added Kaggle E2E validation for Titanic classification and Groceries market-basket analysis.
- Added market-basket detection, duplicate-transaction preservation, and report language that distinguishes sparse item slots from conventional missing values.
- Added executable comparative benchmarks and updated fingerprint model selection to include histogram gradient boosting candidates where appropriate.
