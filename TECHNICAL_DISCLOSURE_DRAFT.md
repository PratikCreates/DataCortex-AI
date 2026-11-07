# Technical Disclosure Draft: DataCortex-AI

## Title

**Systems and Methods for Fingerprint-Guided Multi-Agent Automated Data Science with Cross-Phase Artifact Propagation, Data-Integrity Gating, and Pre-Execution Code Safety Validation**

## Abstract

A computer-implemented automated data-science system receives a tabular dataset, computes a structured data fingerprint, and uses the fingerprint to select an execution plan and bounded candidate model families before model training. The fingerprint includes dataset shape, feature-type distribution, missingness, duplicate-row ratio, correlation metrics, target characteristics, class balance, outlier metrics, noise estimate, and suggested task type. Lifecycle phases emit structured artifacts into a phase-indexed artifact registry, allowing downstream agents to consume upstream schema, quality, cleaning, fingerprint, exploratory-analysis, modeling, and report artifacts. The system compares fingerprints across data mutations to detect row-loss, sparsity, statistical, or target-type anomalies. Generated analysis code is parsed into an abstract syntax tree and rejected when prohibited imports, dynamic execution, filesystem mutation, network access, or object-introspection escape patterns are detected. The system supports checkpointed execution, live progress events, deterministic local modeling fallback, and reproducible validation over benchmark datasets.

## Technical Field

The invention relates to automated data science, automated machine learning, generated-code validation, multi-agent workflow orchestration, data-quality monitoring, checkpointed workflow execution, and reproducible machine-learning evaluation for tabular datasets.

### Suggested IPC/CPC Classification

| Code | Relevance |
| --- | --- |
| G06N 20/00 | Machine learning systems and automated model selection |
| G06F 16/00 | Information retrieval and structured data management |
| G06F 11/36 | Program testing, validation, and execution safety checks |
| G06F 21/56 | Protection against unsafe or unauthorized code behavior |
| G06Q 10/06 | Workflow management and process monitoring |
| G06F 9/54 | Inter-module communication and event-driven workflow orchestration |

## Background

Automated data-science systems commonly rely on broad model and hyperparameter search. Such systems can spend substantial compute on unsuitable candidate models before using the statistical properties of the input dataset. Generative analysis agents can produce executable code, but generated code can be unsafe or unreliable when executed without deterministic validation, provenance tracking, and post-transformation data-integrity checks.

Conventional workflows often treat ingestion, profiling, exploratory analysis, modeling, and reporting as loosely coupled stages. Information from one stage may be discarded, converted into unstructured text, or passed forward through fragile shared state. Weak phase coupling makes it harder to detect silent row loss, target-column mutation, sparsity drift, inappropriate target inference, or unsafe generated code before execution.

There is a need for a technical system that:

1. Characterizes a dataset before selecting a modeling or analysis path.
2. Selects a bounded execution plan and candidate model families using the characterization.
3. Propagates structured artifacts across lifecycle phases.
4. Detects harmful data mutations introduced by automated cleaning or transformation.
5. Validates generated code before execution.
6. Records checkpoints, progress events, and reproducible validation evidence.

## Summary of the Invention

DataCortex-AI provides a multi-agent automated data-science workflow in which an input dataset is parsed, profiled, fingerprinted, analyzed, optionally modeled, and synthesized into a report. The data fingerprint is used as an execution-control signal rather than merely descriptive metadata. The system uses the fingerprint to select a task type, candidate model families, whether supervised modeling should be run, and whether special handling is required for sparse transaction-style data such as market-basket item-slot datasets.

In one implementation, the system includes:

- An ingestion agent that parses CSV, TSV, JSON, Excel, and Parquet files, detects schema, assesses data quality, performs conservative cleaning, guesses a target column only when supported by dataset structure, and computes a data fingerprint.
- A data fingerprinter that computes shape, type distribution, sparsity, duplicate ratio, correlation density, maximum correlation, skewness, outlier ratio, class balance, target type, noise estimate, feature redundancy, suggested task type, suggested model families, and memory footprint.
- A market-basket detector that identifies sparse item-slot transaction tables, prevents unsafe supervised target inference, preserves repeated basket rows as frequency signal, and routes the workflow to basket profiling rather than classification or regression.
- A planner agent that constructs an execution plan from the fingerprint and target-column state.
- An artifact registry that stores phase-indexed artifacts and exposes bounded upstream context to downstream phases.
- A data-integrity gate that compares original and modified fingerprints and emits anomalies for row-loss, sparsity, statistical, and target-type drift.
- A code validator that parses generated Python into an abstract syntax tree and blocks prohibited imports, dangerous builtins, dunder traversal, networking modules, filesystem mutation methods, and runtime injection patterns.
- A modeling agent that trains deterministic local candidates or configured remote sandbox candidates and records model comparisons, metrics, and feature importance.
- A checkpoint manager that persists phase state, execution plans, metadata, metrics, and serialized artifacts.
- An event stream that emits phase start, phase completion, integrity, plan, and pipeline completion events for a user interface or monitoring client.
- A validation harness that evaluates task detection, safety validation, data-integrity checks, and modeling behavior over repository fixtures and available benchmark datasets.

## Inventive Concepts

### 1. Fingerprint-Guided Execution Control

The system computes a structured fingerprint before modeling. The fingerprint is used to determine the execution path and candidate model families. For example, a continuous target routes toward regression models, a binary or multiclass target routes toward classification models, imbalanced classes change scoring and model preferences, high dimensionality can introduce support-vector candidates, high correlation density can prioritize regularized linear models, and market-basket structure routes away from supervised modeling.

### 2. Phase-Indexed Artifact Propagation

Each lifecycle phase emits structured artifacts to an artifact registry indexed by phase and artifact type. Downstream phases retrieve bounded upstream context from the registry. This replaces informal shared state with auditable data dependencies between ingestion, profiling, EDA, modeling, evaluation, and reporting.

### 3. Fingerprint Consistency Gate

The system compares fingerprints before and after data mutation. Anomalies are generated when row loss, sparsity drift, statistical drift, or target-type mutation exceeds configured thresholds. This protects the workflow from silent damage introduced by automatic cleaning or transformation.

### 4. Deterministic Pre-Execution Code Safety Validation

Generated Python code is parsed using the Python AST module before execution. The validator rejects prohibited imports, dynamic execution functions, network access, filesystem mutation, dunder traversal, and related escape patterns. This is deterministic static screening and does not rely on an LLM judgment alone.

### 5. Market-Basket-Aware Cleaning and Planning

The system distinguishes conventional duplicate records from repeated transactions in sparse item-slot market-basket data. For such datasets, duplicate rows are preserved because they encode frequency. Blank item slots are treated as absent items rather than imputation targets. The planner skips supervised model training unless an explicit target is provided.

### 6. Checkpointed and Observable Pipeline Execution

The workflow emits live structured events for phase starts, completions, integrity status, plan generation, and pipeline completion. Checkpoints persist phase status, current phase, execution plan, file metadata, target column, metrics, and serialized artifacts for inspection and resumability.

## Concrete Novelty Matrix

| Novelty | Concrete Mechanism | Technical Effect | Evidence |
| --- | --- | --- | --- |
| Fingerprint-guided execution control | Compute a structured fingerprint before model training and use it to choose task, plan, and bounded model families | Reduces blind model search and makes the selected path auditable | `src/core/data_fingerprint.py`, `src/agents/planner.py`, `VALIDATION_RESULTS.md` |
| Phase-indexed artifact propagation | Store schema, quality, cleaning, fingerprint, EDA, modeling, and report artifacts by phase and artifact type | Preserves cross-agent provenance and downstream context | `src/core/artifact_registry.py`, demo checkpoint JSON files |
| Fingerprint consistency gate | Compare fingerprints across cleaning and transformation boundaries | Detects silent row loss, sparsity drift, statistical drift, and target mutation | `src/core/data_fingerprint.py`, `src/agents/planner.py` |
| Deterministic generated-code safety validation | Parse generated code into an AST and reject prohibited imports, builtins, networking, file mutation, and dunder traversal | Blocks common unsafe generated-code paths before execution | `src/core/sandbox_executor.py`, `tests/test_code_validator.py` |
| Market-basket-aware branch | Detect sparse item-slot transaction tables, suppress unsafe target inference, preserve duplicate baskets, and skip supervised modeling absent a target | Prevents transaction-frequency loss and avoids bogus classification reports | Kaggle Groceries E2E output, `src/agents/ingestion_agent.py` |
| Observable checkpointed pipeline | Emit structured events and persist phase status, plan, artifacts, and metrics | Makes long-running automated analysis inspectable and resumable | `src/agents/planner.py`, `src/core/checkpoint.py`, demo event JSON files |

## Detailed Description

### System Components

#### Ingestion Agent

The ingestion agent receives file bytes and a filename, parses the file into a dataframe, detects schema, assesses data quality, optionally detects a target column, performs conservative cleaning, computes the data fingerprint, registers artifacts, and stores a cleaned CSV representation for downstream phases.

The agent supports CSV, TSV, JSON, Excel, and Parquet inputs. Schema artifacts include column names, dtypes, unique counts, null counts, null percentages, top categorical values, numeric summaries, and semantic type estimates. Quality artifacts include missing-cell counts, duplicate-row counts, duplicate interpretation, column-level issues, and an overall quality score.

#### Data Fingerprinter

The data fingerprinter computes a structured fingerprint containing:

- Row count, column count, dimensionality ratio, and memory footprint.
- Counts of numeric, categorical, temporal, and text-like columns.
- High-cardinality and text-column indicators.
- Sparsity and duplicate-row ratio.
- Correlation density, maximum correlation, and feature redundancy.
- Skewness mean, outlier ratio, and noise estimate.
- Target type, class count, and entropy-based class balance.
- Suggested task and suggested model families.

When no explicit target is provided, the fingerprinter can detect sparse item-slot transaction structure and emit `market_basket` as the suggested task with frequency and association-rule style analysis families.

#### Planner Agent

The planner constructs an execution plan after ingestion and profiling. EDA is included for tabular datasets. Modeling is included when an explicit target column exists or when the fingerprint indicates a supervised classification or regression task. Modeling is skipped for datasets below a minimum statistical threshold and for market-basket datasets without an explicit target. Reporting is included as a synthesis phase.

#### Artifact Registry

The artifact registry stores artifacts by lifecycle phase and artifact type. Example artifacts include schema, data quality, cleaning log, data fingerprint, dataframe sample, statistics, correlation results, generated code, findings, model comparison, model metrics, feature importance, execution logs, and final report. The registry serializes artifacts to JSON-compatible structures and supplies bounded context windows to downstream agents.

#### Data-Integrity Gate

The data-integrity gate compares an original fingerprint against a post-cleaning or post-transformation fingerprint. The gate emits anomalies when:

- Row loss exceeds a configured threshold.
- Sparsity increases beyond a configured threshold.
- Noise estimate changes beyond a configured threshold.
- Target type mutates after being identified.

The gate records integrity status as a structured artifact and can emit an integrity event to the frontend or API client.

#### Code Validator and Executor

Generated Python code is parsed into an abstract syntax tree before execution. The validator rejects prohibited modules including `os`, `subprocess`, `shutil`, `socket`, `requests`, `urllib`, `pickle`, `marshal`, `shelve`, and `pathlib`; prohibited builtins including `eval`, `exec`, `__import__`, `open`, `input`, `globals`, and `locals`; filesystem mutation calls including `unlink`, `remove`, `rmdir`, `rename`, `replace`, `write_text`, and `write_bytes`; and dunder attribute traversal. Accepted code may be executed through a configured sandbox pathway, with deterministic local fallback for development and demos.

#### Modeling Agent

The modeling agent selects candidate models from the fingerprint and trains bounded candidates. For classification, candidates can include logistic regression, random forests, gradient boosting, histogram gradient boosting, support-vector classifiers, k-nearest neighbors, and optional installed libraries such as XGBoost, LightGBM, or CatBoost. For regression, candidates can include linear regression, ridge, lasso, random forests, gradient boosting, histogram gradient boosting, support-vector regression, and optional installed gradient-boosting libraries.

The modeling agent records per-model test score, cross-validation mean, cross-validation standard deviation, scoring metric, iteration number, and optional feature-importance data. It handles invalid states such as missing target columns, missing usable features, insufficient rows, or single-class targets by returning structured errors instead of silent failures.

#### Market-Basket Handling

Sparse item-slot tables are detected using column type distribution, missingness, repeated value ratio, and item-slot column naming. When detected, the workflow:

1. Does not infer a supervised target from item names or from the last sparse item column.
2. Preserves exact duplicate rows as repeated transactions.
3. Removes only fully empty or constant columns where appropriate.
4. Computes item-frequency summaries across item slots.
5. Reports top basket items and market-basket-specific recommendations.
6. Skips supervised modeling unless an explicit target column is provided.

#### Checkpoint Manager and Event Stream

The checkpoint manager records session status, current phase, execution plan, phase summaries, user query, target column, file name, metrics, and artifact registry state. The event stream emits structured events for `pipeline_started`, `phase_started`, `phase_completed`, `integrity_ok`, `integrity_warning`, `plan_generated`, `pipeline_completed`, and `pipeline_failed`.

## Example Method Flow

1. Receive a tabular dataset and optional target column.
2. Parse the dataset into a dataframe.
3. Detect schema and quality characteristics.
4. Determine whether target inference is safe.
5. Perform conservative cleaning.
6. Compute a data fingerprint.
7. Store ingestion artifacts in the artifact registry.
8. Run a fingerprint consistency gate.
9. Generate an execution plan from the fingerprint and target-column state.
10. Run EDA and store statistics, correlations, generated code, and findings.
11. If supervised modeling is selected, train bounded candidate models and store model artifacts.
12. Generate a report from registry artifacts.
13. Emit progress events and persist checkpoint state.

## Condensed Proposed Claim Set

The following condensed set keeps the filing focused on the strongest technical combination and follows the reference style of using broad claim anchors rather than an oversized preliminary claim list.

### Independent Claim 1

A computer-implemented method for automated data-science workflow execution, comprising:

1. receiving a tabular dataset;
2. parsing the tabular dataset into a dataframe;
3. computing, by one or more processors, a data fingerprint comprising dataset shape, feature-type distribution, missingness, duplicate-row ratio, correlation density, target type, class balance, and feature-redundancy metrics;
4. selecting, based at least in part on the data fingerprint, an execution plan comprising one or more lifecycle phases selected from ingestion, profiling, exploratory analysis, modeling, evaluation, and reporting;
5. selecting, based at least in part on the data fingerprint, a bounded set of candidate model families when the execution plan includes supervised modeling;
6. storing outputs of lifecycle phases as structured artifacts in an artifact registry indexed by phase and artifact type;
7. providing a downstream lifecycle phase with context derived from upstream artifacts stored in the artifact registry;
8. comparing a pre-mutation data fingerprint with a post-mutation data fingerprint and generating an anomaly when row loss, sparsity drift, statistical drift, or target-type mutation exceeds a threshold;
9. validating generated executable code before execution by parsing the generated executable code into an abstract syntax tree and rejecting prohibited code patterns; and
10. executing at least one lifecycle phase according to the selected execution plan and storing a result of the executed lifecycle phase as a structured artifact.

### Claim 2

The method of claim 1, wherein the data fingerprint further comprises skewness, outlier ratio, noise estimate, text-column indicators, temporal-column counts, high-cardinality indicators, suggested model families, and memory-use estimate, and wherein the selected candidate model families are modified based on target continuity, class cardinality, or class-balance threshold.

### Claim 3

The method of claim 1, wherein validating generated executable code comprises rejecting at least one of a prohibited import, a dynamic execution function, a network-access module, a file-mutation method, and dunder attribute traversal.

### Claim 4

The method of claim 1, wherein the execution plan skips supervised model training when the data fingerprint indicates a market-basket transaction dataset and no explicit target column is supplied, and wherein duplicate rows in the market-basket transaction dataset are preserved as transaction-frequency signal while item-frequency summaries are computed across sparse item-slot columns.

### Independent Claim 5

A system for automated data-science workflow execution, comprising:

1. an ingestion module configured to parse a tabular dataset, detect schema, assess data quality, and compute ingestion artifacts;
2. a fingerprinting module configured to compute a data fingerprint and suggested task type;
3. a planning module configured to generate an execution plan from the data fingerprint and target-column state;
4. an artifact registry configured to store phase-indexed artifacts and provide upstream context to downstream modules;
5. a data-integrity module configured to compare fingerprints across data mutations and emit anomaly artifacts;
6. a code validation module configured to reject generated executable code based on abstract syntax tree analysis before execution;
7. a modeling module configured to train candidate models selected from the data fingerprint when supervised modeling is selected;
8. a market-basket detection module configured to identify sparse item-slot transaction tables, prevent automatic supervised target selection, and preserve repeated transaction rows; and
9. a reporting module configured to synthesize a report from artifacts stored in the artifact registry.

### Independent Claim 6

A non-transitory computer-readable medium storing instructions that, when executed by one or more processors, cause the one or more processors to perform operations comprising computing a data fingerprint for a tabular dataset, selecting an execution plan based on the data fingerprint, storing phase outputs as structured artifacts in a phase-indexed artifact registry, validating generated executable code by abstract-syntax-tree inspection before execution, comparing fingerprints across data mutations to detect integrity anomalies, detecting market-basket transaction structure when present, checkpointing workflow state, emitting structured progress events, and generating an analytical report from artifacts stored in the artifact registry.

## Validation and Benchmark Evidence

The repository includes reproducible local validation through:

```bash
pytest -q
python scripts\validate_claims.py
python scripts\run_demo.py --dataset <path-to-csv> --target <optional-target>
```

Latest local validation evidence recorded in this workspace:

- Unit and regression tests: 74 passed.
- Completed benchmark datasets in the validation report: 7.
- Skipped benchmark datasets in the validation report: 1.
- Task detection accuracy on completed validation datasets: 7/7.
- Static code-safety scripted attack block rate: 100%.
- Average best benchmark score across completed validation datasets: 0.904.
- Kaggle Titanic E2E run completed ingestion, profiling, EDA, modeling, and reporting; RandomForest test accuracy was 0.8212 and CV accuracy was 0.8216 +/- 0.0203.
- Kaggle Groceries E2E run completed ingestion, profiling, EDA, and reporting; the system detected `market_basket`, preserved 9,835 of 9,835 rows, and kept 2,824 repeated basket rows as frequency signal.

This evidence supports local laboratory validation of the implementation. It does not assert production security certification, external state-of-the-art performance, legal protectability, or mature AutoML superiority.

Evidence is graded in `COMPREHENSIVE_RESULTS.md` so that regression tests, fixture benchmarks, Kaggle E2E demos, dummy-baseline deltas, and qualitative capability comparisons are not treated as equal-strength results. Additional comparison artifacts include:

- `disclosure_artifacts/benchmarks/baseline_comparison.csv` and `.json`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_summary.csv`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_long.csv`
- `disclosure_artifacts/benchmarks/novelty_mechanisms.csv`
- `disclosure_artifacts/benchmarks/evidence_quality_grading.csv` and `.json`
- `disclosure_artifacts/benchmarks/failure_mode_validation.csv` and `.json`
- `disclosure_artifacts/benchmarks/capability_comparison.csv` and `.json`

The executable comparison harness (`scripts/run_comparative_benchmarks.py`) compares dummy, linear, fixed tree, fixed boosting, fingerprint-selected DataCortex candidates, and a broad local sklearn-style model pool on the same folds and preprocessing. On the current six-dataset comparison set, the fingerprint-selected candidate policy has mean lift of 0.3857 over dummy baselines and mean gap of 0.0042 to the broad local pool while evaluating a smaller candidate set.

## Drawings

- Figure 1: Overall architecture showing API gateway, planner, ingestion agent, data fingerprinter, EDA agent, modeling agent, report agent, artifact registry, checkpoint manager, code validator, sandbox executor, and event stream. Attachment: `disclosure_artifacts/figures/figure_1_architecture.svg`.
- Figure 2: Claimed method flow from dataset receipt through parsing, fingerprinting, planning, artifact registration, integrity gate, AST safety gate, analysis, report, and checkpoint. Attachment: `disclosure_artifacts/figures/figure_2_claim_flow.svg`.
- Figure 3: Market-basket branch showing sparse item-slot detection, duplicate preservation, row retention, top item evidence, and supervised-model skip logic. Attachment: `disclosure_artifacts/figures/figure_3_market_basket_branch.svg`.
- Figure 4: Local benchmark score chart. Attachment: `disclosure_artifacts/figures/figure_4_validation_benchmark_scores.png`.
- Figure 5: Baseline comparison chart showing recorded fixture scores against deterministic dummy sanity baselines. Attachment: `disclosure_artifacts/figures/figure_5_baseline_comparison.png`.
- Figure 6: Capability comparison chart for control-architecture mechanisms. Attachment: `disclosure_artifacts/figures/figure_6_capability_comparison.png`.
- Figure 7: Failure-mode validation coverage chart. Attachment: `disclosure_artifacts/figures/figure_7_failure_mode_coverage.png`.
- Figure 8: Executable model comparison chart across dummy, linear, tree, boosting, DataCortex, and broad-pool groups. Attachment: `disclosure_artifacts/figures/figure_8_executable_model_comparison.png`.
- Evidence manifest and validation appendix. Attachments: `disclosure_artifacts/evidence_manifest.json` and `VALIDATION_APPENDIX.md`.

Repository visual assets are located under `visualizations/`.

## Advantages

- Reduces unnecessary model-search breadth by using dataset characteristics before model selection.
- Makes phase dependencies explicit through structured artifact propagation.
- Detects harmful data mutations that would otherwise be silent.
- Blocks common dangerous generated-code patterns before execution.
- Avoids inappropriate supervised training for sparse item-slot market-basket datasets.
- Provides observable progress through structured events.
- Provides reproducible local evidence through tests, benchmark scripts, and demo outputs.

## Implementation Status

The invention is implemented as a Python prototype and validated in a local laboratory setting. The current evidence supports a TRL 4 characterization. Production security hardening, independent benchmark comparison, legal claim review, and deployment-scale reliability testing remain outside the current validation scope.

## Filing Notes for Counsel

- The strongest technical differentiators are the combination of fingerprint-guided execution control, phase-indexed artifact propagation, fingerprint consistency gating, deterministic generated-code AST validation, and market-basket-aware planning.
- The claims should avoid unsupported assertions of state-of-the-art performance, full autonomy, or production security certification.
- The claims should emphasize technical control flow, concrete data structures, validation gates, and reproducible artifacts rather than business outcomes.
- The repository evidence should be treated as prototype evidence, not as proof of commercial readiness.
