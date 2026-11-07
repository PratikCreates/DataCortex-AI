# Invention Disclosure Format (IDF-B)

## 1. Title of the Invention

**DataCortex-AI: Fingerprint-Guided Multi-Agent Automated Data Science with Cross-Phase Artifact Propagation, Data-Integrity Gating, and Pre-Execution Code Safety Validation**

## 2. Field / Area of Invention

Artificial intelligence, automated data science, automated machine learning, multi-agent software orchestration, generated-code validation, data-quality monitoring, checkpointed workflow execution, market-basket dataset analysis, and reproducible machine-learning workflow evaluation.

## 3. Problem Statement

Automated data-science systems face several technical limitations:

1. Broad model-search systems can evaluate unsuitable candidates before using dataset-specific statistical structure.
2. Generated-code systems can execute unsafe or brittle code without deterministic pre-execution validation.
3. Lifecycle stages often pass state informally, making it difficult to audit which upstream artifacts influenced downstream analysis.
4. Automated cleaning can silently damage data through row loss, target mutation, sparsity drift, or inappropriate duplicate removal.
5. Sparse item-slot transaction datasets can be incorrectly treated as supervised classification problems when no target column exists.
6. User-facing systems can leave users without structured visibility into long-running phases.

## 4. Prior Art and Publications from Literature

| Reference / Framework | Summary of Prior Art | Limitation Addressed by This Invention |
| --- | --- | --- |
| TPOT | Genetic-programming approach for discovering machine-learning pipelines. | Searches broad pipeline permutations rather than using a phase-propagated dataset fingerprint as an execution-control signal. |
| Auto-sklearn | Bayesian optimization and meta-learning over machine-learning pipelines. | Primarily optimizes predefined search spaces and does not center a phase-indexed artifact registry with integrity gates and generated-code AST validation. |
| AutoGluon | Automated ensembling and model stacking for tabular ML. | Strong model automation, but not focused on generated-code validation, cross-phase artifact provenance, or market-basket-aware target suppression. |
| Notebook and LLM code agents | Generate executable Python or notebook code from natural-language prompts. | Generated code can be unsafe or non-reproducible when executed without deterministic AST screening, artifact provenance, and data-integrity checks. |
| Market-basket analysis tools | Compute association rules or item co-occurrence for transaction data. | Generally do not integrate with a full automated data-science lifecycle that also performs fingerprinted planning, artifact propagation, code validation, checkpointing, and report synthesis. |

## 5. Summary and Background of the Invention

DataCortex-AI addresses the above limitations by combining:

1. A structured data fingerprint computed before model selection.
2. Fingerprint-guided planning for EDA, supervised modeling, reporting, and market-basket routing.
3. A bounded model-family selection mechanism driven by fingerprint metrics.
4. A phase-indexed artifact registry that propagates structured outputs between lifecycle agents.
5. A fingerprint consistency gate that detects harmful data mutations.
6. A deterministic AST-based safety validator for generated Python code.
7. Checkpointed workflow execution and structured progress events.
8. A reproducible validation harness and Kaggle end-to-end demo evidence.

## 6. Objectives of the Invention

1. Automate major phases of a tabular data-science workflow: ingestion, profiling, exploratory analysis, modeling, evaluation, reporting, event streaming, and checkpointing.
2. Select execution plans and model families using dataset characteristics rather than unbounded brute-force search.
3. Preserve lifecycle phase outputs as structured artifacts for downstream context and auditability.
4. Detect data-integrity anomalies caused by automated cleaning or transformation.
5. Reject unsafe generated code before execution.
6. Correctly identify sparse item-slot transaction datasets and avoid unsupported supervised target inference.
7. Provide reproducible local validation evidence for task detection, data integrity, code safety, modeling behavior, and demo flows.

## 7. Working Principle of the Invention

1. The system receives a tabular dataset and optional target column.
2. The ingestion module parses the dataset, detects schema, assesses quality, performs conservative cleaning, and stores ingestion artifacts.
3. The target-selection logic avoids unsafe target inference when the dataset resembles sparse market-basket item-slot data.
4. The fingerprinting module computes a data fingerprint including shape, feature types, missingness, duplicates, correlations, target type, class balance, outliers, noise estimate, suggested task, suggested model families, and memory footprint.
5. The planner uses the fingerprint and target-column state to construct an execution plan.
6. The artifact registry stores phase outputs and supplies bounded upstream context to downstream agents.
7. A data-integrity gate compares fingerprints across cleaning or transformation boundaries and emits anomaly artifacts or events.
8. Generated code is parsed into an AST and rejected if prohibited imports, dynamic execution calls, network access, filesystem mutation, or dunder traversal patterns are found.
9. The modeling module trains deterministic local candidates or configured sandbox candidates when supervised modeling is selected.
10. The report module synthesizes schema, quality, cleaning, fingerprint, EDA, modeling, and recommendation artifacts into a report.
11. The checkpoint manager serializes phase state and artifacts, while the event stream exposes live progress to the API or frontend.
12. Validation scripts and demo runs produce reproducible local evidence.

## 8. Detailed Description of the Invention

### 8.1 Data Fingerprint

The data fingerprint is a structured representation of the dataset. In the current implementation, it includes:

- Row count and column count.
- Dimensionality ratio.
- Numeric, categorical, temporal, and text-column counts.
- High-cardinality and text-column indicators.
- Sparsity and duplicate-row ratio.
- Correlation density, maximum correlation, and feature redundancy.
- Skewness, outlier ratio, and noise estimate.
- Target type, class count, and class balance.
- Suggested task and suggested model families.
- Memory-use estimate.

### 8.2 Fingerprint-Guided Planning

The planner uses the fingerprint to decide whether to run EDA, supervised modeling, and reporting. If an explicit target column exists or the fingerprint indicates a supervised task, modeling is included. If the dataset is too small for reliable modeling, modeling can be skipped. If the dataset is detected as market-basket transaction data and no explicit target exists, the workflow runs basket profiling and reporting instead of classification or regression.

### 8.3 Artifact Registry

The registry stores artifacts by phase and artifact type. Examples include schema, data quality, cleaning logs, dataframe samples, data fingerprints, statistics, correlations, generated code, findings, model comparisons, model metrics, feature importance, execution logs, and reports. Downstream phases retrieve relevant upstream context through the registry rather than relying on informal shared state.

### 8.4 Data-Integrity Protection

A consistency validator compares fingerprints before and after a data mutation. It reports anomalies for:

- Excessive row loss.
- Sparsity spikes.
- Statistical drift.
- Target-type mutation.

The system emits structured integrity events and stores integrity artifacts for reporting and audit.

### 8.5 Code Safety Validation

Generated Python code is parsed with the Python AST module. The current implementation rejects:

- Prohibited modules including `os`, `subprocess`, `shutil`, `socket`, `requests`, `urllib`, `pickle`, `marshal`, `shelve`, and `pathlib`.
- Prohibited builtins including `eval`, `exec`, `__import__`, `open`, `input`, `globals`, and `locals`.
- Filesystem mutation methods including `unlink`, `remove`, `rmdir`, `rename`, `replace`, `write_text`, and `write_bytes`.
- Dunder attribute traversal.

### 8.6 Market-Basket-Aware Execution

Sparse item-slot transaction tables are detected using column naming, object/category column ratio, missingness, and repeated-value ratio. When detected, the system:

- Does not auto-select item columns as supervised targets.
- Preserves duplicate rows because repeated baskets encode frequency.
- Treats blank item slots as absent items.
- Computes top item frequencies across item slots.
- Skips supervised model training unless a target column is explicitly supplied.
- Generates market-basket-specific methodology, limitations, and recommendations.

### 8.7 Checkpointed and Observable Execution

The checkpoint manager persists session status, phase status, current phase, execution plan, file name, target column, phase metrics, and serialized artifacts. The event stream emits phase and integrity events for user interfaces and monitoring clients.

## 9. Experimental Validation Results

Validation was executed locally with:

```bash
python -m pytest -q
python scripts\validate_claims.py
python scripts\run_demo.py --dataset <path-to-csv> --target <optional-target>
```

Current recorded results:

- Automated tests: 74 passed.
- Completed benchmark datasets in `VALIDATION_RESULTS.md`: 7.
- Skipped datasets in `VALIDATION_RESULTS.md`: 1.
- Task detection accuracy: 7/7 completed validation datasets.
- Static code-safety scripted attack block rate: 100%.
- Average best benchmark score: 0.904 across completed validation datasets.

| Dataset | Task | Best Model / Analysis | Score / Result | Metric / Evidence | Integrity |
| --- | --- | --- | ---: | --- | --- |
| Iris | Classification | RandomForestClassifier | 0.9667 | Accuracy | Pass |
| Churn | Classification | LGBMClassifier | 0.9140 | Accuracy | Pass |
| Credit Risk | Classification | RandomForestClassifier | 0.9187 | Accuracy | Pass |
| Housing | Regression | LinearRegression | 1.0000 | R2 | Pass |
| Hard Benchmark | Classification | XGBClassifier | 0.7300 | Accuracy | Pass |
| Titanic validation fixture | Classification | RandomForestClassifier | 0.8384 | Accuracy | Pass |
| Airline Satisfaction | Classification | LGBMClassifier | 0.9600 | Accuracy | Pass |
| Kaggle Titanic E2E | Classification | RandomForest | 0.8212 | Test accuracy; CV 0.8216 +/- 0.0203 | Pass |
| Kaggle Groceries E2E | Market basket | Item-frequency profiling | 9,835/9,835 rows retained | 2,824 repeated baskets preserved | Pass |

These results are local laboratory validation results. They do not assert production security certification, external state-of-the-art superiority, or legal protectability.

`COMPREHENSIVE_RESULTS.md` adds stricter comparison framing:

- Evidence quality is graded separately for regression tests, fixture benchmarks, Kaggle E2E demos, sanity baselines, and qualitative capability comparisons.
- `COMPARATIVE_BENCHMARKS.md` adds executable same-fold comparisons across dummy, linear, tree, boosting, DataCortex fingerprint-selected, and broad local model-pool groups.
- Dummy-baseline comparison uses deterministic most-frequent classification and mean-regression baselines; it is reported only as a sanity baseline.
- Failure-mode validation maps concrete risks to artifacts, including market-basket target suppression, duplicate transaction preservation, AST code rejection, and checkpointed event visibility.
- Capability comparison is mechanism-level only and is not presented as speed, accuracy, cost, reliability, or leaderboard evidence.

## 10. Aspects of the Invention Requiring Protection

1. Computing a data fingerprint and using it to select a bounded lifecycle execution plan before model training.
2. Selecting model families from fingerprint metrics instead of relying only on brute-force pipeline search.
3. Propagating phase-indexed artifacts between agents through a structured artifact registry.
4. Comparing pre-mutation and post-mutation fingerprints to detect data-integrity anomalies.
5. Validating generated code through deterministic AST inspection before execution.
6. Detecting market-basket item-slot structure and suppressing unsafe supervised target inference.
7. Preserving duplicate transaction rows as frequency signal in market-basket datasets.
8. Combining fingerprint-guided planning, artifact propagation, code validation, checkpointing, event streaming, and report synthesis in a single automated data-science workflow.

## 11. Concrete Novelty Evidence

| Novelty | Why It Is Concrete | Repository Reference |
| --- | --- | --- |
| Fingerprint-guided execution control | The fingerprint is used to select task type, execution plan, and bounded model candidates before training. | `src/core/data_fingerprint.py`, `src/agents/planner.py`, `src/agents/modeling_agent.py` |
| Phase-indexed artifact propagation | Each agent emits typed artifacts, and downstream agents retrieve context through registry APIs. | `src/core/artifact_registry.py`, `src/agents/report_agent.py` |
| Fingerprint consistency gate | The system computes anomalies from pre-mutation and post-mutation fingerprint differences. | `src/core/data_fingerprint.py`, `src/agents/planner.py` |
| Deterministic AST safety validation | Generated code is parsed and rejected on prohibited syntax or API usage before execution. | `src/core/sandbox_executor.py`, `tests/test_code_validator.py` |
| Market-basket target suppression | Sparse item-slot transaction data does not auto-select an item column as a supervised target. | `src/agents/ingestion_agent.py`, `tests/test_ingestion_target_detection.py` |
| Duplicate transaction preservation | Exact duplicate basket rows are retained when they represent transaction frequency. | Kaggle Groceries E2E evidence in `demo_outputs/kaggle_groceries_iter4/` |
| Observable long-running execution | Phase, integrity, plan, and completion events are emitted and persisted. | `src/agents/planner.py`, `demo_outputs/*/demo_events.json` |
| Fingerprint-bounded candidate budget | The data fingerprint narrows the model set before training while retaining near-broad-pool scores on the local comparison harness. | `scripts/run_comparative_benchmarks.py`, `COMPARATIVE_BENCHMARKS.md` |

Attachment bundle:

- `VALIDATION_APPENDIX.md`
- `CLAIM_CHART.md`
- `COMPREHENSIVE_RESULTS.md`
- `COMPARATIVE_BENCHMARKS.md`
- `disclosure_artifacts/evidence_manifest.json`
- `disclosure_artifacts/figures/figure_1_architecture.svg`
- `disclosure_artifacts/figures/figure_2_claim_flow.svg`
- `disclosure_artifacts/figures/figure_3_market_basket_branch.svg`
- `disclosure_artifacts/figures/figure_4_validation_benchmark_scores.png`
- `disclosure_artifacts/figures/figure_5_baseline_comparison.png`
- `disclosure_artifacts/figures/figure_6_capability_comparison.png`
- `disclosure_artifacts/figures/figure_7_failure_mode_coverage.png`
- `disclosure_artifacts/figures/figure_8_executable_model_comparison.png`
- `disclosure_artifacts/benchmarks/validation_benchmarks.json`
- `disclosure_artifacts/benchmarks/validation_benchmarks.csv`
- `disclosure_artifacts/benchmarks/kaggle_e2e_results.json`
- `disclosure_artifacts/benchmarks/baseline_comparison.csv`
- `disclosure_artifacts/benchmarks/capability_comparison.csv`
- `disclosure_artifacts/benchmarks/failure_mode_validation.csv`
- `disclosure_artifacts/benchmarks/evidence_quality_grading.csv`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_summary.csv`
- `disclosure_artifacts/benchmarks/comparative_benchmarks_long.csv`
- `disclosure_artifacts/benchmarks/novelty_mechanisms.csv`

## 12. Proposed Independent Claim Themes

The draft has been condensed to six claims:

1. A broad method claim directed to fingerprint-guided execution planning, artifact propagation, integrity gating, AST validation, and lifecycle execution.
2. A dependent method claim directed to expanded fingerprint fields and model-family modification.
3. A dependent method claim directed to AST rejection categories.
4. A dependent method claim directed to market-basket routing, duplicate preservation, and item-frequency summaries.
5. A system claim directed to modules for ingestion, fingerprinting, planning, artifact registry, integrity checking, code validation, modeling, market-basket detection, and reporting.
6. A computer-readable-medium claim directed to stored instructions for performing the fingerprint-guided workflow, checkpointing, event emission, and report generation.

## 13. Technology Readiness Level

**TRL 4: Technology validated in a laboratory environment.**

Justification:

- Core Python modules are implemented.
- Automated unit and regression tests pass locally.
- Local validation runs over repository fixtures and available benchmark CSVs.
- Kaggle E2E demo runs were completed for supervised classification and market-basket analysis.
- The validation report documents completed and skipped datasets.
- Production deployment, formal security certification, and independent benchmark comparison remain future work.

## 14. Inventor / Applicant Notes

The current repository contains implementation code, tests, validation scripts, validation results, diagrams, demo outputs, and documentation sufficient for technical review and technical drafting refinement. Final formal filing should be reviewed by a qualified qualified professional before submission.
