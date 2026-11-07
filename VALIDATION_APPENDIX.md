# Validation Appendix: DataCortex-AI

This appendix attaches concrete implementation references, benchmark artifacts, failure-mode checks, and generated diagrams for the DataCortex-AI technical disclosure package. It is technical support material, not a legal opinion or production certification.

## Attachment Index

| Attachment | Path | Purpose |
| --- | --- | --- |
| Architecture figure | `disclosure_artifacts/figures/figure_1_architecture.svg` | Current system architecture: API, planner, agents, registry, validator, checkpointing, events |
| Method-flow figure | `disclosure_artifacts/figures/figure_2_claim_flow.svg` | Method flow from dataset receipt through report and checkpoint |
| Market-basket figure | `disclosure_artifacts/figures/figure_3_market_basket_branch.svg` | Kaggle Groceries E2E branch behavior |
| Benchmark score chart | `disclosure_artifacts/figures/figure_4_validation_benchmark_scores.png` | Local validation benchmark score chart |
| Baseline comparison chart | `disclosure_artifacts/figures/figure_5_baseline_comparison.png` | Recorded fixture results versus deterministic dummy sanity baselines |
| Capability comparison chart | `disclosure_artifacts/figures/figure_6_capability_comparison.png` | Mechanism-level control architecture comparison |
| Failure-mode chart | `disclosure_artifacts/figures/figure_7_failure_mode_coverage.png` | Count of validated failure-mode checks |
| Executable model comparison chart | `disclosure_artifacts/figures/figure_8_executable_model_comparison.png` | Same-fold comparison across dummy, linear, tree, boosting, DataCortex, and broad-pool groups |
| Comprehensive results | `COMPREHENSIVE_RESULTS.md` | Graded validation, baseline, E2E, failure-mode, and capability comparison summary |
| Comparative benchmarks | `COMPARATIVE_BENCHMARKS.md` | Executable model benchmark comparison and novelty mechanism summary |
| Benchmark tables | `disclosure_artifacts/benchmarks/*.csv` and `*.json` | Structured benchmark, baseline, quality grading, failure-mode, and E2E tables |
| Manifest | `disclosure_artifacts/evidence_manifest.json` | Regeneration and evidence index |

## Evidence Quality Grading

| Grade | Evidence Type | Supports | Does Not Support |
| --- | --- | --- | --- |
| A | Automated regression tests | Core behavior remains stable across repository changes | Production reliability |
| B | Local fixture benchmarks | Task detection, modeling execution, metric reporting, integrity checks | External leaderboard standing |
| B | Kaggle E2E demos | Full demo flows on independently sourced public CSV datasets | Broad market validation |
| C | Dummy-baseline deltas | Recorded supervised results exceed trivial predictors | Superiority over mature AutoML tools |
| D | Capability comparison | Mechanism-level distinction of control architecture | Performance, speed, or accuracy advantage |

## Concrete Technical Mechanisms

### 1. Fingerprint-Guided Execution Control Before Model Training

**Mechanism:** The system computes a structured fingerprint before model training and uses the fingerprint to determine task type, candidate models, and whether modeling should run.

**Implementation references:**

- `src/core/data_fingerprint.py`: `DataFingerprinter.compute`, `_suggest_task_and_models`
- `src/agents/planner.py`: `_generate_adaptive_plan`
- `src/agents/modeling_agent.py`: `_select_models`, `_build_local_models`

**Validated effect:** Candidate selection is bounded by dataset characteristics before training begins, and task-detection behavior is covered by the local validation harness.

### 2. Phase-Indexed Artifact Registry as Cross-Agent Context

**Mechanism:** Each lifecycle phase registers structured artifacts by phase and artifact type. Downstream phases retrieve bounded upstream context from the registry.

**Implementation references:**

- `src/core/artifact_registry.py`: `ArtifactRegistry.register`, `query`, `get_latest`, `get_context_for_phase`
- `src/agents/report_agent.py`: structured report synthesis from registry artifacts

**Validated effect:** Demo checkpoint files under `demo_outputs/*/checkpoints/` contain serialized phase artifacts used to generate reports.

### 3. Fingerprint Consistency Gate for Automated Data Mutations

**Mechanism:** The system compares fingerprints across mutation boundaries and emits anomalies for row loss, sparsity drift, statistical drift, and target-type mutation.

**Implementation references:**

- `src/core/data_fingerprint.py`: `validate_consistency`
- `src/agents/planner.py`: `_run_integrity_check`

**Validated effect:** Kaggle E2E event logs include integrity status, and `VALIDATION_RESULTS.md` records integrity pass status for completed benchmark datasets.

### 4. Deterministic AST Safety Gate for Generated Analysis Code

**Mechanism:** Generated Python is parsed into an AST and rejected before execution when prohibited imports, dynamic execution, file mutation, networking, or dunder traversal patterns are detected.

**Implementation references:**

- `src/core/sandbox_executor.py`: `CodeValidator`
- `tests/test_code_validator.py`: safety regression tests

**Validated effect:** `VALIDATION_RESULTS.md` records a 100% block rate for the scripted static attack set. This is a covered-attack result, not a complete security guarantee.

### 5. Market-Basket-Aware Target Suppression and Duplicate Preservation

**Mechanism:** Sparse item-slot transaction tables are detected before target guessing and cleaning. The system suppresses automatic supervised target inference and preserves repeated basket rows as frequency signal.

**Implementation references:**

- `src/agents/ingestion_agent.py`: `_looks_like_market_basket`, `_guess_target_column`, `_auto_clean`
- `src/core/data_fingerprint.py`: `_looks_like_market_basket`
- `src/agents/eda_agent.py`: market-basket top item computation
- `src/agents/report_agent.py`: market-basket methodology, limitations, and recommendations

**Validated effect:** Kaggle Groceries E2E detected `market_basket`, preserved 9,835/9,835 rows, kept 2,824 repeated basket rows, skipped supervised modeling, and reported top items.

### 6. Observable and Checkpointed Long-Running Pipeline

**Mechanism:** The planner emits structured events and persists checkpoint state with phase status, current phase, execution plan, target column, file name, metrics, and artifacts.

**Implementation references:**

- `src/agents/planner.py`: `_emit_event`
- `src/core/checkpoint.py`: checkpoint persistence
- `src/api/gateway.py`: status, report, artifact, event, and WebSocket routes

**Validated effect:** Demo outputs include `demo_events.json`, `demo_summary.json`, and checkpoint JSON files for Kaggle E2E runs.

## Failure-Mode Validation Summary

| Risk Tested | Expected Behavior | Artifact | Status |
| --- | --- | --- | --- |
| Market-basket data misread as supervised classification | Detect `market_basket`, suppress automatic target selection, skip supervised training without explicit target | `demo_outputs/kaggle_groceries_iter4/demo_summary.json` | Pass |
| Duplicate basket rows discarded as ordinary duplicates | Retain 9,835/9,835 rows and preserve 2,824 repeated baskets | `demo_outputs/kaggle_groceries_iter4/demo_report.md` | Pass |
| Unsafe generated Python accepted | Reject prohibited imports, dynamic execution, networking, file mutation, and dunder traversal | `tests/test_code_validator.py` | Pass |
| Long-running modeling phase appears opaque | Emit phase events and persist checkpoints | `demo_outputs/kaggle_titanic_iter3/demo_events.json` | Pass |

## Benchmark Summary

| Scope | Comparator | Result | Interpretation |
| --- | --- | --- | --- |
| Executable model comparison | Dummy, linear, tree, boosting, DataCortex, broad local pool | Mean lift over dummy 0.3857; mean gap to broad pool 0.0042 | Concrete same-fold benchmark comparison |
| Fixture benchmarks | Local validation harness | 7 completed datasets with integrity status | Core benchmark flows are reproducible locally |
| Dummy baselines | Most-frequent classifier or mean regressor | Mean absolute lift 0.396 across 6 local comparisons | Sanity check only |
| Kaggle Titanic E2E | Held-out local evaluation | RandomForest test accuracy 0.8212; CV 0.8216 +/- 0.0203 | Supervised demo flow executes |
| Kaggle Groceries E2E | Input/output integrity | 9,835/9,835 rows retained; 2,824 repeats preserved | Market-basket branch behaves correctly |
| Static safety attack set | AST validator | 100% block rate recorded | Covered unsafe code patterns are rejected |

## Rebuild Commands

```bash
python scripts\generate_disclosure_artifacts.py
python scripts\run_comparative_benchmarks.py
python scripts\generate_comprehensive_results.py
python scripts\build_disclosure_docx.py
```

## Evidence Limits

- This is local laboratory evidence for a prototype, not production certification.
- Static AST validation is a deterministic pre-execution gate, not a complete security sandbox.
- Benchmark results are repository evidence, not external leaderboard or mature AutoML superiority claims.
- The executable comparison uses local sklearn-style baselines and a broad local model pool under shared folds; external AutoML tools remain future comparison work.
- Capability comparisons describe mechanisms; they do not claim speed, accuracy, cost, or reliability superiority.
