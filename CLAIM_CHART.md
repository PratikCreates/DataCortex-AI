# DataCortex-AI Claim Chart

This chart maps the proposed technical claim themes to implementation evidence in the repository. It is intended as drafting support for counsel and technical reviewers, not as a legal opinion.

## Core Claim Mapping

| Proposed Claim Element | Implementation Evidence | Validation Evidence |
| --- | --- | --- |
| Receive and parse a tabular dataset | `src/agents/ingestion_agent.py`, `IngestionAgent._parse_file` | `tests/`, Kaggle Titanic and Groceries E2E outputs |
| Detect schema and quality characteristics | `src/agents/ingestion_agent.py`, `_detect_schema`, `_assess_quality` | Demo reports include schema, quality, missingness, duplicates, cleaning log |
| Compute a structured data fingerprint | `src/core/data_fingerprint.py`, `DataFingerprinter.compute` | `VALIDATION_RESULTS.md`, task detection accuracy 7/7 completed datasets |
| Use fingerprint to select task and model families | `src/core/data_fingerprint.py`, `_suggest_task_and_models`; `src/agents/planner.py`, `_generate_adaptive_plan` | Validation dataset table; Titanic E2E classification run |
| Store phase outputs as structured artifacts | `src/core/artifact_registry.py`, `ArtifactRegistry.register`, `get_context_for_phase` | Demo checkpoint artifact JSON files under `demo_outputs/` |
| Provide downstream context from upstream artifacts | `src/core/artifact_registry.py`, context construction; `src/agents/report_agent.py` artifact retrieval | Generated reports synthesize schema, quality, cleaning, EDA, modeling, and fingerprint artifacts |
| Compare fingerprints across mutations | `src/core/data_fingerprint.py`, `validate_consistency`; `src/agents/planner.py`, `_run_integrity_check` | E2E events include `integrity_ok`; validation report records integrity pass |
| Validate generated code before execution | `src/core/sandbox_executor.py`, `CodeValidator` | Static code-safety scripted attack block rate 100% in `VALIDATION_RESULTS.md` |
| Reject unsafe AST patterns | `src/core/sandbox_executor.py`, prohibited imports, builtins, filesystem methods, dunder traversal | Safety regression tests under `tests/` |
| Execute bounded local modeling fallback | `src/agents/modeling_agent.py`, `_train_locally`, `_build_local_models` | Titanic E2E RandomForest accuracy 0.8212; pytest regression coverage |
| Record model comparison and feature importance | `src/agents/modeling_agent.py`, model artifact registration; `src/agents/report_agent.py`, `_generate_model_section` | Titanic E2E report model comparison table |
| Detect market-basket item-slot structure | `src/agents/ingestion_agent.py`, `_looks_like_market_basket`; `src/core/data_fingerprint.py`, `_looks_like_market_basket` | Groceries E2E detected `market_basket` |
| Preserve repeated transaction rows | `src/agents/ingestion_agent.py`, `_auto_clean` duplicate branch | Groceries E2E preserved 9,835/9,835 rows and kept 2,824 repeated baskets |
| Skip supervised modeling for market-basket data without target | `src/agents/planner.py`, `_generate_adaptive_plan`; `src/agents/report_agent.py`, market-basket report branch | Groceries E2E phases: ingestion, profiling, EDA, reporting |
| Emit structured progress events | `src/agents/planner.py`, `_emit_event`; `src/api/gateway.py` event/WebSocket endpoints | Demo event JSON files under `demo_outputs/` |
| Persist checkpoints and artifacts | `src/core/checkpoint.py` | Demo checkpoint JSON files under `demo_outputs/*/checkpoints/` |

## Condensed Claim Strategy

The filing package now uses a condensed 6-claim set:

1. **Claim 1**: broad method claim covering fingerprint-guided planning, artifact propagation, integrity gating, AST validation, and lifecycle execution.
2. **Claim 2**: dependent method claim covering expanded fingerprint fields and model-family modification.
3. **Claim 3**: dependent method claim covering AST rejection categories.
4. **Claim 4**: dependent method claim covering market-basket routing, duplicate preservation, and item-frequency summaries.
5. **Claim 5**: system claim covering the concrete modules.
6. **Claim 6**: computer-readable medium claim covering stored instructions, checkpointing, events, and report generation.

## Concrete Dependent Claim Candidates

| Dependent Theme | Concrete Feature |
| --- | --- |
| Expanded fingerprint | Includes skewness, outlier ratio, noise estimate, high-cardinality indicator, text-column indicator, memory footprint |
| Imbalanced classification | Changes scoring or model preference when entropy-based class balance is below a threshold |
| Integrity gate | Emits anomaly for row loss, sparsity spike, noise drift, or target-type mutation |
| AST validator | Blocks imports, builtins, dunder traversal, network access, filesystem mutation |
| Market-basket branch | Detects sparse item-slot tables and skips target inference |
| Duplicate preservation | Keeps repeated basket rows as frequency signal |
| Event stream | Emits phase and integrity events to API or frontend |
| Local fallback | Uses deterministic local sklearn training when remote sandbox is disabled or unavailable |
| Checkpointing | Serializes phase status, execution plan, metadata, metrics, and artifacts |

## Evidence Boundaries

- Current evidence supports a local Python prototype and TRL 4 laboratory validation.
- Current evidence does not support production security certification.
- Current evidence does not support external state-of-the-art performance claims.
- Static AST validation is a safety gate, not a complete sandbox.
- Kaggle E2E evidence confirms demo behavior for one supervised classification dataset and one market-basket dataset.
