# DataCortex-AI Demo TODO

This file tracks demo-facing work that should be implemented after the current stabilization pass.

## High Priority

- **Accessible live training progress**
  - Emit backend progress events during modeling, not only phase start/end.
  - Include current model name, model index, cross-validation step, elapsed time, and best score so far.
  - Add frontend ARIA live-region announcements for blind users.
  - Add heartbeat events such as "Still training RandomForest, elapsed 90s" for long-running jobs.

- **Plan modes**
  - Add explicit user-selectable execution modes:
    - `fast_demo`: EDA plus lightweight local modeling only.
    - `profile_only`: ingestion, profiling, EDA, and report; no modeling.
    - `full_validation`: full modeling and stricter validation.
    - `market_basket`: transaction/frequency/association analysis for item-slot datasets.
  - Surface the selected mode in the report and event log.
  - Prevent unsafe auto-target selection in `market_basket` mode.

- **Multi-file dataset visualizer**
  - Allow uploading multiple related files in one session.
  - Show per-file schema, row/column counts, missingness, duplicate rate, and inferred relationships.
  - Add join-key candidate detection across files.
  - Add a relationship graph view for tables/files.
  - Generate a combined report with per-file sections and cross-file observations.

## Medium Priority

- **Market-basket analysis**
  - Detect transaction item-slot datasets and skip supervised modeling by default.
  - Compute top item frequencies, pair frequencies, basket size distribution, and optional association rules.
  - Report support/confidence/lift when association-rule dependencies are available.

- **Report polish**
  - Clearly distinguish raw rows/columns from cleaned rows/columns.
  - Display model error messages in a compact "why modeling failed" section.
  - Avoid recommendations about supervised validation when no meaningful supervised target exists.

- **Frontend polish**
  - Replace corrupted legacy text symbols in comments and status strings.
  - Add a "current backend activity" panel separate from the event log.
  - Add a cancel/stop button for long-running training.

## Done in Current Pass

- Remote LLM and sandbox execution are opt-in for reliable local demos.
- API status polling remains responsive during local training.
- Market-basket-like datasets no longer auto-select columns such as `yams` as supervised targets.
- Progress accounting now caps at 100%.
- Technical Disclosure Draft and invention disclosure were rewritten around concrete, validated mechanisms.
