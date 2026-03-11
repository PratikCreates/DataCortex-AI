"""
DatacortexAI — Report Generation Agent

Synthesizes structured outputs from all lifecycle stages into
coherent, executive-quality analytical reports with embedded
statistical tables, model evaluations, and actionable recommendations.
"""

import logging
from typing import Optional
from datetime import datetime

from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    AI-Generated Narrative Report Synthesis.

    Synthesizes all upstream artifacts into a comprehensive Markdown report:
    - Executive summary
    - Methodology overview
    - Data quality and profiling
    - EDA findings with statistical detail
    - Model comparison and feature importance
    - Limitations and caveats
    - Actionable recommendations
    """

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def run(
        self,
        user_query: str,
        file_name: str,
        registry: ArtifactRegistry,
    ) -> dict:
        """Generate the final analytical report."""
        logger.info("Report Agent started")

        # Gather ALL upstream artifacts (Claim 2)
        upstream_context = registry.get_context_for_phase(Phase.REPORTING)

        # Get specific artifacts for structured sections
        schema = registry.get_latest(Phase.INGESTION.value, ArtifactType.SCHEMA.value)
        quality = registry.get_latest(Phase.INGESTION.value, ArtifactType.DATA_QUALITY.value)
        fingerprint = registry.get_latest(Phase.INGESTION.value, ArtifactType.DATA_FINGERPRINT.value)
        cleaning_log = registry.get_latest(Phase.INGESTION.value, ArtifactType.CLEANING_LOG.value)
        statistics = registry.get_latest(Phase.EDA.value, ArtifactType.STATISTICS.value)
        correlations = registry.get_latest(Phase.EDA.value, ArtifactType.CORRELATION.value)
        findings = registry.get_latest(Phase.EDA.value, ArtifactType.FINDINGS.value)
        model_comparison = registry.get_latest(Phase.TRAINING.value, ArtifactType.MODEL_COMPARISON.value)
        model_metrics = registry.get_latest(Phase.EVALUATION.value, ArtifactType.MODEL_METRICS.value)
        feature_imp = registry.get_latest(Phase.EVALUATION.value, ArtifactType.FEATURE_IMPORTANCE.value)

        # Generate report sections
        report_sections = []

        # 1. Title & metadata
        report_sections.append(self._generate_header(file_name, user_query, schema, fingerprint))

        # 2. Executive Summary
        exec_summary = await self._generate_executive_summary(
            file_name, user_query, fingerprint, findings, model_metrics
        )
        report_sections.append(exec_summary)

        # 3. Methodology
        report_sections.append(self._generate_methodology(fingerprint))

        # 4. Data Overview & Quality
        data_section = self._generate_data_section(schema, quality, cleaning_log, fingerprint)
        report_sections.append(data_section)

        # 5. EDA Findings
        eda_section = self._generate_eda_section(statistics, correlations, findings)
        report_sections.append(eda_section)

        # 6. Distribution Analysis
        dist_section = self._generate_distribution_section(statistics)
        report_sections.append(dist_section)

        # 7. Modeling Results
        model_section = self._generate_model_section(model_comparison, model_metrics, feature_imp)
        report_sections.append(model_section)

        # 8. Limitations & Caveats
        limitations = self._generate_limitations(schema, quality, model_metrics, fingerprint)
        report_sections.append(limitations)

        # 9. Recommendations
        recommendations = await self._generate_recommendations(
            upstream_context, user_query, findings, model_metrics, fingerprint
        )
        report_sections.append(recommendations)

        # 10. Technical Appendix
        appendix = self._generate_appendix(schema, correlations, fingerprint)
        report_sections.append(appendix)

        # Assemble final report
        report_markdown = "\n\n".join([s for s in report_sections if s.strip()])

        # Register the report
        registry.register(Phase.REPORTING, ArtifactType.REPORT, {
            "report_markdown": report_markdown,
            "sections": len(report_sections),
        })

        result = {
            "report_markdown": report_markdown,
            "sections": len(report_sections),
        }

        logger.info(f"Report generated: {len(report_markdown)} chars, {len(report_sections)} sections")
        return result

    # ─── Section Generators ──────────────────────────

    def _generate_header(self, file_name, user_query, schema, fingerprint) -> str:
        """Generate report title and metadata block."""
        s = schema.data if schema else {}
        fp = fingerprint.data if fingerprint else {}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""# DatacortexAI Analysis Report

| Property | Value |
|----------|-------|
| **Dataset** | `{file_name}` |
| **Generated** | {timestamp} |
| **Rows x Columns** | {s.get('n_rows', 'N/A')} x {s.get('n_cols', 'N/A')} |
| **Detected Task** | {fp.get('suggested_task', 'N/A')} |
| **Analysis Query** | {user_query or 'Full automated analysis'} |
| **Platform** | DatacortexAI v0.1 — AI-Orchestrated Data Science |

---"""

    async def _generate_executive_summary(
        self, file_name, user_query, fingerprint, findings, model_metrics
    ) -> str:
        """Generate executive summary — LLM with template fallback."""
        fp_data = fingerprint.data if fingerprint else {}
        findings_data = findings.data if findings else {}
        metrics_data = model_metrics.data if model_metrics else {}

        # Try LLM narrative
        try:
            prompt = f"""Write a concise executive summary (2-3 paragraphs) for a data analysis report.
File analyzed: {file_name}
User's question: {user_query or 'General analysis'}
Dataset: {str(fp_data)[:1500]}
Findings: {str(findings_data)[:1500]}
Model results: {str(metrics_data)[:500]}
Write in professional language. No markdown headers. No emojis."""

            response = await self.llm.narrate(prompt)
            narrative = response.content
        except Exception as e:
            logger.info(f"LLM executive summary skipped: {e}")
            # Template fallback
            task = fp_data.get('suggested_task', 'analysis')
            n_rows = fp_data.get('n_rows', 'N/A')
            n_cols = fp_data.get('n_cols', 'N/A')
            n_findings = len(findings_data.get('key_findings', [])) if findings_data else 0
            best_model = metrics_data.get('best_model', 'N/A') if metrics_data else 'N/A'
            best_score = metrics_data.get('best_score', 0) if metrics_data else 0
            models = fp_data.get('suggested_models', [])

            narrative = (
                f"This report presents the results of an automated analytical pipeline applied to "
                f"`{file_name}`, a dataset comprising **{n_rows} observations** across **{n_cols} features**. "
                f"The platform classified this as a **{task}** problem based on the computed data fingerprint, "
                f"which accounts for sparsity, correlation density, class distribution, and noise characteristics.\n\n"
                f"The exploratory analysis phase identified **{n_findings} statistically significant findings** "
                f"spanning feature distributions, inter-variable correlations, and data quality metrics."
            )
            if best_model and best_model != 'N/A':
                narrative += (
                    f" In the modeling phase, **{best_model}** achieved the highest test score of "
                    f"**{best_score:.4f}**, selected from a pool of {len(models)} candidate algorithms "
                    f"({', '.join(models[:3])}) chosen via adaptive fingerprint-based model selection."
                )
            narrative += (
                "\n\nThe following sections detail each phase of the analysis, from data ingestion "
                "through model evaluation, concluding with actionable recommendations."
            )

        return f"## Executive Summary\n\n{narrative}"

    def _generate_methodology(self, fingerprint) -> str:
        """Generate methodology section describing the analysis approach."""
        fp = fingerprint.data if fingerprint else {}
        task = fp.get('suggested_task', 'analysis')
        models = fp.get('suggested_models', ['N/A'])

        return f"""## Methodology

The analysis follows a structured five-phase pipeline orchestrated by the DatacortexAI meta-cognitive planner:

1. **Data Ingestion** — File parsing, schema detection (type inference for each column), automated data quality assessment, and cleaning (duplicate removal, null handling, constant-column pruning).

2. **Data Fingerprinting** — Computation of a multi-dimensional dataset signature capturing sparsity ({fp.get('sparsity', 0):.1%}), correlation density ({fp.get('correlation_density', 0):.3f}), class balance ({fp.get('class_balance', 0):.3f}), and noise estimate ({fp.get('noise_estimate', 0):.3f}). This fingerprint drives adaptive model selection.

3. **Exploratory Data Analysis** — Statistical profiling (mean, median, standard deviation, skewness), pairwise correlation analysis with significance filtering, and distribution characterization for all features.

4. **Model Training & Evaluation** — Fingerprint-guided selection of {len(models)} candidate algorithms ({', '.join(models[:5])}). Each model is trained with stratified hold-out validation (80/20 split) and evaluated via k-fold cross-validation. Detected task type: **{task}**.

5. **Report Synthesis** — Aggregation of cross-phase artifacts into this structured analytical narrative."""

    def _generate_data_section(self, schema, quality, cleaning_log, fingerprint) -> str:
        """Generate the data overview section."""
        sections = ["## Data Overview\n"]

        if schema and schema.data:
            s = schema.data
            sections.append(f"### Dataset Dimensions\n\n- **Rows**: {s.get('n_rows', 'N/A'):,}\n- **Columns**: {s.get('n_cols', 'N/A')}\n")

            # Column table
            columns = s.get("columns", [])
            if columns:
                sections.append("### Column Profiles\n")
                sections.append("| Column | Data Type | Semantic Type | Unique Values | Missing % | Notes |")
                sections.append("|--------|-----------|---------------|---------------|-----------|-------|")
                for col in columns[:30]:
                    dtype = col.get('dtype', '')
                    stype = col.get('semantic_type', '')
                    null_pct = col.get('null_pct', 0)
                    n_unique = col.get('n_unique', '')
                    notes = []
                    if null_pct > 20:
                        notes.append("High nulls")
                    if stype == 'categorical' and n_unique and n_unique <= 2:
                        notes.append("Binary")
                    if col.get('std') and col.get('mean') and col['mean'] != 0:
                        cv = abs(col['std'] / col['mean'])
                        if cv > 1:
                            notes.append(f"High variance (CV={cv:.1f})")
                    sections.append(
                        f"| {col.get('name', '')} | `{dtype}` | {stype} | "
                        f"{n_unique} | {null_pct:.1f}% | {', '.join(notes) if notes else '—'} |"
                    )
                sections.append("")

        if quality and quality.data:
            q = quality.data
            sections.append("### Data Quality Assessment\n")
            overall = q.get('overall_score', 0)
            rating = "Excellent" if overall >= 0.95 else "Good" if overall >= 0.8 else "Fair" if overall >= 0.6 else "Poor"
            sections.append(f"| Metric | Value | Rating |")
            sections.append(f"|--------|-------|--------|")
            sections.append(f"| Overall Quality | {overall:.1%} | {rating} |")
            sections.append(f"| Missing Values | {q.get('missing_pct', 0):.1f}% ({q.get('missing_cells', 0):,} cells) | {'Pass' if q.get('missing_pct', 0) < 5 else 'Review'} |")
            sections.append(f"| Duplicate Rows | {q.get('duplicate_pct', 0):.1f}% | {'Pass' if q.get('duplicate_pct', 0) < 1 else 'Review'} |")
            sections.append("")

            # Column-level issues
            col_quality = q.get('column_quality', {})
            problematic = {k: v for k, v in col_quality.items() if v.get('issues')}
            if problematic:
                sections.append("**Column-Level Issues Detected:**\n")
                for col_name, cq in list(problematic.items())[:10]:
                    for issue in cq.get('issues', []):
                        sections.append(f"- `{col_name}`: {issue}")
                sections.append("")

        if cleaning_log and cleaning_log.data:
            cl = cleaning_log.data
            actions = cl.get("actions", [])
            if actions:
                sections.append("### Automated Cleaning Applied\n")
                for action in actions:
                    sections.append(f"- {action}")
                sections.append(f"\n*Rows: {cl.get('rows_before', '?')} → {cl.get('rows_after', '?')} | "
                                f"Columns: {cl.get('cols_before', '?')} → {cl.get('cols_after', '?')}*\n")

        if fingerprint and fingerprint.data:
            fp = fingerprint.data
            sections.append("### Data Fingerprint\n")
            sections.append("The data fingerprint is a multi-dimensional signature used for adaptive model selection:\n")
            sections.append("| Dimension | Value | Interpretation |")
            sections.append("|-----------|-------|----------------|")
            sparsity = fp.get('sparsity', 0)
            sections.append(f"| Sparsity | {sparsity:.1%} | {'Dense' if sparsity < 0.1 else 'Moderate' if sparsity < 0.5 else 'Sparse'} |")
            corr_d = fp.get('correlation_density', 0)
            sections.append(f"| Correlation Density | {corr_d:.3f} | {'Low' if corr_d < 0.3 else 'Moderate' if corr_d < 0.6 else 'High'} inter-feature dependency |")
            noise = fp.get('noise_estimate', 0)
            sections.append(f"| Noise Estimate | {noise:.3f} | {'Low noise' if noise < 0.2 else 'Moderate noise' if noise < 0.5 else 'High noise'} |")
            balance = fp.get('class_balance', 0)
            sections.append(f"| Class Balance | {balance:.3f} | {'Balanced' if balance > 0.8 else 'Imbalanced — consider resampling'} |")
            sections.append(f"\n**Suggested Task**: {fp.get('suggested_task', 'N/A')}")
            if fp.get("suggested_models"):
                sections.append(f"**Recommended Algorithms**: {', '.join(fp['suggested_models'][:5])}")
            sections.append("")

        return "\n".join(sections)

    def _generate_eda_section(self, statistics, correlations, findings) -> str:
        """Generate the EDA section."""
        sections = ["## Exploratory Data Analysis\n"]

        if findings and findings.data:
            fd = findings.data
            if fd.get("data_story"):
                sections.append(f"### Narrative Summary\n\n{fd['data_story']}\n")

            key_findings = fd.get("key_findings", [])
            if key_findings:
                sections.append("### Key Findings\n")
                for i, f in enumerate(key_findings[:10], 1):
                    importance = f.get("importance", "medium")
                    marker = "**HIGH**" if importance == "high" else "MEDIUM" if importance == "medium" else "LOW"
                    category = f.get('category', '').title()
                    sections.append(f"{i}. [{marker}] *{category}* — {f.get('finding', '')}")
                sections.append("")

        if correlations and correlations.data:
            strong_pairs = correlations.data.get("strong_pairs", [])
            if strong_pairs:
                sections.append("### Correlation Analysis\n")
                sections.append("Pairs with |r| > 0.5 are listed below. Strong correlations may indicate "
                               "redundant features or meaningful relationships.\n")
                sections.append("| Feature A | Feature B | Pearson r | Strength | Implication |")
                sections.append("|-----------|-----------|-----------|----------|-------------|")
                for pair in strong_pairs[:10]:
                    r = pair['correlation']
                    strength = pair['strength']
                    direction = "positive" if r > 0 else "negative"
                    implication = "Consider dropping one" if abs(r) > 0.9 else "Potential interaction feature" if abs(r) > 0.7 else "Worth monitoring"
                    sections.append(
                        f"| {pair['col1']} | {pair['col2']} | "
                        f"{r:.3f} | {strength} ({direction}) | {implication} |"
                    )
                sections.append("")

            target_corrs = correlations.data.get("target_correlations", {})
            if target_corrs:
                sections.append("### Feature-Target Correlations\n")
                sections.append("Ranked by absolute correlation with the target variable:\n")
                sections.append("| Rank | Feature | |Correlation| | Predictive Signal |")
                sections.append("|------|---------|-------------|-------------------|")
                for rank, (feat, corr) in enumerate(list(target_corrs.items())[:10], 1):
                    signal = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                    bar = "#" * int(abs(corr) * 20)
                    sections.append(f"| {rank} | {feat} | {abs(corr):.3f} `{bar}` | {signal} |")
                sections.append("")

        return "\n".join(sections)

    def _generate_distribution_section(self, statistics) -> str:
        """Generate detailed distribution analysis."""
        sections = ["## Statistical Profiles\n"]

        if not statistics or not statistics.data:
            return ""

        stats_data = statistics.data

        # Numeric columns
        numeric_stats = {k: v for k, v in stats_data.items()
                         if isinstance(v, dict) and v.get("type") == "numeric"}
        if numeric_stats:
            sections.append("### Numeric Features\n")
            sections.append("| Feature | Mean | Median | Std Dev | Min | Max | Skewness | Nulls |")
            sections.append("|---------|------|--------|---------|-----|-----|----------|-------|")
            for col, s in list(numeric_stats.items())[:20]:
                skew = s.get('skew', 0)
                skew_note = ""
                if skew and abs(skew) > 1.5:
                    skew_note = " (highly skewed)"
                elif skew and abs(skew) > 0.5:
                    skew_note = " (moderately skewed)"
                sections.append(
                    f"| {col} | {s.get('mean', '—')} | {s.get('median', '—')} | "
                    f"{s.get('std', '—')} | {s.get('min', '—')} | {s.get('max', '—')} | "
                    f"{s.get('skew', '—')}{skew_note} | {s.get('null_count', 0)} |"
                )
            sections.append("")

            # Add IQR analysis
            sections.append("**Distribution Notes:**\n")
            for col, s in list(numeric_stats.items())[:10]:
                q25 = s.get('q25')
                q75 = s.get('q75')
                if q25 is not None and q75 is not None:
                    iqr = q75 - q25
                    lower = q25 - 1.5 * iqr
                    upper = q75 + 1.5 * iqr
                    min_val = s.get('min', 0)
                    max_val = s.get('max', 0)
                    if min_val < lower or max_val > upper:
                        sections.append(f"- `{col}`: Potential outliers detected (IQR method: values outside [{lower:.2f}, {upper:.2f}])")
            sections.append("")

        # Categorical columns
        cat_stats = {k: v for k, v in stats_data.items()
                     if isinstance(v, dict) and v.get("type") == "categorical"}
        if cat_stats:
            sections.append("### Categorical Features\n")
            for col, s in list(cat_stats.items())[:10]:
                sections.append(f"**`{col}`** — {s.get('n_unique', '?')} unique values, mode: `{s.get('mode', '?')}`")
                top5 = s.get('top_5', {})
                if top5:
                    total = sum(top5.values())
                    sections.append("| Value | Count | Proportion |")
                    sections.append("|-------|-------|------------|")
                    for val, count in list(top5.items())[:5]:
                        pct = count / total * 100 if total > 0 else 0
                        sections.append(f"| {val} | {count} | {pct:.1f}% |")
                    sections.append("")

        return "\n".join(sections)

    def _generate_model_section(self, model_comparison, model_metrics, feature_imp) -> str:
        """Generate the modeling results section."""
        sections = ["## Modeling Results\n"]

        if model_comparison and model_comparison.data:
            models = model_comparison.data.get("models", [])
            exec_mode = model_comparison.data.get("execution_mode", "sandbox")
            if exec_mode == "local":
                sections.append("*Models trained locally via in-process scikit-learn execution.*\n")

            if models:
                sections.append("### Model Comparison\n")
                sections.append("| Rank | Model | Test Score | CV Mean +/- Std | Metric |")
                sections.append("|------|-------|-----------|-----------------|--------|")
                for rank, m in enumerate(models, 1):
                    if "error" in m:
                        sections.append(f"| {rank} | {m['name']} | Error | — | — |")
                    else:
                        cv_str = f"{m.get('cv_mean', 0):.4f} +/- {m.get('cv_std', 0):.4f}"
                        sections.append(
                            f"| {rank} | {m['name']} | {m.get('test_score', 0):.4f} | "
                            f"{cv_str} | {m.get('scoring', 'N/A')} |"
                        )
                sections.append("")

                # Add model interpretation
                if len(models) > 1:
                    best = models[0]
                    second = models[1] if len(models) > 1 else None
                    if not best.get("error") and best.get("test_score", 0) > 0:
                        gap = best.get("test_score", 0) - (second.get("test_score", 0) if second and not second.get("error") else 0)
                        sections.append(f"**Analysis**: {best['name']} outperforms the runner-up by "
                                       f"{gap:.4f} on the test set. ")
                        cv_std = best.get("cv_std", 0)
                        if cv_std > 0.1:
                            sections.append(f"However, the cross-validation standard deviation ({cv_std:.4f}) "
                                           f"suggests notable variance — results may differ on new data splits.\n")
                        else:
                            sections.append(f"The low cross-validation variance ({cv_std:.4f}) "
                                           f"indicates stable generalization.\n")

        if model_metrics and model_metrics.data:
            mm = model_metrics.data
            if mm.get('best_model') and mm['best_model'] != 'None':
                sections.append(f"### Best Model: **{mm.get('best_model', 'N/A')}**\n")
                sections.append(f"- **Test Score**: {mm.get('best_score', 0):.4f}")
                sections.append(f"- **Task Type**: {mm.get('task_type', 'N/A')}\n")

        if feature_imp and feature_imp.data:
            imp = feature_imp.data
            if imp:
                sections.append("### Feature Importance (Top 10)\n")
                sections.append("| Rank | Feature | Importance | Relative Contribution |")
                sections.append("|------|---------|------------|----------------------|")
                max_imp = max(imp.values()) if imp.values() else 1
                for rank, (feat, score) in enumerate(list(imp.items())[:10], 1):
                    bar = "#" * int(score / max_imp * 20) if isinstance(score, (int, float)) and max_imp > 0 else ""
                    pct = score / max_imp * 100 if max_imp > 0 else 0
                    sections.append(f"| {rank} | {feat} | {score:.4f} | {pct:.1f}% `{bar}` |")
                sections.append("")

        if (not model_comparison or not model_comparison.data or not model_comparison.data.get("models")) and \
           (not model_metrics or not model_metrics.data or not model_metrics.data.get("best_model")):
            sections.append("*No model training was performed for this dataset (insufficient data or no target column specified).*\n")

        return "\n".join(sections)

    def _generate_limitations(self, schema, quality, model_metrics, fingerprint) -> str:
        """Generate limitations and caveats section."""
        sections = ["## Limitations and Caveats\n"]
        items = []

        s = schema.data if schema else {}
        n_rows = s.get('n_rows', 0)
        if n_rows < 100:
            items.append(f"**Small sample size** ({n_rows} rows): Results may not generalize. "
                        f"Cross-validation helps but cannot fully compensate. Collect more data if possible.")
        if n_rows < 30:
            items.append("**Critically small dataset**: Statistical tests have low power. "
                        "Treat all findings as preliminary hypotheses requiring validation.")

        q = quality.data if quality else {}
        missing_pct = q.get('missing_pct', 0)
        if missing_pct > 5:
            items.append(f"**Missing data** ({missing_pct:.1f}%): Imputation was applied but may "
                        f"introduce bias, particularly if data is not missing at random.")

        fp = fingerprint.data if fingerprint else {}
        noise = fp.get('noise_estimate', 0)
        if noise > 0.3:
            items.append(f"**High noise estimate** ({noise:.3f}): The dataset exhibits substantial noise, "
                        f"which may limit achievable model accuracy.")

        mm = model_metrics.data if model_metrics else {}
        score = mm.get('best_score', 0)
        if score and score < 0.7:
            items.append(f"**Moderate model performance** (best score: {score:.4f}): "
                        f"Consider feature engineering, additional data collection, or domain-specific transformations.")

        items.append("**Automated analysis**: This report was generated algorithmically. "
                    "Domain expert review is recommended before acting on findings.")
        items.append("**No causal claims**: Correlations identified do not imply causation. "
                    "Experimental or quasi-experimental designs are needed for causal inference.")

        for item in items:
            sections.append(f"- {item}")
        sections.append("")

        return "\n".join(sections)

    async def _generate_recommendations(
        self, upstream_context, user_query, findings, model_metrics, fingerprint
    ) -> str:
        """Generate recommendations — LLM with template fallback."""
        try:
            prompt = f"""Based on this data analysis, generate 5-7 specific, actionable recommendations.
Context: {upstream_context[:3000]}
Question: {user_query or 'General analysis'}
Cover: data quality improvements, feature engineering ideas, model tuning, deployment considerations, and next steps.
Use numbered list. No headers. No emojis. Professional language."""

            response = await self.llm.narrate(prompt)
            recs_text = response.content
        except Exception as e:
            logger.info(f"LLM recommendations skipped: {e}")
            fp = fingerprint.data if fingerprint else {}
            mm = model_metrics.data if model_metrics else {}
            fd = findings.data if findings else {}

            recs = []
            # Data quality
            recs.append("1. **Data collection**: Increase sample size to improve model stability and statistical power. "
                       "Target at least 500-1000 observations for reliable results.")

            # Feature engineering from findings
            for rec in fd.get('recommendations', []):
                recs.append(f"1. {rec}")

            # Model-specific
            if mm.get('best_model'):
                recs.append(f"1. **Hyperparameter tuning**: Perform grid or Bayesian search on {mm['best_model']} "
                           f"to potentially improve the {mm.get('best_score', 0):.4f} score.")
            recs.append("1. **Feature engineering**: Create interaction terms for the most correlated feature pairs "
                       "and test polynomial features for numeric columns.")
            recs.append("1. **Ensemble methods**: Consider stacking the top 2-3 models for improved accuracy.")
            recs.append("1. **Validation strategy**: Before deployment, validate on a completely held-out dataset "
                       "collected at a different time period to assess temporal stability.")
            recs.append("1. **Monitoring**: Implement data drift detection to alert when incoming data "
                       "deviates significantly from the training distribution.")

            recs_text = "\n".join(recs)

        return f"## Recommendations\n\n{recs_text}"

    def _generate_appendix(self, schema, correlations, fingerprint) -> str:
        """Generate technical appendix with raw details."""
        sections = ["## Technical Appendix\n"]

        sections.append("### Pipeline Configuration\n")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        sections.append("| Train/Test Split | 80/20, stratified |")
        sections.append("| Cross-Validation | k-fold (k=5 or adjusted for small data) |")
        sections.append("| Feature Scaling | StandardScaler (zero mean, unit variance) |")
        sections.append("| Missing Value Strategy | Median imputation |")
        sections.append("| Categorical Encoding | Label encoding |")
        sections.append("")

        if fingerprint and fingerprint.data:
            fp = fingerprint.data
            sections.append("### Raw Fingerprint Vector\n")
            sections.append("```")
            for key, val in fp.items():
                if key not in ('suggested_models',):
                    sections.append(f"  {key}: {val}")
            sections.append("```\n")

        sections.append("---\n")
        sections.append("*Report generated by DatacortexAI — Enterprise Data Science Platform.*")

        return "\n".join(sections)
