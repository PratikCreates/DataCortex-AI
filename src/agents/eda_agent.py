"""
DatacortexAI — Exploratory Data Analysis Agent

Generates statistical profiles, correlations, distributions,
and visual analysis. Uses direct pandas computation as primary path
with LLM enrichment as optional enhancement.

FIX: Removed dependency on LLM for core analysis. Direct computation
is used for all statistics, correlations, and findings. LLM is only
used for narrative synthesis (with graceful fallback if unavailable).
"""

import io
import json
import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType
from src.core.data_fingerprint import DataFingerprint

logger = logging.getLogger(__name__)


class EDAAgent:
    """
    Specialized agent for Exploratory Data Analysis.

    ALL core analysis (stats, correlations, findings) is done via
    direct pandas/numpy computation — no LLM dependency.
    LLM is only used optionally for narrative synthesis.
    """

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def run(
        self,
        df_info: dict,
        fingerprint: Optional[DataFingerprint],
        target_column: Optional[str],
        registry: ArtifactRegistry,
    ) -> dict:
        """Execute the EDA phase using direct computation."""
        logger.info("EDA Agent started")

        # Reconstruct DataFrame from ingestion CSV
        df = self._load_dataframe(df_info)

        # ── Step 1: Compute statistics directly ───────
        stats_result = self._compute_statistics(df)
        registry.register(Phase.EDA, ArtifactType.STATISTICS, stats_result)
        logger.info(f"Statistics computed for {len(stats_result) - 1} columns")

        # ── Step 2: Correlation analysis ──────────────
        corr_result = self._compute_correlations(df, target_column)
        registry.register(Phase.EDA, ArtifactType.CORRELATION, corr_result)
        logger.info(f"Found {len(corr_result.get('strong_pairs', []))} strong correlations")

        # ── Step 3: Generate visualization code ───────
        viz_code = self._generate_viz_code(df, fingerprint, target_column)
        registry.register(Phase.EDA, ArtifactType.CODE, {
            "visualization_code": viz_code,
            "purpose": "EDA visualizations",
        })

        # ── Step 4: Synthesize key findings ───────────
        findings = self._extract_findings(df, stats_result, corr_result, fingerprint, target_column)

        # Optionally enhance with LLM narrative (non-blocking)
        findings = await self._try_llm_narrative(findings, stats_result, corr_result, df_info, fingerprint, target_column)
        registry.register(Phase.EDA, ArtifactType.FINDINGS, findings)

        result = {
            "statistics": stats_result,
            "correlations": corr_result,
            "visualization_code": viz_code,
            "findings": findings.get("key_findings", []),
        }

        logger.info(f"EDA complete: {len(result['findings'])} key findings")
        return result

    # ─── Direct Computation Methods ──────────────────

    def _load_dataframe(self, df_info: dict) -> pd.DataFrame:
        """Reconstruct DataFrame from ingestion output."""
        csv_data = df_info.get("df_csv", "")
        if csv_data:
            return pd.read_csv(io.StringIO(csv_data))
        # Fallback: build from sample
        sample = df_info.get("schema", {}).get("columns", [])
        return pd.DataFrame()

    def _compute_statistics(self, df: pd.DataFrame) -> dict:
        """Compute descriptive statistics directly with pandas."""
        stats = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            stats[col] = {
                "type": "numeric",
                "mean": round(float(s.mean()), 4) if len(s) > 0 else None,
                "median": round(float(s.median()), 4) if len(s) > 0 else None,
                "std": round(float(s.std()), 4) if len(s) > 0 else None,
                "min": round(float(s.min()), 4) if len(s) > 0 else None,
                "max": round(float(s.max()), 4) if len(s) > 0 else None,
                "q25": round(float(s.quantile(0.25)), 4) if len(s) > 0 else None,
                "q75": round(float(s.quantile(0.75)), 4) if len(s) > 0 else None,
                "skew": round(float(s.skew()), 4) if len(s) > 0 else None,
                "null_count": int(df[col].isnull().sum()),
            }

        for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
            top_vals = df[col].value_counts().head(5).to_dict()
            stats[col] = {
                "type": "categorical",
                "n_unique": int(df[col].nunique()),
                "mode": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None,
                "top_5": {str(k): int(v) for k, v in top_vals.items()},
                "null_count": int(df[col].isnull().sum()),
            }

        stats["_meta"] = {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "total_missing": int(df.isnull().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        return stats

    def _compute_correlations(self, df: pd.DataFrame, target_column: Optional[str]) -> dict:
        """Compute correlation matrix directly with pandas."""
        result = {"correlations": {}, "strong_pairs": []}

        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return result

        corr = numeric_df.corr()
        result["correlations"] = {
            str(k): {str(kk): round(vv, 3) for kk, vv in v.items()}
            for k, v in corr.to_dict().items()
        }

        # Find strong correlations
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if val > 0.5 and not np.isnan(val):
                    result["strong_pairs"].append({
                        "col1": str(corr.columns[i]),
                        "col2": str(corr.columns[j]),
                        "correlation": round(float(corr.iloc[i, j]), 3),
                        "strength": "strong" if val > 0.7 else "moderate",
                    })

        # Target correlations
        if target_column and target_column in numeric_df.columns:
            target_corr = corr[target_column].drop(target_column).abs().sort_values(ascending=False)
            result["target_correlations"] = {
                str(k): round(float(v), 3) for k, v in target_corr.items()
            }

        return result

    def _extract_findings(
        self, df: pd.DataFrame, stats: dict, correlations: dict,
        fingerprint: Optional[DataFingerprint], target_column: Optional[str],
    ) -> dict:
        """Extract key findings from computed statistics — no LLM needed."""
        findings = []

        # Dataset size finding
        n_rows = len(df)
        n_cols = len(df.columns)
        findings.append({
            "finding": f"Dataset contains {n_rows:,} rows and {n_cols} columns",
            "importance": "medium",
            "category": "distribution",
        })

        # Missing data
        total_missing = int(df.isnull().sum().sum())
        total_cells = n_rows * n_cols
        if total_missing > 0:
            pct = total_missing / max(total_cells, 1) * 100
            findings.append({
                "finding": f"{pct:.1f}% of values are missing ({total_missing:,} cells)",
                "importance": "high" if pct > 10 else "medium",
                "category": "quality",
            })

        # Highly correlated pairs
        strong_pairs = correlations.get("strong_pairs", [])
        if strong_pairs:
            top = strong_pairs[0]
            findings.append({
                "finding": f"Strong correlation ({top['correlation']:.2f}) between {top['col1']} and {top['col2']}",
                "importance": "high",
                "category": "correlation",
            })

        # Skewed distributions
        for col, s in stats.items():
            if isinstance(s, dict) and s.get("type") == "numeric" and s.get("skew"):
                if abs(s["skew"]) > 1.5:
                    findings.append({
                        "finding": f"Column '{col}' is highly skewed (skew={s['skew']:.2f})",
                        "importance": "medium",
                        "category": "distribution",
                    })

        # Class imbalance for target
        if target_column and target_column in df.columns:
            if df[target_column].dtype == "object" or df[target_column].nunique() <= 10:
                vc = df[target_column].value_counts()
                if len(vc) > 0:
                    ratio = vc.min() / vc.max()
                    if ratio < 0.3:
                        findings.append({
                            "finding": f"Target '{target_column}' has class imbalance (ratio={ratio:.2f})",
                            "importance": "high",
                            "category": "pattern",
                        })

        # Target correlations
        target_corrs = correlations.get("target_correlations", {})
        if target_corrs:
            top_feature = list(target_corrs.keys())[0]
            top_val = list(target_corrs.values())[0]
            findings.append({
                "finding": f"Most predictive feature for target: '{top_feature}' (|r|={top_val:.3f})",
                "importance": "high",
                "category": "correlation",
            })

        # Fingerprint insights
        if fingerprint:
            findings.append({
                "finding": f"Data fingerprint suggests task type: {fingerprint.suggested_task}",
                "importance": "medium",
                "category": "pattern",
            })
            if fingerprint.suggested_models:
                findings.append({
                    "finding": f"Recommended models: {', '.join(fingerprint.suggested_models[:3])}",
                    "importance": "medium",
                    "category": "pattern",
                })

        return {
            "key_findings": findings,
            "recommendations": self._generate_recommendations(df, stats, correlations, target_column),
            "data_story": self._generate_data_story(df, findings, fingerprint, target_column),
        }

    def _generate_recommendations(self, df, stats, correlations, target_column):
        """Generate recommendations from statistics without LLM."""
        recs = []
        total_missing = int(df.isnull().sum().sum())
        if total_missing > 0:
            recs.append("Consider imputing missing values or dropping columns with >50% nulls")
        strong_pairs = correlations.get("strong_pairs", [])
        if len(strong_pairs) > 3:
            recs.append("Multiple strongly correlated features detected — consider dimensionality reduction (PCA)")
        if len(df) < 100:
            recs.append("Small dataset — use cross-validation and simpler models to avoid overfitting")
        if target_column:
            recs.append(f"Focus feature engineering on top correlated features with '{target_column}'")
        return recs

    def _generate_data_story(self, df, findings, fingerprint, target_column):
        """Generate a brief narrative from findings without LLM."""
        task = fingerprint.suggested_task if fingerprint else "analysis"
        n_rows = len(df)
        n_cols = len(df.columns)
        n_findings = len(findings)
        target_str = f" with target column '{target_column}'" if target_column else ""
        return (
            f"This dataset has {n_rows:,} rows and {n_cols} columns{target_str}. "
            f"The suggested analysis task is {task}. "
            f"The automated EDA identified {n_findings} key findings across "
            f"distribution patterns, correlations, and data quality dimensions."
        )

    def _generate_viz_code(
        self, df: pd.DataFrame, fingerprint: Optional[DataFingerprint],
        target_column: Optional[str],
    ) -> str:
        """Generate matplotlib/seaborn visualization code."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]

        code = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, os

os.makedirs('/tmp/viz', exist_ok=True)
sns.set_style('darkgrid')

df = pd.read_csv(io.StringIO(csv_data))
"""
        if len(numeric_cols) >= 2:
            code += f"""
# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('/tmp/viz/correlation_heatmap.png', dpi=100)
    plt.close()
"""
        for i, col in enumerate(numeric_cols[:4]):
            code += f"""
fig, ax = plt.subplots(figsize=(8, 5))
df['{col}'].hist(bins=20, ax=ax, color='steelblue', edgecolor='white')
ax.set_title('{col} Distribution')
ax.set_xlabel('{col}')
plt.tight_layout()
plt.savefig('/tmp/viz/dist_{col}.png', dpi=100)
plt.close()
"""
        if target_column:
            code += f"""
if '{target_column}' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    df['{target_column}'].value_counts().plot(kind='bar', ax=ax, color='coral', edgecolor='white')
    ax.set_title('{target_column} Distribution')
    plt.tight_layout()
    plt.savefig('/tmp/viz/target_distribution.png', dpi=100)
    plt.close()
"""
        return code

    async def _try_llm_narrative(
        self, findings: dict, stats: dict, correlations: dict,
        df_info: dict, fingerprint, target_column,
    ) -> dict:
        """Optionally enhance findings with LLM narrative. Non-blocking."""
        try:
            prompt = f"""Summarize this data analysis in 2-3 sentences:
Findings: {json.dumps(findings.get('key_findings', [])[:5], default=str)}
Target: {target_column or 'None'}
Task type: {fingerprint.suggested_task if fingerprint else 'unknown'}"""

            response = await self.llm.narrate(prompt)
            if response.content.strip():
                findings["data_story"] = response.content.strip()
        except Exception as e:
            logger.info(f"LLM narrative enrichment skipped: {e}")

        return findings
