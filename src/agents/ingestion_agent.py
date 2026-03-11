"""
DatacortexAI — Data Ingestion Agent

Handles automated data ingestion:
- Multi-format parsing (CSV, JSON, Excel, Parquet)
- Schema auto-detection and type inference
- Data quality assessment
- Cleaning recommendations
- Data fingerprint computation
"""

import io
import logging
from typing import Optional, Any

import pandas as pd
import numpy as np

from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.artifact_registry import ArtifactRegistry, Phase, ArtifactType
from src.core.data_fingerprint import DataFingerprinter, DataFingerprint

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Specialized agent for data ingestion and initial quality assessment.

    Responsibilities:
    - Parse uploaded files into DataFrames
    - Detect and infer schema / data types
    - Compute data quality metrics
    - Generate data fingerprint (Claim 4)
    - Register all findings in the artifact registry (Claim 2)
    """

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.fingerprinter = DataFingerprinter()

    async def run(
        self,
        file_content: bytes,
        file_name: str,
        registry: ArtifactRegistry,
        target_column: Optional[str] = None,
    ) -> dict:
        """
        Execute the ingestion phase.

        Returns dict with: df_json, n_rows, n_cols, schema, quality_report, fingerprint
        """
        logger.info(f"Ingestion started: {file_name} ({len(file_content)} bytes)")

        # ── Step 1: Parse the file ────────────────────
        df = self._parse_file(file_content, file_name)
        logger.info(f"Parsed: {df.shape[0]} rows × {df.shape[1]} cols")

        # ── Step 2: Schema detection ──────────────────
        schema = self._detect_schema(df)
        registry.register(Phase.INGESTION, ArtifactType.SCHEMA, schema)

        # ── Step 3: Data quality assessment ───────────
        quality = self._assess_quality(df)
        registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, quality)

        # ── Step 4: Auto-detect target column ─────────
        if not target_column:
            target_column = self._guess_target_column(df, schema)

        # ── Step 5: Basic cleaning ────────────────────
        df_cleaned, cleaning_log = self._auto_clean(df)
        registry.register(Phase.INGESTION, ArtifactType.CLEANING_LOG, cleaning_log)

        # ── Step 6: Compute data fingerprint (Claim 4) ─
        fingerprint = self.fingerprinter.compute(df_cleaned, target_column)
        registry.register(Phase.INGESTION, ArtifactType.DATA_FINGERPRINT, fingerprint.to_dict())

        # ── Step 7: Store sample for downstream agents ─
        sample = df_cleaned.head(20).to_dict(orient="records")
        registry.register(Phase.INGESTION, ArtifactType.DATAFRAME_SAMPLE, {
            "sample": sample,
            "columns": list(df_cleaned.columns),
            "dtypes": {col: str(dtype) for col, dtype in df_cleaned.dtypes.items()},
        })

        # ── Step 8: Store full DataFrame as CSV string ─
        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False)
        df_csv = csv_buffer.getvalue()
        
        registry.register(Phase.INGESTION, ArtifactType.DATA_QUALITY, {
            "full_csv": df_csv,
            "filename": file_name,
        }, metadata={"note": "Full cleaned data for downstream modeling"})

        result = {
            "n_rows": df_cleaned.shape[0],
            "n_cols": df_cleaned.shape[1],
            "schema": schema,
            "quality": quality,
            "cleaning_log": cleaning_log,
            "fingerprint": fingerprint,
            "target_column": target_column,
            "df_csv": df_csv,
            "columns": list(df_cleaned.columns),
        }

        logger.info(
            f"Ingestion complete: {result['n_rows']}×{result['n_cols']}, "
            f"sparsity={fingerprint.sparsity:.1%}, task={fingerprint.suggested_task}"
        )
        return result

    # ─── File Parsing ─────────────────────────────────

    def _parse_file(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse file content into a DataFrame."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"

        try:
            if ext == "csv":
                # Try multiple encodings
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        return pd.read_csv(io.BytesIO(content), encoding=encoding)
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                return pd.read_csv(io.BytesIO(content), encoding="utf-8", on_bad_lines="skip")

            elif ext in ("xls", "xlsx"):
                return pd.read_excel(io.BytesIO(content))

            elif ext == "json":
                return pd.read_json(io.BytesIO(content))

            elif ext == "parquet":
                return pd.read_parquet(io.BytesIO(content))

            elif ext in ("tsv", "tab"):
                return pd.read_csv(io.BytesIO(content), sep="\t")

            else:
                # Default: try CSV
                return pd.read_csv(io.BytesIO(content))

        except Exception as e:
            raise ValueError(f"Failed to parse {filename}: {e}")

    # ─── Schema Detection ─────────────────────────────

    def _detect_schema(self, df: pd.DataFrame) -> dict:
        """Auto-detect column types and statistics."""
        schema = {
            "columns": [],
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }

        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "n_null": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
            }

            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
                col_info["std"] = float(df[col].std()) if not df[col].isnull().all() else None
                col_info["semantic_type"] = "numeric"
            elif df[col].dtype == "object":
                top_values = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                col_info["avg_length"] = float(df[col].dropna().astype(str).str.len().mean()) if len(df[col].dropna()) > 0 else 0
                col_info["semantic_type"] = "text" if col_info["avg_length"] > 50 else "categorical"
            elif "datetime" in str(df[col].dtype):
                col_info["min"] = str(df[col].min())
                col_info["max"] = str(df[col].max())
                col_info["semantic_type"] = "temporal"
            else:
                col_info["semantic_type"] = "other"

            schema["columns"].append(col_info)

        return schema

    # ─── Data Quality Assessment ──────────────────────

    def _assess_quality(self, df: pd.DataFrame) -> dict:
        """Assess overall data quality."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()

        # Column-level quality
        column_quality = {}
        for col in df.columns:
            issues = []
            null_pct = df[col].isnull().mean()
            if null_pct > 0.5:
                issues.append(f"High missing rate ({null_pct:.0%})")
            if null_pct > 0 and null_pct <= 0.5:
                issues.append(f"Has missing values ({null_pct:.0%})")
            if df[col].dtype in [np.float64, np.int64]:
                if df[col].nunique() == 1:
                    issues.append("Constant column (zero variance)")
            if df[col].dtype == "object" and df[col].nunique() == len(df):
                issues.append("Potentially unique identifier")
            column_quality[col] = {
                "issues": issues,
                "quality_score": max(0, 1.0 - len(issues) * 0.25),
            }

        overall_score = 1.0
        if missing_cells > 0:
            overall_score -= min(0.3, (missing_cells / total_cells))
        if duplicate_rows > 0:
            overall_score -= min(0.2, (duplicate_rows / df.shape[0]))

        return {
            "overall_score": round(max(0, overall_score), 3),
            "total_cells": total_cells,
            "missing_cells": int(missing_cells),
            "missing_pct": round(missing_cells / max(total_cells, 1) * 100, 2),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_pct": round(duplicate_rows / max(df.shape[0], 1) * 100, 2),
            "column_quality": column_quality,
        }

    # ─── Target Column Guess ──────────────────────────

    def _guess_target_column(self, df: pd.DataFrame, schema: dict) -> Optional[str]:
        """Heuristically guess the target column."""
        candidates = []

        for col_info in schema["columns"]:
            name = col_info["name"].lower()
            # Common target column names
            target_keywords = [
                "target", "label", "class", "outcome", "result",
                "survived", "price", "salary", "revenue", "y",
                "diagnosis", "default", "churn", "approved",
            ]
            for keyword in target_keywords:
                if keyword in name:
                    candidates.append((col_info["name"], 10))
                    break

            # Last column heuristic
            if col_info["name"] == df.columns[-1]:
                candidates.append((col_info["name"], 3))

        # Return highest scoring candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    # ─── Auto Cleaning ────────────────────────────────

    def _auto_clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Basic automatic data cleaning."""
        log = {"actions": [], "rows_before": len(df), "cols_before": len(df.columns)}

        df_clean = df.copy()

        # Remove fully empty columns
        empty_cols = [c for c in df_clean.columns if df_clean[c].isnull().all()]
        if empty_cols:
            df_clean.drop(columns=empty_cols, inplace=True)
            log["actions"].append(f"Removed {len(empty_cols)} fully empty columns: {empty_cols}")

        # Remove fully empty rows
        empty_rows = df_clean.isnull().all(axis=1).sum()
        if empty_rows > 0:
            df_clean.dropna(how="all", inplace=True)
            log["actions"].append(f"Removed {empty_rows} fully empty rows")

        # Remove exact duplicate rows
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean.drop_duplicates(inplace=True)
            log["actions"].append(f"Removed {dup_count} duplicate rows")

        # Remove constant columns (zero variance)
        const_cols = [c for c in df_clean.columns if df_clean[c].nunique() <= 1]
        if const_cols:
            df_clean.drop(columns=const_cols, inplace=True)
            log["actions"].append(f"Removed {len(const_cols)} constant columns: {const_cols}")

        log["rows_after"] = len(df_clean)
        log["cols_after"] = len(df_clean.columns)

        return df_clean, log
