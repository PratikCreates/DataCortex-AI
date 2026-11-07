"""
DatacortexAI — Data Fingerprinting Module

Adaptive Model Selection via Data Fingerprinting

Computes a multi-dimensional "data fingerprint" from a dataset:
dimensionality, sparsity, class balance, correlation distribution,
noise level, temporal patterns. The fingerprint drives intelligent
model selection and hyperparameter configuration, bypassing
brute-force AutoML grid search.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataFingerprint:
    """
    Data Fingerprint Structure:
    Multi-dimensional characterization of a dataset.
    """
    # Shape
    n_rows: int = 0
    n_cols: int = 0
    dimensionality_ratio: float = 0.0  # n_cols / n_rows

    # Type distribution
    n_numeric: int = 0
    n_categorical: int = 0
    n_temporal: int = 0
    n_text: int = 0
    has_text_cols: bool = False
    has_high_cardinality: bool = False  # categorical with >50 unique

    # Quality
    sparsity: float = 0.0  # % missing values
    duplicate_row_ratio: float = 0.0

    # Statistical
    correlation_density: float = 0.0  # avg absolute correlation
    max_correlation: float = 0.0
    skewness_mean: float = 0.0
    outlier_ratio: float = 0.0  # % values beyond 3σ

    # Target characteristics (for supervised tasks)
    class_balance: float = 1.0  # entropy for classification; 1.0 = balanced
    target_type: str = "unknown"  # "binary", "multiclass", "continuous"
    n_classes: int = 0

    # Noise & complexity
    noise_estimate: float = 0.0  # estimated via spectral analysis
    feature_redundancy: float = 0.0  # ratio of highly correlated feature pairs

    # Task suggestion
    suggested_task: str = "unknown"  # "classification", "regression", "clustering"
    suggested_models: list[str] = None  # type: ignore

    # Metadata
    memory_mb: float = 0.0

    def __post_init__(self):
        if self.suggested_models is None:
            self.suggested_models = []

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_string(self) -> str:
        """Human-readable summary for LLM context."""
        return (
            f"Dataset: {self.n_rows:,} rows × {self.n_cols} cols "
            f"({self.n_numeric} numeric, {self.n_categorical} categorical, {self.n_temporal} temporal)\n"
            f"Sparsity: {self.sparsity:.1%} missing | Duplicates: {self.duplicate_row_ratio:.1%}\n"
            f"Correlation density: {self.correlation_density:.3f} | Max corr: {self.max_correlation:.3f}\n"
            f"Outlier ratio: {self.outlier_ratio:.2%} | Noise estimate: {self.noise_estimate:.3f}\n"
            f"Suggested task: {self.suggested_task} | Target type: {self.target_type}\n"
            f"Suggested models: {', '.join(self.suggested_models)}\n"
            f"Memory: {self.memory_mb:.1f} MB"
        )


class DataFingerprinter:
    """
    Computes DataFingerprint from a pandas DataFrame.
    """

    def compute(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> DataFingerprint:
        """Compute full fingerprint for a DataFrame."""
        fp = DataFingerprint()

        # ─── Shape ────────────────────────────────────
        fp.n_rows, fp.n_cols = df.shape
        fp.dimensionality_ratio = fp.n_cols / max(fp.n_rows, 1)
        fp.memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # ─── Type Distribution ────────────────────────
        fp.n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        fp.n_categorical = len(df.select_dtypes(include=["category", "object", "bool"]).columns)
        fp.n_temporal = len(df.select_dtypes(include=["datetime64", "timedelta64"]).columns)

        # Detect text columns (object cols with high average string length)
        text_candidates = []
        for col in df.select_dtypes(include=["object"]).columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_len = non_null.astype(str).str.len().mean()
                if avg_len > 50:
                    text_candidates.append(col)
        fp.n_text = len(text_candidates)
        fp.has_text_cols = fp.n_text > 0

        # High cardinality detection
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if df[col].nunique() > 50:
                fp.has_high_cardinality = True
                break

        # ─── Quality ──────────────────────────────────
        total_cells = fp.n_rows * fp.n_cols
        fp.sparsity = df.isnull().sum().sum() / max(total_cells, 1)
        fp.duplicate_row_ratio = df.duplicated().sum() / max(fp.n_rows, 1)

        # ─── Statistical Analysis ─────────────────────
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr().abs()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper = corr_matrix.where(mask)
            corr_values = upper.stack().dropna().values
            if len(corr_values) > 0:
                fp.correlation_density = float(np.mean(corr_values))
                fp.max_correlation = float(np.max(corr_values))
                fp.feature_redundancy = float(
                    np.sum(corr_values > 0.9) / max(len(corr_values), 1)
                )

        if len(numeric_df.columns) > 0:
            fp.skewness_mean = float(numeric_df.skew().abs().mean())

            # Outlier estimation (beyond 3σ)
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std().replace(0, np.inf))
            total_numeric_cells = z_scores.size
            fp.outlier_ratio = float(
                (z_scores > 3).sum().sum() / max(total_numeric_cells, 1)
            )

            # Noise estimation via coefficient of variation
            cvs = (numeric_df.std() / numeric_df.mean().abs().replace(0, np.inf)).dropna()
            fp.noise_estimate = float(cvs.mean()) if len(cvs) > 0 else 0.0

        # ─── Target Analysis ──────────────────────────
        if target_col and target_col in df.columns:
            target = df[target_col]
            unique_count = target.nunique()

            if target.dtype in [np.float64, np.float32, np.int64, np.int32]:
                if unique_count <= 2:
                    fp.target_type = "binary"
                    fp.n_classes = unique_count
                elif pd.api.types.is_integer_dtype(target) and unique_count <= min(20, max(3, int(fp.n_rows * 0.2))):
                    fp.target_type = "binary" if unique_count == 2 else "multiclass"
                    fp.n_classes = unique_count
                else:
                    fp.target_type = "continuous"
                    fp.n_classes = 0
            else:
                fp.target_type = "binary" if unique_count == 2 else "multiclass"
                fp.n_classes = unique_count

            # Class balance (entropy-based)
            if fp.target_type in ("binary", "multiclass"):
                value_counts = target.value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                max_entropy = np.log2(max(unique_count, 2))
                fp.class_balance = float(entropy / max(max_entropy, 1e-10))

        if target_col is None and self._looks_like_market_basket(df):
            fp.suggested_task = "market_basket"
            fp.suggested_models = ["ItemFrequency", "PairFrequency", "AssociationRules"]
            return fp

        # ─── Task Suggestion ──────────────────────────
        fp.suggested_task, fp.suggested_models = self._suggest_task_and_models(fp, target_col)

        return fp

    def _looks_like_market_basket(self, df: pd.DataFrame) -> bool:
        """Detect item-slot transaction tables where nulls mean absent items."""
        if df.empty or len(df.columns) < 4:
            return False
        item_slot_names = sum(
            str(col).lower().startswith("item") for col in df.columns
        ) / max(len(df.columns), 1)
        object_cols = df.select_dtypes(include=["object", "category"]).columns
        object_ratio = len(object_cols) / max(len(df.columns), 1)
        if object_ratio < 0.5 and item_slot_names < 0.5:
            return False
        missing_ratio = df[object_cols].isna().mean().mean()
        repeated_value_ratio = sum(
            df[col].nunique(dropna=True) < len(df) * 0.25 for col in object_cols
        ) / max(len(object_cols), 1)
        return missing_ratio > 0.2 and (repeated_value_ratio >= 0.5 or item_slot_names >= 0.5)

    def validate_consistency(
        self,
        original: DataFingerprint,
        modified: DataFingerprint,
        threshold_row_loss: float = 0.15,
        threshold_sparsity_drift: float = 0.2,
    ) -> list[str]:
        """
        🛡️ CLAIM 2: Cross-Phase Data Integrity Protection
        
        Compare two fingerprints to detect anomalies introduced by 
        automated data mutations.
        
        Returns:
            List of anomaly descriptions (empty if consistent)
        """
        anomalies = []

        # 1. Critical Row Loss
        row_delta = (original.n_rows - modified.n_rows) / max(original.n_rows, 1)
        if row_delta > threshold_row_loss:
            anomalies.append(
                f"Significant Row Loss: {row_delta:.1%} of data dropped "
                f"({original.n_rows} -> {modified.n_rows})"
            )
        elif row_delta < -0.01: # Small growth is fine (augmentation), but large is suspicious
             pass

        # 2. Sparsity Spike (Increased missingness)
        if modified.sparsity > original.sparsity + threshold_sparsity_drift:
            anomalies.append(
                f"Sparsity Spike: Missing values increased from "
                f"{original.sparsity:.1%} to {modified.sparsity:.1%}"
            )

        # 3. Statistical Drift (Noise/Complexity)
        if abs(original.noise_estimate - modified.noise_estimate) > 0.5:
             anomalies.append(
                 f"Statistical Drift: Noise estimate shifted significantly "
                 f"({original.noise_estimate:.3f} -> {modified.noise_estimate:.3f})"
             )

        # 4. Target Type Mutation (Should NOT happen)
        if original.target_type != modified.target_type and original.target_type != "unknown":
             anomalies.append(
                 f"Target Mutation: Data task changed from "
                 f"{original.target_type} to {modified.target_type}"
             )

        return anomalies

    def _suggest_task_and_models(
        self,
        fp: DataFingerprint,
        target_col: Optional[str],
    ) -> tuple[str, list[str]]:
        """
        INTELLIGENCE of CLAIM 4: Suggest task type and model families
        based on fingerprint characteristics.
        """
        if target_col is None:
            return "clustering", ["KMeans", "DBSCAN", "AgglomerativeClustering"]

        if fp.target_type == "continuous":
            task = "regression"
            models = ["LinearRegression", "Ridge", "RandomForestRegressor", "HistGradientBoostingRegressor"]

            if fp.n_rows > 10000:
                models.append("GradientBoostingRegressor")
            if fp.correlation_density > 0.5:
                models.insert(0, "Lasso")  # Feature selection via L1
            if fp.noise_estimate > 1.0:
                models.append("SVR")
            if fp.dimensionality_ratio < 0.01:
                models.append("XGBRegressor")

        else:
            task = "classification"
            models = ["LogisticRegression", "RandomForestClassifier", "HistGradientBoostingClassifier"]

            if fp.n_classes > 5:
                models.append("GradientBoostingClassifier")
            if fp.n_rows > 5000:
                models.append("XGBClassifier")
            if fp.class_balance < 0.7:
                # Imbalanced: prefer models with class_weight support
                models = [
                    "RandomForestClassifier",
                    "HistGradientBoostingClassifier",
                    "GradientBoostingClassifier",
                    "LogisticRegression",
                ]
            if fp.dimensionality_ratio > 0.1:
                models.insert(0, "SVC")  # Good for high-dimensional
            if fp.n_numeric > 0 and fp.n_categorical == 0:
                models.append("KNeighborsClassifier")

        return task, models
