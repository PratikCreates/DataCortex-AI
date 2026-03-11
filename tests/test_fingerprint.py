"""
Test — Data Fingerprinting Module

Tests fingerprint computation on various synthetic datasets:
small, sparse, imbalanced, temporal.
"""

import pytest
import pandas as pd
import numpy as np

from src.core.data_fingerprint import DataFingerprinter, DataFingerprint


@pytest.fixture
def fingerprinter():
    return DataFingerprinter()


@pytest.fixture
def iris_df():
    """Load iris test fixture."""
    return pd.read_csv("tests/fixtures/iris.csv")


@pytest.fixture
def sparse_df():
    """Create sparse dataset with lots of missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.choice(["x", "y", "z"], 100),
        "target": np.random.choice([0, 1], 100),
    })
    # Insert 30% nulls
    mask = np.random.random(df.shape) < 0.3
    df = df.mask(mask)
    df["target"] = np.random.choice([0, 1], 100)  # Restore target
    return df


@pytest.fixture
def imbalanced_df():
    """Create class-imbalanced dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "target": [0] * 180 + [1] * 20,  # 90% class 0
    })
    return df


class TestDataFingerprint:

    def test_basic_fingerprint(self, fingerprinter, iris_df):
        fp = fingerprinter.compute(iris_df, target_col="Species")
        assert fp.n_rows == 150
        assert fp.n_cols == 5
        assert fp.n_numeric == 4
        assert fp.n_categorical == 1  # Species
        assert fp.suggested_task == "classification"
        assert fp.target_type == "multiclass"
        assert fp.n_classes == 3
        assert len(fp.suggested_models) > 0

    def test_sparsity_detection(self, fingerprinter, sparse_df):
        fp = fingerprinter.compute(sparse_df, target_col="target")
        assert fp.sparsity > 0.1  # At least 10% missing
        assert fp.n_rows == 100

    def test_imbalanced_detection(self, fingerprinter, imbalanced_df):
        fp = fingerprinter.compute(imbalanced_df, target_col="target")
        assert fp.class_balance < 0.9  # Detects imbalance
        assert fp.target_type == "binary"
        assert fp.n_classes == 2
        # Should suggest models with class_weight support
        assert "RandomForestClassifier" in fp.suggested_models

    def test_regression_detection(self, fingerprinter):
        np.random.seed(42)
        df = pd.DataFrame({
            "x1": np.random.randn(200),
            "x2": np.random.randn(200),
            "price": np.random.randn(200) * 100 + 500,
        })
        fp = fingerprinter.compute(df, target_col="price")
        assert fp.suggested_task == "regression"
        assert fp.target_type == "continuous"

    def test_clustering_suggestion_no_target(self, fingerprinter, iris_df):
        fp = fingerprinter.compute(iris_df, target_col=None)
        assert fp.suggested_task == "clustering"

    def test_fingerprint_summary_string(self, fingerprinter, iris_df):
        fp = fingerprinter.compute(iris_df, target_col="Species")
        summary = fp.to_summary_string()
        assert "150" in summary
        assert "classification" in summary

    def test_fingerprint_dict_serialization(self, fingerprinter, iris_df):
        fp = fingerprinter.compute(iris_df, target_col="Species")
        d = fp.to_dict()
        assert isinstance(d, dict)
        assert d["n_rows"] == 150
        assert "suggested_models" in d

    def test_correlation_density(self, fingerprinter):
        np.random.seed(42)
        x = np.random.randn(100)
        df = pd.DataFrame({
            "a": x,
            "b": x + np.random.randn(100) * 0.1,  # Highly correlated
            "c": np.random.randn(100),  # Independent
            "target": np.random.choice([0, 1], 100),
        })
        fp = fingerprinter.compute(df, target_col="target")
        assert fp.max_correlation > 0.8  # a and b should be highly correlated

    def test_high_cardinality(self, fingerprinter):
        df = pd.DataFrame({
            "id": [f"user_{i}" for i in range(100)],
            "value": np.random.randn(100),
        })
        fp = fingerprinter.compute(df)
        assert fp.has_high_cardinality is True


class TestDataFingerprintModelSuggestion:

    def test_small_dataset_models(self, fingerprinter):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "target": np.random.choice([0, 1], 50),
        })
        fp = fingerprinter.compute(df, target_col="target")
        assert len(fp.suggested_models) >= 2

    def test_large_dataset_includes_boosting(self, fingerprinter):
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        })
        fp = fingerprinter.compute(df, target_col="target")
        assert any("Boosting" in m or "XGB" in m for m in fp.suggested_models)
