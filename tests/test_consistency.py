"""
Tests — Cross-Phase Data Integrity (PATENT CLAIM 2)

Validates that DataFingerprinter.validate_consistency() correctly
detects data corruption from automated cleaning. This constitutes
formal evidence for the Cross-Phase Knowledge Propagation claim.
"""
import pytest
import pandas as pd
import numpy as np
from src.core.data_fingerprint import DataFingerprinter, DataFingerprint


@pytest.fixture
def fingerprinter():
    return DataFingerprinter()


@pytest.fixture
def base_fp():
    """A clean, normal dataset fingerprint."""
    return DataFingerprint(
        n_rows=1000,
        n_cols=10,
        sparsity=0.05,
        noise_estimate=0.2,
        target_type="binary",
    )


class TestConsistencyNoAnomalies:
    """Claim 2: No false positives on clean data."""

    def test_identical_fingerprints_pass(self, fingerprinter, base_fp):
        """Same fingerprint should produce zero anomalies."""
        anomalies = fingerprinter.validate_consistency(base_fp, base_fp)
        assert len(anomalies) == 0

    def test_acceptable_row_reduction_passes(self, fingerprinter, base_fp):
        """Removing duplicates (< 15% rows lost) should not trigger alert."""
        modified = DataFingerprint(n_rows=870, n_cols=10, sparsity=0.05,
                                   noise_estimate=0.2, target_type="binary")
        # 870/1000 = 13% loss — below threshold
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert not any("Row Loss" in a for a in anomalies)

    def test_slight_sparsity_change_passes(self, fingerprinter, base_fp):
        """Small sparsity increase (< 0.2 delta) should not trigger."""
        modified = DataFingerprint(n_rows=1000, n_cols=10, sparsity=0.10,
                                   noise_estimate=0.2, target_type="binary")
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert not any("Sparsity" in a for a in anomalies)


class TestConsistencyAnomalyDetection:
    """Claim 2: Correctly detects data corruption."""

    def test_detects_critical_row_loss(self, fingerprinter, base_fp):
        """Losing > 15% of rows triggers Row Loss alert."""
        modified = DataFingerprint(n_rows=700, n_cols=10, sparsity=0.05,
                                   noise_estimate=0.2, target_type="binary")
        # 700/1000 = 30% loss — above threshold
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert any("Row Loss" in a for a in anomalies)
        assert "700" in anomalies[0] or "30" in anomalies[0]

    def test_detects_sparsity_spike(self, fingerprinter, base_fp):
        """Sudden increase in missing values triggers Sparsity Spike alert."""
        modified = DataFingerprint(n_rows=1000, n_cols=10, sparsity=0.40,
                                   noise_estimate=0.2, target_type="binary")
        # 0.40 - 0.05 = 0.35 delta — above 0.2 threshold
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert any("Sparsity" in a for a in anomalies)

    def test_detects_statistical_drift(self, fingerprinter, base_fp):
        """Large change in noise estimate triggers statistical drift alert."""
        modified = DataFingerprint(n_rows=1000, n_cols=10, sparsity=0.05,
                                   noise_estimate=1.5, target_type="binary")
        # 1.5 - 0.2 = 1.3 delta — above 0.5 threshold
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert any("Drift" in a for a in anomalies)

    def test_detects_target_mutation(self, fingerprinter, base_fp):
        """Target type changing (e.g., binary → continuous) triggers alert."""
        modified = DataFingerprint(n_rows=1000, n_cols=10, sparsity=0.05,
                                   noise_estimate=0.2, target_type="continuous")
        anomalies = fingerprinter.validate_consistency(base_fp, modified)
        assert any("Mutation" in a for a in anomalies)

    def test_detects_multiple_anomalies_simultaneously(self, fingerprinter, base_fp):
        """Multiple issues should all be reported."""
        worst_case = DataFingerprint(
            n_rows=400, n_cols=10, sparsity=0.60,
            noise_estimate=2.0, target_type="continuous",
        )
        anomalies = fingerprinter.validate_consistency(base_fp, worst_case)
        # Should catch: row loss + sparsity spike + drift + target mutation
        assert len(anomalies) >= 3


class TestConsistencyWithRealDataframes:
    """End-to-end: fingerprint real DataFrames and check consistency."""

    def test_clean_pipeline_no_anomalies(self, fingerprinter):
        """Dataset that loses no data should produce clean integrity report."""
        np.random.seed(42)
        df_original = pd.DataFrame({
            "a": np.random.randn(200),
            "b": np.random.randn(200),
            "target": np.random.choice([0, 1], 200),
        })
        df_cleaned = df_original.copy()  # No changes

        fp_before = fingerprinter.compute(df_original, target_col="target")
        fp_after = fingerprinter.compute(df_cleaned, target_col="target")
        anomalies = fingerprinter.validate_consistency(fp_before, fp_after)
        assert len(anomalies) == 0

    def test_aggressive_cleaning_triggers_alert(self, fingerprinter):
        """Dropping 40% of rows in cleaning should be flagged."""
        np.random.seed(42)
        df_original = pd.DataFrame({
            "a": np.random.randn(500),
            "b": np.random.randn(500),
            "target": np.random.choice([0, 1], 500),
        })
        # Simulate overly aggressive cleaning
        df_cleaned = df_original.head(250)

        fp_before = fingerprinter.compute(df_original, target_col="target")
        fp_after = fingerprinter.compute(df_cleaned, target_col="target")
        anomalies = fingerprinter.validate_consistency(fp_before, fp_after)
        assert any("Row Loss" in a for a in anomalies)
