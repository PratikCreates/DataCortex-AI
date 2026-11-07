import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_comparative_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_comparative_benchmarks", SCRIPT)
benchmarks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmarks
SPEC.loader.exec_module(benchmarks)


def test_comparative_benchmark_runs_on_iris():
    spec = next(ds for ds in benchmarks.DATASETS if ds.name == "Iris")

    result = benchmarks.run_dataset(spec)

    assert result["dataset"] == "Iris"
    assert result["metric"] == "accuracy"
    assert result["datacortex_score"] is not None
    assert result["dummy_score"] is not None
    assert result["models_evaluated_datacortex"] < result["models_evaluated_broad_pool"]
    assert "datacortex_fingerprint" in result["groups"]
    assert "broad_pool_reference" in result["groups"]


def test_comparative_benchmark_rows_are_csv_ready():
    spec = next(ds for ds in benchmarks.DATASETS if ds.name == "Iris")
    result = benchmarks.run_dataset(spec)

    summaries = benchmarks.summary_rows([result])
    long_rows = benchmarks.flat_rows([result])

    assert summaries[0]["dataset"] == "Iris"
    assert "gap_to_broad_pool" in summaries[0]
    assert {row["group"] for row in long_rows} >= {
        "dummy",
        "linear",
        "fixed_tree",
        "fixed_boosting",
        "datacortex_fingerprint",
        "broad_pool_reference",
    }
