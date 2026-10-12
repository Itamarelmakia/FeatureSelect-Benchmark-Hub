"""End-to-end proof that the benchmark runs without the external dataset archive.

Uses only the small .mat files committed under `sample_data_medium/`, so it runs
on a clean clone. Marked `slow` because it executes real cross-validation.
"""

import pandas as pd
import pytest

import Main_FS

pytestmark = pytest.mark.slow

EXPECTED_COLUMNS = {
    "Repository", "Dataset", "FS/HD", "Algorithm", "N", "P", "K",
    "Average AUC", "Std_dev AUC", "Average Accuracy",
    "Best Indices (frequency)", "Pipeline Runtime [Sec]",
}


@pytest.fixture(scope="module")
def benchmark_results(tmp_path_factory, sample_data_dir):
    """Run ETree on the smallest committed dataset and return the results frame."""
    out_dir = tmp_path_factory.mktemp("results") / "FS"

    exit_code = Main_FS.main([
        "--data-dir", sample_data_dir,
        "--datasets", "glass",
        "--algorithms", "ETree",
        "--experiment-name", "pytest_e2e",
        "--output-dir", str(out_dir),
    ])

    workbook = out_dir / "pytest_e2e.xlsx"
    assert exit_code == 0
    assert workbook.is_file(), f"benchmark produced no results at {workbook}"
    return pd.read_excel(workbook)


def test_results_workbook_has_the_expected_schema(benchmark_results):
    missing = EXPECTED_COLUMNS - set(benchmark_results.columns)
    assert not missing, f"result table is missing columns: {sorted(missing)}"


def test_every_evaluation_model_is_scored(benchmark_results, config):
    """One row per (K, evaluation model) combination."""
    assert len(benchmark_results) == len(benchmark_results["K"].unique()) * len(
        config.evaluation_models
    )


def test_both_the_fs_and_all_features_arms_ran(benchmark_results):
    """The hardness criterion compares FS against the no-FS (HD) baseline."""
    assert set(benchmark_results["FS/HD"]) == {"FS", "No FS"}


def test_auc_values_are_valid_probabilities(benchmark_results):
    auc = benchmark_results["Average AUC"].dropna()
    assert not auc.empty
    assert auc.between(0.0, 1.0).all(), "AUC outside [0, 1]"


def test_selected_feature_counts_match_k(benchmark_results):
    """Feature selection must return exactly K features for each FS row."""
    fs_rows = benchmark_results[benchmark_results["FS/HD"] == "FS"]
    assert not fs_rows.empty
    for _, row in fs_rows.iterrows():
        indices = [i for i in str(row["Best Indices (frequency)"]).strip("[]").split(",") if i.strip()]
        assert len(indices) == int(row["K"])


def test_run_is_repeatable_with_the_configured_seed(benchmark_results, tmp_path, sample_data_dir):
    """The config pins seed=0; a second identical run must reproduce the AUCs."""
    out_dir = tmp_path / "FS"
    Main_FS.main([
        "--data-dir", sample_data_dir,
        "--datasets", "glass",
        "--algorithms", "ETree",
        "--experiment-name", "pytest_repeat",
        "--output-dir", str(out_dir),
    ])

    repeat = pd.read_excel(out_dir / "pytest_repeat.xlsx")
    key = ["Dataset", "Algorithm", "FS/HD", "K", "Classifier Model"]
    first = benchmark_results.sort_values(key).reset_index(drop=True)
    second = repeat.sort_values(key).reset_index(drop=True)
    pd.testing.assert_series_equal(
        first["Average AUC"], second["Average AUC"], check_names=False
    )


def test_adaboost_completes_a_benchmark_run(tmp_path, sample_data_dir):
    """AdaBoost aborted the whole run on scikit-learn 1.8; it must complete here."""
    out_dir = tmp_path / "FS"
    exit_code = Main_FS.main([
        "--data-dir", sample_data_dir,
        "--datasets", "hepatitis",
        "--algorithms", "AdaBoost",
        "--experiment-name", "pytest_adaboost",
        "--output-dir", str(out_dir),
    ])

    assert exit_code == 0
    results = pd.read_excel(out_dir / "pytest_adaboost.xlsx")
    fs_rows = results[(results["FS/HD"] == "FS") & (results["Algorithm"] == "AdaBoost")]
    assert not fs_rows.empty, "AdaBoost produced no feature-selection rows"
    assert results["Average AUC"].dropna().between(0.0, 1.0).all()
