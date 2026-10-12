"""LASSO_Stability, the selector used by the stability study (articles/article_stability).

The implementation is the published one, unchanged. These tests pin its
selections on the committed sample datasets and check that the stability
study's own resolution code finds it with its published hyperparameters.
"""

import importlib.util
import os

import pytest
import scipy.io

# Selections of the published implementation with the configured seed on the
# full committed datasets, recorded with scikit-learn 1.7.2 (identical with 1.4.2).
# Not pinned: twonorm, where all 20 features are selected in every bootstrap, so
# the order is numpy's tie-breaking rather than the selector's; and glass at
# K=10, which exceeds its 9 features.
REFERENCE_SELECTIONS = {
    ("glass", 1): [2],
    ("glass", 5): [2, 1, 3, 7, 5],
    ("hepatitis", 1): [0],
    ("hepatitis", 5): [0, 11, 13, 12, 4],
    ("hepatitis", 10): [0, 11, 13, 12, 4, 17, 14, 10, 5, 16],
    ("PeriodChanger", 1): [617],
    ("PeriodChanger", 5): [617, 714, 265, 431, 161],
    ("PeriodChanger", 10): [617, 714, 265, 431, 161, 133, 618, 197, 200, 1088],
    ("tumors_C", 1): [5229],
    ("tumors_C", 5): [5229, 6057, 2694, 2657, 236],
    ("tumors_C", 10): [5229, 6057, 2694, 2657, 236, 551, 1460, 4148, 2376, 2367],
}

STABILITY_SCRIPT = os.path.join(
    "articles", "article_stability", "scripts", "run_real_data_r2_stability_slice.py"
)


def load_sample(sample_data_dir, name):
    """Load a committed .mat dataset with the same label rule as Main_FS.py."""
    data = scipy.io.loadmat(os.path.join(sample_data_dir, f"{name}.mat"))
    y = data["Y"][:, 0]
    return data["X"].astype(float), (y if y.min() == 0 else y - 1)


@pytest.fixture(scope="module")
def utilities():
    import utilities as utilities_module

    return utilities_module


def test_hyperparameters_match_the_stability_study(utilities):
    row = utilities.get_algorithms_mapping()["LASSO_Stability"]["hyper"].iloc[0]
    assert int(row["n_bootstrap"]) == 50
    assert float(row["C"]) == 0.1


@pytest.mark.parametrize("dataset,k", sorted(REFERENCE_SELECTIONS))
def test_selection_matches_the_published_implementation(utilities, sample_data_dir, dataset, k):
    X, y = load_sample(sample_data_dir, dataset)
    info = utilities.get_algorithms_mapping()["LASSO_Stability"]
    row = info["hyper"].iloc[0]
    selected = utilities.run_fs(dataset, info["function"], "LASSO_Stability", X, y, X, y, k, row, None)
    assert [int(i) for i in selected] == REFERENCE_SELECTIONS[(dataset, k)]


def test_stability_study_resolves_lasso_stability(repo_root, config):
    """The study looks selectors up through its own resolve_algorithm()."""
    spec = importlib.util.spec_from_file_location(
        "r2_stability_slice", os.path.join(repo_root, STABILITY_SCRIPT)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fs_fn, hyper_row, _family, error = module.resolve_algorithm("LASSO_Stability")
    assert error is None, error
    assert fs_fn is config.feature_selection_mapping["LASSO_Stability"]
    assert int(hyper_row["n_bootstrap"]) == 50
    assert float(hyper_row["C"]) == 0.1
