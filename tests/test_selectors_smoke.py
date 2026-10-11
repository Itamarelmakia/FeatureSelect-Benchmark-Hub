"""Smoke tests for the registered feature selection algorithms.

Each selector runs through the pipeline's own dispatcher (`utilities.run_fs`) on
one cross-validation fold of a committed sample dataset and must return exactly
K distinct, valid feature indices. A planted-signal check on synthetic data
catches selectors that quietly return a fallback selection instead of raising.

The 23 core selectors and LASSO_Stability need only requirements.txt and always
run. The four neural selectors need requirements-optional.txt: they are
skipped, with the import error as the reason, when those packages are missing,
and they are marked `slow`.
"""

import os

import numpy as np
import pytest
import scipy.io
import sklearn
from packaging.requirements import Requirement
from packaging.version import Version
from sklearn.model_selection import StratifiedKFold

# The default `--alg-group All` order of the published 27-algorithm benchmark.
KBS_DEFAULT_ALL = [
    "ETree", "AdaBoost", "Univariate", "low_variance", "SVC", "LS", "MCFS",
    "DT", "Skewness", "alpha_investing", "SHAP", "ReliefF", "mRMR", "CT",
    "LeadingEV", "RFS", "SVM_Forward", "CEI_GA", "SVM_RBF",
    "DecisionTree_Forward", "DecisionTree_Backward", "SVM_Backward",
    "DLFS", "GRAPH", "CAE", "GRACES", "UFS",
]
NEURAL_SELECTORS = ["DLFS", "CAE", "GRACES", "UFS"]
CORE_SELECTORS = [name for name in KBS_DEFAULT_ALL if name not in NEURAL_SELECTORS]
# Registered for the stability study, but not part of the KBS default group.
STABILITY_SELECTORS = ["LASSO_Stability"]
NON_NEURAL_SELECTORS = CORE_SELECTORS + STABILITY_SELECTORS

# Supervised selectors that recover a planted class-mean shift. Excluded: the
# unsupervised selectors (CT, low_variance, Skewness, LS, MCFS, DT, LeadingEV),
# which never see the labels, and GRAPH, which does not recover this signal
# even when it runs correctly.
PLANTED_SIGNAL_SELECTORS = [
    "ETree", "AdaBoost", "Univariate", "SVC", "SVM_RBF", "SHAP", "ReliefF",
    "alpha_investing", "DecisionTree_Forward", "SVM_Forward", "RFS", "mRMR",
    "DecisionTree_Backward", "SVM_Backward", "CEI_GA", "LASSO_Stability",
]

K_VALUES = (1, 5)

DLFS_LABEL_XFAIL = pytest.mark.xfail(
    reason=(
        "DLFS uses class labels as row indices, and the glass labels are not "
        "0..C-1 after the loader's -1 shift"
    ),
    raises=IndexError,
    strict=True,
)


def load_sample(sample_data_dir, name):
    """Load a committed .mat dataset with the same label rule as Main_FS.py."""
    data = scipy.io.loadmat(os.path.join(sample_data_dir, f"{name}.mat"))
    X = data["X"].astype(float)
    y = data["Y"][:, 0]
    y = y if y.min() == 0 else y - 1
    return X, y


def planted_signal_data(seed=0, n_samples=200, n_features=20, informative=3, shift=3.0):
    """Binary data whose first `informative` features carry the class signal."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_samples // 2)
    X = rng.normal(size=(n_samples, n_features))
    X[:, :informative] += shift * y[:, None]
    return X - X.min(axis=0), y  # non-negative, as chi2-based selectors require


def run_selector(utilities, name, data_name, X_train, y_train, X_test, y_test, k):
    """Run one selector exactly as the pipeline does, with its first hyperparameter row."""
    info = utilities.get_algorithms_mapping()[name]
    hyper = info["hyper"]
    row = hyper.iloc[0] if hasattr(hyper, "iloc") else hyper
    device = utilities.get_cpu_device()
    selected = utilities.run_fs(
        data_name, info["function"], name, X_train, y_train, X_test, y_test, k, row, device
    )
    return [int(i) for i in np.asarray(selected).ravel()]


def assert_valid_selection(selected, k, n_features):
    assert len(selected) == k, f"expected {k} indices, got {len(selected)}: {selected}"
    assert len(set(selected)) == k, f"duplicate indices: {selected}"
    assert all(0 <= i < n_features for i in selected), f"index out of range: {selected}"


@pytest.fixture(scope="module")
def utilities():
    import utilities as utilities_module

    return utilities_module


def check_selector_on_sample(utilities, config, sample_data_dir, dataset, name):
    if name not in config.feature_selection_mapping:
        pytest.skip(f"{name} unavailable: {config.unavailable_algorithms.get(name)}")
    X, y = load_sample(sample_data_dir, dataset)
    train, test = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed).split(X, y)
    )
    for k in K_VALUES:
        selected = run_selector(
            utilities, name, dataset, X[train], y[train], X[test], y[test], k
        )
        assert_valid_selection(selected, k, X.shape[1])


def test_installed_scikit_learn_is_in_the_supported_range(repo_root):
    """Published selectors break outside this range (see requirements.txt)."""
    with open(os.path.join(repo_root, "requirements.txt")) as handle:
        lines = [line.split("#")[0].strip() for line in handle]
    requirement = next(Requirement(line) for line in lines if line.startswith("scikit-learn"))
    installed = Version(sklearn.__version__)
    assert requirement.specifier.contains(installed), (
        f"scikit-learn {installed} is outside the supported range "
        f"'{requirement.specifier}' in requirements.txt"
    )


def test_kbs_default_group_is_unchanged(utilities, config, monkeypatch):
    """`--alg-group All` must stay exactly the published 27 algorithms, in order."""
    declared = {name for name, _, _ in config.ALGORITHM_IMPORTS}
    assert set(KBS_DEFAULT_ALL) <= declared
    # Pretend every declared selector is installed, so the check does not
    # depend on the optional neural packages.
    monkeypatch.setattr(
        utilities,
        "get_algorithms_mapping",
        lambda: {name: {"function": None, "hyper": None} for name in declared},
    )
    names = [algo["name"] for algo in utilities.get_sorted_algorithms("All")]
    assert names == KBS_DEFAULT_ALL
    assert len(names) == 27
    for name in STABILITY_SELECTORS:
        assert name in declared and name not in names


@pytest.mark.parametrize("name", NON_NEURAL_SELECTORS)
def test_core_selector_on_binary_data(utilities, config, sample_data_dir, name):
    check_selector_on_sample(utilities, config, sample_data_dir, "hepatitis", name)


# SHAP is covered on binary data only (test_core_selector_on_binary_data).
@pytest.mark.parametrize("name", [name for name in NON_NEURAL_SELECTORS if name != "SHAP"])
def test_core_selector_on_multiclass_data(utilities, config, sample_data_dir, name):
    check_selector_on_sample(utilities, config, sample_data_dir, "glass", name)


@pytest.mark.parametrize("name", PLANTED_SIGNAL_SELECTORS)
def test_selector_recovers_a_planted_signal(utilities, name):
    """At least 2 of the 3 informative features must be selected at K=3.

    A selector that silently falls back to an arbitrary or index-order
    selection fails this: a random choice of 3 of the 20 features hits 2 or
    more informative ones with probability below 5%.
    """
    X, y = planted_signal_data()
    selected = run_selector(utilities, name, "planted_signal", X, y, X, y, 3)
    assert_valid_selection(selected, 3, X.shape[1])
    hits = len(set(selected) & {0, 1, 2})
    assert hits >= 2, f"{name} selected {selected}; only {hits} of the informative features 0, 1, 2"


@pytest.mark.slow
@pytest.mark.parametrize("name", NEURAL_SELECTORS)
def test_neural_selector_on_binary_data(utilities, config, sample_data_dir, name):
    check_selector_on_sample(utilities, config, sample_data_dir, "hepatitis", name)


@pytest.mark.slow
@DLFS_LABEL_XFAIL
def test_dlfs_on_multiclass_data_with_non_contiguous_labels(utilities, config, sample_data_dir):
    check_selector_on_sample(utilities, config, sample_data_dir, "glass", "DLFS")
