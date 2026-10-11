"""The pipeline must import on a clean install of requirements.txt alone.

These tests exist because the benchmark used to import tensorflow, torch and
every feature selection module unconditionally, which made `Main_FS.py`
impossible to even start without the optional neural dependencies installed.
"""

import glob
import os
import py_compile


def test_config_imports(config):
    assert config.feature_selection_mapping, "no feature selection algorithms registered"


def test_utilities_imports():
    import utilities

    assert hasattr(utilities, "process_fs_algorithm")
    assert hasattr(utilities, "run_cross_validation")


def test_core_selectors_are_available_without_optional_deps(config):
    """The non-neural selectors must load with only requirements.txt installed."""
    core = {
        "ETree", "AdaBoost", "Univariate", "CT", "low_variance", "SVC",
        "SVM_RBF", "SHAP", "LS", "Skewness", "MCFS", "DT", "ReliefF",
        "alpha_investing", "DecisionTree_Forward", "SVM_Forward", "GRAPH",
        "RFS", "LeadingEV", "mRMR", "DecisionTree_Backward", "SVM_Backward",
        "CEI_GA", "LASSO_Stability",
    }
    missing = core - set(config.feature_selection_mapping)
    assert not missing, (
        f"core selectors failed to import: "
        f"{ {name: config.unavailable_algorithms.get(name) for name in missing} }"
    )


def test_unavailable_algorithms_are_reported_not_silently_dropped(config):
    """Anything that fails to import must carry a reason, not vanish."""
    registered = set(config.feature_selection_mapping)
    unavailable = set(config.unavailable_algorithms)
    declared = {name for name, _, _ in config.ALGORITHM_IMPORTS}

    assert registered | unavailable == declared
    assert not registered & unavailable
    for name, reason in config.unavailable_algorithms.items():
        assert reason, f"{name} is unavailable with no reason recorded"


def test_availability_summary_is_printable(config):
    summary = config.describe_algorithm_availability()
    assert "Feature selection algorithms available:" in summary


def test_every_source_file_compiles(repo_root, tmp_path):
    """Guards against unparseable modules sitting in the tree."""
    failures = []
    patterns = [
        os.path.join(repo_root, "src", "**", "*.py"),
        os.path.join(repo_root, "articles", "**", "*.py"),
        os.path.join(repo_root, "tests", "*.py"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            target = str(tmp_path / (os.path.basename(path) + "c"))
            try:
                py_compile.compile(path, cfile=target, doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path}: {exc}")
    assert not failures, "files failed to compile:\n" + "\n".join(failures)
