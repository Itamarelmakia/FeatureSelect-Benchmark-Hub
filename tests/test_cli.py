"""CLI contract for the benchmark entry points.

`Main_FS.py` used to be a bare script with no argument parsing at all, while the
README documented a `--config` flag, and a hard-coded dataset filter meant it
silently processed nothing. These tests pin the documented behaviour down.
"""

import os

import pytest

import Main_FS


def test_default_config_path_exists():
    assert os.path.isfile(Main_FS.DEFAULT_CONFIG)


def test_documented_config_flag_is_accepted():
    args = Main_FS.parse_args(["--config", Main_FS.DEFAULT_CONFIG])
    assert args.config == Main_FS.DEFAULT_CONFIG


def test_default_arguments_select_everything():
    args = Main_FS.parse_args([])
    assert args.datasets == ""
    assert args.algorithms == ""
    assert args.alg_group == "All"
    assert args.experiment_name == "feature_selection_benchmark"


def test_selection_flags_are_parsed():
    args = Main_FS.parse_args(
        ["--datasets", "glass,colon", "--algorithms", "ETree", "--alg-group", "Fast"]
    )
    assert args.datasets == "glass,colon"
    assert args.algorithms == "ETree"
    assert args.alg_group == "Fast"


def test_missing_config_fails_with_actionable_message(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        Main_FS.load_config(str(tmp_path / "nope.py"))
    assert "Config file not found" in str(excinfo.value)


def test_discover_datasets_finds_committed_samples(sample_data_dir):
    found = Main_FS.discover_datasets(sample_data_dir)
    names = {name for name, _, _ in found}
    # Committed for local sanity checks; see the repository README.
    assert {"glass", "hepatitis", "twonorm", "PeriodChanger"} <= names
    for _, _, path in found:
        assert os.path.isfile(path)


def test_discover_datasets_reads_nested_and_flat_layouts(sample_data_dir):
    found = Main_FS.discover_datasets(sample_data_dir)
    repositories = {repo for _, repo, _ in found}
    # Flat .mat files at the top level plus the nested "Peeling datasets" folder.
    assert "Peeling datasets" in repositories
    assert "sample_data_medium" in repositories


def test_empty_data_directory_explains_the_missing_archive(tmp_path):
    """A clean clone has no datasets; the failure must say so, not no-op."""
    empty = tmp_path / "data"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        Main_FS.main(["--data-dir", str(empty)])
    message = str(excinfo.value)
    assert "No .mat datasets found" in message
    assert "data/README.md" in message


def test_missing_data_directory_explains_the_missing_archive(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        Main_FS.main(["--data-dir", str(tmp_path / "absent")])
    assert "Data directory not found" in str(excinfo.value)


def test_unknown_dataset_lists_what_is_available(sample_data_dir):
    with pytest.raises(SystemExit) as excinfo:
        Main_FS.main(["--data-dir", sample_data_dir, "--datasets", "not_a_dataset"])
    message = str(excinfo.value)
    assert "not_a_dataset" in message
    assert "glass" in message


def test_unknown_algorithm_reports_availability(sample_data_dir):
    with pytest.raises(SystemExit) as excinfo:
        Main_FS.main(
            ["--data-dir", sample_data_dir, "--datasets", "glass",
             "--algorithms", "NotAnAlgorithm"]
        )
    message = str(excinfo.value)
    assert "NotAnAlgorithm" in message
    assert "Feature selection algorithms available:" in message


def test_list_algorithms_exits_cleanly(capsys):
    assert Main_FS.main(["--list-algorithms"]) == 0
    assert "Feature selection algorithms available:" in capsys.readouterr().out
