"""
Main feature selection benchmarking pipeline.

Runs the configured feature selection algorithms over every ``.mat`` dataset
found in the benchmark data directory and writes the results to
``results/FS/<experiment_name>.xlsx``.

Usage:
    python src/Main_FS.py
    python src/Main_FS.py --config src/configs/config.py
    python src/Main_FS.py --datasets colon,SMK-CAN-187 --algorithms ETree
    python src/Main_FS.py --data-dir sample_data_medium --alg-group Fast
    python src/Main_FS.py --list-algorithms

The dataset archive is not part of this repository; see ``data/README.md`` for
how to populate the local ``data/`` directory.
"""

import argparse
import glob
import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

DEFAULT_CONFIG = os.path.join(_THIS_DIR, "configs", "config.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the feature selection benchmark over a directory of .mat datasets.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the configuration module (default: src/configs/config.py).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory holding the dataset sub-folders. Overrides `data_path` "
             "from the config (default: the repository's data/ directory).",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to run (without the .mat extension). "
             "Default: every dataset found under the data directory.",
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help="Comma-separated algorithm names to run. "
             "Default: every algorithm in the selected --alg-group.",
    )
    parser.add_argument(
        "--alg-group",
        default="All",
        choices=["All", "Fast", "Slow", "GPU"],
        help="Which algorithm group to benchmark (default: All).",
    )
    parser.add_argument(
        "--experiment-name",
        default="feature_selection_benchmark",
        help="Name of the output workbook written to results/FS/ "
             "(default: feature_selection_benchmark).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the results workbook. Overrides `path_output_excel_FS` "
             "from the config (default: the repository's results/FS/ directory).",
    )
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="Print which algorithms this environment can run, then exit.",
    )
    return parser.parse_args(argv)


def load_config(config_path):
    """Load ``config_path`` and register it as ``configs.config``.

    ``utilities`` does ``from configs.config import *`` at import time, so the
    selected configuration has to be in ``sys.modules`` before that happens.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise SystemExit(
            f"Config file not found: {config_path}\n"
            f"Pass an existing file with --config, or omit --config to use "
            f"the bundled {DEFAULT_CONFIG}."
        )
    spec = importlib.util.spec_from_file_location("configs.config", config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["configs.config"] = module
    spec.loader.exec_module(module)
    return module


def discover_datasets(data_dir):
    """Return ``(dataset_name, repository, path)`` for every .mat under data_dir.

    Datasets are expected one level deep (``<data_dir>/<Source>/<name>.mat``),
    which is how the benchmark archive is organised. Files sitting directly in
    ``data_dir`` are also picked up, so a flat folder of samples works too.
    """
    found = []
    for entry in sorted(os.listdir(data_dir)):
        if entry == ".ipynb_checkpoints":
            continue
        full = os.path.join(data_dir, entry)
        if os.path.isdir(full):
            for path in sorted(glob.glob(os.path.join(full, "*.mat"))):
                name = os.path.splitext(os.path.basename(path))[0]
                found.append((name, entry, path))
        elif entry.endswith(".mat"):
            name = os.path.splitext(entry)[0]
            found.append((name, os.path.basename(data_dir.rstrip(os.sep)), full))
    return found


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)

    import scipy.io
    import utilities

    if args.list_algorithms:
        print(config.describe_algorithm_availability())
        return 0

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir) + os.sep
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = config.path_output_excel_FS

    data_dir = os.path.abspath(args.data_dir) if args.data_dir else config.data_path
    if not os.path.isdir(data_dir):
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            f"The benchmark dataset archive is not committed to this repository. "
            f"See data/README.md for how to download and populate it."
        )

    available = discover_datasets(data_dir)
    if not available:
        raise SystemExit(
            f"No .mat datasets found under: {data_dir}\n"
            f"The benchmark dataset archive is not committed to this repository. "
            f"Download it and extract it into data/ (see data/README.md), or point "
            f"--data-dir at a folder of .mat files. For a quick local check without "
            f"the archive, try:\n"
            f"    python src/Main_FS.py --data-dir sample_data_medium "
            f"--datasets glass --algorithms ETree"
        )

    wanted_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if wanted_datasets:
        known = {name for name, _, _ in available}
        missing = [d for d in wanted_datasets if d not in known]
        if missing:
            raise SystemExit(
                f"Dataset(s) not found under {data_dir}: {', '.join(missing)}\n"
                f"Available: {', '.join(sorted(known))}"
            )
        available = [row for row in available if row[0] in wanted_datasets]

    algorithms = utilities.get_sorted_algorithms(args.alg_group)
    wanted_algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if wanted_algorithms:
        known = {a["name"] for a in algorithms}
        missing = [a for a in wanted_algorithms if a not in known]
        if missing:
            raise SystemExit(
                f"Algorithm(s) not available: {', '.join(missing)}\n"
                f"{config.describe_algorithm_availability()}"
            )
        algorithms = [a for a in algorithms if a["name"] in wanted_algorithms]

    if not algorithms:
        raise SystemExit(
            f"No feature selection algorithms available to run.\n"
            f"{config.describe_algorithm_availability()}"
        )

    print(config.describe_algorithm_availability())
    print(
        f"Benchmarking {len(algorithms)} algorithm(s) over {len(available)} dataset(s) "
        f"from {data_dir}"
    )
    print(f"Results will be written to {output_dir}{args.experiment_name}.xlsx")

    run_benchmark(config, utilities, scipy, algorithms, available,
                  args.experiment_name, output_dir)
    return 0


def run_benchmark(config, utilities, scipy, algorithms, datasets, experiment_name,
                  output_dir):
    """Run every algorithm over every dataset, preserving the original protocol."""
    fixed_threshold = config.fixed_threshold

    for algo_info in algorithms:
        algo_name = algo_info["name"]
        alg = algo_info["function"]
        hyperparams = algo_info["hyper"]

        for data_name, repository, dataset_path in datasets:
            Data = scipy.io.loadmat(dataset_path)
            data_arr = Data["X"]
            label_arr = (
                Data["Y"][:, 0] if min(Data["Y"][:, 0]) == 0 else Data["Y"][:, 0] - 1
            )
            rows, columns_len = data_arr.shape
            df_FS, columns = utilities.read_or_create_result_table(
                output_dir, experiment_name
            )

            filtered_k_fs, filtered_k_hd = utilities.check_k_limitation_and_run(
                algo_name, hyperparams, df_FS, data_name, rows, columns_len,
                fixed_threshold, config.perform_cv, config.initial_splits,
                config.train_size, config.evaluation_models,
                config.no_hyperparameters, config.K_List,
            )

            # Only for ETree and AdaBoost do we also run the all-features (HD)
            # baseline that the hardness criteria are measured against.
            if algo_name in ["ETree", "AdaBoost"]:
                if not filtered_k_fs and filtered_k_hd == "Y":
                    print(f"For Dataset: {data_name}, all K values already run for algorithm: {algo_name}")
                    continue
                if not filtered_k_fs and filtered_k_hd == "N":
                    # Not run HD yet but ran FS (all K values)
                    print(
                        f"For Dataset : {data_name} All K values already run for algorithm: "
                        f"{algo_name} Not run High Dimentional dataset and Experiment Type: "
                        f"{config.Experimnet_Type}"
                    )
                    _run(utilities, config, label_arr, experiment_name, fixed_threshold,
                         data_name, algo_name, filtered_k_fs, dataset_path, hyperparams,
                         repository, columns_len, alg, columns, output_dir, "No FS")
                elif filtered_k_hd == "Y":
                    # Not run FS - run HD
                    print(
                        f"For Dataset : {data_name} Partial run for K values already run for "
                        f"algorithm: {algo_name} For High Dimentional dataset and Experiment "
                        f"Type: {config.Experimnet_Type}"
                    )
                    _run(utilities, config, label_arr, experiment_name, fixed_threshold,
                         data_name, algo_name, filtered_k_fs, dataset_path, hyperparams,
                         repository, columns_len, alg, columns, output_dir, "FS")
                else:
                    _run(utilities, config, label_arr, experiment_name, fixed_threshold,
                         data_name, algo_name, filtered_k_fs, dataset_path, hyperparams,
                         repository, columns_len, alg, columns, output_dir, "No FS")
                    _run(utilities, config, label_arr, experiment_name, fixed_threshold,
                         data_name, algo_name, filtered_k_fs, dataset_path, hyperparams,
                         repository, columns_len, alg, columns, output_dir, "FS")
            else:
                if not filtered_k_fs:
                    print(
                        f"For Dataset : {data_name} All K values already run for algorithm: {algo_name}"
                    )
                    continue
                _run(utilities, config, label_arr, experiment_name, fixed_threshold,
                     data_name, algo_name, filtered_k_fs, dataset_path, hyperparams,
                     repository, columns_len, alg, columns, output_dir, "FS")


def _run(utilities, config, label_arr, experiment_name, fixed_threshold, data_name,
         algo_name, filtered_k_fs, dataset_path, hyperparams, repository, columns_len,
         alg, columns, output_dir, fs_hd):
    from datetime import datetime

    start_time = datetime.now()
    print("Start time:", start_time.strftime("%H:%M"),
          "| Day:", start_time.strftime("%A, %d %B %Y"))
    utilities.process_fs_algorithm(
        label_arr, experiment_name, fixed_threshold, data_name, algo_name,
        filtered_k_fs, dataset_path, hyperparams, repository, columns_len,
        output_dir, alg, columns, fs_hd, config.debug,
    )


if __name__ == "__main__":
    sys.exit(main())
