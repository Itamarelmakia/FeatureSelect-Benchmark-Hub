"""
Script to run feature selection benchmark on a single dataset.

Usage:
    python run_single_dataset.py path/to/dataset.mat [--alg_group All] [--experiment_name custom]

This script reads a MATLAB `.mat` file containing your dataset (with keys `X` for features and `Y` for labels),
runs the configured feature selection algorithms on it using the existing `utilities` and `config` modules,
and writes the results to an Excel file in the results/FS directory.
"""

import argparse
import os
import scipy.io
import numpy as np
from typing import Optional, List, Dict

# Config can be either in configs/config or local config.py
try:
    from configs import config as config
except ImportError:
    import config

import utilities


# ----------------------------------------------------------------------
# Helper: ensure HD baselines (all features) for ETree & AdaBoost
# ----------------------------------------------------------------------
def ensure_hd_baselines(
    df_results,
    label_arr: np.ndarray,
    data_name: str,
    repository: str,
    n_features: int,
    mat_path: str,
    experiment_name: str,
    columns: List[str],
    algo_config_by_name: Dict[str, dict],
) -> None:
    """
    Ensure that HD (all-features) baselines exist for this dataset with
    algorithms 'ETree' and 'AdaBoost'.

    We look for rows in the Excel results where:
        Dataset == data_name
        FS/HD  == 'HD'
        Algorithm in {'ETree', 'AdaBoost'}

    If such a row does NOT exist for a given algorithm, we run that algorithm
    ONCE with K = [n_features] and FS_HD = 'HD'.
    """
    hd_algo_names = ["ETree", "AdaBoost"]

    for algo_name in hd_algo_names:
        mask_hd = (
            (df_results.get("Dataset") == data_name)
            & (df_results.get("Algorithm") == algo_name)
            & (df_results.get("FS/HD") == "HD")
        )

        has_hd = bool(mask_hd.any()) if mask_hd is not None else False

        if has_hd:
            print(
                f"[HD] Baseline already exists for '{algo_name}' "
                f"on dataset '{data_name}'. Skipping HD run."
            )
            continue

        # Need to run HD baseline
        algo_info = algo_config_by_name.get(algo_name)
        if algo_info is None:
            print(
                f"[HD] WARNING: Algorithm '{algo_name}' not found in "
                f"utilities.get_sorted_algorithms(). Cannot run HD baseline."
            )
            continue

        alg_func = algo_info["function"]
        hyperparams = algo_info["hyper"]   # <-- this is a DataFrame

        filtered_k_hd = [n_features]       # all features

        print(
            f"[HD] No baseline found for '{algo_name}' on dataset '{data_name}'. "
            f"Running all-features baseline (K = {n_features})."
        )

        utilities.process_fs_algorithm(
            label_arr=label_arr,
            specific_experiment=experiment_name,
            fixed_threshold=config.fixed_threshold,
            data_name=data_name,
            algo_name=algo_name,
            filtered_k=filtered_k_hd,
            dataset_path=mat_path,
            hyperparamters=hyperparams,     # DataFrame, works with .iterrows()
            repository=repository,
            columns_len=n_features,
            path_output_excel_FS=config.path_output_excel_FS,
            alg=alg_func,
            columns=columns,
            FS_HD="HD",                     # <-- mark as HD baseline
            debug=config.debug,
        )


# ----------------------------------------------------------------------
# Main logic for single dataset
# ----------------------------------------------------------------------
def run_feature_selection(
    mat_path: str,
    alg_group: str,
    experiment_name: str,
    selected_algorithms: Optional[list] = None,
) -> None:
    """
    Run feature selection algorithms on a single dataset.

    Logic:
    1. Load MAT dataset (X, Y).
    2. Read/create the result Excel table.
    3. Determine which FS algorithms are requested.
    4. For each requested FS algorithm:
       - If (Dataset, Algorithm, FS/HD='FS') already exists -> SKIP.
       - Otherwise, mark it as "to run".
    5. If nothing to run -> print message and exit.
    6. If there is at least one FS algorithm to run:
       - Ensure HD baselines (FS/HD='HD') exist for ETree & AdaBoost.
         If missing -> run them once each.
    7. Run the remaining FS algorithms with utilities.process_fs_algorithm().
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Dataset not found: {mat_path}")

    # 1. Load dataset
    data = scipy.io.loadmat(mat_path)
    X = data.get("X")
    Y = data.get("Y")
    if X is None or Y is None:
        raise ValueError("MAT file must contain 'X' and 'Y' keys.")

    X = np.array(X)
    label_arr = np.array(Y).flatten()
    if label_arr.min() != 0:
        label_arr = label_arr - label_arr.min()

    n_samples, n_features = X.shape

    data_name = os.path.splitext(os.path.basename(mat_path))[0]
    repository = os.path.basename(os.path.dirname(mat_path)) or "UserUpload"

    print(f"Loaded dataset '{data_name}' from '{mat_path}'")
    print(f"Repository: {repository} | Shape: {X.shape}")

    # 2. Prepare result table and algorithms list
    df_results, columns = utilities.read_or_create_result_table(
        config.path_output_excel_FS,
        experiment_name,
    )

    # Get all configured algorithms in this group
    all_algos_info = utilities.get_sorted_algorithms(alg_group)

    # Build lookup by name (used for HD baselines)
    algo_config_by_name = {info["name"]: info for info in all_algos_info}

    # Maybe filter by user-selected algorithms
    if selected_algorithms:
        selected_set = set(a.strip() for a in selected_algorithms if a.strip())
        algos_info = [info for info in all_algos_info if info["name"] in selected_set]
        if not algos_info:
            raise ValueError(
                f"None of the requested algorithms {selected_set} "
                f"were found in the configuration."
            )
    else:
        algos_info = all_algos_info

    # 3. Determine which FS algorithms still need to run
    filtered_k_fs = config.K_List
    algos_to_run = []

    for algo_info in algos_info:
        algo_name = algo_info["name"]

        # Has this FS algorithm already produced results for this dataset?
        mask_fs = (
            (df_results.get("Dataset") == data_name)
            & (df_results.get("Algorithm") == algo_name)
            & (df_results.get("FS/HD") == "FS")
        )

        already_have_fs = bool(mask_fs.any()) if mask_fs is not None else False

        if already_have_fs:
            print(
                f"[SKIP] FS results already exist for dataset '{data_name}', "
                f"algorithm '{algo_name}', experiment '{experiment_name}'."
            )
            continue

        algos_to_run.append(algo_info)

    # 4. If nothing to run – exit nicely
    if not algos_to_run:
        print(
            f"All requested FS algorithms already have results for dataset "
            f"'{data_name}' in experiment '{experiment_name}'. Nothing to run."
        )
        return

    # 5. Ensure HD baselines exist (ETree & AdaBoost on all features)
    ensure_hd_baselines(
        df_results=df_results,
        label_arr=label_arr,
        data_name=data_name,
        repository=repository,
        n_features=n_features,
        mat_path=mat_path,
        experiment_name=experiment_name,
        columns=columns,
        algo_config_by_name=algo_config_by_name,
    )

    # 6. Run remaining FS algorithms
    for algo_info in algos_to_run:
        algo_name = algo_info["name"]
        alg_func = algo_info["function"]
        hyperparams = algo_info["hyper"]

        print(
            f"[FS] Running algorithm '{algo_name}' on dataset '{data_name}' "
            f"with K_list={filtered_k_fs}"
        )

        utilities.process_fs_algorithm(
            label_arr=label_arr,
            specific_experiment=experiment_name,
            fixed_threshold=config.fixed_threshold,
            data_name=data_name,
            algo_name=algo_name,
            filtered_k=filtered_k_fs,
            dataset_path=mat_path,
            hyperparamters=hyperparams,
            repository=repository,
            columns_len=n_features,
            path_output_excel_FS=config.path_output_excel_FS,
            alg=alg_func,
            columns=columns,
            FS_HD="FS",
            debug=config.debug,
        )

    print(
        "Feature selection completed. Check the results Excel file in:",
        config.path_output_excel_FS,
    )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature selection algorithms on a single .mat dataset."
    )
    parser.add_argument(
        "mat_path",
        help="Path to the .mat file containing your dataset",
    )
    parser.add_argument(
        "--alg_group",
        choices=["All", "Fast", "Slow", "GPU"],
        default="All",
        help="Algorithm group to run",
    )
    parser.add_argument(
        "--experiment_name",
        default="custom_experiment",
        help="Name of the experiment for the output file",
    )
    parser.add_argument(
        "--algorithms",
        default="",
        help="Comma-separated list of algorithm names to run instead of the entire group",
    )
    args = parser.parse_args()

    # If specific algorithms are requested, restrict the group accordingly
    if args.algorithms:
        selected = [name.strip() for name in args.algorithms.split(",") if name.strip()]
        # Always fetch all algorithms, then filter inside run_feature_selection
        run_feature_selection(
            args.mat_path,
            "All",
            args.experiment_name,
            selected_algorithms=selected,
        )
    else:
        run_feature_selection(
            args.mat_path,
            args.alg_group,
            args.experiment_name,
        )


if __name__ == "__main__":
    main()
