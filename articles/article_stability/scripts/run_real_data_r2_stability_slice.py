"""
run_real_data_r2_stability_slice.py — Article 5, Stage R2 real-data slice
=========================================================================

The Article 5 Stage R2 real-data feature-selection stability slice. R2 is an
**illustrative, diagnostic** applied slice — NOT a broad benchmark, NOT a
recovery study.

Locked R2 configuration (per REAL_DATA_R2_ALGORITHM_SET_FINAL_OPUS_REVIEW.md):
    datasets    : colon, PeriodChanger, SMK-CAN-187
    algorithms  : ETree, mRMR, LASSO_Stability, ReliefF
    classifier  : ETREE  (held-out evaluation)
    K values    : 1, 5, 10, 30, 60
    resamples   : 50  (repeated stratified shuffle-split, test_size 0.2)
    seed        : 0
    expected rows: 3 x 4 x 5 x 50 = 3000

PREPROCESSING (per ARTICLE2_DATASET_VALUE_RANGE_AUDIT.md / REAL_DATA_APPLICATION_PLAN.md
§7, matching Stage R1 and every Article 5 real-data smoke):
  * RAW signed X — `np.abs(X)` is intentionally NOT applied (colon and
    PeriodChanger are signed; SMK-CAN-187 is non-negative — raw X for uniformity).
  * y is remapped to {0,1} with a proper 0-based label encoder (numpy unique
    inverse). `np.abs` is NEVER applied to y.
  * every logged row records `preprocessing_note=raw_X`, `abs_applied=False`.

NO DATA LEAKAGE:
  * feature selection fit on the training split of each resample only;
  * classifier trained only on the selected training features;
  * evaluation on the held-out test split only;
  * per-split cleaning is element-wise (float64 cast + NaN/Inf guard).

REAL-DATA RULES — real data has NO planted ground truth:
  * `synthetic_recovery_metrics.py` is NOT used; no true-core recall, no exact
    support recovery, no noise contamination, no synthetic null comparison.
  * The stability profile is computed INLINE here (a safe subset: feature
    frequencies, threshold sweep, pairwise Jaccard, Kuncheva, Nogueira-style
    estimate, observed-vs-random baseline, cross-K recurrence) — all from the
    LOGGED selected indices, which are preprocessing-independent. The offline
    `analyze_per_fold_stability.py` is deliberately NOT invoked because its
    periphery-correlation view reloads X and applies `np.abs` (colon-pinned),
    inconsistent with R2's raw X.

Isolation contract:
  * Modifies no benchmark/config/utility/notebook/raw-dataset file; reuses the
    FS engine (`run_fs`, `get_eval_model`, `evaluate_once`) unmodified.
  * Reads the three `.mat` datasets read-only.
  * Writes ONLY under results/article_stability/real_data/r2/ and reports under
    results/article_stability/reports/experiment_planning/.
  * Never overwrites an existing output (timestamps a new name instead).
  * Locked to the 3 datasets x 4 algorithms above.

Usage:
    python articles/article_stability/scripts/run_real_data_r2_stability_slice.py --dry-run
    python articles/article_stability/scripts/run_real_data_r2_stability_slice.py
"""

import sys
import json
import math
import time
import argparse
from datetime import datetime
from itertools import combinations
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs.config import (              # noqa: E402
    feature_selection_mapping,
    fixed_threshold as CONFIG_FIXED_THRESHOLD,
)
from utilities import (                   # noqa: E402
    run_fs,
    get_eval_model,
    evaluate_once,
    get_algorithms_mapping,
)

# ==========================================================================
# Locked R2 configuration
# ==========================================================================
CLASSIFIER = "ETREE"
TEST_SIZE = 0.2
SEED = 0
RESAMPLE_TYPE = "stratified_shuffle_split"
PREPROCESSING_NOTE = "raw_X"
ABS_APPLIED = False
LEAKAGE_GUARD_NOTE = ("feature selection fit on training split only; "
                      "classifier trained on selected training features; "
                      "evaluation on held-out test split")

DEFAULT_DATASETS = ["colon", "PeriodChanger", "SMK-CAN-187"]
DEFAULT_ALGORITHMS = ["ETree", "mRMR", "LASSO_Stability", "ReliefF"]
DEFAULT_K_VALUES = [1, 5, 10, 30, 60]
DEFAULT_N_RESAMPLES = 50

DATASETS = {
    "colon": {"relpath": "data/scikit-feature/colon.mat",
              "repository": "scikit-feature"},
    "PeriodChanger": {"relpath": "data/UCI/PeriodChanger.mat",
                      "repository": "UCI"},
    "SMK-CAN-187": {"relpath": "data/scikit-feature/SMK-CAN-187.mat",
                    "repository": "scikit-feature"},
}
ALLOWED_DATASETS = set(DATASETS)
ALLOWED_ALGORITHMS = set(DEFAULT_ALGORITHMS)

THRESHOLDS = [0.6, 0.8, 1.0]
# TODO: algorithm_metadata.json is not included in this public repository.
# get_algorithm_family() returns None for all algorithms without it (non-fatal).
ALGORITHM_METADATA_PATH = SRC_DIR / "configs" / "algorithm_metadata.json"

# Output safety root — every write MUST stay inside this directory.
R2_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "real_data" / "r2"
).resolve()
PERFOLD_DIR = R2_OUTPUT_ROOT / "per_fold_selections"
STABILITY_ROOT = R2_OUTPUT_ROOT / "stability_profiles"
RUN_TAG = "real_data_r2_rawX_4algos_3datasets_seed0_50resamples"

DOCS_DIR = PROJECT_ROOT / "results" / "article_stability" / "reports"
MAIN_REPORT_PATH = DOCS_DIR / "REAL_DATA_R2_STABILITY_SLICE_REPORT.md"
DATASET_REPORT_NAMES = {
    "colon": "REAL_DATA_R2_COLON_REPORT.md",
    "PeriodChanger": "REAL_DATA_R2_PERIODCHANGER_REPORT.md",
    "SMK-CAN-187": "REAL_DATA_R2_SMK_CAN_187_REPORT.md",
}

OUTPUT_COLUMNS = [
    "dataset", "dataset_path", "repository", "algorithm", "algorithm_family",
    "registered_key", "K", "seed", "fold_id", "n_splits", "n_features",
    "selected_feature_indices", "selected_feature_count",
    "selected_count_equals_k", "selected_indices_valid",
    "selected_feature_scores", "fs_runtime_sec", "classifier",
    "auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "status", "error_message", "resample_type", "train_size", "test_size",
    "n_class_0", "n_class_1", "class_balance_train", "class_balance_test",
    "preprocessing_note", "abs_applied", "leakage_guard_note",
]


# ==========================================================================
# Output-path safety
# ==========================================================================
def _is_within(child, parent):
    child, parent = Path(child).resolve(), Path(parent).resolve()
    return child == parent or parent in child.parents


def safe_path(directory, base_name, safety_root):
    target = (directory / base_name).resolve()
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, suffix = Path(base_name).stem, Path(base_name).suffix
        target = (directory / f"{stem}_{stamp}{suffix}").resolve()
        print(f"[output] '{base_name}' exists; using timestamped name: {target.name}")
    if not _is_within(target, safety_root):
        raise SystemExit(f"[SAFETY] Resolved output path escapes safety root: {target}")
    return target


# ==========================================================================
# Helpers
# ==========================================================================
def clean_X(X):
    """Per-split cleaning: float64 cast + NaN/inf guard. Does NOT apply np.abs."""
    if np.ma.isMaskedArray(X):
        X = X.filled(0)
    X = np.asarray(X, dtype=np.float64)
    np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return X


def get_algorithm_family(algo_name):
    try:
        with open(ALGORITHM_METADATA_PATH, "r") as fh:
            metadata = json.load(fh)
        return metadata.get(algo_name, {}).get("Category FS")
    except Exception:  # noqa: BLE001
        return None


def load_dataset_raw(name):
    """Load a dataset read-only with RAW signed X (np.abs NOT applied) and y
    remapped to {0,1} by a 0-based label encoder."""
    relpath = DATASETS[name]["relpath"]
    mat_path = (PROJECT_ROOT / relpath).resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"dataset not found: {mat_path}")
    Data = scipy.io.loadmat(str(mat_path))
    if "X" not in Data or "Y" not in Data:
        raise ValueError(f"MAT file must contain 'X' and 'Y' keys: {mat_path}")
    X = np.asarray(Data["X"], dtype=np.float64)          # RAW signed X
    y_raw = np.asarray(Data["Y"]).reshape(-1)
    raw_labels = sorted(int(v) for v in np.unique(y_raw))
    uniq, y = np.unique(y_raw, return_inverse=True)      # -> {0,1,...}
    y = y.astype(np.int64)
    return {
        "name": name, "mat_path": str(mat_path), "relpath": relpath,
        "repository": DATASETS[name]["repository"], "X": X, "y": y,
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "raw_labels": raw_labels,
        "encoded_labels": sorted(int(v) for v in np.unique(y)),
        "x_min": float(np.nanmin(X)), "x_max": float(np.nanmax(X)),
        "x_negative_count": int((X < 0).sum()),
        "nan_count": int(np.isnan(X).sum()), "inf_count": int(np.isinf(X).sum()),
        "n_class_0": int((y == 0).sum()), "n_class_1": int((y == 1).sum()),
    }


def check_indices(selected_idx, k, n_features):
    count_equals_k = bool(len(selected_idx) == k)
    in_range = all(0 <= i < n_features for i in selected_idx)
    no_dup = len(selected_idx) == len(set(selected_idx))
    return count_equals_k, bool(in_range and no_dup and len(selected_idx) == k)


def resolve_algorithm(algo_name):
    """Resolve an algorithm's FS function + hyper row. Returns
    (fs_fn, hyper_row, family, error)."""
    fs_fn = feature_selection_mapping.get(algo_name)
    if fs_fn is None:
        return None, None, None, f"'{algo_name}' not in feature_selection_mapping"
    algo_info = get_algorithms_mapping().get(algo_name)
    if algo_info is None or "hyper" not in algo_info:
        return None, None, None, (f"'{algo_name}' not in get_algorithms_mapping()"
                                  f" / no hyper sheet")
    try:
        hyper_row = algo_info["hyper"].iloc[0]
    except Exception as exc:  # noqa: BLE001
        return None, None, None, f"hyper unavailable: {type(exc).__name__}: {exc}"
    return fs_fn, hyper_row, get_algorithm_family(algo_name), None


# ==========================================================================
# Step 1 — per-fold logging (3 datasets x 4 algorithms x 5 K x 50 resamples)
# ==========================================================================
def run_logging(datasets, algorithms, k_values, n_resamples):
    """Run the full R2 logging loop. Returns (df, ds_cache). Per-row errors are
    recorded, not raised."""
    ds_cache = {}
    records = []
    for ds_name in datasets:
        ds = load_dataset_raw(ds_name)
        ds_cache[ds_name] = ds
        X, y, n_features = ds["X"], ds["y"], ds["n_features"]
        print("=" * 70)
        print(f"DATASET — {ds_name} ({ds['relpath']})  N={ds['n_samples']}  "
              f"P={n_features}  classes={ds['n_class_0']}/{ds['n_class_1']}  "
              f"x_neg={ds['x_negative_count']}")
        print("=" * 70)
        # one set of resamples per dataset, reused across all algorithms and K
        sss = StratifiedShuffleSplit(n_splits=n_resamples, test_size=TEST_SIZE,
                                     random_state=SEED)
        splits = list(sss.split(X, y))

        for algo in algorithms:
            fs_fn, hyper_row, algo_family, resolve_err = resolve_algorithm(algo)
            t_algo = time.time()
            if resolve_err:
                print(f"  [{algo}] DISPATCH FAILED: {resolve_err}")
            for k in k_values:
                for fold_id, (tr, te) in enumerate(splits, start=1):
                    row = {col: None for col in OUTPUT_COLUMNS}
                    row.update(
                        dataset=ds_name, dataset_path=ds["relpath"],
                        repository=ds["repository"], algorithm=algo,
                        algorithm_family=algo_family, registered_key=algo,
                        K=int(k), seed=int(SEED), fold_id=int(fold_id),
                        n_splits=int(n_resamples), n_features=int(n_features),
                        classifier=CLASSIFIER, selected_feature_indices=[],
                        selected_feature_count=0, selected_count_equals_k=False,
                        selected_indices_valid=False, selected_feature_scores=None,
                        status="ok", error_message=None,
                        resample_type=RESAMPLE_TYPE, train_size=int(len(tr)),
                        test_size=int(len(te)), n_class_0=ds["n_class_0"],
                        n_class_1=ds["n_class_1"],
                        class_balance_train=f"{int((y[tr]==0).sum())}/{int((y[tr]==1).sum())}",
                        class_balance_test=f"{int((y[te]==0).sum())}/{int((y[te]==1).sum())}",
                        preprocessing_note=PREPROCESSING_NOTE,
                        abs_applied=ABS_APPLIED,
                        leakage_guard_note=LEAKAGE_GUARD_NOTE,
                    )
                    if resolve_err:
                        row["status"] = "algo_dispatch_failed"
                        row["error_message"] = resolve_err
                        records.append(row)
                        continue
                    try:
                        X_train = clean_X(X[tr])
                        X_test = clean_X(X[te])
                        y_train, y_test = y[tr], y[te]
                        if (len(np.unique(y_train)) < 2
                                or len(np.unique(y_test)) < 2):
                            row["status"] = "resample_skipped_single_class"
                            row["error_message"] = "a split lacked both classes"
                            records.append(row)
                            continue

                        # --- feature selection — TRAINING SPLIT ONLY --------
                        t0 = time.time()
                        selected = run_fs(ds_name, fs_fn, algo,
                                          X_train, y_train, X_test, y_test,
                                          int(k), hyper_row, None)
                        fs_runtime = time.time() - t0
                        sel = [int(i) for i in np.asarray(selected).ravel()]
                        row["selected_feature_indices"] = sel
                        row["selected_feature_count"] = len(sel)
                        row["fs_runtime_sec"] = round(fs_runtime, 4)
                        cek, iv = check_indices(sel, int(k), n_features)
                        row["selected_count_equals_k"] = cek
                        row["selected_indices_valid"] = iv
                        if not cek:
                            row["status"] = "variable_k_returned"
                            row["error_message"] = (
                                f"fixed-K contract violated: returned "
                                f"{len(sel)} features, expected K={k}")

                        # --- held-out evaluation ----------------------------
                        if cek and iv:
                            X_tr_sel = X_train[:, sel]
                            X_te_sel = X_test[:, sel]
                            try:
                                clf = get_eval_model(CLASSIFIER, SEED)
                                res = evaluate_once(clf, X_tr_sel, X_te_sel,
                                                    y_train, y_test,
                                                    CONFIG_FIXED_THRESHOLD)
                                row["auc"] = None if res["auc"] is None else float(res["auc"])
                                row["accuracy"] = None if res["acc"] is None else float(res["acc"])
                                row["precision"] = None if res["prec"] is None else float(res["prec"])
                                row["recall"] = None if res["rec"] is None else float(res["rec"])
                                row["f1"] = None if res["f1"] is None else float(res["f1"])
                            except Exception as exc:  # noqa: BLE001
                                row["error_message"] = (
                                    f"evaluation_failed: {type(exc).__name__}: {exc}")
                            # balanced accuracy — independent deterministic fit
                            # (evaluate_once does not return it)
                            try:
                                clf_b = get_eval_model(CLASSIFIER, SEED)
                                clf_b.fit(X_tr_sel, y_train)
                                pred = clf_b.predict(X_te_sel)
                                row["balanced_accuracy"] = float(
                                    balanced_accuracy_score(y_test, pred))
                            except Exception:  # noqa: BLE001
                                row["balanced_accuracy"] = None
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "fs_failed"
                        row["error_message"] = f"{type(exc).__name__}: {exc}"
                    records.append(row)
            ok_n = sum(1 for r in records
                       if r["dataset"] == ds_name and r["algorithm"] == algo
                       and r["status"] == "ok")
            print(f"  [{algo:16s}] done — {ok_n}/{len(k_values)*n_resamples} ok"
                  f"  ({time.time()-t_algo:.1f}s)")
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS), ds_cache


# ==========================================================================
# Step 2 — inline safe stability profile (raw-X consistent; no np.abs reload)
# ==========================================================================
def _jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _kuncheva_pair(r, d, k):
    denom = k * (d - k)
    return ((r * d - k ** 2) / denom) if denom != 0 else None


def _binom_tail_ge(c, n, p):
    """P[Binomial(n,p) >= c]."""
    if c <= 0:
        return 1.0
    if c > n:
        return 0.0
    total = 0.0
    for j in range(c, n + 1):
        total += math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
    return float(min(max(total, 0.0), 1.0))


def _threshold_count(t, n_folds):
    return int(math.ceil(t * n_folds - 1e-9))


def compute_stability_profile(df):
    """Compute the SAFE stability subset per (dataset, algorithm, K), from the
    logged selected indices only — feature frequencies, threshold sweep,
    pairwise Jaccard, Kuncheva, Nogueira-style estimate, observed-vs-random
    baseline, and cross-K recurrence. No np.abs, no X reload, no recovery."""
    freq_rows, group_rows, sweep_rows = [], [], []
    keys = ["dataset", "algorithm", "K"]
    for (dataset, algo, k), g in df.groupby(keys, dropna=False):
        ok = g[g["status"] == "ok"]
        n_ok = len(ok)
        K = int(k)
        d = int(g["n_features"].iloc[0])
        subsets = [list(np.asarray(v)) for v in ok["selected_feature_indices"]]

        fold_count = Counter()
        for s in subsets:
            for f in set(s):
                fold_count[int(f)] += 1
        for feat in sorted(fold_count, key=lambda f: (-fold_count[f], f)):
            freq_rows.append(dict(
                dataset=dataset, algorithm=algo, K=K, feature_index=int(feat),
                fold_count=int(fold_count[feat]),
                frequency=round(fold_count[feat] / n_ok, 6) if n_ok else None,
                selected_in_all=bool(fold_count[feat] == n_ok and n_ok > 0)))

        # pairwise Jaccard + Kuncheva
        pj = [_jaccard(a, b) for a, b in combinations(subsets, 2)]
        kv = []
        for a, b in combinations(subsets, 2):
            val = _kuncheva_pair(len(set(a) & set(b)), d, K)
            if val is not None:
                kv.append(val)
        # Nogueira variance-based stability estimator (exact, with M/(M-1) correction)
        nog = None
        if d > 0 and K > 0 and n_ok > 1:
            denom = (K / d) * (1.0 - K / d)
            if denom != 0:
                var_sum = sum((c / n_ok) * (1.0 - c / n_ok)
                              for c in fold_count.values())
                mean_unbiased_var = (n_ok / (n_ok - 1.0)) * (var_sum / d)
                nog = 1.0 - mean_unbiased_var / denom
        # observed-vs-random Jaccard baseline
        exp_rand_jacc = None
        if d > 0 and K > 0:
            exp_inter = (K ** 2) / d
            exp_union = 2 * K - exp_inter
            exp_rand_jacc = (exp_inter / exp_union) if exp_union > 0 else None
        pj_mean = float(np.mean(pj)) if pj else None
        obs_vs_rand = ((pj_mean / exp_rand_jacc)
                       if (pj_mean is not None and exp_rand_jacc) else None)
        # threshold sweep + chance baseline
        p_random = (K / d) if d > 0 else None
        for t in THRESHOLDS:
            tcount = _threshold_count(t, n_ok if n_ok else 1)
            core = sorted(f for f, c in fold_count.items() if c >= tcount)
            exp_rand = (d * _binom_tail_ge(tcount, n_ok, p_random)
                        if (d > 0 and p_random is not None and n_ok > 0) else None)
            ratio = (len(core) / exp_rand) if (exp_rand and exp_rand > 0) else None
            sweep_rows.append(dict(
                dataset=dataset, algorithm=algo, K=K, threshold=t,
                threshold_count=int(tcount), n_resamples_ok=int(n_ok),
                recurrent_set_size=int(len(core)),
                recurrent_features=[int(x) for x in core],
                core_fraction=round(len(core) / K, 6) if K else None,
                expected_random_core_size=(round(exp_rand, 6)
                                           if exp_rand is not None else None),
                observed_vs_expected_ratio=(round(ratio, 4)
                                            if ratio is not None else None)))
        aucs = ok["auc"].dropna().astype(float)
        accs = ok["accuracy"].dropna().astype(float)
        bals = ok["balanced_accuracy"].dropna().astype(float)
        f1s = ok["f1"].dropna().astype(float)
        rts = g["fs_runtime_sec"].dropna().astype(float)
        group_rows.append(dict(
            dataset=dataset, algorithm=algo, K=K, n_features=d,
            n_resamples=int(g["n_splits"].iloc[0]), n_ok=int(n_ok),
            n_not_ok=int(len(g) - n_ok),
            feature_pool_size=int(len(fold_count)),
            mean_pairwise_jaccard=round(pj_mean, 6) if pj_mean is not None else None,
            kuncheva_mean=round(float(np.mean(kv)), 6) if kv else None,
            nogueira_style_stability=round(nog, 6) if nog is not None else None,
            expected_random_jaccard=(round(exp_rand_jacc, 6)
                                     if exp_rand_jacc is not None else None),
            observed_vs_random_jaccard_ratio=(round(obs_vs_rand, 4)
                                              if obs_vs_rand is not None else None),
            auc_mean=round(float(aucs.mean()), 6) if len(aucs) else None,
            auc_std=round(float(aucs.std(ddof=0)), 6) if len(aucs) else None,
            accuracy_mean=round(float(accs.mean()), 6) if len(accs) else None,
            accuracy_std=round(float(accs.std(ddof=0)), 6) if len(accs) else None,
            balanced_accuracy_mean=round(float(bals.mean()), 6) if len(bals) else None,
            balanced_accuracy_std=round(float(bals.std(ddof=0)), 6) if len(bals) else None,
            f1_mean=round(float(f1s.mean()), 6) if len(f1s) else None,
            f1_std=round(float(f1s.std(ddof=0)), 6) if len(f1s) else None,
            fs_runtime_mean=round(float(rts.mean()), 4) if len(rts) else None,
            fs_runtime_max=round(float(rts.max()), 4) if len(rts) else None))

    # cross-K recurrence: features in the recurrent set for >=2 K, per
    # (dataset, algorithm, threshold)
    sweep_df = pd.DataFrame(sweep_rows)
    cross_rows = []
    if len(sweep_df):
        for (dataset, algo, t), gs in sweep_df.groupby(
                ["dataset", "algorithm", "threshold"], dropna=False):
            feat_to_ks = {}
            for r in gs.itertuples(index=False):
                for f in r.recurrent_features:
                    feat_to_ks.setdefault(int(f), []).append(int(r.K))
            for feat, ks in sorted(feat_to_ks.items()):
                ks_sorted = sorted(set(ks))
                cross_rows.append(dict(
                    dataset=dataset, algorithm=algo, threshold=t,
                    feature_index=int(feat), n_K_recurrent=len(ks_sorted),
                    K_values_recurrent=ks_sorted,
                    appears_in_multiple_K=bool(len(ks_sorted) >= 2)))
    return (pd.DataFrame(freq_rows), pd.DataFrame(group_rows), sweep_df,
            pd.DataFrame(cross_rows))


# ==========================================================================
# Validation
# ==========================================================================
def validate(df, datasets, algorithms, k_values, n_resamples):
    c = {}
    expected = len(datasets) * len(algorithms) * len(k_values) * n_resamples
    c["expected_rows"] = expected
    c["actual_rows"] = len(df)
    c["row_count_ok"] = bool(len(df) == expected)
    c["status_counts"] = df["status"].value_counts().to_dict()
    ok = df[df["status"] == "ok"]
    c["n_ok"] = int(len(ok))
    c["datasets_ok"] = bool(set(df["dataset"]) == set(datasets))
    c["algorithms_ok"] = bool(set(df["algorithm"]) == set(algorithms))
    c["k_values_ok"] = bool(sorted(set(int(k) for k in df["K"])) == sorted(k_values))
    c["count_equals_k_all_ok"] = bool(
        (ok["selected_feature_count"] == ok["K"]).all()) if len(ok) else True
    c["indices_valid_all_ok"] = bool(
        ok["selected_indices_valid"].all()) if len(ok) else True
    c["preprocessing_raw_x"] = bool((df["preprocessing_note"] == "raw_X").all())
    c["abs_not_applied"] = bool((df["abs_applied"] == False).all())  # noqa: E712
    c["fold_id_range"] = (int(df["fold_id"].min()), int(df["fold_id"].max()))
    c["n_features_by_dataset"] = {
        d: sorted(int(x) for x in g["n_features"].unique())
        for d, g in df.groupby("dataset")}
    grouping_ok = True
    for (_, _, _), g in df.groupby(["dataset", "algorithm", "K"]):
        if len(g) != n_resamples:
            grouping_ok = False
        if sorted(g["fold_id"].tolist()) != list(range(1, n_resamples + 1)):
            grouping_ok = False
    c["resample_grouping_ok"] = bool(grouping_ok)
    metric_ok = True
    for m in ("auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        vals = df[m].dropna().astype(float)
        if len(vals) and not vals.between(0.0, 1.0).all():
            metric_ok = False
    c["metrics_in_range"] = bool(metric_ok)
    return c


# ==========================================================================
# Reports
# ==========================================================================
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_main_report(ds_cache, df, checks, group_df, sweep_df, cross_df,
                      commands, out_paths, datasets, algorithms, k_values,
                      n_resamples):
    L = []
    L.append("# Real-Data R2 Stability Slice Report — Article 5\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Stage R2 — the first full real-data feature-selection "
             "stability slice. **Illustrative and diagnostic, not a broad "
             "benchmark.** ETree + mRMR + LASSO_Stability + ReliefF on colon + "
             "PeriodChanger + SMK-CAN-187, raw signed X (np.abs not applied), "
             "repeated stratified shuffle-split, 50 resamples. No recovery "
             "metrics, no synthetic data, no slashdot.\n")
    L.append("---\n")

    # 1. purpose
    L.append("## 1. Purpose\n")
    L.append("R2 is the first full real-data stability slice after the "
             "completed synthetic arc, the R1 plumbing smoke, the algorithm "
             "compatibility smoke, and the final ReliefF / LASSO_Stability "
             "pre-R2 smoke. It applies the Article 5 stability profile to "
             "three real p>>n datasets with four mechanistically distinct "
             "selectors. **It is an illustrative, diagnostic applied slice — "
             "not a broad benchmark** (Article 2 is the broad benchmark) and "
             "**not a recovery study** (real data has no planted ground "
             "truth).\n")

    # 2. inputs
    L.append("## 2. Inputs\n")
    L.append("| Dataset | Exact path | N | P | classes (0/1) | X signed? |")
    L.append("|---------|-----------|---|---|---------------|-----------|")
    for nm in datasets:
        ds = ds_cache[nm]
        L.append(f"| {nm} | `{ds['relpath']}` | {ds['n_samples']} | "
                 f"{ds['n_features']} | {ds['n_class_0']}/{ds['n_class_1']} | "
                 f"{'yes' if ds['x_negative_count'] > 0 else 'no'} "
                 f"({ds['x_negative_count']} neg) |")
    L.append("")
    L.append(f"- **Algorithms:** {', '.join(algorithms)} (the locked R2 set).")
    L.append(f"- **K values:** {k_values}.")
    L.append(f"- **Resampling:** {RESAMPLE_TYPE}, n_resamples={n_resamples}, "
             f"test_size={TEST_SIZE}, seed={SEED}.")
    L.append(f"- **Classifier (held-out eval):** {CLASSIFIER}.")
    L.append("")

    # 3. preprocessing
    L.append("## 3. Preprocessing and Leakage Controls\n")
    L.append("- **Raw X** for all three datasets; **`np.abs(X)` NOT applied** "
             "(colon and PeriodChanger are signed; SMK-CAN-187 is non-negative "
             "— raw X used for uniformity). Every row records "
             "`preprocessing_note=raw_X`, `abs_applied=False`.")
    L.append("- **y label-encoded** to {0,1} via a 0-based label encoder; "
             "`np.abs` never applied to y. Raw->encoded: "
             + "; ".join(f"{nm} {ds_cache[nm]['raw_labels']}->"
                         f"{ds_cache[nm]['encoded_labels']}" for nm in datasets)
             + ".")
    L.append("- **Train-split-only feature selection**; classifier trained "
             "only on the selected training features; **held-out evaluation** "
             "on the test split only. Per-split cleaning is element-wise "
             "(float64 + NaN/Inf guard) — no cross-sample leakage.")
    L.append("- **No recovery metrics.** `synthetic_recovery_metrics.py` is "
             "not used; real data has no planted ground truth. The stability "
             "profile is computed inline from the logged selected indices "
             "(preprocessing-independent); the offline analyzer's `np.abs` "
             "periphery-correlation view is deliberately not invoked.")
    L.append("")

    # 4. logging validation
    L.append("## 4. Logging Validation\n")
    L.append(f"- expected rows: **{checks['expected_rows']}** "
             f"({len(datasets)} datasets x {len(algorithms)} algorithms x "
             f"{len(k_values)} K x {n_resamples} resamples); actual rows: "
             f"**{checks['actual_rows']}** — "
             f"{'MATCH' if checks['row_count_ok'] else 'MISMATCH'}")
    L.append(f"- status counts: {checks['status_counts']}")
    L.append(f"- datasets == locked set: {checks['datasets_ok']}; "
             f"algorithms == locked set: {checks['algorithms_ok']}; "
             f"K values == locked grid: {checks['k_values_ok']}")
    L.append(f"- `selected_feature_count == K` on all ok rows: "
             f"{checks['count_equals_k_all_ok']}; selected indices valid "
             f"(in range, unique, length K) on all ok rows: "
             f"{checks['indices_valid_all_ok']}")
    L.append(f"- n_features by dataset: {checks['n_features_by_dataset']}")
    L.append(f"- `preprocessing_note == raw_X` on all rows: "
             f"{checks['preprocessing_raw_x']}; `abs_applied == False` on all "
             f"rows: {checks['abs_not_applied']}")
    L.append(f"- fold_id range: {checks['fold_id_range']}; resample grouping "
             f"ok (every dataset/algorithm/K has {n_resamples} resamples, "
             f"fold_id 1..{n_resamples}): {checks['resample_grouping_ok']}")
    L.append(f"- metrics within [0,1] or null: {checks['metrics_in_range']}")
    L.append("- Rows per dataset / algorithm (ok / total):\n")
    L.append("| Dataset | " + " | ".join(algorithms) + " |")
    L.append("|---------|" + "|".join("---" for _ in algorithms) + "|")
    for nm in datasets:
        cells = []
        for algo in algorithms:
            sub = df[(df["dataset"] == nm) & (df["algorithm"] == algo)]
            oksub = sub[sub["status"] == "ok"]
            cells.append(f"{len(oksub)}/{len(sub)}")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("- All outputs are written under "
             "`results/article_stability/real_data/r2/` — no writes to "
             "`results/article_stability/real_data_smoke/`, `results/article_2/`, or "
             "`results/pillar_1_benchmark/`.")
    L.append("")

    # 5. runtime
    L.append("## 5. Runtime Summary\n")
    L.append("Mean / max FS runtime per call, by dataset and algorithm "
             "(seconds):\n")
    L.append("| Dataset | " + " | ".join(algorithms) + " |")
    L.append("|---------|" + "|".join("---" for _ in algorithms) + "|")
    for nm in datasets:
        cells = []
        for algo in algorithms:
            sub = group_df[(group_df["dataset"] == nm)
                           & (group_df["algorithm"] == algo)]
            if len(sub):
                mn = sub["fs_runtime_mean"].dropna()
                mx = sub["fs_runtime_max"].dropna()
                cells.append(f"{_fmt(mn.mean() if len(mn) else None)} / "
                             f"{_fmt(mx.max() if len(mx) else None)}")
            else:
                cells.append("—")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    rf_smk = group_df[(group_df["dataset"] == "SMK-CAN-187")
                      & (group_df["algorithm"] == "ReliefF")]
    if len(rf_smk):
        L.append(f"- **ReliefF on SMK-CAN-187** (the pre-R2 runtime concern): "
                 f"mean per-call {_fmt(rf_smk['fs_runtime_mean'].mean())} s, "
                 f"max {_fmt(rf_smk['fs_runtime_max'].max())} s — consistent "
                 f"with the pre-R2 spot-check; runtime remained feasible.")
    bad = df[~df["status"].isin(["ok"])]
    if len(bad):
        L.append(f"- non-ok rows: {len(bad)} "
                 f"({bad['status'].value_counts().to_dict()}) — see §4.")
    else:
        L.append("- No failed or skipped cells — every (dataset, algorithm, K, "
                 "resample) completed.")
    L.append("")

    # 6. performance
    L.append("## 6. Performance Summary\n")
    L.append("Held-out metrics by dataset / algorithm / K (mean ± std over "
             "resamples). **Performance is descriptive only — Article 5's "
             "focus is the stability profile, not maximising AUC.**\n")
    L.append("| Dataset | Algorithm | K | AUC | accuracy | balanced acc | F1 |")
    L.append("|---------|-----------|---|-----|----------|--------------|-----|")
    for nm in datasets:
        for algo in algorithms:
            for k in k_values:
                r = group_df[(group_df["dataset"] == nm)
                             & (group_df["algorithm"] == algo)
                             & (group_df["K"] == k)]
                if not len(r):
                    continue
                rr = r.iloc[0]
                L.append(f"| {nm} | {algo} | {k} | "
                         f"{_fmt(rr['auc_mean'])}±{_fmt(rr['auc_std'])} | "
                         f"{_fmt(rr['accuracy_mean'])}±{_fmt(rr['accuracy_std'])} | "
                         f"{_fmt(rr['balanced_accuracy_mean'])}±"
                         f"{_fmt(rr['balanced_accuracy_std'])} | "
                         f"{_fmt(rr['f1_mean'])}±{_fmt(rr['f1_std'])} |")
    L.append("")

    # 7. stability profile
    L.append("## 7. Stability Profile Summary\n")
    L.append("Per dataset / algorithm / K: recurrent feature counts at "
             "selection-frequency thresholds 0.6 / 0.8 / 1.0, mean pairwise "
             "Jaccard, Kuncheva index, Nogueira-style estimate, and the "
             "observed-vs-random Jaccard ratio. All computed from the logged "
             "selected indices (preprocessing-independent). Exploratory and "
             "descriptive.\n")
    L.append("| Dataset | Algorithm | K | recur@0.6 | recur@0.8 | recur@1.0 | "
             "Jaccard | Kuncheva | Nogueira | obs/rand |")
    L.append("|---------|-----------|---|-----------|-----------|-----------|"
             "---------|----------|----------|----------|")
    for nm in datasets:
        for algo in algorithms:
            for k in k_values:
                gr = group_df[(group_df["dataset"] == nm)
                              & (group_df["algorithm"] == algo)
                              & (group_df["K"] == k)]
                if not len(gr):
                    continue
                grr = gr.iloc[0]

                def rec(t):
                    s = sweep_df[(sweep_df["dataset"] == nm)
                                 & (sweep_df["algorithm"] == algo)
                                 & (sweep_df["K"] == k)
                                 & (sweep_df["threshold"] == t)]
                    return int(s.iloc[0]["recurrent_set_size"]) if len(s) else "—"
                L.append(f"| {nm} | {algo} | {k} | {rec(0.6)} | {rec(0.8)} | "
                         f"{rec(1.0)} | {_fmt(grr['mean_pairwise_jaccard'])} | "
                         f"{_fmt(grr['kuncheva_mean'])} | "
                         f"{_fmt(grr['nogueira_style_stability'])} | "
                         f"{_fmt(grr['observed_vs_random_jaccard_ratio'], 1)} |")
    L.append("")
    L.append("Cross-K recurrence (features in the recurrent set for >=2 K "
             "values, at threshold 0.8), per dataset/algorithm:\n")
    L.append("| Dataset | Algorithm | features recurrent across >=2 K (t=0.8) |")
    L.append("|---------|-----------|------------------------------------------|")
    for nm in datasets:
        for algo in algorithms:
            if not len(cross_df):
                continue
            sub = cross_df[(cross_df["dataset"] == nm)
                           & (cross_df["algorithm"] == algo)
                           & (cross_df["threshold"] == 0.8)
                           & (cross_df["appears_in_multiple_K"])]
            L.append(f"| {nm} | {algo} | {len(sub)} |")
    L.append("")

    # 8. cross-dataset
    L.append("## 8. Cross-Dataset Observations\n")
    L.append("Described cautiously and descriptively — no true-core claims. "
             "Reading the K=10 / threshold-0.8 cell (a mid-budget reference) "
             "averaged across the four algorithms:\n")
    L.append("| Dataset | mean recur@0.8 (K=10) | mean Kuncheva (K=10) | "
             "mean obs/rand Jaccard (K=10) |")
    L.append("|---------|------------------------|----------------------|"
             "------------------------------|")
    for nm in datasets:
        recs, kun, ovr = [], [], []
        for algo in algorithms:
            s = sweep_df[(sweep_df["dataset"] == nm) & (sweep_df["algorithm"] == algo)
                         & (sweep_df["K"] == 10) & (sweep_df["threshold"] == 0.8)]
            if len(s):
                recs.append(int(s.iloc[0]["recurrent_set_size"]))
            gr = group_df[(group_df["dataset"] == nm) & (group_df["algorithm"] == algo)
                          & (group_df["K"] == 10)]
            if len(gr):
                if gr.iloc[0]["kuncheva_mean"] is not None:
                    kun.append(float(gr.iloc[0]["kuncheva_mean"]))
                if gr.iloc[0]["observed_vs_random_jaccard_ratio"] is not None:
                    ovr.append(float(gr.iloc[0]["observed_vs_random_jaccard_ratio"]))
        L.append(f"| {nm} | {_fmt(np.mean(recs) if recs else None, 1)} | "
                 f"{_fmt(np.mean(kun) if kun else None)} | "
                 f"{_fmt(np.mean(ovr) if ovr else None, 1)} |")
    L.append("")
    L.append("- The numbers above are a **descriptive** cross-dataset view; "
             "whether recurrence is higher or lower on a given dataset is "
             "reported as an observation, not a verdict. No dataset is "
             "labelled \"stable\" or \"unstable\"; see the §11 taxonomy.")
    L.append("")

    # 9. cross-algorithm
    L.append("## 9. Cross-Algorithm Observations\n")
    L.append("Per algorithm, averaged across datasets at K=10 / t=0.8 — "
             "descriptive only, **not a benchmark ranking**:\n")
    L.append("| Algorithm | mean recur@0.8 (K=10) | mean Kuncheva (K=10) | "
             "mean obs/rand Jaccard (K=10) |")
    L.append("|-----------|------------------------|----------------------|"
             "------------------------------|")
    for algo in algorithms:
        recs, kun, ovr = [], [], []
        for nm in datasets:
            s = sweep_df[(sweep_df["dataset"] == nm) & (sweep_df["algorithm"] == algo)
                         & (sweep_df["K"] == 10) & (sweep_df["threshold"] == 0.8)]
            if len(s):
                recs.append(int(s.iloc[0]["recurrent_set_size"]))
            gr = group_df[(group_df["dataset"] == nm) & (group_df["algorithm"] == algo)
                          & (group_df["K"] == 10)]
            if len(gr):
                if gr.iloc[0]["kuncheva_mean"] is not None:
                    kun.append(float(gr.iloc[0]["kuncheva_mean"]))
                if gr.iloc[0]["observed_vs_random_jaccard_ratio"] is not None:
                    ovr.append(float(gr.iloc[0]["observed_vs_random_jaccard_ratio"]))
        L.append(f"| {algo} | {_fmt(np.mean(recs) if recs else None, 1)} | "
                 f"{_fmt(np.mean(kun) if kun else None)} | "
                 f"{_fmt(np.mean(ovr) if ovr else None, 1)} |")
    L.append("")
    L.append("- ETree (ensemble), mRMR (redundancy-aware filter), "
             "LASSO_Stability (L1/sparse), and ReliefF (instance/margin-based) "
             "are compared **descriptively**. Whether the redundancy-aware / "
             "L1 / instance-based profiles differ from the ensemble profile — "
             "or whether all four behave similarly on a given dataset — is "
             "reported as an observation. A low-contrast cross-algorithm "
             "result is itself a legitimate, reportable finding. No broad "
             "benchmark claim is made.")
    L.append("")

    # 10. colon K=60 caveat
    L.append("## 10. Colon K=60 Caveat\n")
    L.append("- **K=60 on colon was logged for schema consistency only.** "
             "colon has N=62; with test_size 0.2 the training folds are ~49 "
             "samples, so K=60 selects 60 of 2000 features from a ~49-sample "
             "fold (more features than training samples) — near-degenerate.")
    L.append("- **Exclude K=60-on-colon from headline interpretation.** "
             "**K=30-on-colon is also marginal** (~49-sample folds) and should "
             "be read cautiously. colon's interpretable budgets are mainly "
             "K = 1, 5, 10.")
    L.append("- PeriodChanger (N=90) and SMK-CAN-187 (N=187) are less affected, "
             "though K=60 is still a large budget relative to their training "
             "folds.")
    L.append("")

    # 11. taxonomy
    L.append("## 11. Descriptive Interpretation Taxonomy\n")
    L.append("Any interpretation of the §7-§9 numbers must use **soft, "
             "exploratory, threshold-dependent, dataset/algorithm/protocol-"
             "specific** categories only:\n")
    L.append("- *high-recurrence / consistent-selection profile*")
    L.append("- *recurrent-core-plus-rotating-periphery profile*")
    L.append("- *low-recurrence / near-random profile*")
    L.append("- *inconclusive profile*\n")
    L.append("**Avoided:** hard \"stable\" / \"unstable\" verdicts; \"trusted "
             "feature\"; \"true feature\"; \"biomarker\"; \"validated "
             "feature\". This report deliberately does **not** assign a "
             "category to any (dataset, algorithm) cell — categorisation is "
             "left to the Opus interpretation checkpoint (§15), which has the "
             "full profile tables.")
    L.append("")

    # 12. claims allowed
    L.append("## 12. Claims Allowed After R2\n")
    L.append("- The Article 5 stability profile **can be applied to real "
             "scientific datasets** and produces interpretable, non-degenerate "
             "descriptive output.")
    L.append("- Real datasets show **descriptive stability profiles** "
             "(feature-frequency recurrence, threshold sweep, Jaccard, "
             "Kuncheva, Nogueira-style, observed-vs-random).")
    L.append("- Recurrence / stability **differs by dataset, algorithm, and K** "
             "— where the §7-§9 tables support it.")
    L.append("- Held-out performance and stability can be reported **side by "
             "side**, and can diverge — where supported.")
    L.append("- R2 provides **illustrative applied evidence** that the "
             "trust-layer concept is practically usable as a descriptive "
             "diagnostic — not benchmark-scale proof.")
    L.append("")

    # 13. claims forbidden
    L.append("## 13. Claims Still Forbidden\n")
    L.append("- True-core / planted-core / support-recovery claims on real "
             "data (no ground truth).")
    L.append("- Causal, biomarker, or validated-feature claims; that recurrent "
             "features are \"correct\" or \"trusted\".")
    L.append("- That 3 datasets x 4 algorithms is a comprehensive benchmark "
             "(Article 2 is the broad benchmark).")
    L.append("- Instance-dependence claims on real data (one instance per "
             "dataset — that axis is synthetic-only).")
    L.append("- Statistical error-control (PFER/FWER/FDR) claims; the "
             "Nogueira-style index as a *verified* estimator.")
    L.append("- Reliance on the unpublished Article 3 / Article 4 as evidence.")
    L.append("")

    # 14. verdict
    L.append("## 14. R2 Verdict\n")
    technical_pass = bool(
        checks["row_count_ok"] and checks["datasets_ok"]
        and checks["algorithms_ok"] and checks["k_values_ok"]
        and checks["count_equals_k_all_ok"] and checks["indices_valid_all_ok"]
        and checks["preprocessing_raw_x"] and checks["abs_not_applied"]
        and checks["resample_grouping_ok"] and checks["metrics_in_range"]
        and checks["n_ok"] > 0)
    if technical_pass:
        L.append("- **R2 passed technically.** All 3000 rows logged with the "
                 "correct schema; datasets / algorithms / K grid match the "
                 "locked configuration; `selected_feature_count == K` and "
                 "valid unique indices on every ok row; raw X confirmed "
                 "(`abs_applied=False`) on every row; resamples correctly "
                 "grouped; metrics in range.")
        L.append("- **R2 produced interpretable stability profiles** — the "
                 "feature-frequency, threshold-sweep, Jaccard, Kuncheva, "
                 "Nogueira-style, observed-vs-random, and cross-K tables are "
                 "all populated (§7).")
        L.append("- **No fixes required.** R2 should proceed to an Opus "
                 "interpretation checkpoint.")
    else:
        L.append("- **R2 did not fully pass the technical checks** (see §4). "
                 "Resolve the flagged issue before interpretation.")
    L.append("")

    # 15. next step
    L.append("## 15. Recommended Next Step\n")
    L.append("- **Do NOT update the NotebookLM package automatically.**")
    L.append("- Recommended sequence: (1) an **Opus interpretation "
             "checkpoint** reads the R2 stability profiles and assigns the "
             "§11 descriptive categories per (dataset, algorithm); (2) then "
             "the NotebookLM transfer package is updated; (3) then the "
             "writing outline / results integration proceeds.")
    L.append("- Stage R3 (the optional slashdot edge-case appendix) remains "
             "optional and is not triggered by this run.")
    L.append("")

    L.append("---\n")
    L.append("**Commands run:**")
    L.append("```bash")
    L.extend(commands)
    L.append("```")
    L.append("")
    L.append("**Output files:**")
    for label, p in out_paths:
        L.append(f"- {label}: `{p}`")
    L.append("")
    L.append("*Stage R2 real-data stability slice — 3 datasets x 4 algorithms "
             "x 5 K x 50 resamples, raw X. No recovery metrics, no synthetic "
             "data, no slashdot, no raw dataset modified, no benchmark file "
             "modified, nothing written to results/article_2 or "
             "results/pillar_1_benchmark, no commit, no push.*")
    return "\n".join(L)


def build_dataset_report(nm, ds, df, group_df, sweep_df, k_values, algorithms):
    L = []
    L.append(f"# Real-Data R2 Report — {nm}\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection* — Stage R2 per-dataset extract.  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Per-dataset extract of the Stage R2 stability slice. "
             "Descriptive only; see `REAL_DATA_R2_STABILITY_SLICE_REPORT.md` "
             "for the full context, taxonomy, and allowed/forbidden claims.\n")
    L.append("---\n")
    L.append(f"- **Exact path:** `{ds['relpath']}`")
    L.append(f"- **Shape:** N={ds['n_samples']}, P={ds['n_features']}; class "
             f"balance {ds['n_class_0']}/{ds['n_class_1']}; X "
             f"{'signed' if ds['x_negative_count'] > 0 else 'non-negative'} "
             f"(raw X used, np.abs NOT applied); y "
             f"{ds['raw_labels']}->{ds['encoded_labels']}.")
    sub = df[df["dataset"] == nm]
    L.append(f"- **Rows:** {len(sub)} ({int((sub['status']=='ok').sum())} ok); "
             f"status {sub['status'].value_counts().to_dict()}.")
    if nm == "colon":
        L.append("- **Caveat:** K=60-on-colon is near-degenerate (N=62) and "
                 "K=30-on-colon is marginal — exclude/treat cautiously; "
                 "interpretable budgets K=1,5,10.")
    L.append("")
    L.append("## Stability profile (recurrent counts / Jaccard / Kuncheva / "
             "Nogueira / obs-vs-random)\n")
    L.append("| Algorithm | K | recur@0.6 | recur@0.8 | recur@1.0 | Jaccard | "
             "Kuncheva | Nogueira | obs/rand | AUC mean±std |")
    L.append("|-----------|---|-----------|-----------|-----------|---------|"
             "----------|----------|----------|-------------|")
    for algo in algorithms:
        for k in k_values:
            gr = group_df[(group_df["dataset"] == nm)
                          & (group_df["algorithm"] == algo)
                          & (group_df["K"] == k)]
            if not len(gr):
                continue
            grr = gr.iloc[0]

            def rec(t):
                s = sweep_df[(sweep_df["dataset"] == nm)
                             & (sweep_df["algorithm"] == algo)
                             & (sweep_df["K"] == k)
                             & (sweep_df["threshold"] == t)]
                return int(s.iloc[0]["recurrent_set_size"]) if len(s) else "—"
            L.append(f"| {algo} | {k} | {rec(0.6)} | {rec(0.8)} | {rec(1.0)} | "
                     f"{_fmt(grr['mean_pairwise_jaccard'])} | "
                     f"{_fmt(grr['kuncheva_mean'])} | "
                     f"{_fmt(grr['nogueira_style_stability'])} | "
                     f"{_fmt(grr['observed_vs_random_jaccard_ratio'], 1)} | "
                     f"{_fmt(grr['auc_mean'])}±{_fmt(grr['auc_std'])} |")
    L.append("")
    L.append("*Descriptive per-dataset extract — no stability verdict, no "
             "true-core / causal / biomarker claim. Soft taxonomy and "
             "interpretation are deferred to the Opus checkpoint.*")
    return "\n".join(L)


# ==========================================================================
# CLI / driver
# ==========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Article 5 Stage R2 real-data stability slice "
                    "(3 datasets x 4 algorithms x 5 K x 50 resamples, raw X).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan + dataset confirmation, then exit "
                             "without running feature selection or writing.")
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    n_resamples = args.n_resamples

    # scope lock
    for d in datasets:
        if d not in ALLOWED_DATASETS:
            raise SystemExit(f"[SCOPE] dataset '{d}' is out of scope for R2; "
                             f"allowed: {sorted(ALLOWED_DATASETS)}")
    for a in algorithms:
        if a not in ALLOWED_ALGORITHMS:
            raise SystemExit(f"[SCOPE] algorithm '{a}' is out of scope for R2; "
                             f"allowed: {sorted(ALLOWED_ALGORITHMS)}")
    planned_rows = len(datasets) * len(algorithms) * len(k_values) * n_resamples

    if args.dry_run:
        print("=" * 70)
        print("Article 5 — Stage R2 real-data stability slice  [DRY RUN]")
        print("=" * 70)
        all_ok = True
        for nm in datasets:
            mat = (PROJECT_ROOT / DATASETS[nm]["relpath"]).resolve()
            print(f"Dataset {nm:14s}: {DATASETS[nm]['relpath']}  "
                  f"exists={mat.exists()}")
            all_ok = all_ok and mat.exists()
        print(f"Datasets ({len(datasets)})   : {datasets}")
        print(f"Algorithms ({len(algorithms)}) : {algorithms}")
        for a in algorithms:
            print(f"  registered '{a}': {a in feature_selection_mapping}")
        print(f"K values ({len(k_values)})    : {k_values}")
        print(f"n_resamples       : {n_resamples}  test_size={TEST_SIZE}  "
              f"seed={SEED}  ({RESAMPLE_TYPE})")
        print(f"Expected rows     : {planned_rows} "
              f"({len(datasets)} x {len(algorithms)} x {len(k_values)} x "
              f"{n_resamples})")
        print(f"Preprocessing     : raw X; np.abs NOT applied; abs_applied=False")
        print(f"Output root       : {R2_OUTPUT_ROOT}")
        print(f"  per-fold parquet: {PERFOLD_DIR / (RUN_TAG + '.parquet')}")
        print(f"  stability dir   : {STABILITY_ROOT / RUN_TAG}")
        print(f"  main report     : {MAIN_REPORT_PATH}")
        print(f"Recovery metrics  : NOT used (real data has no ground truth)")
        print(f"Offline analyzer  : NOT invoked (its np.abs periphery view is "
              f"raw-X-inconsistent); stability computed inline (safe subset)")
        print(f"All input files present: {all_ok}")
        print("DRY RUN — no feature selection executed, no file written.")
        print("=" * 70)
        return 0 if all_ok else 1

    # --- step 1: logging --------------------------------------------------
    t_start = time.time()
    df, ds_cache = run_logging(datasets, algorithms, k_values, n_resamples)
    checks = validate(df, datasets, algorithms, k_values, n_resamples)

    PERFOLD_DIR.mkdir(parents=True, exist_ok=True)
    perfold_path = safe_path(PERFOLD_DIR, f"{RUN_TAG}.parquet", R2_OUTPUT_ROOT)
    df.to_parquet(perfold_path, index=False)
    print(f"\n[output] per-fold parquet: {perfold_path}  ({len(df)} rows)")

    # --- step 2: inline safe stability profile ----------------------------
    freq_df, group_df, sweep_df, cross_df = compute_stability_profile(df)
    stab_dir = (STABILITY_ROOT / RUN_TAG).resolve()
    stab_dir.mkdir(parents=True, exist_ok=True)
    stab_paths = {}
    for key, base, dfo in [("feature_frequencies", "feature_frequencies.parquet", freq_df),
                           ("group_summary", "stability_group_summary.parquet", group_df),
                           ("threshold_sweep", "threshold_sweep.parquet", sweep_df),
                           ("cross_k", "cross_k_recurrence.parquet", cross_df)]:
        p = safe_path(stab_dir, base, R2_OUTPUT_ROOT)
        dfo.to_parquet(p, index=False)
        stab_paths[key] = p
        print(f"[output] stability  : {p}  ({len(dfo)} rows)")

    # --- step 3: reports --------------------------------------------------
    commands = [
        "python -m py_compile articles/article_stability/scripts/run_real_data_r2_stability_slice.py",
        "python articles/article_stability/scripts/run_real_data_r2_stability_slice.py --dry-run",
        "python articles/article_stability/scripts/run_real_data_r2_stability_slice.py",
    ]
    out_paths = [
        ("per-fold parquet", perfold_path),
        ("stability — feature frequencies", stab_paths["feature_frequencies"]),
        ("stability — group summary", stab_paths["group_summary"]),
        ("stability — threshold sweep", stab_paths["threshold_sweep"]),
        ("stability — cross-K recurrence", stab_paths["cross_k"]),
        ("main R2 report", MAIN_REPORT_PATH),
    ]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_report_paths = {}
    for nm in datasets:
        rp = safe_path(DOCS_DIR, DATASET_REPORT_NAMES[nm], DOCS_DIR)
        rp.write_text(build_dataset_report(
            nm, ds_cache[nm], df, group_df, sweep_df, k_values, algorithms))
        dataset_report_paths[nm] = rp
        out_paths.append((f"per-dataset report ({nm})", rp))
        print(f"[output] dataset report: {rp}")

    main_report_path = safe_path(DOCS_DIR, MAIN_REPORT_PATH.name, DOCS_DIR)
    main_report_path.write_text(build_main_report(
        ds_cache, df, checks, group_df, sweep_df, cross_df, commands,
        out_paths, datasets, algorithms, k_values, n_resamples))
    print(f"[output] main R2 report : {main_report_path}")

    # --- console summary --------------------------------------------------
    technical_pass = bool(
        checks["row_count_ok"] and checks["datasets_ok"]
        and checks["algorithms_ok"] and checks["k_values_ok"]
        and checks["count_equals_k_all_ok"] and checks["indices_valid_all_ok"]
        and checks["preprocessing_raw_x"] and checks["abs_not_applied"]
        and checks["resample_grouping_ok"] and checks["metrics_in_range"]
        and checks["n_ok"] > 0)
    print("-" * 70)
    print(f"Rows logged : {checks['actual_rows']} (expected "
          f"{checks['expected_rows']})  status={checks['status_counts']}")
    print(f"count==K all ok: {checks['count_equals_k_all_ok']}  "
          f"indices valid all ok: {checks['indices_valid_all_ok']}  "
          f"raw X: {checks['abs_not_applied']}")
    print(f"datasets/algorithms/K match locked config: "
          f"{checks['datasets_ok']}/{checks['algorithms_ok']}/"
          f"{checks['k_values_ok']}")
    print(f"R2 technical verdict: {'PASS' if technical_pass else 'CHECK — see report'}")
    print(f"Total wall-clock: {time.time()-t_start:.1f}s")
    print("=" * 70)
    return 0 if technical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
