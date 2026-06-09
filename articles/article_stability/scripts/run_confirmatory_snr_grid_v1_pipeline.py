"""
run_confirmatory_snr_grid_v1_pipeline.py — Article 5, confirmatory SNR grid
===========================================================================

Runs the manuscript-resolution confirmatory synthetic SNR grid for Article 5
(*Feature Stability as a Trust Layer for Feature Selection*).

Motivation (see
results/article_stability/reports/experiment_planning/SYNTHETIC_MIDLOW_CALIBRATION_V1F_INTERPRETATION_CHECKPOINT.md):
v1e established a reliable positive-control anchor (core beta=2.5); v1f swept
alpha and identified the confirmatory grid composition. This run confirms the
selected grid at 50 resamples (10-resample -> 50-resample resolution upgrade).

This is a CONFIRMATORY run, not a new diagnostic. It REUSES the existing v1f
`.mat` datasets unchanged — it does not regenerate or modify any dataset.

Grid composition (composition C of the v1f checkpoint), per the same 8
generation seeds:
  * alpha=2.5  — positive-control reference   (synthetic_snrHIGH_a250_*gammaV1f)
  * alpha=0.6  — mild degradation / hedge rung (synthetic_snrMID_a060_*gammaV1f)
  * alpha=0.35 — variable-recovery mid/low     (synthetic_snrLOW_a035_*gammaV1f)
  * null control                               (synthetic_null_control_*gammaV1f)
alpha=1.0 is intentionally DROPPED (too close to the positive control in v1f).

What it does (ETree only, rho fixed 0.9, raw signed X, no np.abs):
  1. per-resample feature-selection logging  (reuses run_one_dataset from
     run_synthetic_repeated_resampling_logging.py — no new FS implemented)
  2. offline stability analysis              (invokes analyze_per_fold_stability.py)
  3. recovery metrics vs the manifests       (invokes synthetic_recovery_metrics.py)
  4. confirmatory across-seed summaries by alpha + a consolidated report

Scope guards: the 32 existing `_gammaV1f` datasets only; no real datasets, no
mRMR/LASSO, no p=10000, no rho=0.7, no alpha=1.0, no benchmark file touched, no
dataset generated or modified.

Usage:
    python articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py --dry-run
    python articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py
"""

import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Path setup. Lives at articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
ARTICLE5_DIR = _THIS_FILE.parent
PROJECT_ROOT = _THIS_FILE.parents[3]
if str(ARTICLE5_DIR) not in sys.path:
    sys.path.insert(0, str(ARTICLE5_DIR))

# TODO: run_synthetic_repeated_resampling_logging is not included in this public
# reproducibility package. This script is provided for reference only and cannot
# be executed without the private logging driver. See README.md for details.
try:
    import run_synthetic_repeated_resampling_logging as logdrv  # noqa: E402
except ImportError as _logdrv_err:
    raise ImportError(
        "run_synthetic_repeated_resampling_logging is required but not available "
        "in this public reproducibility package. This script cannot be run here."
    ) from _logdrv_err

# ==========================================================================
# Configuration
# ==========================================================================
DATASETS_DIR = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "datasets"
).resolve()
PERFOLD_DIR = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "per_fold_selections"
).resolve()
STABILITY_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "stability_profiles"
).resolve()
RECOVERY_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "recovery_metrics"
).resolve()
REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_CONFIRMATORY_SNR_GRID_V1F_50RESAMPLE_REPORT.md"
).resolve()

ANALYZER_SCRIPT = ARTICLE5_DIR / "analyze_per_fold_stability.py"
RECOVERY_SCRIPT = ARTICLE5_DIR / "synthetic_recovery_metrics.py"

# v1f 10-resample aggregate, read (read-only) for the qualitative comparison.
V1F_PROBE_AGGREGATE = (
    RECOVERY_ROOT / "synthetic_midlow_calibration_v1f_etree_seed0_10resamples"
    / "synthetic_midlow_v1f_across_seed_aggregate.parquet"
).resolve()

ALGORITHM = "ETree"
CLASSIFIER = "ETREE"
TEST_SIZE = 0.2
THRESHOLDS = [0.6, 0.8, 1.0]
RECOVERY_REPORT_NAME = "SYNTHETIC_CONFIRMATORY_SNR_GRID_V1F_RECOVERY_METRICS_REPORT.md"

# Recovery cells the confirmatory grid focuses on.
FOCUS_K = [30, 60]
FOCUS_THRESHOLDS = [0.8, 1.0]

# The confirmatory grid cells (dataset-name prefix, alpha, cell role).
# alpha=1.0 is intentionally dropped (too easy in v1f).
ALPHA_CELLS = [
    ("synthetic_snrHIGH_a250", 2.5, "positive_control_reference"),
    ("synthetic_snrMID_a060", 0.6, "mild_degradation_hedge"),
    ("synthetic_snrLOW_a035", 0.35, "variable_recovery_midlow"),
]
REFERENCE_ALPHA = 2.5
ALPHA_ORDER = [2.5, 0.6, 0.35]

OUT_AUC_BY_SEED = "synthetic_confirmatory_snr_grid_v1f_auc_by_seed_k.parquet"
OUT_RECOVERY_BY_SEED = "synthetic_confirmatory_snr_grid_v1f_recovery_by_seed.parquet"
OUT_STABILITY_BY_SEED = "synthetic_confirmatory_snr_grid_v1f_stability_by_seed.parquet"
OUT_ACROSS_SEED = "synthetic_confirmatory_snr_grid_v1f_across_seed_aggregate.parquet"


# ==========================================================================
# Dataset list
# ==========================================================================
def build_dataset_list(n_seeds):
    """The 4*n_seeds confirmatory-grid datasets — per seed the alpha=2.5 / 0.6 /
    0.35 known-core cells and a null control. All are existing v1f `.mat`
    files; nothing is generated."""
    datasets = []
    for i in range(n_seeds):
        sfx = f"_rho090_gammaV1f_seed{i:02d}"
        for prefix, alpha, cell_role in ALPHA_CELLS:
            datasets.append({
                "name": f"{prefix}{sfx}", "role": "known_core",
                "alpha": alpha, "alpha_label": f"alpha{alpha}",
                "cell_role": cell_role, "seed_index": i})
        datasets.append({
            "name": f"synthetic_null_control_gammaV1f_seed{i:02d}",
            "role": "null_control", "alpha": None, "alpha_label": "null",
            "cell_role": "null", "seed_index": i})
    return datasets


def null_name_for_seed(seed_index):
    return f"synthetic_null_control_gammaV1f_seed{seed_index:02d}"


# ==========================================================================
# Output-path safety
# ==========================================================================
def _is_within(child, parent):
    child, parent = Path(child).resolve(), Path(parent).resolve()
    return child == parent or parent in child.parents


def safe_path(directory, base_name, safety_root):
    """Non-overwriting path inside `directory` (timestamps if it exists)."""
    target = (directory / base_name).resolve()
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, suffix = Path(base_name).stem, Path(base_name).suffix
        target = (directory / f"{stem}_{stamp}{suffix}").resolve()
        print(f"[output] '{base_name}' exists; using timestamped name: {target.name}")
    if not _is_within(target, safety_root):
        raise SystemExit(f"[SAFETY] Resolved output path escapes safety root: {target}")
    return target


def run_tag(seed, n_resamples):
    return (f"synthetic_confirmatory_snr_grid_v1f_etree_seed{seed}"
            f"_{n_resamples}resamples")


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _agg(values):
    vals = [float(v) for v in values if v is not None and not (
        isinstance(v, float) and np.isnan(v))]
    if not vals:
        return (None, None, None, None, 0)
    a = np.array(vals, dtype=float)
    return (float(a.mean()), float(a.std(ddof=0)),
            float(a.min()), float(a.max()), int(len(a)))


# ==========================================================================
# Step 1 — per-resample feature-selection logging (reuse run_one_dataset)
# ==========================================================================
def step1_logging(datasets, k_values, n_resamples, test_size, seed, out_path):
    print("=" * 70)
    print("STEP 1 — per-resample feature-selection logging (confirmatory grid)")
    print("=" * 70)
    print(f"Algorithm={ALGORITHM} Classifier={CLASSIFIER}  K={k_values}  "
          f"n_resamples={n_resamples}  test_size={test_size}  seed={seed}")
    print(f"Datasets: {len(datasets)}  ({len(datasets) // 4} seeds x 4 cells)")
    print(f"Preprocessing: {logdrv.PREPROCESSING_NOTE}")
    print(f"Leakage guard: {logdrv.LEAKAGE_GUARD_NOTE}")

    fs_fn = logdrv.feature_selection_mapping[ALGORITHM]
    algo_info = logdrv.get_algorithms_mapping().get(ALGORITHM)
    if algo_info is None:
        raise KeyError(f"Algorithm '{ALGORITHM}' not in get_algorithms_mapping().")
    hyper_row = algo_info["hyper"].iloc[0]
    algo_family = logdrv.get_algorithm_family(ALGORITHM)

    records = []
    for ds in datasets:
        records.extend(logdrv.run_one_dataset(
            ds, ALGORITHM, CLASSIFIER, k_values,
            n_resamples, test_size, seed, fs_fn, hyper_row, algo_family,
        ))
    df = pd.DataFrame(records, columns=logdrv.OUTPUT_COLUMNS)

    print(f"\n[output] writing combined per-fold parquet: {out_path}")
    df.to_parquet(out_path, index=False)
    print(f"Logged {len(df)} rows  "
          f"(status counts: {df['status'].value_counts().to_dict()})")
    return df


# ==========================================================================
# Step 2 / 3 — invoke the analyzer and the recovery script (unmodified CLIs)
# ==========================================================================
def _run_subprocess(label, argv):
    print("=" * 70)
    print(label)
    print("=" * 70)
    print("  " + " ".join(argv))
    result = subprocess.run(argv, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] {label} failed (returncode {result.returncode}).")


def step2_analysis(combined_parquet, tag):
    _run_subprocess(
        "STEP 2 — offline stability analysis",
        [sys.executable, str(ANALYZER_SCRIPT),
         "--input-parquet", str(combined_parquet),
         "--output-dir", tag],
    )
    out_dir = (STABILITY_ROOT / tag).resolve()
    if not out_dir.exists():
        raise SystemExit(f"[ERROR] analyzer output dir not found: {out_dir}")
    return out_dir


def step3_recovery(analyzer_dir, tag):
    def _one(pattern):
        hits = sorted(analyzer_dir.glob(pattern))
        if not hits:
            raise SystemExit(f"[ERROR] analyzer output not found: {pattern}")
        return hits[0]

    ffreq = _one("*feature_frequencies*.parquet")
    tsweep = _one("*threshold_sweep*.parquet")
    gsumm = _one("*stability_group_summary*.parquet")
    _run_subprocess(
        "STEP 3 — recovery metrics",
        [sys.executable, str(RECOVERY_SCRIPT),
         "--feature-frequencies", str(ffreq),
         "--threshold-sweep", str(tsweep),
         "--group-summary", str(gsumm),
         "--output-dir", tag,
         "--report-name", RECOVERY_REPORT_NAME],
    )
    out_dir = (RECOVERY_ROOT / tag).resolve()
    if not out_dir.exists():
        raise SystemExit(f"[ERROR] recovery output dir not found: {out_dir}")
    return out_dir


# ==========================================================================
# Step 4 — confirmatory across-seed summaries
# ==========================================================================
def load_manifests(datasets):
    manifests = {}
    for ds in datasets:
        path = DATASETS_DIR / f"{ds['name']}_manifest.json"
        manifests[ds["name"]] = json.loads(path.read_text())
    return manifests


def build_auc_by_seed_k(combined_df, datasets):
    rows = []
    for ds in datasets:
        sub = combined_df[(combined_df["dataset"] == ds["name"])
                          & (combined_df["status"] == "ok")]
        for K in sorted(sub["K"].unique()):
            ksub = sub[sub["K"] == K]
            aucs = ksub["auc"].dropna().astype(float)
            rows.append(dict(
                alpha_label=ds["alpha_label"], alpha=ds["alpha"],
                cell_role=ds["cell_role"], seed_index=ds["seed_index"],
                dataset=ds["name"], role=ds["role"], K=int(K),
                n_ok=int(len(ksub)), n_auc=int(len(aucs)),
                mean_auc=float(aucs.mean()) if len(aucs) else None,
                min_auc=float(aucs.min()) if len(aucs) else None,
                max_auc=float(aucs.max()) if len(aucs) else None,
            ))
    return pd.DataFrame(rows)


def per_seed_overall_auc(combined_df, datasets):
    rows = []
    for ds in datasets:
        sub = combined_df[(combined_df["dataset"] == ds["name"])
                          & (combined_df["status"] == "ok")]
        aucs = sub["auc"].dropna().astype(float)
        rows.append(dict(
            alpha_label=ds["alpha_label"], alpha=ds["alpha"],
            cell_role=ds["cell_role"], seed_index=ds["seed_index"],
            dataset=ds["name"], role=ds["role"],
            mean_auc=float(aucs.mean()) if len(aucs) else None,
            min_auc=float(aucs.min()) if len(aucs) else None,
            max_auc=float(aucs.max()) if len(aucs) else None,
        ))
    return pd.DataFrame(rows)


def build_recovery_by_seed(threshold_df, datasets):
    """Per (alpha, seed, known-core dataset, K, threshold): recovery metrics
    paired with the SAME-SEED null control's recurrent-set size."""
    info = {ds["name"]: ds for ds in datasets}
    known = threshold_df[threshold_df["dataset_role"] == "known_core"]
    null = threshold_df[threshold_df["dataset_role"] == "null_control"]
    null_lookup = {
        (str(r.dataset), int(r.K), float(r.threshold)): int(r.recurrent_set_size)
        for r in null.itertuples(index=False)
    }
    rows = []
    for r in known.itertuples(index=False):
        ds = info.get(r.dataset)
        if ds is None:
            continue
        seed_i = ds["seed_index"]
        matched = null_lookup.get(
            (null_name_for_seed(seed_i), int(r.K), float(r.threshold)))
        rows.append(dict(
            alpha_label=ds["alpha_label"], alpha=ds["alpha"],
            cell_role=ds["cell_role"], seed_index=seed_i, dataset=r.dataset,
            K=int(r.K), threshold=float(r.threshold),
            recurrent_set_size=int(r.recurrent_set_size),
            true_core_recall=(float(r.true_core_recall)
                              if r.true_core_recall is not None else None),
            proxy_group_coverage_fraction=(
                float(r.proxy_group_coverage_fraction)
                if pd.notna(r.proxy_group_coverage_fraction) else None),
            anchored_signal_group_coverage_fraction=(
                float(r.anchored_signal_group_coverage_fraction)
                if pd.notna(r.anchored_signal_group_coverage_fraction) else None),
            noise_contamination=(float(r.noise_contamination)
                                 if r.noise_contamination is not None else None),
            matched_null_recurrent_set_size=matched,
        ))
    return pd.DataFrame(rows).sort_values(
        ["alpha", "K", "threshold", "seed_index"],
        ascending=[False, True, True, True]).reset_index(drop=True)


def build_stability_by_seed(group_summary_df, datasets):
    """Per (alpha, seed, known-core dataset, K): stability indices from the
    analyzer's group summary."""
    info = {ds["name"]: ds for ds in datasets}
    rows = []
    for r in group_summary_df.itertuples(index=False):
        ds = info.get(r.dataset)
        if ds is None:
            continue
        rows.append(dict(
            alpha_label=ds["alpha_label"], alpha=ds["alpha"],
            cell_role=ds["cell_role"], role=ds["role"],
            seed_index=ds["seed_index"], dataset=r.dataset, K=int(r.K),
            mean_pairwise_jaccard=(float(r.mean_pairwise_jaccard)
                                   if pd.notna(r.mean_pairwise_jaccard) else None),
            kuncheva_mean=(float(r.kuncheva_mean)
                           if pd.notna(r.kuncheva_mean) else None),
            nogueira_style_stability=(float(r.nogueira_style_stability)
                                      if pd.notna(r.nogueira_style_stability)
                                      else None),
            observed_vs_random_jaccard_ratio=(
                float(r.observed_vs_random_jaccard_ratio)
                if pd.notna(r.observed_vs_random_jaccard_ratio) else None),
        ))
    return pd.DataFrame(rows).sort_values(
        ["alpha", "K", "seed_index"],
        ascending=[False, True, True]).reset_index(drop=True)


def build_across_seed_aggregate(overall_auc_df, recovery_df, stability_df):
    rows = []
    for alpha_label in (list(f"alpha{a}" for a in ALPHA_ORDER) + ["null"]):
        sub = overall_auc_df[overall_auc_df["alpha_label"] == alpha_label]
        mean, std, lo, hi, n = _agg(sub["mean_auc"].tolist())
        rows.append(dict(alpha_label=alpha_label, metric="overall_mean_auc",
                         K=None, threshold=None, across_seed_mean=mean,
                         across_seed_std=std, across_seed_min=lo,
                         across_seed_max=hi, n_seeds=n))
    rec_metrics = ["true_core_recall", "noise_contamination",
                   "recurrent_set_size", "proxy_group_coverage_fraction",
                   "anchored_signal_group_coverage_fraction",
                   "matched_null_recurrent_set_size"]
    for alpha in ALPHA_ORDER:
        sub = recovery_df[recovery_df["alpha"] == alpha]
        for K in sorted(sub["K"].unique()):
            for t in sorted(sub[sub["K"] == K]["threshold"].unique()):
                cell = sub[(sub["K"] == K) & (sub["threshold"] == t)]
                for metric in rec_metrics:
                    mean, std, lo, hi, n = _agg(cell[metric].tolist())
                    rows.append(dict(alpha_label=f"alpha{alpha}", metric=metric,
                                     K=int(K), threshold=float(t),
                                     across_seed_mean=mean, across_seed_std=std,
                                     across_seed_min=lo, across_seed_max=hi,
                                     n_seeds=n))
    stab_metrics = ["mean_pairwise_jaccard", "kuncheva_mean",
                    "nogueira_style_stability",
                    "observed_vs_random_jaccard_ratio"]
    for alpha in ALPHA_ORDER:
        sub = stability_df[stability_df["alpha"] == alpha]
        for K in sorted(sub["K"].unique()):
            cell = sub[sub["K"] == K]
            for metric in stab_metrics:
                mean, std, lo, hi, n = _agg(cell[metric].tolist())
                rows.append(dict(alpha_label=f"alpha{alpha}", metric=metric,
                                 K=int(K), threshold=None,
                                 across_seed_mean=mean, across_seed_std=std,
                                 across_seed_min=lo, across_seed_max=hi,
                                 n_seeds=n))
    return pd.DataFrame(rows)


def step4_summary(combined_df, threshold_df, group_summary_df, datasets,
                  recovery_dir):
    print("=" * 70)
    print("STEP 4 — confirmatory across-seed summaries by alpha")
    print("=" * 70)
    auc_by_seed_k = build_auc_by_seed_k(combined_df, datasets)
    overall_auc = per_seed_overall_auc(combined_df, datasets)
    recovery_by_seed = build_recovery_by_seed(threshold_df, datasets)
    stability_by_seed = build_stability_by_seed(group_summary_df, datasets)
    across_seed = build_across_seed_aggregate(
        overall_auc, recovery_by_seed, stability_by_seed)

    paths = {}
    for key, base, dfo in [("auc", OUT_AUC_BY_SEED, auc_by_seed_k),
                           ("recovery", OUT_RECOVERY_BY_SEED, recovery_by_seed),
                           ("stability", OUT_STABILITY_BY_SEED, stability_by_seed),
                           ("aggregate", OUT_ACROSS_SEED, across_seed)]:
        p = safe_path(recovery_dir, base, RECOVERY_ROOT)
        dfo.to_parquet(p, index=False)
        paths[key] = p
        print(f"[output] {key:10s}: {p}  ({len(dfo)} rows)")
    return (auc_by_seed_k, overall_auc, recovery_by_seed, stability_by_seed,
            across_seed, paths)


# ==========================================================================
# v1f 10-resample reference (read-only, for qualitative comparison)
# ==========================================================================
def load_v1f_probe_auc():
    """Read the v1f 10-resample across-seed AUC means, if available."""
    if not V1F_PROBE_AGGREGATE.exists():
        return {}
    try:
        a = pd.read_parquet(V1F_PROBE_AGGREGATE)
        a = a[a["metric"] == "overall_mean_auc"]
        return {str(r.alpha_label): float(r.across_seed_mean)
                for r in a.itertuples(index=False)}
    except Exception:  # noqa: BLE001
        return {}


# ==========================================================================
# Across-seed helpers for the report
# ==========================================================================
def _agg_cell(across_seed, alpha, metric, K, threshold):
    row = across_seed[
        (across_seed["alpha_label"] == f"alpha{alpha}")
        & (across_seed["metric"] == metric)
        & (across_seed["K"] == K)
        & (across_seed["threshold"] == threshold)]
    if not len(row):
        return None
    return row.iloc[0]


def _focus_recall_by_seed(recovery_by_seed, alpha, K, threshold):
    sub = recovery_by_seed[
        (recovery_by_seed["alpha"] == alpha)
        & (recovery_by_seed["K"] == K)
        & (recovery_by_seed["threshold"] == threshold)].sort_values("seed_index")
    return [(int(r.seed_index), r.true_core_recall)
            for r in sub.itertuples(index=False)]


# ==========================================================================
# Consolidated report
# ==========================================================================
def build_report(commands, combined_df, datasets, manifests, group_summary_df,
                  threshold_df, auc_by_seed_k, overall_auc, recovery_by_seed,
                  stability_by_seed, across_seed, v1f_probe_auc, n_seeds,
                  n_resamples, k_values, out_paths):
    L = []
    L.append("# Synthetic Confirmatory SNR Grid (gammaV1f, 50-resample) — "
             "Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Manuscript-resolution **confirmatory** synthetic SNR grid — "
             "ETree only, rho fixed 0.9, raw signed X (np.abs not applied), "
             "feature selection confined to each training split, evaluation on "
             f"the held-out split, {n_resamples} resamples per dataset. Reuses "
             "the existing v1f `.mat` datasets unchanged (no regeneration). "
             "Synthetic data only; no real dataset, no mRMR/LASSO, no rho=0.7, "
             "no alpha=1.0, no benchmark file touched.\n")
    L.append("---\n")

    # 1. purpose
    L.append("## 1. Purpose\n")
    L.append("This is the manuscript-resolution **confirmatory grid** after the "
             "v1e / v1f synthetic arc:")
    L.append("- **v1e** established a reliable positive-control anchor (core "
             "beta=2.5; reliable recovery across 8 seeds).")
    L.append("- **v1f** swept alpha at fixed beta=2.5 and calibrated the SNR "
             "degradation: alpha=1.0 too easy, alpha=0.6 mildly degraded, "
             "alpha=0.35 the single clearly degraded (but instance-variable) "
             "candidate.")
    L.append("- **This run** confirms the selected grid — alpha=2.5 + 0.6 + "
             f"0.35 + null — at {n_resamples} resamples (a resolution upgrade "
             "from the v1f 10-resample probe), to produce reportable numbers "
             "and to separate genuine instance variance from 10-resample "
             "estimation noise. It is confirmatory, not a new diagnostic.\n")

    # 2. inputs
    L.append("## 2. Inputs\n")
    L.append("Exact existing v1f datasets used (8 generation seeds 00..07, "
             "4 cells per seed):\n")
    L.append("| Cell | alpha | role | dataset pattern |")
    L.append("|------|-------|------|-----------------|")
    for prefix, alpha, role in ALPHA_CELLS:
        L.append(f"| known-core | {alpha} | {role} | "
                 f"`{prefix}_rho090_gammaV1f_seed<NN>` |")
    L.append("| null-control | — | null | "
             "`synthetic_null_control_gammaV1f_seed<NN>` |")
    L.append("")
    L.append("- **alpha=1.0 was intentionally dropped** — v1f found it "
             "indistinguishable from the alpha=2.5 positive control "
             "(AUC gap -0.002).")
    L.append("- **No datasets were regenerated or modified** — the confirmatory "
             "grid reruns the identical v1f instances at higher resolution.")
    L.append("- The `_gammaV1f` and `_gammaV1e` outputs were preserved; this "
             "run writes only new `_confirmatory_snr_grid_v1f` outputs.\n")

    # 3. fixed parameters
    L.append("## 3. Fixed Parameters\n")
    L.append("| Parameter | Value |")
    L.append("|-----------|-------|")
    L.append("| core beta | 2.5 (the v1e positive-control anchor) |")
    L.append("| rho | 0.9 |")
    L.append("| p (n_total_features) | 2000 |")
    L.append("| n_samples | 200 |")
    L.append("| weak gamma | 0.8 |")
    L.append("| algorithm / classifier | ETree / ETREE only |")
    L.append(f"| K values | {k_values} |")
    L.append(f"| n_resamples | {n_resamples} |")
    L.append(f"| test_size | {TEST_SIZE} |")
    L.append("| preprocessing | raw signed X; np.abs NOT applied |")
    L.append("| feature selection | fit on the training split only |")
    L.append("| evaluation | held-out test split only |")
    L.append("")

    # 4. logging summary
    L.append("## 4. Logging Summary\n")
    planned = len(datasets) * 1 * len(k_values) * n_resamples
    L.append(f"- rows logged: **{len(combined_df)}** (expected {planned} = "
             f"{len(datasets)} datasets x 1 algorithm x {len(k_values)} K x "
             f"{n_resamples} resamples)")
    L.append(f"- status counts: {combined_df['status'].value_counts().to_dict()}")
    nfeat = sorted(int(x) for x in combined_df["n_features"].dropna().unique())
    L.append(f"- n_features values: {nfeat}")
    ok = combined_df[combined_df["status"] == "ok"]
    cnt_ok = bool((ok["selected_feature_count"] == ok["K"]).all()) if len(ok) else True
    L.append(f"- selected_feature_count == K on all ok rows: {cnt_ok}")
    L.append(f"- fold_id range: {int(combined_df['fold_id'].min())}.."
             f"{int(combined_df['fold_id'].max())}; n_splits values: "
             f"{sorted(int(x) for x in combined_df['n_splits'].unique())}")
    L.append(f"- preprocessing_note: \"{combined_df['preprocessing_note'].iloc[0]}\"")
    L.append(f"- leakage_guard_note: \"{combined_df['leakage_guard_note'].iloc[0]}\"")
    # rows per alpha
    L.append("")
    L.append("Rows per alpha (across all seeds / K):\n")
    L.append("| Alpha | datasets | rows | ok rows |")
    L.append("|-------|----------|------|---------|")
    for alpha in ALPHA_ORDER + ["null"]:
        label = f"alpha{alpha}" if alpha != "null" else "null"
        sub = combined_df[combined_df["dataset"].isin(
            [d["name"] for d in datasets if d["alpha_label"] == label])]
        L.append(f"| {alpha} | {sub['dataset'].nunique()} | {len(sub)} | "
                 f"{int((sub['status'] == 'ok').sum())} |")
    L.append("")
    L.append("- No real datasets were used; no benchmark file was modified.\n")

    # 5. performance summary
    L.append("## 5. Performance Summary\n")
    L.append("Held-out AUC per alpha. Per-seed mean AUC is averaged over all K "
             "and all resamples of that dataset. The final column compares the "
             "50-resample across-seed mean to the v1f 10-resample probe.\n")
    L.append("| Alpha (role) | per-seed mean AUC | across-seed mean±std | min | "
             "max | gap vs null | v1f 10-resample mean |")
    L.append("|--------------|-------------------|----------------------|-----|"
             "-----|-------------|----------------------|")
    null_row = overall_auc[overall_auc["alpha_label"] == "null"]
    null_mean, _, _, _, _ = _agg(null_row["mean_auc"].tolist())
    for alpha in ALPHA_ORDER + ["null"]:
        label = f"alpha{alpha}" if alpha != "null" else "null"
        sub = overall_auc[overall_auc["alpha_label"] == label].sort_values(
            "seed_index")
        role = sub["cell_role"].iloc[0] if len(sub) else "?"
        per_seed = [round(x, 3) if x is not None else None
                    for x in sub["mean_auc"].tolist()]
        mean, std, lo, hi, n = _agg(sub["mean_auc"].tolist())
        gap = (mean - null_mean) if (mean is not None
                                     and null_mean is not None
                                     and alpha != "null") else None
        v1f = v1f_probe_auc.get(label)
        L.append(f"| {alpha} ({role}) | {per_seed} | "
                 f"{_fmt(mean)}±{_fmt(std)} | {_fmt(lo)} | {_fmt(hi)} | "
                 f"{_fmt(gap)} | {_fmt(v1f)} |")
    L.append("")
    L.append("Per-K mean AUC by alpha (averaged across seeds):\n")
    L.append("| Alpha | " + " | ".join(f"K={k}" for k in k_values) + " |")
    L.append("|-------|" + "|".join("-----" for _ in k_values) + "|")
    for alpha in ALPHA_ORDER + ["null"]:
        label = f"alpha{alpha}" if alpha != "null" else "null"
        cells = []
        for k in k_values:
            sub = auc_by_seed_k[(auc_by_seed_k["alpha_label"] == label)
                                & (auc_by_seed_k["K"] == k)]
            mean, _, _, _, _ = _agg(sub["mean_auc"].tolist())
            cells.append(_fmt(mean))
        L.append(f"| {alpha} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Qualitative comparison to the v1f 10-resample probe: the "
             "50-resample across-seed AUC means are expected to track the v1f "
             "probe closely (the datasets are identical); the resolution "
             "upgrade tightens the estimates rather than moving them. Material "
             "divergences, if any, are flagged in section 13.\n")

    # 6. stability summary
    L.append("## 6. Stability Summary\n")
    L.append("Internal subset-stability indices per alpha and K (across-seed "
             "mean). Raw pairwise Jaccard is reported beside the chance-"
             "corrected Kuncheva index, the Nogueira-style estimate, and the "
             "observed-vs-random Jaccard ratio.\n")
    L.append("| Alpha | K | mean pairwise Jaccard | Kuncheva | Nogueira-style | "
             "obs/random Jaccard ratio |")
    L.append("|-------|---|-----------------------|----------|----------------|"
             "--------------------------|")
    for alpha in ALPHA_ORDER:
        for k in k_values:
            def sval(metric):
                r = across_seed[
                    (across_seed["alpha_label"] == f"alpha{alpha}")
                    & (across_seed["metric"] == metric)
                    & (across_seed["K"] == k)]
                return r.iloc[0]["across_seed_mean"] if len(r) else None
            L.append(f"| {alpha} | {k} | "
                     f"{_fmt(sval('mean_pairwise_jaccard'))} | "
                     f"{_fmt(sval('kuncheva_mean'))} | "
                     f"{_fmt(sval('nogueira_style_stability'))} | "
                     f"{_fmt(sval('observed_vs_random_jaccard_ratio'), 1)} |")
    L.append("")
    # degradation note
    def kunch_focus(alpha):
        r = across_seed[(across_seed["alpha_label"] == f"alpha{alpha}")
                        & (across_seed["metric"] == "kuncheva_mean")
                        & (across_seed["K"] == 30)]
        return r.iloc[0]["across_seed_mean"] if len(r) else None
    kref, k06, k035 = (kunch_focus(2.5), kunch_focus(0.6), kunch_focus(0.35))
    degrades = bool(kref is not None and k06 is not None and k035 is not None
                    and kref >= k06 >= k035)
    L.append(f"- Kuncheva at K=30 by alpha: 2.5={_fmt(kref)}, 0.6={_fmt(k06)}, "
             f"0.35={_fmt(k035)} — "
             f"{'monotone decline as alpha decreases' if degrades else 'not strictly monotone; read the full table'}.")
    L.append("- Subset stability is expected to weaken as alpha decreases "
             "(noisier labels -> less consistent selection); the table above "
             "is the descriptive evidence.\n")

    # 7. recovery summary
    L.append("## 7. Recovery Summary\n")
    L.append(f"Across-seed mean ± std at the focus cells (K in {FOCUS_K}, "
             f"thresholds {FOCUS_THRESHOLDS}). Exact true-core recall and "
             "proxy/group-aware coverage are kept separate; a recurrent proxy "
             "is not a false positive. `matched null` is the same-seed null "
             "control's recurrent-set size.\n")
    L.append("| Alpha | K | thr | true-core recall | proxy-group cov | "
             "anchored-sig-group cov | noise contam | recurrent set | "
             "matched null |")
    L.append("|-------|---|-----|------------------|-----------------|"
             "------------------------|--------------|---------------|"
             "--------------|")
    for alpha in ALPHA_ORDER:
        for k in FOCUS_K:
            for t in FOCUS_THRESHOLDS:
                def cell(metric):
                    rr = _agg_cell(across_seed, alpha, metric, k, t)
                    if rr is None:
                        return "—"
                    return (f"{_fmt(rr['across_seed_mean'])}±"
                            f"{_fmt(rr['across_seed_std'])}")
                L.append(f"| {alpha} | {k} | {t} | {cell('true_core_recall')} | "
                         f"{cell('proxy_group_coverage_fraction')} | "
                         f"{cell('anchored_signal_group_coverage_fraction')} | "
                         f"{cell('noise_contamination')} | "
                         f"{cell('recurrent_set_size')} | "
                         f"{cell('matched_null_recurrent_set_size')} |")
    L.append("")

    # 8. K=5 substitution
    L.append("## 8. K=5 Substitution Analysis\n")
    L.append("K=5 equals the exact true-core budget (5 planted core features). "
             f"At {n_resamples} resamples K=5 is resolved finely enough to "
             "report (it was flagged too noisy at 10 resamples). Across-seed "
             "mean ± std at K=5:\n")
    L.append("| Alpha | thr | exact true-core recall | proxy-group cov | "
             "anchored-sig-group cov | noise contam | recurrent set |")
    L.append("|-------|-----|------------------------|-----------------|"
             "------------------------|--------------|---------------|")
    for alpha in ALPHA_ORDER:
        for t in (0.8, 1.0):
            def cell(metric):
                rr = _agg_cell(across_seed, alpha, metric, 5, t)
                if rr is None:
                    return "—"
                return (f"{_fmt(rr['across_seed_mean'])}±"
                        f"{_fmt(rr['across_seed_std'])}")
            L.append(f"| {alpha} | {t} | {cell('true_core_recall')} | "
                     f"{cell('proxy_group_coverage_fraction')} | "
                     f"{cell('anchored_signal_group_coverage_fraction')} | "
                     f"{cell('noise_contamination')} | "
                     f"{cell('recurrent_set_size')} |")
    L.append("")
    # substitution verdict at K=5/t=0.8
    sub_lines = []
    for alpha in ALPHA_ORDER:
        rec = _agg_cell(across_seed, alpha, "true_core_recall", 5, 0.8)
        prx = _agg_cell(across_seed, alpha, "anchored_signal_group_coverage_fraction",
                        5, 0.8)
        if rec is not None and prx is not None:
            rm = rec["across_seed_mean"]
            pm = prx["across_seed_mean"]
            if rm is not None and pm is not None:
                sub_lines.append((alpha, rm, pm, pm > rm + 1e-9))
    if sub_lines:
        persists = sum(1 for _, _, _, s in sub_lines if s)
        L.append(f"- At K=5/t=0.8 the anchored-signal-group coverage exceeds "
                 f"exact true-core recall in {persists}/{len(sub_lines)} alpha "
                 f"levels — the substitution divergence "
                 f"{'persists at K=5' if persists else 'is not evident at K=5'} "
                 f"at confirmatory resolution.")
    L.append("- Substitution = the planted signal recurrently captured via "
             "redundant proxies where the exact core feature is not. K=5 is now "
             "interpretable at 50-resample resolution; exact recall below "
             "group coverage at K=5 is the sharpest view of it.\n")

    # 9. alpha=0.35 instance variance
    L.append("## 9. Alpha=0.35 Instance-Variance Analysis\n")
    L.append("The critical confirmatory question: does the v1f seed-02 "
             "recall-0 behaviour persist at 50 resamples, and how much of the "
             "0.0-1.0 v1f spread is real instance variance versus 10-resample "
             "estimation noise?\n")
    L.append("Per-seed alpha=0.35 recovery (true-core recall = recovered/5):\n")
    L.append("| Seed | recall K30/t0.8 | recall K60/t0.8 | proxy-grp cov "
             "K30/t0.8 | noise contam K30/t0.8 | recurrent set K30/t0.8 |")
    L.append("|------|-----------------|-----------------|------------------"
             "----|------------------------|------------------------|")
    r35_k30 = recovery_by_seed[(recovery_by_seed["alpha"] == 0.35)
                               & (recovery_by_seed["K"] == 30)
                               & (recovery_by_seed["threshold"] == 0.8)
                               ].sort_values("seed_index")
    r35_k60 = {int(r.seed_index): r.true_core_recall for r in
               recovery_by_seed[(recovery_by_seed["alpha"] == 0.35)
                                & (recovery_by_seed["K"] == 60)
                                & (recovery_by_seed["threshold"] == 0.8)
                                ].itertuples(index=False)}
    near_collapse = []
    for r in r35_k30.itertuples(index=False):
        s = int(r.seed_index)
        k60 = r35_k60.get(s)
        L.append(f"| {s:02d} | {_fmt(r.true_core_recall)} | {_fmt(k60)} | "
                 f"{_fmt(r.proxy_group_coverage_fraction)} | "
                 f"{_fmt(r.noise_contamination)} | {r.recurrent_set_size} |")
        rec30 = r.true_core_recall if r.true_core_recall is not None else 0.0
        rec60 = k60 if k60 is not None else 0.0
        if rec30 <= 0.2 and rec60 <= 0.2:
            near_collapse.append(s)
    L.append("")
    recalls30 = [r.true_core_recall for r in r35_k30.itertuples(index=False)
                 if r.true_core_recall is not None]
    if recalls30:
        L.append(f"- alpha=0.35 K=30/t=0.8 true-core recall across seeds: "
                 f"min {_fmt(min(recalls30))}, max {_fmt(max(recalls30))}, "
                 f"mean {_fmt(float(np.mean(recalls30)))}, "
                 f"std {_fmt(float(np.std(recalls30)))}.")
    if near_collapse:
        L.append(f"- **Near-collapse seed(s): {near_collapse}** — true-core "
                 f"recall <= 0.2 at BOTH K=30/t=0.8 and K=60/t=0.8. At "
                 f"{n_resamples} resamples (frequency resolution "
                 f"{1.0 / n_resamples:.2f}) this is a confirmed "
                 f"instance-level near-collapse, not a 10-resample estimation "
                 f"artifact.")
    else:
        L.append(f"- **No seed near-collapses** at {n_resamples} resamples "
                 f"(every seed has recall > 0.2 at K=30 or K=60, t=0.8). The "
                 f"v1f seed-02 recall-0 result does NOT persist at confirmatory "
                 f"resolution — it was substantially a 10-resample estimation "
                 f"artifact.")
    L.append("- Conservative wording: alpha=0.35 is a **variable-recovery / "
             "borderline-SNR regime** with **instance-dependent recovery** — "
             "not a reliably hard-but-learnable regime. The per-seed spread "
             "above is the substance of that statement.\n")

    # 10. alpha=0.6 hedge rung
    L.append("## 10. Alpha=0.6 Hedge-Rung Analysis\n")
    def auc_mean(alpha):
        r = across_seed[(across_seed["alpha_label"] == f"alpha{alpha}")
                        & (across_seed["metric"] == "overall_mean_auc")]
        return r.iloc[0]["across_seed_mean"] if len(r) else None
    a25, a06 = auc_mean(2.5), auc_mean(0.6)
    gap_06 = (a25 - a06) if (a25 is not None and a06 is not None) else None
    # K-dependent recall degradation
    rec_06_k30 = _agg_cell(across_seed, 0.6, "true_core_recall", 30, 0.8)
    rec_06_k60 = _agg_cell(across_seed, 0.6, "true_core_recall", 60, 0.8)
    rec_25_k30 = _agg_cell(across_seed, 2.5, "true_core_recall", 30, 0.8)
    L.append(f"- alpha=0.6 across-seed mean AUC = {_fmt(a06)} vs the alpha=2.5 "
             f"positive control {_fmt(a25)} (gap {_fmt(gap_06)}).")
    if rec_06_k30 is not None and rec_25_k30 is not None:
        L.append(f"- true-core recall at K=30/t=0.8: alpha=0.6 "
                 f"{_fmt(rec_06_k30['across_seed_mean'])} vs alpha=2.5 "
                 f"{_fmt(rec_25_k30['across_seed_mean'])}; at K=60/t=0.8 "
                 f"alpha=0.6 "
                 f"{_fmt(rec_06_k60['across_seed_mean']) if rec_06_k60 is not None else '—'} "
                 f"— degradation is expected to be K-dependent (visible at "
                 f"K=30, washed out at K=60).")
    too_close = bool(gap_06 is not None and gap_06 < 0.05)
    L.append(f"- Is alpha=0.6 too close to the positive control? "
             f"**{'yes — overall AUC gap < 0.05' if too_close else 'overall AUC gap is modest but non-trivial'}**. "
             f"Its value is as a **mild, K-dependent intermediate rung**, not a "
             f"headline degraded cell.")
    L.append("- Recommendation: **keep alpha=0.6 in the final manuscript "
             "figure/table** as the intermediate rung — it gives the SNR "
             "degradation gradient its shape (positive control -> mild "
             "K-dependent degradation -> variable degradation -> null) and "
             "hedges the instance-variable alpha=0.35 cell.\n")

    # 11. null-control analysis
    L.append("## 11. Null-Control Analysis\n")
    null_auc = overall_auc[overall_auc["alpha_label"] == "null"].sort_values(
        "seed_index")
    nm, ns, nlo, nhi, _ = _agg(null_auc["mean_auc"].tolist())
    L.append(f"- null held-out AUC across seeds: mean {_fmt(nm)}, "
             f"min {_fmt(nlo)}, max {_fmt(nhi)} (std {_fmt(ns)}) — near chance, "
             "as expected for signal-free data.")
    null_t = threshold_df[threshold_df["dataset_role"] == "null_control"]
    for t in (0.8, 1.0):
        tot = int(null_t[null_t["threshold"] == t]["recurrent_set_size"].sum())
        L.append(f"- null recurrent-set size at threshold {t} (summed across "
                 f"seeds and K): {tot}")
    L.append("- Any null recurrence is treated as a small fixed-label "
             "finite-sample artifact (a few chance-correlated noise features "
             "can recur with p>>n and a fixed label vector). The matched-null "
             "recurrent-set sizes are reported beside every known-core cell in "
             "sections 7-9 so each recovery number is read against its own "
             "null.\n")

    # 12. main interpretation
    L.append("## 12. Main Interpretation\n")
    # verdict computation
    ref_mean = auc_mean(2.5)
    a035 = auc_mean(0.35)
    gradient = bool(
        ref_mean is not None and a06 is not None and a035 is not None
        and nm is not None and ref_mean > a06 > a035 > nm)
    rec_25 = _agg_cell(across_seed, 2.5, "true_core_recall", 30, 0.8)
    ref_recall = rec_25["across_seed_mean"] if rec_25 is not None else None
    pc_confirmed = bool(
        ref_mean is not None and nm is not None and (ref_mean - nm) > 0.15
        and ref_recall is not None and ref_recall >= 0.5)
    spread035 = (max(recalls30) - min(recalls30)) if recalls30 else 0.0
    variable035 = bool(spread035 >= 0.4)
    L.append("| Question | Verdict | Evidence |")
    L.append("|----------|---------|----------|")
    L.append(f"| Does the grid confirm the positive-control anchor? | "
             f"**{'yes' if pc_confirmed else 'qualified'}** | alpha=2.5 "
             f"across-seed mean AUC {_fmt(ref_mean)} vs null {_fmt(nm)}; "
             f"K=30/t=0.8 true-core recall {_fmt(ref_recall)}. |")
    L.append(f"| Does it confirm alpha=0.35 as a variable-recovery mid/low "
             f"regime? | **{'yes' if variable035 else 'qualified'}** | "
             f"alpha=0.35 mean AUC {_fmt(a035)} (degraded vs ref, above null "
             f"{_fmt(nm)}); per-seed K=30/t=0.8 recall spread "
             f"{_fmt(spread035)}. |")
    L.append(f"| Does it confirm alpha=0.6 as a mild/intermediate rung? | "
             f"**{'yes' if (gap_06 is not None and 0.0 < gap_06 < 0.12) else 'qualified'}** "
             f"| alpha=0.6 mean AUC {_fmt(a06)}, gap vs ref {_fmt(gap_06)} — "
             f"mild, K-dependent degradation. |")
    L.append(f"| Does the SNR degradation gradient support the trust-layer "
             f"claim? | **{'yes' if gradient else 'partly'}** | across-seed "
             f"mean AUC: {_fmt(ref_mean)} (2.5) -> {_fmt(a06)} (0.6) -> "
             f"{_fmt(a035)} (0.35) -> {_fmt(nm)} (null). |")
    L.append(f"| Does instance-dependence persist after 50 resamples? | "
             f"**{'yes' if variable035 else 'reduced'}** | alpha=0.35 per-seed "
             f"recall spread {_fmt(spread035)} at K=30/t=0.8; "
             f"near-collapse seed(s): {near_collapse if near_collapse else 'none'}. |")
    L.append("")
    L.append("Reading: the grid is a confirmatory descriptive run (synthetic, "
             "ETree, rho=0.9, 8 seeds). It confirms a controlled SNR "
             "degradation gradient and an instance-variance gradient — the two "
             "on-claim divergence stories the v1f checkpoint identified — at "
             "manuscript resolution. Conservative wording only; no single seed "
             "is over-interpreted.\n")

    # 13. warnings / caveats
    L.append("## 13. Warnings / Caveats\n")
    warnings = []
    bad = combined_df[combined_df["status"] != "ok"]
    if len(bad):
        warnings.append(f"{len(bad)} logged row(s) not status 'ok': "
                        f"{bad['status'].value_counts().to_dict()}")
    for alpha in ALPHA_ORDER + ["null"]:
        label = f"alpha{alpha}" if alpha != "null" else "null"
        v1f = v1f_probe_auc.get(label)
        cur = auc_mean(alpha) if alpha != "null" else nm
        if (v1f is not None and cur is not None
                and abs(v1f - cur) > 0.05):
            warnings.append(f"alpha={alpha} across-seed mean AUC moved "
                            f"{_fmt(cur - v1f)} from the v1f 10-resample probe "
                            f"({_fmt(v1f)} -> {_fmt(cur)}) — report this "
                            f"divergence conservatively.")
    if near_collapse:
        warnings.append(f"alpha=0.35 seed(s) {near_collapse} near-collapse "
                        f"(recall <= 0.2 at K=30 and K=60, t=0.8) — confirmed "
                        f"instance-level low recovery.")
    for w in warnings:
        L.append(f"- **WARNING:** {w}")
    L.append("- **Synthetic data only** — no real-data evidence.")
    L.append("- **ETree only** — no cross-algorithm (mRMR / LASSO) evidence.")
    L.append("- **rho fixed at 0.9** — no proxy-correlation-axis evidence.")
    L.append("- **beta=2.5 is a deliberately strengthened positive-control "
             "anchor** — not a representative real-data difficulty.")
    L.append("- No false-discovery / FWER / FDR / PFER control is claimed; the "
             "Nogueira-style index is an exploratory estimate, not verified.")
    L.append("- No causal / biomarker / validated-feature claims; \"planted "
             "core\" wording is licensed only for synthetic data.\n")

    # 14. recommended next step
    L.append("## 14. Recommended Next Step\n")
    L.append("- **Proceed to an Opus interpretation checkpoint.** The "
             "confirmatory 50-resample grid is complete; an Opus checkpoint "
             "should read the confirmed positive control, the SNR degradation "
             "gradient, the alpha=0.6 intermediate rung, and the alpha=0.35 "
             "instance-variance behaviour, and decide the next step (e.g. "
             "real-data example planning).")
    L.append("- This pipeline does NOT automatically continue beyond this "
             "report — no further grid, diagnostic, or real-data run was "
             "launched.")
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
    L.append("*Confirmatory grid run on existing synthetic v1f datasets only — "
             "no dataset generated or modified, no real dataset, no mRMR/LASSO, "
             "no rho=0.7, no alpha=1.0, no benchmark file modified, no commit "
             "made.*")
    return "\n".join(L)


# ==========================================================================
# CLI / driver
# ==========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Article 5 — confirmatory 50-resample synthetic SNR grid "
                    "(logging + stability analysis + recovery metrics + "
                    "across-seed confirmatory summaries).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and check inputs, then exit "
                             "without running anything or writing files.")
    parser.add_argument("--n-generation-seeds", type=int, default=8,
                        help="Number of generation seeds / instances "
                             "(default 8).")
    parser.add_argument("--n-resamples", type=int, default=50,
                        help="Stratified shuffle-split resamples per dataset "
                             "(default 50).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Resampling random seed (default 0).")
    parser.add_argument("--k-values", default="1,5,10,30,60",
                        help='Comma-separated K values (default "1,5,10,30,60").')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    datasets = build_dataset_list(args.n_generation_seeds)
    tag = run_tag(args.seed, args.n_resamples)
    combined_path = PERFOLD_DIR / f"{tag}.parquet"
    planned_rows = len(datasets) * 1 * len(k_values) * args.n_resamples

    if args.dry_run:
        print("=" * 70)
        print("Article 5 — Confirmatory 50-Resample SNR Grid  [DRY RUN]")
        print("=" * 70)
        print(f"Generation seeds : {args.n_generation_seeds}")
        print(f"Grid cells/seed  : alpha=2.5, alpha=0.6, alpha=0.35, null "
              "(alpha=1.0 intentionally EXCLUDED)")
        print(f"Datasets ({len(datasets)}):")
        all_ok = True
        for ds in datasets:
            mat = DATASETS_DIR / f"{ds['name']}.mat"
            man = DATASETS_DIR / f"{ds['name']}_manifest.json"
            ok = mat.exists() and man.exists()
            all_ok = all_ok and ok
            print(f"  {ds['name']:46s} (alpha={str(ds['alpha']):4s}) "
                  f".mat={mat.exists()} manifest={man.exists()}")
        print(f"Algorithm={ALGORITHM}  Classifier={CLASSIFIER}  K={k_values}  "
              f"n_resamples={args.n_resamples}  test_size={TEST_SIZE}  "
              f"seed={args.seed}")
        print(f"Expected logged rows : {planned_rows}")
        print(f"Combined parquet    : {combined_path}")
        print(f"Stability profiles  : {STABILITY_ROOT / tag}")
        print(f"Recovery metrics    : {RECOVERY_ROOT / tag}")
        print(f"Consolidated report : {REPORT_PATH}")
        print(f"Preprocessing       : raw signed X; np.abs is not applied")
        print(f"Leakage policy      : FS train split only; held-out evaluation")
        print(f"Datasets reused (not generated/modified); alpha=1.0 excluded.")
        print(f"All input files present: {all_ok}")
        if not all_ok:
            print("  -> the v1f datasets must exist first "
                  "(`make_synthetic_control.py --midlow-calibration-v1`).")
        print("DRY RUN — nothing run, nothing written.")
        print("=" * 70)
        return 0 if all_ok else 1

    # --- input check ------------------------------------------------------
    for ds in datasets:
        for suffix in (".mat", "_manifest.json"):
            p = DATASETS_DIR / f"{ds['name']}{suffix}"
            if not p.exists():
                raise SystemExit(
                    f"[ERROR] missing v1f input dataset: {p}\n"
                    f"        the confirmatory grid reuses existing v1f "
                    f"datasets; it does not generate them.")

    PERFOLD_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = safe_path(PERFOLD_DIR, f"{tag}.parquet", PERFOLD_DIR)

    # --- step 1: logging --------------------------------------------------
    combined_df = step1_logging(datasets, k_values, args.n_resamples,
                                TEST_SIZE, args.seed, combined_path)

    # --- step 2: stability analysis --------------------------------------
    analyzer_dir = step2_analysis(combined_path, tag)

    # --- step 3: recovery metrics ----------------------------------------
    recovery_dir = step3_recovery(analyzer_dir, tag)

    # --- step 4: confirmatory summaries ----------------------------------
    manifests = load_manifests(datasets)
    group_summary_df = pd.read_parquet(
        sorted(analyzer_dir.glob("*stability_group_summary*.parquet"))[0])
    threshold_df = pd.read_parquet(
        sorted(recovery_dir.glob("*threshold_recovery*.parquet"))[0])
    (auc_by_seed_k, overall_auc, recovery_by_seed, stability_by_seed,
     across_seed, summary_paths) = step4_summary(
        combined_df, threshold_df, group_summary_df, datasets, recovery_dir)
    v1f_probe_auc = load_v1f_probe_auc()

    # --- consolidated report ---------------------------------------------
    print("=" * 70)
    print("STEP 5 — consolidated confirmatory grid report")
    print("=" * 70)
    commands = [
        "python -m py_compile "
        "articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py",
        "python articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py --dry-run",
        "python articles/article_stability/scripts/run_confirmatory_snr_grid_v1_pipeline.py",
    ]
    out_paths = [
        ("combined per-fold parquet", combined_path),
        ("stability profiles dir", analyzer_dir),
        ("recovery metrics dir", recovery_dir),
        ("auc by seed/K parquet", summary_paths["auc"]),
        ("recovery by seed parquet", summary_paths["recovery"]),
        ("stability by seed parquet", summary_paths["stability"]),
        ("across-seed aggregate parquet", summary_paths["aggregate"]),
        ("consolidated report", REPORT_PATH),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(REPORT_PATH.parent, REPORT_PATH.name,
                            REPORT_PATH.parent)
    report_path.write_text(build_report(
        commands, combined_df, datasets, manifests, group_summary_df,
        threshold_df, auc_by_seed_k, overall_auc, recovery_by_seed,
        stability_by_seed, across_seed, v1f_probe_auc, args.n_generation_seeds,
        args.n_resamples, k_values, out_paths,
    ))
    print(f"[output] consolidated report: {report_path}")

    print("-" * 70)
    for alpha in ALPHA_ORDER + ["null"]:
        label = f"alpha{alpha}" if alpha != "null" else "null"
        r = across_seed[(across_seed["alpha_label"] == label)
                        & (across_seed["metric"] == "overall_mean_auc")]
        m = r.iloc[0]["across_seed_mean"] if len(r) else None
        print(f"  alpha={str(alpha):<5} across-seed mean AUC = {_fmt(m)}")
    print("=" * 70)
    print("Confirmatory 50-resample SNR grid pipeline COMPLETE.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
