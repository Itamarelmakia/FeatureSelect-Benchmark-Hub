"""
analyze_per_fold_stability.py — Article 5, offline stability analysis
=====================================================================

Offline, descriptive stability analysis of an Article 5 per-fold feature-selection
parquet (produced by `run_per_fold_stability_logging.py`). It does NOT run feature
selection, does NOT rerun Stage A, and does NOT touch the benchmark pipeline.

Working title: "Feature Stability as a Trust Layer for Feature Selection".

Design constraints (from results/article_stability/reports/literature_notes/FOLLOWUP_DEEP_RESEARCH_SYNTHESIS.md):
  * No single stable-core threshold — report a sweep over {0.6, 0.8, 1.0}.
  * All thresholds are EXPLORATORY / DESCRIPTIVE. No PFER / FWER / FDR claim.
  * `core_fraction` is descriptive only — never a decision rule.
  * No hard labels (stable / unstable / trusted / validated / true / causal).
  * Conservative wording only: "recurrently selected features", "recurrent candidate
    core", "variable component", "preliminary stability profile".
  * Report Nogueira-style and Kuncheva indices beside raw Jaccard.
  * Report chance / random-selection baselines next to observed values.

Isolation contract:
  * Does NOT modify utilities.py, src/benchmark_core/, or src/configs/config.py.
  * Reads the per-fold parquet (read-only) and, optionally, colon.mat (read-only).
  * Writes analysis parquets ONLY under results/article_stability/stability_profiles/.
  * Writes one markdown report under results/article_stability/reports/experiment_planning/.
  * Never overwrites an existing output (timestamps a new name instead).

Usage:
    python articles/article_stability/scripts/analyze_per_fold_stability.py            # analyze Stage A parquet
    python articles/article_stability/scripts/analyze_per_fold_stability.py --dry-run  # plan + path checks only
    python articles/article_stability/scripts/analyze_per_fold_stability.py --input-parquet PATH --output-dir SUBDIR
"""

import os
import sys
import math
import argparse
from datetime import datetime
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Path setup. This script lives at articles/article_stability/scripts/analyze_per_fold_stability.py
# so parents[3] == project root.
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]

# ==========================================================================
# Configuration
# ==========================================================================
DEFAULT_INPUT_PARQUET = (
    PROJECT_ROOT / "results" / "article_stability" / "per_fold_selections"
    / "per_fold_selections_stageA_colon_etree_seed0.parquet"
).resolve()

# Safety root — all analysis parquet artifacts MUST stay inside this directory.
ARTICLE5_STABILITY_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "stability_profiles"
).resolve()

# Markdown report — fixed documented path (separate from the parquet output root).
REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "STAGE_A_OFFLINE_STABILITY_ANALYSIS_REPORT.md"
).resolve()

# Exploratory selection-frequency thresholds (NOT final; pending literature synthesis).
THRESHOLDS = [0.6, 0.8, 1.0]

# Output parquet base names.
OUT_GROUP_SUMMARY = "stageA_colon_etree_stability_group_summary.parquet"
OUT_FEATURE_FREQ = "stageA_colon_etree_feature_frequencies.parquet"
OUT_THRESHOLD_SWEEP = "stageA_colon_etree_threshold_sweep.parquet"
OUT_CROSS_K = "stageA_colon_etree_cross_k_recurrence.parquet"

GROUP_KEYS = ["dataset", "repository", "algorithm", "classifier", "seed", "K"]
DATASET_KEYS = ["dataset", "repository", "algorithm", "classifier", "seed"]
METRIC_COLS = ["auc", "accuracy", "precision", "recall", "f1"]


# ==========================================================================
# Small numeric helpers
# ==========================================================================
def jaccard(a, b):
    """Jaccard index of two index sets. Empty-vs-empty defined as 1.0."""
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def kuncheva_pair(r, d, k):
    """Kuncheva (2007) consistency index for an equal-cardinality subset pair.

        index = (r * d - k**2) / (k * (d - k))

    r = intersection size, d = number of available features, k = subset size.
    Valid only for equal-cardinality subsets (Stage A is fixed-K, so this holds).
    Returns None if the denominator is zero.
    """
    denom = k * (d - k)
    if denom == 0:
        return None
    return (r * d - k ** 2) / denom


def binom_tail_ge(c, n, p):
    """P[Binomial(n, p) >= c], computed directly (no scipy dependency)."""
    if c <= 0:
        return 1.0
    if c > n:
        return 0.0
    total = 0.0
    for j in range(c, n + 1):
        total += math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j))
    return float(min(max(total, 0.0), 1.0))


def threshold_count_for(t, n_folds):
    """Number of folds a feature must be selected in to meet frequency cutoff t.

    threshold_count = ceil(t * n_folds), with a small epsilon to avoid float
    artifacts (e.g. 0.8 * 5 == 4.0000000000000004).
    """
    return int(math.ceil(t * n_folds - 1e-9))


def _r(x, nd=4):
    """Round, passing through None / NaN."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except TypeError:
        pass
    return round(float(x), nd)


# ==========================================================================
# Output-path safety
# ==========================================================================
def _is_within(child, parent):
    """True if `child` equals or is nested under `parent`."""
    child = Path(child).resolve()
    parent = Path(parent).resolve()
    return child == parent or parent in child.parents


def resolve_output_dir(output_dir_arg):
    """Resolve + validate the parquet output directory.

    Must resolve inside ARTICLE5_STABILITY_ROOT. A relative --output-dir is
    interpreted relative to that root. Raises SystemExit if it escapes.
    """
    if output_dir_arg is None:
        out_dir = ARTICLE5_STABILITY_ROOT
    else:
        candidate = Path(output_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = ARTICLE5_STABILITY_ROOT / candidate
        out_dir = candidate.resolve()

    if not _is_within(out_dir, ARTICLE5_STABILITY_ROOT):
        raise SystemExit(
            f"[SAFETY] Refusing to write outside the Article 5 stability root.\n"
            f"         requested : {out_dir}\n"
            f"         allowed   : {ARTICLE5_STABILITY_ROOT}"
        )
    return out_dir


def safe_path(directory, base_name, safety_root):
    """Return a non-overwriting path inside `directory`.

    If the default name exists, a timestamped name is used. The final path is
    re-checked against `safety_root`.
    """
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
# Core analysis
# ==========================================================================
def determine_d(dataset_df):
    """Determine the feature-universe size d for one dataset group.

    Preference order:
      1. If the parquet carries a valid, consistent `n_features` column, use it
         (d_source = "n_features_column", d_was_inferred = False).
      2. Otherwise fall back to (maximum observed selected feature index + 1)
         (d_source = "max_selected_feature_index_plus_one",
         d_was_inferred = True) and record a warning.

    `d` feeds the Kuncheva index, the Nogueira-style estimate, the
    recurrent-core chance baseline, and the random-Jaccard baseline, so a
    correct d matters.

    Returns (d, d_was_inferred, d_source).
    """
    # 1. Prefer an explicit, consistent n_features column.
    if "n_features" in dataset_df.columns:
        vals = dataset_df["n_features"].dropna().unique()
        if len(vals) == 1:
            try:
                d_explicit = int(vals[0])
            except (TypeError, ValueError):
                d_explicit = None
            if d_explicit is not None and d_explicit > 0:
                return d_explicit, False, "n_features_column"
        elif len(vals) > 1:
            print(f"  [warn] n_features column has inconsistent values "
                  f"{sorted(int(x) for x in vals)}; falling back to "
                  f"max-selected-index inference.")

    # 2. Fallback: infer from the maximum observed selected feature index.
    max_idx = -1
    for v in dataset_df["selected_feature_indices"]:
        if v is not None and len(v) > 0:
            max_idx = max(max_idx, int(np.max(np.asarray(v))))
    d = max_idx + 1 if max_idx >= 0 else 0
    print(f"  [warn] no usable `n_features` column; d inferred as "
          f"(max selected feature index + 1) = {d}.")
    return d, True, "max_selected_feature_index_plus_one"


def analyze_group(gdf, d, d_was_inferred, d_source):
    """Analyze one (dataset, repo, algorithm, classifier, seed, K) group.

    `d`, `d_was_inferred`, and `d_source` are produced by `determine_d`.
    Returns (summary_row, feature_freq_rows, threshold_sweep_rows).
    """
    g = gdf.iloc[0]
    K = int(g["K"])
    n_splits = int(g["n_splits"]) if pd.notna(g["n_splits"]) else len(gdf)

    ok = gdf[gdf["status"] == "ok"]
    n_ok = len(ok)
    n_failed = len(gdf) - n_ok

    # ---- A. Basic validation ------------------------------------------------
    subsets = [list(np.asarray(v)) for v in ok["selected_feature_indices"]]
    all_list_like = all(
        isinstance(v, (list, np.ndarray)) for v in gdf["selected_feature_indices"]
    )
    count_consistent = bool(
        (ok["selected_feature_count"].astype(int)
         == ok["selected_feature_indices"].apply(lambda v: len(np.asarray(v)))).all()
    ) if n_ok else True
    all_ok_size_K = bool(all(len(s) == K for s in subsets)) if n_ok else True
    metrics_valid = True
    for m in METRIC_COLS:
        vals = gdf[m].dropna()
        if len(vals) and not vals.between(0.0, 1.0).all():
            metrics_valid = False

    # ---- B. Performance summary --------------------------------------------
    perf = {}
    for m in METRIC_COLS:
        vals = ok[m].dropna().astype(float)
        perf[f"{m}_mean"] = _r(vals.mean()) if len(vals) else None
        perf[f"{m}_std"] = _r(vals.std(ddof=0)) if len(vals) else None

    # ---- C. Feature frequency profile --------------------------------------
    fold_count = Counter()
    for s in subsets:
        for f in set(s):
            fold_count[f] += 1
    freq_rows = []
    for feat in sorted(fold_count, key=lambda f: (-fold_count[f], f)):
        fc = fold_count[feat]
        freq_rows.append(dict(
            dataset=g["dataset"], repository=g["repository"],
            algorithm=g["algorithm"], classifier=g["classifier"],
            seed=int(g["seed"]), K=K,
            feature_index=int(feat), fold_count=int(fc),
            frequency=_r(fc / n_ok) if n_ok else None,
            selected_in_all_folds=bool(fc == n_ok and n_ok > 0),
        ))
    feature_pool_size = len(fold_count)

    # ---- D + H. Threshold sweep + chance baseline --------------------------
    p_random = (K / d) if d > 0 else None
    sweep_rows = []
    core_by_threshold = {}
    for t in THRESHOLDS:
        tcount = threshold_count_for(t, n_ok if n_ok else n_splits)
        core = sorted([f for f, c in fold_count.items() if c >= tcount])
        core_by_threshold[t] = core
        core_size = len(core)
        # H. expected core size under random independent selection.
        if d > 0 and p_random is not None and n_ok > 0:
            exp_rand = d * binom_tail_ge(tcount, n_ok, p_random)
        else:
            exp_rand = None
        ratio = (core_size / exp_rand) if (exp_rand and exp_rand > 0) else None
        sweep_rows.append(dict(
            dataset=g["dataset"], repository=g["repository"],
            algorithm=g["algorithm"], classifier=g["classifier"],
            seed=int(g["seed"]), K=K,
            threshold_frequency=t,
            threshold_count=int(tcount),
            n_folds=int(n_ok),
            recurrent_candidate_core_size=int(core_size),
            recurrent_candidate_core_features=[int(x) for x in core],
            # core_fraction is DESCRIPTIVE ONLY — never used as a decision rule.
            core_fraction=_r(core_size / K) if K else None,
            expected_random_core_size=_r(exp_rand),
            observed_core_size=int(core_size),
            observed_vs_expected_ratio=_r(ratio),
            p_random=_r(p_random, 6),
            d_used=int(d),
            d_was_inferred=bool(d_was_inferred),
        ))

    # ---- E. Pairwise overlap (raw Jaccard) ---------------------------------
    pj = [jaccard(a, b) for a, b in combinations(subsets, 2)]
    if pj:
        pj_mean, pj_std = float(np.mean(pj)), float(np.std(pj))
        pj_min, pj_max = float(np.min(pj)), float(np.max(pj))
    else:
        pj_mean = pj_std = pj_min = pj_max = None

    # ---- F. Kuncheva stability ---------------------------------------------
    kv = []
    for a, b in combinations(subsets, 2):
        r = len(set(a) & set(b))
        val = kuncheva_pair(r, d, K)
        if val is not None:
            kv.append(val)
    if kv:
        k_mean, k_std = float(np.mean(kv)), float(np.std(kv))
        k_min, k_max = float(np.min(kv)), float(np.max(kv))
    else:
        k_mean = k_std = k_min = k_max = None

    # ---- G. Nogueira-style threshold-free stability ------------------------
    # NOTE: this is a Nogueira-STYLE *exploratory* estimate, not the exact
    # JMLR 2018 estimator (which uses the unbiased sample variance with an
    # M/(M-1) correction). It MUST be verified against Nogueira et al. (2018)
    # before any final-manuscript use. Formula used here:
    #   stability = 1 - mean_i[ p_i*(1-p_i) ] / ( (k/d)*(1-k/d) )
    # with p_i the empirical per-feature selection probability over folds.
    nog_stability = None
    nog_denom = None
    if d > 0 and K > 0 and n_ok > 0:
        nog_denom = (K / d) * (1.0 - K / d)
        if nog_denom == 0:
            print(f"  [warn] Nogueira-style denominator is zero for K={K}; returning null.")
        else:
            # Sum p_i*(1-p_i) over ALL d features; never-selected features add 0.
            var_sum = sum((c / n_ok) * (1.0 - c / n_ok) for c in fold_count.values())
            nog_stability = 1.0 - (var_sum / d) / nog_denom

    # ---- I. Random-selection baseline for pairwise Jaccard -----------------
    if d > 0 and K > 0:
        exp_inter = (K ** 2) / d
        exp_union = 2 * K - exp_inter
        exp_rand_jacc = (exp_inter / exp_union) if exp_union > 0 else None
    else:
        exp_rand_jacc = None
    jacc_vs_random = (
        (pj_mean / exp_rand_jacc)
        if (pj_mean is not None and exp_rand_jacc and exp_rand_jacc > 0)
        else None
    )

    # ---- J. Consensus comparison (internal frequency-consensus diagnostic) --
    # Frequency-consensus set = top-K features by (fold_count desc, index asc).
    # This is an internal Stage A diagnostic — NOT Article 2's three
    # "Best Indices" variants.
    consensus = [r["feature_index"] for r in freq_rows[:K]]
    cj = [jaccard(s, consensus) for s in subsets] if subsets else []
    if cj:
        cj_mean, cj_std = float(np.mean(cj)), float(np.std(cj))
        cj_min, cj_max = float(np.min(cj)), float(np.max(cj))
    else:
        cj_mean = cj_std = cj_min = cj_max = None

    # ---- M. Conservative interpretation text -------------------------------
    core_sizes = {t: len(core_by_threshold[t]) for t in THRESHOLDS}
    interp = (
        f"Under the exploratory selection-frequency cutoff sweep, the recurrent "
        f"candidate core for K={K} contains "
        f"{core_sizes[0.6]} feature(s) at >=0.6, "
        f"{core_sizes[0.8]} at >=0.8, and {core_sizes[1.0]} at ==1.0; this remains "
        f"descriptive and threshold-dependent. "
        f"Overall subset overlap is low in raw Jaccard terms "
        f"(mean pairwise Jaccard {_r(pj_mean)}); chance-corrected metrics "
        f"(Kuncheva, Nogueira-style) and the random baseline should be considered "
        f"before describing the selection as recurrent or variable. "
        f"Interpretation remains preliminary: one dataset, one algorithm, one seed, "
        f"and {n_ok} fold(s). No claim is made that recurrent features are true, "
        f"causal, validated, or trusted."
    )

    # ---- assemble summary row ----------------------------------------------
    summary = dict(
        dataset=g["dataset"], repository=g["repository"],
        algorithm=g["algorithm"], classifier=g["classifier"],
        seed=int(g["seed"]), K=K,
        n_splits=n_splits, n_successful_folds=int(n_ok), n_failed_folds=int(n_failed),
        all_subsets_list_like=bool(all_list_like),
        selected_feature_count_consistent=bool(count_consistent),
        all_ok_subsets_size_K=bool(all_ok_size_K),
        metrics_valid=bool(metrics_valid),
        feature_pool_size=int(feature_pool_size),
        **perf,
        mean_pairwise_jaccard=_r(pj_mean), std_pairwise_jaccard=_r(pj_std),
        min_pairwise_jaccard=_r(pj_min), max_pairwise_jaccard=_r(pj_max),
        n_pairs=len(pj),
        kuncheva_mean=_r(k_mean), kuncheva_std=_r(k_std),
        kuncheva_min=_r(k_min), kuncheva_max=_r(k_max),
        nogueira_style_stability=_r(nog_stability),
        nogueira_denominator=_r(nog_denom, 8),
        d_used=int(d), d_was_inferred=bool(d_was_inferred),
        d_source=str(d_source),
        inferred_n_features=int(d),
        expected_random_jaccard=_r(exp_rand_jacc, 6),
        observed_mean_pairwise_jaccard=_r(pj_mean),
        observed_vs_random_jaccard_ratio=_r(jacc_vs_random),
        consensus_jaccard_mean=_r(cj_mean), consensus_jaccard_std=_r(cj_std),
        consensus_jaccard_min=_r(cj_min), consensus_jaccard_max=_r(cj_max),
        interpretation=interp,
    )
    return summary, freq_rows, sweep_rows, core_by_threshold


def compute_cross_k_recurrence(sweep_df):
    """K. Cross-K recurrence: features appearing in recurrent candidate cores
    for multiple K values, reported separately per threshold."""
    rows = []
    for keys, sub in sweep_df.groupby(DATASET_KEYS, dropna=False):
        keymap = dict(zip(DATASET_KEYS, keys))
        for t in THRESHOLDS:
            tsub = sub[sub["threshold_frequency"] == t]
            feat_to_Ks = {}
            for _, r in tsub.iterrows():
                for f in r["recurrent_candidate_core_features"]:
                    feat_to_Ks.setdefault(int(f), []).append(int(r["K"]))
            for feat, ks in sorted(feat_to_Ks.items()):
                ks_sorted = sorted(set(ks))
                rows.append(dict(
                    **keymap,
                    threshold_frequency=t,
                    feature_index=int(feat),
                    n_K_in_core=len(ks_sorted),
                    K_values_in_core=ks_sorted,
                    appears_in_multiple_K=bool(len(ks_sorted) >= 2),
                ))
    return pd.DataFrame(rows)


def periphery_correlation_view(df, sweep_df, d):
    """L. Optional periphery-correlation view for colon only.

    Loads colon.mat read-only and computes absolute Pearson correlations between
    recurrent candidate core features and non-core selected features, to indicate
    whether the variable component looks correlated with the recurrent core.
    Returns a markdown-ready string. On any failure, returns a TODO note.
    """
    try:
        import scipy.io
        if "colon" not in set(df["dataset"]):
            return "_Not applicable — input does not contain the `colon` dataset._"

        mat_path = (PROJECT_ROOT / "data" / "scikit-feature" / "colon.mat").resolve()
        if not mat_path.exists():
            return f"_TODO: colon.mat not found at {mat_path}; periphery-correlation view skipped._"

        Data = scipy.io.loadmat(str(mat_path))
        # Use np.abs(X) to match the preprocessing the FS step actually saw.
        X = np.abs(np.asarray(Data["X"], dtype=np.float64))

        lines = []
        for K in sorted(df["K"].unique()):
            t = 0.8  # descriptive reference cutoff for the periphery view
            srow = sweep_df[(sweep_df["K"] == K)
                            & (sweep_df["threshold_frequency"] == t)]
            if srow.empty:
                continue
            core = list(srow.iloc[0]["recurrent_candidate_core_features"])
            gdf = df[(df["K"] == K) & (df["status"] == "ok")]
            selected_any = set()
            for v in gdf["selected_feature_indices"]:
                selected_any.update(int(x) for x in np.asarray(v))
            periphery = sorted(selected_any - set(core))
            if not core or not periphery:
                lines.append(f"- K={K} (cutoff >=0.8): core={len(core)}, "
                             f"periphery={len(periphery)} — insufficient for a "
                             f"correlation view.")
                continue
            # Max |Pearson r| of each periphery feature to any core feature.
            core_mat = X[:, core]
            peri_max = []
            for f in periphery:
                col = X[:, f]
                cors = []
                for j in range(core_mat.shape[1]):
                    cmat = np.corrcoef(col, core_mat[:, j])
                    rij = cmat[0, 1]
                    if np.isfinite(rij):
                        cors.append(abs(rij))
                if cors:
                    peri_max.append(max(cors))
            # Mean |r| within the core.
            within = []
            for i in range(len(core)):
                for j in range(i + 1, len(core)):
                    cmat = np.corrcoef(core_mat[:, i], core_mat[:, j])
                    if np.isfinite(cmat[0, 1]):
                        within.append(abs(cmat[0, 1]))
            lines.append(
                f"- K={K} (cutoff >=0.8): core={len(core)}, periphery={len(periphery)}. "
                f"Periphery-to-core max |Pearson r|: "
                f"mean={_r(np.mean(peri_max)) if peri_max else None}, "
                f"median={_r(np.median(peri_max)) if peri_max else None}, "
                f"max={_r(np.max(peri_max)) if peri_max else None}. "
                f"Within-core mean |Pearson r|="
                f"{_r(np.mean(within)) if within else None}."
            )
        note = (
            "Correlations computed on `np.abs(X)` (the matrix the feature-selection "
            "step actually saw). This view is **descriptive only**: it indicates "
            "whether variable-component features tend to be correlated with "
            "recurrent-core features. It does not establish redundancy — that "
            "requires the dedicated redundancy-aware analysis (future work)."
        )
        return "\n".join(lines) + "\n\n" + note
    except Exception as exc:  # noqa: BLE001
        return (f"_TODO: periphery-correlation view skipped "
                f"({type(exc).__name__}: {exc}). Revisit when convenient._")


# ==========================================================================
# Markdown report
# ==========================================================================
def build_report(input_path, out_paths, summary_df, freq_df, sweep_df,
                  cross_k_df, periphery_text, commands):
    L = []
    L.append("# Stage A — Offline Stability Analysis Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> All thresholds and quantities below are **exploratory and descriptive**. "
             "No false-discovery / PFER / FWER / FDR control is claimed. `core_fraction` "
             "is descriptive only and is not used as a decision rule. No hard "
             "stable/unstable/trusted/validated labels are produced.\n")
    L.append("---\n")

    # 1. input + commands
    L.append("## 1. Input and Commands\n")
    L.append(f"- **Input parquet:** `{input_path}`")
    L.append(f"- **Output parquets (under `results/article_stability/stability_profiles/`):**")
    for p in out_paths:
        L.append(f"  - `{p}`")
    L.append("\n**Commands run:**\n")
    L.append("```bash")
    for c in commands:
        L.append(c)
    L.append("```\n")

    # 2. validation
    L.append("## 2. Validation Summary\n")
    L.append("| K | n_splits | ok folds | failed | subsets list-like | count consistent | ok subsets size K | metrics in [0,1] |")
    L.append("|---|----------|----------|--------|-------------------|------------------|-------------------|------------------|")
    for _, r in summary_df.iterrows():
        L.append(f"| {r['K']} | {r['n_splits']} | {r['n_successful_folds']} | "
                 f"{r['n_failed_folds']} | {r['all_subsets_list_like']} | "
                 f"{r['selected_feature_count_consistent']} | {r['all_ok_subsets_size_K']} | "
                 f"{r['metrics_valid']} |")
    L.append("")

    # 3. performance
    L.append("## 3. Per-K Performance Summary\n")
    L.append("| K | AUC mean±std | Acc mean±std | Prec mean±std | Rec mean±std | F1 mean±std |")
    L.append("|---|--------------|--------------|---------------|--------------|-------------|")
    for _, r in summary_df.iterrows():
        L.append(f"| {r['K']} | {r['auc_mean']}±{r['auc_std']} | "
                 f"{r['accuracy_mean']}±{r['accuracy_std']} | "
                 f"{r['precision_mean']}±{r['precision_std']} | "
                 f"{r['recall_mean']}±{r['recall_std']} | "
                 f"{r['f1_mean']}±{r['f1_std']} |")
    L.append("")

    # 4. overlap / chance-corrected indices
    L.append("## 4. Per-K Overlap and Chance-Corrected Stability\n")
    L.append("Raw Jaccard is reported **beside** chance-corrected indices (Kuncheva, "
             "Nogueira-style) and a random baseline, per the follow-up synthesis.\n")
    L.append("| K | mean pairwise Jaccard | Kuncheva mean | Nogueira-style | expected random Jaccard | observed/random ratio |")
    L.append("|---|-----------------------|---------------|----------------|-------------------------|-----------------------|")
    for _, r in summary_df.iterrows():
        L.append(f"| {r['K']} | {r['mean_pairwise_jaccard']} | {r['kuncheva_mean']} | "
                 f"{r['nogueira_style_stability']} | {r['expected_random_jaccard']} | "
                 f"{r['observed_vs_random_jaccard_ratio']} |")
    L.append("")
    d_used = int(summary_df['d_used'].iloc[0]) if len(summary_df) else None
    d_inf = bool(summary_df['d_was_inferred'].iloc[0]) if len(summary_df) else None
    d_source = (str(summary_df['d_source'].iloc[0])
                if len(summary_df) and 'd_source' in summary_df.columns else None)
    any_inferred = bool(summary_df['d_was_inferred'].any()) if len(summary_df) else False
    L.append(f"- **d (feature-universe size) used:** {d_used}")
    L.append(f"- **d_source:** `{d_source}`")
    L.append(f"- **d_was_inferred:** {d_inf}")
    if any_inferred:
        L.append("- *Caveat:* for at least one dataset group no usable `n_features` "
                 "column was found, so `d` was inferred as (max observed selected "
                 "feature index + 1). Inference can under-count the true feature "
                 "universe when the highest-indexed features are never selected; log "
                 "an explicit `n_features` column to avoid this.")
    else:
        L.append("- `d` was taken directly from the explicit `n_features` column in "
                 "the input parquet for every dataset group — not inferred.")
    L.append("- **Nogueira-style** is an *exploratory* variance-based estimate, not the "
             "exact JMLR-2018 estimator; verify before manuscript use.\n")

    # 5. threshold sweep
    L.append("## 5. Threshold Sensitivity (Recurrent Candidate Core)\n")
    L.append("Exploratory cutoffs only. `core_fraction` is descriptive.\n")
    L.append("| K | cutoff | folds needed | recurrent candidate core size | core_fraction (descriptive) | expected random core size | observed/expected ratio |")
    L.append("|---|--------|--------------|-------------------------------|-----------------------------|---------------------------|-------------------------|")
    for _, r in sweep_df.iterrows():
        L.append(f"| {r['K']} | >={r['threshold_frequency']} | {r['threshold_count']} | "
                 f"{r['recurrent_candidate_core_size']} | {r['core_fraction']} | "
                 f"{r['expected_random_core_size']} | {r['observed_vs_expected_ratio']} |")
    L.append("")
    L.append("**Recurrent candidate core members:**\n")
    for _, r in sweep_df.iterrows():
        L.append(f"- K={r['K']}, cutoff >={r['threshold_frequency']}: "
                 f"{list(r['recurrent_candidate_core_features'])}")
    L.append("")

    # 6. chance baseline narrative
    L.append("## 6. Chance Baselines\n")
    L.append("- **Recurrent-core chance baseline** (Section H): expected core size if "
             "features were selected independently at random with p = K/d per fold "
             "(see table in §5, columns *expected random core size* / "
             "*observed/expected ratio*).")
    L.append("- **Random pairwise-Jaccard baseline** (Section I): expected Jaccard for "
             "random K-subsets of d features (see §4, *expected random Jaccard* / "
             "*observed/random ratio*).")
    L.append("- An observed/expected ratio above 1 indicates more recurrence (or more "
             "overlap) than random selection would produce; this is **descriptive "
             "evidence**, not a significance test.\n")

    # 7. consensus jaccard
    L.append("## 7. Frequency-Consensus Jaccard (Internal Diagnostic)\n")
    L.append("Jaccard of each fold subset against the internal top-K "
             "frequency-consensus set. **Not** Article 2's three `Best Indices` "
             "variants — this is a Stage A internal diagnostic only.\n")
    L.append("| K | consensus Jaccard mean | std | min | max |")
    L.append("|---|------------------------|-----|-----|-----|")
    for _, r in summary_df.iterrows():
        L.append(f"| {r['K']} | {r['consensus_jaccard_mean']} | "
                 f"{r['consensus_jaccard_std']} | {r['consensus_jaccard_min']} | "
                 f"{r['consensus_jaccard_max']} |")
    L.append("")

    # 8. cross-K recurrence
    L.append("## 8. Cross-K Recurrence\n")
    L.append("Features appearing in recurrent candidate cores for multiple K values. "
             "This is a **budget-robustness** view (K is a hyperparameter), distinct "
             "from perturbation stability.\n")
    if cross_k_df is not None and len(cross_k_df):
        for t in THRESHOLDS:
            tsub = cross_k_df[cross_k_df["threshold_frequency"] == t]
            multi = tsub[tsub["appears_in_multiple_K"]]
            L.append(f"- **cutoff >={t}:** "
                     f"{len(multi)} feature(s) appear in the recurrent candidate core "
                     f"for >=2 K values.")
            for _, r in multi.sort_values(["n_K_in_core", "feature_index"],
                                          ascending=[False, True]).iterrows():
                L.append(f"  - feature {r['feature_index']}: in core for "
                         f"K={list(r['K_values_in_core'])}")
    else:
        L.append("- No cross-K recurrence rows produced.")
    L.append("")

    # 9. periphery correlation
    L.append("## 9. Periphery-Correlation View (Optional)\n")
    L.append(periphery_text)
    L.append("")

    # 10. interpretation
    L.append("## 10. Conservative Interpretation\n")
    for _, r in summary_df.iterrows():
        L.append(f"**K={r['K']}.** {r['interpretation']}\n")

    # 11. caveats
    L.append("## 11. Methodological Caveats\n")
    L.append("- **One dataset, one algorithm, one seed, five folds** — not generalizable.")
    L.append("- **Five folds give only six possible selection-frequency values** "
             "(0, 0.2, 0.4, 0.6, 0.8, 1.0); the 0.6/0.8/1.0 sweep is therefore coarse. "
             "Finer thresholds require many resamples (repeated subsampling / bootstrap).")
    L.append("- **Thresholds are exploratory**, pending the threshold-selection "
             "literature synthesis. They are conventional reference points "
             "(0.6/0.8 within the Meinshausen & Bühlmann 0.6–0.9 range; 1.0 the strict "
             "endpoint), **not** PFER/FWER/FDR-calibrated cutoffs.")
    L.append("- **`d` is inferred** from observed indices (no `n_features` column).")
    L.append("- **Nogueira-style stability is an exploratory estimate** — verify against "
             "Nogueira et al. (2018) before manuscript use.")
    L.append("- **No redundancy claim:** the periphery-correlation view is descriptive; "
             "it does not establish that variable-component features are redundant.")
    L.append("- No hard stable/unstable/trusted/validated/causal labels are asserted.\n")

    # 12. next step
    L.append("## 12. Recommended Next Step\n")
    L.append("Per `FOLLOWUP_DEEP_RESEARCH_SYNTHESIS.md` §4: the offline tooling (this "
             "script) is now in place. The recommended next step is the **synthetic "
             "control with planted correlated equivalence classes** (a known core of "
             "unique relevant features + correlated redundant proxy groups + noise), "
             "run under a **higher-resolution resampling protocol** (repeated "
             "subsampling / multiple seeds, tens of resamples) so selection frequency "
             "is estimated finely enough for the cutoff sweep to be meaningful. "
             "Repeated subsampling is adopted as the *perturbation protocol* only — "
             "not as a stability-selection *selection method*.\n")

    L.append("---\n")
    L.append("*Analysis is offline and descriptive. No feature selection was run, no "
             "benchmark file modified, no Stage A rerun performed.*")
    return "\n".join(L)


# ==========================================================================
# Driver
# ==========================================================================
def run_analysis(input_path, out_dir):
    """Run the full offline analysis and write all artifacts."""
    print("=" * 70)
    print("Article 5 — Offline per-fold stability analysis")
    print("=" * 70)
    print(f"Input : {input_path}")

    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df)} rows x {df.shape[1]} columns")

    summary_rows, freq_rows_all, sweep_rows_all = [], [], []

    for dkeys, ddf in df.groupby(DATASET_KEYS, dropna=False):
        d, d_inf, d_source = determine_d(ddf)
        print(f"Dataset {dict(zip(DATASET_KEYS, dkeys))}: "
              f"d={d} (source={d_source}, inferred={d_inf})")
        for gkeys, gdf in ddf.groupby(["K"], dropna=False):
            summary, freq_rows, sweep_rows, _ = analyze_group(gdf, d, d_inf, d_source)
            summary_rows.append(summary)
            freq_rows_all.extend(freq_rows)
            sweep_rows_all.extend(sweep_rows)
            print(f"  K={summary['K']}: ok folds={summary['n_successful_folds']}, "
                  f"pairwise Jaccard={summary['mean_pairwise_jaccard']}, "
                  f"Kuncheva={summary['kuncheva_mean']}, "
                  f"Nogueira-style={summary['nogueira_style_stability']}")

    summary_df = pd.DataFrame(summary_rows)
    freq_df = pd.DataFrame(freq_rows_all)
    sweep_df = pd.DataFrame(sweep_rows_all)
    cross_k_df = compute_cross_k_recurrence(sweep_df)

    # L. periphery-correlation view (colon only; descriptive)
    periphery_text = periphery_correlation_view(df, sweep_df, int(summary_df['d_used'].iloc[0]))

    # ---- resolve output paths (never overwrite) ----------------------------
    p_summary = safe_path(out_dir, OUT_GROUP_SUMMARY, ARTICLE5_STABILITY_ROOT)
    p_freq = safe_path(out_dir, OUT_FEATURE_FREQ, ARTICLE5_STABILITY_ROOT)
    p_sweep = safe_path(out_dir, OUT_THRESHOLD_SWEEP, ARTICLE5_STABILITY_ROOT)
    p_crossk = safe_path(out_dir, OUT_CROSS_K, ARTICLE5_STABILITY_ROOT)
    p_report = safe_path(REPORT_PATH.parent, REPORT_PATH.name, REPORT_PATH.parent)

    print("-" * 70)
    for label, p in [("group summary", p_summary), ("feature frequencies", p_freq),
                     ("threshold sweep", p_sweep), ("cross-K recurrence", p_crossk)]:
        print(f"[output] {label}: {p}")
    print(f"[output] markdown report: {p_report}")

    # ---- write -------------------------------------------------------------
    summary_df.to_parquet(p_summary, index=False)
    freq_df.to_parquet(p_freq, index=False)
    sweep_df.to_parquet(p_sweep, index=False)
    cross_k_df.to_parquet(p_crossk, index=False)

    commands = [
        "python -m py_compile articles/article_stability/scripts/analyze_per_fold_stability.py",
        "python articles/article_stability/scripts/analyze_per_fold_stability.py --dry-run",
        "python articles/article_stability/scripts/analyze_per_fold_stability.py",
    ]
    report_md = build_report(
        input_path,
        [p_summary.name, p_freq.name, p_sweep.name, p_crossk.name],
        summary_df, freq_df, sweep_df, cross_k_df, periphery_text, commands,
    )
    p_report.write_text(report_md)

    # ---- concise console summary ------------------------------------------
    print("-" * 70)
    print("Analysis summary")
    print("-" * 70)
    print(f"Groups analyzed     : {len(summary_df)}")
    print(f"Feature-freq rows   : {len(freq_df)}")
    print(f"Threshold-sweep rows: {len(sweep_df)}")
    print(f"Cross-K recur. rows : {len(cross_k_df)}")
    print(f"Markdown report     : {p_report}")
    print("=" * 70)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Article 5 offline per-fold stability analysis (descriptive; "
                    "exploratory thresholds; no false-discovery claims).",
    )
    parser.add_argument(
        "--input-parquet", default=None,
        help="Per-fold selection parquet to analyze. "
             "Default: the Stage A colon/ETree parquet.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for analysis parquets. Relative paths are interpreted "
             "under results/article_stability/stability_profiles/; the resolved path must "
             "stay inside it.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and resolved paths, then exit without reading the input, "
             "computing, or writing anything.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    input_path = (Path(args.input_parquet).expanduser().resolve()
                  if args.input_parquet else DEFAULT_INPUT_PARQUET)
    out_dir = resolve_output_dir(args.output_dir)

    if args.dry_run:
        print("=" * 70)
        print("Article 5 — Offline stability analysis  [DRY RUN]")
        print("=" * 70)
        print(f"Input parquet        : {input_path}")
        print(f"Input exists         : {input_path.exists()}")
        print(f"Stability output root: {ARTICLE5_STABILITY_ROOT}")
        print(f"Output directory     : {out_dir}")
        print(f"Thresholds (sweep)   : {THRESHOLDS}")
        print("Planned parquet outputs:")
        for n in (OUT_GROUP_SUMMARY, OUT_FEATURE_FREQ, OUT_THRESHOLD_SWEEP, OUT_CROSS_K):
            print(f"  - {out_dir / n}")
        print(f"Planned markdown report: {REPORT_PATH}")
        print("DRY RUN — input not read, nothing computed, nothing written.")
        print("=" * 70)
        return 0

    if not input_path.exists():
        raise SystemExit(f"[ERROR] Input parquet not found: {input_path}")

    # Create the output directory only for a real run.
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return run_analysis(input_path, out_dir)


if __name__ == "__main__":
    sys.exit(main())
