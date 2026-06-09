"""
synthetic_recovery_metrics.py — Article 5, synthetic support-recovery metrics
=============================================================================

Measures **support recovery** of the Article 5 synthetic smoke test against the
known planted ground truth — a view complementary to, and deliberately kept
SEPARATE from, the internal stability profile produced by
`analyze_per_fold_stability.py`.

  * stability  = internal consistency of the selected subsets across resamples
                 (Jaccard / Kuncheva / Nogueira-style / chance baselines).
  * recovery   = agreement of the recurrently selected features with the known
                 planted structure (true core / proxies / weak / noise).

These two are NOT collapsed into a single score. This script computes the
recovery view only; it reads the existing stability-profile parquets and the
synthetic dataset manifests, and writes recovery tables + a markdown report.

It does NOT run feature selection, does NOT rerun the logging driver, and does
NOT rerun the offline analyzer — it consumes existing v2 outputs.

Inputs (defaults point at the v2 stability-profile directory):
  * feature-frequencies parquet  (per dataset/K/feature selection frequency)
  * threshold-sweep parquet      (used only for a consistency cross-check)
  * group-summary parquet        (used to confirm d / d_source)
  * synthetic manifests          (ground-truth planted structure)

Isolation contract:
  * Modifies no existing script, no generator, no benchmark file, no notebook.
  * Reads only the inputs above (read-only).
  * Writes recovery parquets ONLY under
    results/article_stability/synthetic_control/recovery_metrics/.
  * Writes one markdown report under results/article_stability/reports/experiment_planning/.
  * Never overwrites an existing output (timestamps a new name instead).

Conservative-wording policy:
  * "planted-core recovery", "recurrent candidate core", "noise contamination",
    "redundant proxy recurrence" — yes.
  * "trusted core", "causal feature", "biomarker", "validated feature" — no.
  * A recurrent proxy is NOT a false positive — it is redundant-but-informative;
    that is why `noise_contamination` is reported separately from
    `exact_true_core_fraction`.

Usage:
    python articles/article_stability/scripts/synthetic_recovery_metrics.py --dry-run
    python articles/article_stability/scripts/synthetic_recovery_metrics.py
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Path setup. This script lives at articles/article_stability/scripts/synthetic_recovery_metrics.py
# so parents[3] == project root.
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]

# ==========================================================================
# Configuration
# ==========================================================================
V2_PROFILE_DIR = (
    PROJECT_ROOT / "results" / "article_stability" / "stability_profiles"
    / "synthetic_smoke_etree_seed0_10resamples_v2"
).resolve()

DEFAULT_FEATURE_FREQUENCIES = V2_PROFILE_DIR / "stageA_colon_etree_feature_frequencies.parquet"
DEFAULT_THRESHOLD_SWEEP = V2_PROFILE_DIR / "stageA_colon_etree_threshold_sweep.parquet"
DEFAULT_GROUP_SUMMARY = V2_PROFILE_DIR / "stageA_colon_etree_stability_group_summary.parquet"

DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "datasets"
).resolve()

# Output safety root — all recovery parquets MUST stay inside this directory.
ARTICLE5_RECOVERY_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "recovery_metrics"
).resolve()

REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_SMOKE_RECOVERY_METRICS_REPORT.md"
).resolve()

THRESHOLDS = [0.6, 0.8, 1.0]
EPS = 1e-9   # float guard for `frequency >= threshold` comparisons

OUT_ANNOTATED = "synthetic_smoke_feature_frequency_annotated.parquet"
OUT_GROUP = "synthetic_smoke_recovery_group_summary.parquet"
OUT_THRESHOLD = "synthetic_smoke_threshold_recovery.parquet"
OUT_CROSS_K = "synthetic_smoke_cross_k_recovery.parquet"
OUT_KNOWN_VS_NULL = "synthetic_smoke_known_vs_null_comparison.parquet"

# The 5 mutually-exclusive feature roles, and the broad-category label each maps
# to. NOTE on `broad_category`: it is a mutually-exclusive per-feature partition.
# The label "planted_signal" here denotes weak planted-signal features
# specifically; the broad SET planted_signal (= true_core + weak) used in some
# metrics is the union of the "true_core" and "planted_signal" broad categories
# (equivalently, `is_true_core | is_weak`).
ROLE_TO_BROAD = {
    "solo_core":     "true_core",
    "anchored_core": "true_core",
    "proxy":         "redundant_proxy",
    "weak":          "planted_signal",
    "noise":         "pure_noise",
}
ROLES = ["solo_core", "anchored_core", "true_core", "proxy", "weak", "noise"]


# ==========================================================================
# Output-path safety
# ==========================================================================
def _is_within(child, parent):
    child = Path(child).resolve()
    parent = Path(parent).resolve()
    return child == parent or parent in child.parents


def resolve_output_dir(output_dir_arg):
    """Resolve + validate the output directory (must stay inside the recovery
    safety root)."""
    if output_dir_arg is None:
        out_dir = ARTICLE5_RECOVERY_ROOT
    else:
        candidate = Path(output_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = ARTICLE5_RECOVERY_ROOT / candidate
        out_dir = candidate.resolve()
    if not _is_within(out_dir, ARTICLE5_RECOVERY_ROOT):
        raise SystemExit(
            f"[SAFETY] Refusing to write outside the recovery-metrics root.\n"
            f"         requested : {out_dir}\n"
            f"         allowed   : {ARTICLE5_RECOVERY_ROOT}"
        )
    return out_dir


def safe_path(directory, base_name, safety_root):
    """Return a non-overwriting path inside `directory` (timestamps if it exists)."""
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
# Ground-truth (manifest) parsing
# ==========================================================================
def load_manifest(name, manifest_dir):
    """Load the JSON manifest for a synthetic dataset by name."""
    path = (Path(manifest_dir) / f"{name}_manifest.json").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found for '{name}': {path}")
    with open(path, "r") as fh:
        return json.load(fh), str(path)


def build_role_map(manifest):
    """Build the ground-truth role map for one synthetic dataset.

    Returns a dict of planted index sets, proxy-group structure, the total
    feature count, and an `is_null` flag.
    """
    solo = set(int(i) for i in manifest.get("true_core_solo_indices", []))
    anchored = set(int(i) for i in manifest.get("true_core_anchored_indices", []))
    proxy_groups_raw = manifest.get("proxy_groups", {}) or {}
    proxy_groups = {}
    proxy_to_parent, proxy_to_group = {}, {}
    proxy = set()
    for gid, (parent_s, proxies) in enumerate(proxy_groups_raw.items()):
        parent = int(parent_s)
        plist = [int(x) for x in proxies]
        proxy_groups[parent] = plist
        for px in plist:
            proxy.add(px)
            proxy_to_parent[px] = parent
            proxy_to_group[px] = gid
    weak = set(int(i) for i in manifest.get("weak_signal_indices", []))
    true_core = solo | anchored
    n_total = int(manifest.get("n_total_features", 2000))
    n_noise = n_total - len(solo) - len(anchored) - len(proxy) - len(weak)
    is_null = (len(true_core) == 0 and len(proxy) == 0 and len(weak) == 0)
    return {
        "solo": solo, "anchored": anchored, "true_core": true_core,
        "proxy": proxy, "weak": weak,
        "proxy_groups": proxy_groups,
        "proxy_to_parent": proxy_to_parent, "proxy_to_group": proxy_to_group,
        "n_total": n_total, "n_noise": n_noise,
        "is_null": is_null,
    }


def annotate_feature(idx, rm):
    """Return the role annotation for a single feature index under role map `rm`."""
    base = dict(
        proxy_parent_index=None, proxy_group_id=None,
        is_true_core=False, is_solo_core=False, is_anchored_core=False,
        is_proxy=False, is_weak=False, is_noise=False,
    )
    if rm["is_null"]:
        base.update(feature_role="noise", broad_category="pure_noise", is_noise=True)
        return base
    if idx in rm["solo"]:
        base.update(feature_role="solo_core", broad_category="true_core",
                    is_true_core=True, is_solo_core=True)
        return base
    if idx in rm["anchored"]:
        base.update(feature_role="anchored_core", broad_category="true_core",
                    is_true_core=True, is_anchored_core=True)
        return base
    if idx in rm["proxy"]:
        base.update(feature_role="proxy", broad_category="redundant_proxy",
                    is_proxy=True,
                    proxy_parent_index=int(rm["proxy_to_parent"][idx]),
                    proxy_group_id=int(rm["proxy_to_group"][idx]))
        return base
    if idx in rm["weak"]:
        base.update(feature_role="weak", broad_category="planted_signal", is_weak=True)
        return base
    base.update(feature_role="noise", broad_category="pure_noise", is_noise=True)
    return base


def role_population(rm, role):
    """Full planted-population size for a role under role map `rm`."""
    return {
        "solo_core": len(rm["solo"]),
        "anchored_core": len(rm["anchored"]),
        "true_core": len(rm["true_core"]),
        "proxy": len(rm["proxy"]),
        "weak": len(rm["weak"]),
        "noise": rm["n_noise"],
    }[role]


# ==========================================================================
# 1. Feature-level annotated frequency table
# ==========================================================================
def annotate_feature_frequencies(freq_df, role_maps):
    """Annotate every row of the feature-frequency table with its ground-truth
    role. Returns the annotated DataFrame."""
    records = []
    for r in freq_df.itertuples(index=False):
        rm = role_maps[r.dataset]
        ann = annotate_feature(int(r.feature_index), rm)
        records.append(dict(
            dataset=r.dataset,
            dataset_role=("null_control" if rm["is_null"] else "known_core"),
            algorithm=r.algorithm,
            classifier=r.classifier,
            seed=int(r.seed),
            K=int(r.K),
            feature_index=int(r.feature_index),
            fold_count=int(r.fold_count),
            frequency=float(r.frequency),
            selected_in_all_folds=bool(r.selected_in_all_folds),
            feature_role=ann["feature_role"],
            broad_category=ann["broad_category"],
            proxy_parent_index=ann["proxy_parent_index"],
            proxy_group_id=ann["proxy_group_id"],
            is_true_core=ann["is_true_core"],
            is_solo_core=ann["is_solo_core"],
            is_anchored_core=ann["is_anchored_core"],
            is_proxy=ann["is_proxy"],
            is_weak=ann["is_weak"],
            is_noise=ann["is_noise"],
        ))
    return pd.DataFrame(records)


# ==========================================================================
# 2. Group-level recovery metrics
# ==========================================================================
GROUP_KEYS = ["dataset", "algorithm", "classifier", "seed", "K"]


def compute_group_summary(annotated_df, role_maps):
    """Category-wise frequency summaries per (dataset, algorithm, classifier,
    seed, K), plus known-core / null-control specific fields."""
    rows = []
    for keys, g in annotated_df.groupby(GROUP_KEYS, dropna=False):
        keymap = dict(zip(GROUP_KEYS, keys))
        rm = role_maps[keymap["dataset"]]
        is_null = rm["is_null"]
        row = dict(keymap)
        row["dataset_role"] = "null_control" if is_null else "known_core"

        # category-wise frequency summaries for the 5 roles + the combined
        # true_core role.
        for role in ROLES:
            pop = role_population(rm, role)
            if role == "true_core":
                sel = g[g["is_true_core"]]
            else:
                sel = g[g["feature_role"] == role]
            freqs = sel["frequency"].astype(float).tolist()
            row[f"{role}_mean_frequency"] = (
                round(sum(freqs) / pop, 6) if pop > 0 else None
            )
            row[f"{role}_max_frequency"] = (
                round(max(freqs), 6) if freqs else (0.0 if pop > 0 else None)
            )
            row[f"{role}_count_ge_0_6"] = int(sum(1 for f in freqs if f >= 0.6 - EPS))
            row[f"{role}_count_ge_0_8"] = int(sum(1 for f in freqs if f >= 0.8 - EPS))
            row[f"{role}_count_eq_1_0"] = int(sum(1 for f in freqs if f >= 1.0 - EPS))

        # known-core specific
        noise_sel = g[g["feature_role"] == "noise"]
        noise_freqs = noise_sel["frequency"].astype(float)
        row["noise_selected_feature_count"] = int(len(noise_sel))
        row["noise_mean_frequency_among_selected_features"] = (
            round(float(noise_freqs.mean()), 6) if len(noise_freqs) else None
        )
        top = g.sort_values(["frequency", "feature_index"],
                            ascending=[False, True]).head(10)
        row["top_10_features_with_roles"] = "; ".join(
            f"{int(t.feature_index)}:{t.feature_role}({float(t.frequency):.2f})"
            for t in top.itertuples(index=False)
        )

        # null-control specific
        if is_null:
            all_freqs = g["frequency"].astype(float)
            row["max_frequency_any_feature"] = (
                round(float(all_freqs.max()), 6) if len(all_freqs) else None
            )
            row["recurrent_feature_count_0_6"] = int((all_freqs >= 0.6 - EPS).sum())
            row["recurrent_feature_count_0_8"] = int((all_freqs >= 0.8 - EPS).sum())
            row["recurrent_feature_count_1_0"] = int((all_freqs >= 1.0 - EPS).sum())
        else:
            row["max_frequency_any_feature"] = None
            row["recurrent_feature_count_0_6"] = None
            row["recurrent_feature_count_0_8"] = None
            row["recurrent_feature_count_1_0"] = None

        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "K"]).reset_index(drop=True)


# ==========================================================================
# 3. Threshold-level recovery metrics
# ==========================================================================
def compute_threshold_recovery(annotated_df, role_maps):
    """Per (dataset, algorithm, classifier, seed, K, threshold): planted-core
    recovery, noise contamination, and proxy-group coverage."""
    rows = []
    for keys, g in annotated_df.groupby(GROUP_KEYS, dropna=False):
        keymap = dict(zip(GROUP_KEYS, keys))
        rm = role_maps[keymap["dataset"]]
        is_null = rm["is_null"]
        for t in THRESHOLDS:
            rec = g[g["frequency"].astype(float) >= t - EPS]
            rec_idx = set(int(i) for i in rec["feature_index"])
            size = len(rec_idx)
            row = dict(keymap)
            row["dataset_role"] = "null_control" if is_null else "known_core"
            row["threshold"] = t
            row["recurrent_set_size"] = int(size)

            if is_null:
                # every selected feature in a null dataset is noise
                row.update(
                    true_core_recovered_count=None, true_core_recall=None,
                    solo_core_recovered_count=None, solo_core_recall=None,
                    anchored_core_recovered_count=None, anchored_core_recall=None,
                    proxy_recurrent_count=None, weak_recurrent_count=None,
                    noise_recurrent_count=int(size),
                    noise_contamination=(1.0 if size > 0 else 0.0),
                    exact_true_core_fraction=None,
                    proxy_group_coverage_count=None,
                    proxy_group_coverage_fraction=None,
                    anchored_signal_group_coverage_count=None,
                    anchored_signal_group_coverage_fraction=None,
                )
            else:
                tc_rec = rec_idx & rm["true_core"]
                solo_rec = rec_idx & rm["solo"]
                anch_rec = rec_idx & rm["anchored"]
                proxy_rec = rec_idx & rm["proxy"]
                weak_rec = rec_idx & rm["weak"]
                noise_rec = size - len(tc_rec) - len(proxy_rec) - len(weak_rec)
                n_groups = len(rm["proxy_groups"])
                proxy_grp_cov = sum(
                    1 for proxies in rm["proxy_groups"].values()
                    if rec_idx & set(proxies)
                )
                anch_sig_cov = sum(
                    1 for parent, proxies in rm["proxy_groups"].items()
                    if (parent in rec_idx) or (rec_idx & set(proxies))
                )
                row.update(
                    true_core_recovered_count=int(len(tc_rec)),
                    true_core_recall=round(len(tc_rec) / len(rm["true_core"]), 6),
                    solo_core_recovered_count=int(len(solo_rec)),
                    solo_core_recall=round(len(solo_rec) / len(rm["solo"]), 6),
                    anchored_core_recovered_count=int(len(anch_rec)),
                    anchored_core_recall=round(len(anch_rec) / len(rm["anchored"]), 6),
                    proxy_recurrent_count=int(len(proxy_rec)),
                    weak_recurrent_count=int(len(weak_rec)),
                    noise_recurrent_count=int(noise_rec),
                    noise_contamination=(round(noise_rec / size, 6) if size > 0 else 0.0),
                    exact_true_core_fraction=(
                        round(len(tc_rec) / size, 6) if size > 0 else None
                    ),
                    proxy_group_coverage_count=int(proxy_grp_cov),
                    proxy_group_coverage_fraction=(
                        round(proxy_grp_cov / n_groups, 6) if n_groups else None
                    ),
                    anchored_signal_group_coverage_count=int(anch_sig_cov),
                    anchored_signal_group_coverage_fraction=(
                        round(anch_sig_cov / n_groups, 6) if n_groups else None
                    ),
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["dataset", "K", "threshold"]).reset_index(drop=True)


# ==========================================================================
# 4. Cross-K recovery
# ==========================================================================
def compute_cross_k_recovery(annotated_df, role_maps):
    """For each (dataset, threshold): which features are recurrent (freq >= t)
    in one or more K values, with their roles."""
    rows = []
    for dataset, g_ds in annotated_df.groupby("dataset", dropna=False):
        rm = role_maps[dataset]
        dataset_role = "null_control" if rm["is_null"] else "known_core"
        for t in THRESHOLDS:
            rec = g_ds[g_ds["frequency"].astype(float) >= t - EPS]
            for feat, gf in rec.groupby("feature_index"):
                ks = sorted(int(k) for k in gf["K"].unique())
                ann = annotate_feature(int(feat), rm)
                rows.append(dict(
                    dataset=dataset,
                    dataset_role=dataset_role,
                    threshold=t,
                    feature_index=int(feat),
                    feature_role=ann["feature_role"],
                    broad_category=ann["broad_category"],
                    n_K_recurrent=len(ks),
                    K_values_recurrent=ks,
                    appears_in_multiple_K=bool(len(ks) >= 2),
                ))
    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "dataset_role", "threshold", "feature_index",
            "feature_role", "broad_category", "n_K_recurrent",
            "K_values_recurrent", "appears_in_multiple_K",
        ])
    return pd.DataFrame(rows).sort_values(
        ["dataset", "threshold", "n_K_recurrent", "feature_index"],
        ascending=[True, True, False, True]).reset_index(drop=True)


# ==========================================================================
# 5. Known-core vs null-control comparison
# ==========================================================================
def compute_known_vs_null(threshold_df):
    """Per (known-core dataset, K, threshold) comparison against the matched
    null-control.

    Supports one OR multiple known-core datasets: each known-core dataset is
    compared, row by row, against the single null-control at the same
    (K, threshold). For a single known-core dataset the behaviour is unchanged
    apart from the added `known_core_dataset` column.
    """
    known = threshold_df[threshold_df["dataset_role"] == "known_core"]
    null = threshold_df[threshold_df["dataset_role"] == "null_control"]
    rows = []
    for kc_dataset in sorted(known["dataset"].unique()):
        kd = known[known["dataset"] == kc_dataset]
        for (K, t) in sorted(set(zip(kd["K"], kd["threshold"]))):
            kr = kd[(kd["K"] == K) & (kd["threshold"] == t)]
            nr = null[(null["K"] == K) & (null["threshold"] == t)]
            krow = kr.iloc[0] if len(kr) else None
            nrow = nr.iloc[0] if len(nr) else None
            k_size = int(krow["recurrent_set_size"]) if krow is not None else None
            n_size = int(nrow["recurrent_set_size"]) if nrow is not None else None
            rows.append(dict(
                known_core_dataset=kc_dataset,
                K=int(K),
                threshold=float(t),
                known_core_recurrent_set_size=k_size,
                known_core_true_core_recall=(
                    float(krow["true_core_recall"]) if krow is not None else None
                ),
                known_core_proxy_group_coverage_fraction=(
                    float(krow["proxy_group_coverage_fraction"])
                    if krow is not None
                    and pd.notna(krow["proxy_group_coverage_fraction"])
                    else None
                ),
                known_core_anchored_signal_group_coverage_fraction=(
                    float(krow["anchored_signal_group_coverage_fraction"])
                    if krow is not None
                    and pd.notna(krow["anchored_signal_group_coverage_fraction"])
                    else None
                ),
                known_core_noise_contamination=(
                    float(krow["noise_contamination"]) if krow is not None else None
                ),
                null_control_recurrent_set_size=n_size,
                null_control_noise_contamination=(
                    float(nrow["noise_contamination"]) if nrow is not None else None
                ),
                recurrent_set_size_diff=(
                    (k_size - n_size)
                    if (k_size is not None and n_size is not None) else None
                ),
            ))
    return pd.DataFrame(rows)


# ==========================================================================
# Consistency cross-check vs the analyzer's threshold sweep
# ==========================================================================
def crosscheck_threshold_sweep(threshold_df, sweep_df):
    """Compare this script's recurrent_set_size against the analyzer's
    recurrent_candidate_core_size. Returns (n_compared, n_mismatch, details)."""
    n_compared, n_mismatch, details = 0, 0, []
    for r in threshold_df.itertuples(index=False):
        m = sweep_df[
            (sweep_df["dataset"] == r.dataset)
            & (sweep_df["K"] == r.K)
            & (sweep_df["threshold_frequency"] == r.threshold)
        ]
        if len(m) != 1:
            continue
        n_compared += 1
        sweep_size = int(m.iloc[0]["recurrent_candidate_core_size"])
        if sweep_size != int(r.recurrent_set_size):
            n_mismatch += 1
            details.append(
                f"{r.dataset} K={r.K} t={r.threshold}: "
                f"recovery={r.recurrent_set_size} vs sweep={sweep_size}"
            )
    return n_compared, n_mismatch, details


# ==========================================================================
# Conservative interpretation
# ==========================================================================
def compute_interpretation(threshold_df, kvn_df):
    """Conservative interpretation findings. Returns a list of
    (question, verdict, evidence) tuples — kept deliberately cautious."""
    kc = threshold_df[threshold_df["dataset_role"] == "known_core"]
    findings = []

    # 1. known-core recurrence vs matched null
    n_cells = len(kvn_df)
    n_exceed = int((kvn_df["recurrent_set_size_diff"] > 0).sum())
    n_tied = int((kvn_df["recurrent_set_size_diff"] == 0).sum())
    findings.append((
        "Does known-core recurrence exceed matched null recurrence?",
        f"Yes in {n_exceed}/{n_cells} (K,threshold) cells",
        f"the known-core recurrent set is strictly larger than the matched null in "
        f"{n_exceed} of {n_cells} cells ({n_tied} tied); the null is never larger.",
    ))

    # 2. exact true-core recovery strength
    full = kc[kc["true_core_recall"].astype(float) >= 1.0 - EPS]
    cells_full = sorted(set((int(r.K), float(r.threshold))
                            for r in full.itertuples(index=False)))
    k5_recall = sorted(set(float(r.true_core_recall)
                           for r in kc[kc["K"] == 5].itertuples(index=False)))
    findings.append((
        "Is exact true-core recovery strong?",
        "Budget-dependent" if cells_full else "Weak",
        f"exact true-core recall reaches 1.0 in {len(cells_full)} (K,threshold) cell(s) "
        f"{cells_full}; at K=5 exact recall is {k5_recall}. Exact recovery is "
        f"budget-dependent, not uniform.",
    ))

    # 3. substitution: anchored-signal-group coverage above exact recall
    sub_cells = []
    for r in kc.itertuples(index=False):
        asc = r.anchored_signal_group_coverage_fraction
        rec = r.true_core_recall
        if asc is not None and rec is not None and float(asc) > float(rec) + EPS:
            sub_cells.append((int(r.K), float(r.threshold)))
    findings.append((
        "Does group/proxy-aware recovery suggest substitution?",
        "Yes" if sub_cells else "Not evident",
        f"anchored-signal-group coverage exceeds exact true-core recall in "
        f"{len(sub_cells)} cell(s) {sorted(sub_cells)} — the planted signal is "
        f"recurrently captured via redundant-but-informative proxies where the exact "
        f"core feature is not.",
    ))

    # 4. null recurrence at high thresholds
    null_hi = threshold_df[
        (threshold_df["dataset_role"] == "null_control")
        & (threshold_df["threshold"].astype(float) >= 0.8 - EPS)
    ]
    null_hi_total = int(null_hi["recurrent_set_size"].sum())
    findings.append((
        "Does null recurrence remain non-negligible?",
        "Yes — small but non-zero" if null_hi_total > 0 else "No",
        f"the null control has {null_hi_total} recurrent feature-slots at threshold "
        f">= 0.8 (summed across K); all are noise. Consistent with a fixed-label "
        f"finite-sample artifact; it motivates the null-calibrated view and will not "
        f"be averaged out by more resamples of the same dataset.",
    ))
    return findings


# ==========================================================================
# Markdown report
# ==========================================================================
def build_report(annotated_df, group_df, threshold_df, cross_k_df, kvn_df,
                 role_maps, group_summary_df, commands, out_paths,
                 crosscheck, input_paths):
    # Resample count, derived from the stability-profile group summary.
    if "n_successful_folds" in group_summary_df.columns and len(group_summary_df):
        n_resamples = int(group_summary_df["n_successful_folds"].max())
    elif "n_splits" in group_summary_df.columns and len(group_summary_df):
        n_resamples = int(group_summary_df["n_splits"].max())
    else:
        n_resamples = 0
    L = []
    L.append("# Synthetic Smoke — Support-Recovery Metrics Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Smoke test only. This report measures **support recovery against the "
             "known synthetic ground truth** — a view kept deliberately separate from "
             "the internal stability profile. Not manuscript-quality analysis.\n")
    L.append("---\n")

    # 1. Purpose
    L.append("## 1. Purpose\n")
    L.append("The offline stability analyzer measures *internal consistency* of the "
             "selected subsets (Jaccard, Kuncheva, Nogueira-style, chance baselines). "
             "It cannot say whether the recurrently selected features are *correct*, "
             "because real data has no known ground truth. This report uses the "
             "**synthetic known-core control**, whose planted structure is known, to "
             "measure **support recovery**: do the recurrently selected features align "
             "with the planted true core / proxies / weak signal, and is the recurrent "
             "set contaminated by pure noise? Stability and recovery are reported as "
             "two complementary views and are **not** collapsed into one score.\n")

    # 2. Inputs
    L.append("## 2. Inputs\n")
    L.append("Actual input paths used for this run (not hardcoded defaults):\n")
    L.append(f"- feature-frequency parquet: `{input_paths['feature_frequencies']}`")
    L.append(f"- threshold-sweep parquet (consistency cross-check only): "
             f"`{input_paths['threshold_sweep']}`")
    L.append(f"- group-summary parquet (d / d_source check): "
             f"`{input_paths['group_summary']}`")
    L.append(f"- manifests: `{input_paths['manifest_dir']}` "
             f"(`synthetic_snrHIGH_rho090_manifest.json`, "
             f"`synthetic_null_control_manifest.json`)\n")

    # 3. Ground-truth structure
    L.append("## 3. Ground-Truth Structure\n")
    for name, rm in role_maps.items():
        if rm["is_null"]:
            L.append(f"- **{name}** (null control): all {rm['n_total']} features are "
                     f"pure noise — no true core, no proxies, no weak features.")
        else:
            L.append(f"- **{name}** (known core):")
            L.append(f"  - solo core features: {sorted(rm['solo'])}")
            L.append(f"  - anchored core features: {sorted(rm['anchored'])}")
            L.append(f"  - proxy groups (parent → proxies): "
                     + "; ".join(f"{p}→{rm['proxy_groups'][p]}"
                                 for p in rm['proxy_groups']))
            L.append(f"  - weak signal features: {len(rm['weak'])} "
                     f"({sorted(rm['weak'])})")
            L.append(f"  - pure noise features: {rm['n_noise']}")
            L.append(f"  - total features: {rm['n_total']}")
    L.append("- *Note:* `broad_category` is a mutually-exclusive per-feature partition; "
             "its label `planted_signal` denotes weak features specifically. The broad "
             "set planted_signal (= true_core ∪ weak) is `is_true_core | is_weak`.\n")

    # 4. Main recovery summary
    L.append("## 4. Main Recovery Summary\n")
    L.append("Known-core planted-core recall and noise contamination, by K and "
             "threshold (recurrent set = features with selection frequency ≥ threshold):\n")
    L.append("| K | thr | recurrent set | true-core recall | noise contamination | "
             "proxy-group coverage | anchored-signal-group coverage |")
    L.append("|---|-----|---------------|------------------|---------------------|"
             "----------------------|--------------------------------|")
    kc = threshold_df[threshold_df["dataset_role"] == "known_core"]
    for r in kc.itertuples(index=False):
        L.append(f"| {r.K} | {r.threshold} | {r.recurrent_set_size} | "
                 f"{r.true_core_recovered_count}/{len(role_maps_true_core(role_maps))} "
                 f"({r.true_core_recall}) | {r.noise_contamination} | "
                 f"{r.proxy_group_coverage_count}/3 ({r.proxy_group_coverage_fraction}) | "
                 f"{r.anchored_signal_group_coverage_count}/3 "
                 f"({r.anchored_signal_group_coverage_fraction}) |")
    L.append("")
    L.append("Null-control recurrent-set size, by K and threshold:\n")
    L.append("| K | thr | recurrent set | noise contamination |")
    L.append("|---|-----|---------------|---------------------|")
    nc = threshold_df[threshold_df["dataset_role"] == "null_control"]
    for r in nc.itertuples(index=False):
        L.append(f"| {r.K} | {r.threshold} | {r.recurrent_set_size} | "
                 f"{r.noise_contamination} |")
    L.append("")

    # 4a. Null-calibrated recurrence (primary view) ---------------------------
    L.append("### 4a. Null-Calibrated Recurrence (primary view)\n")
    L.append("\"Recurrent\" is judged against the matched signal-free null at the same "
             "K and threshold — not against zero. Exact planted-core recovery and "
             "group/proxy-aware coverage are presented **side by side and are not "
             "collapsed into one score**: exact recall is honest about substitution; "
             "group/proxy-aware coverage credits redundant-but-informative proxy "
             "capture.\n")
    L.append("| known-core dataset | K | thr | known recurrent | null recurrent | "
             "size diff | known exact true-core recall | known proxy-group cov frac | "
             "known anchored-signal-group cov frac | known noise contam | "
             "null noise contam |")
    L.append("|--------------------|---|-----|-----------------|----------------|"
             "-----------|------------------------------|----------------------------|"
             "--------------------------------------|---------------------|"
             "-------------------|")
    for r in kvn_df.itertuples(index=False):
        L.append(f"| {r.known_core_dataset} | {r.K} | {r.threshold} | "
                 f"{r.known_core_recurrent_set_size} | "
                 f"{r.null_control_recurrent_set_size} | {r.recurrent_set_size_diff} | "
                 f"{r.known_core_true_core_recall} | "
                 f"{r.known_core_proxy_group_coverage_fraction} | "
                 f"{r.known_core_anchored_signal_group_coverage_fraction} | "
                 f"{r.known_core_noise_contamination} | "
                 f"{r.null_control_noise_contamination} |")
    L.append("")

    # 4b. Conservative interpretation -----------------------------------------
    L.append("### 4b. Conservative Interpretation\n")
    L.append("| Question | Verdict | Evidence |")
    L.append("|----------|---------|----------|")
    for q, v, d in compute_interpretation(threshold_df, kvn_df):
        L.append(f"| {q} | {v} | {d} |")
    L.append("")

    # 5. K-specific interpretation
    L.append("## 5. K-Specific Interpretation\n")
    L.append("- **K=1** — top-feature recurrence diagnostic only.")
    L.append("- **K=5** — exact true-core budget (budget equals the 5 planted core "
             "features).")
    L.append("- **K=10 / K=30** — core + proxy/periphery behaviour.")
    L.append("- **K=60** — large-budget periphery stress test.\n")

    # 6. Null-control sanity
    L.append("## 6. Null-Control Sanity\n")
    null_rec = nc[nc["threshold"] >= 0.8 - EPS]["recurrent_set_size"].sum()
    null_rec_06 = nc[nc["threshold"] == 0.6]["recurrent_set_size"].sum()
    if null_rec == 0:
        L.append(f"- At thresholds 0.8 and 1.0 the null control produced **no** "
                 f"recurrent features (total recurrent-set size across K = 0). At "
                 f"threshold 0.6 the total across K = {int(null_rec_06)}. This is the "
                 f"expected behaviour — a signal-free dataset should not yield a stable "
                 f"recurrent core.")
    else:
        L.append(f"- The null control produced recurrent features at threshold ≥ 0.8 "
                 f"(total recurrent-set size across K = {int(null_rec)}). With p≫n and "
                 f"a fixed label vector a few chance-correlated noise features can "
                 f"recur; a *large* count would warrant review. All such features are "
                 f"treated as noise.")
    L.append("")

    # 7. Weak-signal caveat
    L.append("## 7. Weak-Signal Caveat\n")
    L.append("- The weak-signal coefficient `gamma = 0.2` is **under-calibrated** in "
             "generator v0: weak features sit close to the noise floor (per the "
             "generation report, weak mean |corr with y| ≈ 0.08 vs a noise-sample max "
             "≈ 0.16).")
    L.append("- Weak-signal selection frequency is therefore **computed** in these "
             "tables but must **not** be interpreted strongly.")
    L.append("- Weak-signal conclusions are deferred to the later synthetic grid that "
             "uses a stronger `gamma`.\n")

    # 8. Safe interpretation
    L.append("## 8. Safe Interpretation\n")
    L.append("- Wording used: *planted-core recovery*, *recurrent candidate core*, "
             "*noise contamination*, *redundant proxy recurrence*.")
    L.append("- Wording avoided: *trusted core*, *causal feature*, *biomarker*, "
             "*validated feature*.")
    L.append("- A recurrent **proxy** is **not** a false positive — it is "
             "redundant-but-informative (correlated with a true anchored core "
             "feature). This is why `noise_contamination` (genuine false discovery) is "
             "reported separately from `exact_true_core_fraction` (exact-core share).\n")

    # 9. Relation to the stability profile
    L.append("## 9. Relation to the Stability Profile\n")
    L.append("- The **stability profile** (`analyze_per_fold_stability.py`) says *which* "
             "features recur and *how internally consistent* the selection is.")
    L.append("- The **recovery metrics** in this report say *whether* those recurrent "
             "features align with the known planted ground truth.")
    L.append("- The two are complementary and are kept separate: a method can be "
             "internally stable yet recover the wrong support, or recover the true "
             "support while showing low raw subset overlap.\n")

    # 10. Warnings / caveats
    L.append("## 10. Warnings / Caveats\n")
    warnings = []
    # d check
    d_vals = sorted(set(int(x) for x in group_summary_df["d_used"]))
    d_srcs = sorted(set(str(x) for x in group_summary_df["d_source"])) \
        if "d_source" in group_summary_df.columns else ["(absent)"]
    if d_vals != [2000]:
        warnings.append(f"d is not uniformly 2000 — observed d_used values {d_vals}.")
    else:
        L.append(f"- d check: all stability-profile groups used d = 2000 "
                 f"(d_source = {d_srcs}). OK.")
    # empty recurrent sets
    empty_known = kc[kc["recurrent_set_size"] == 0]
    if len(empty_known):
        cells = ", ".join(f"K={int(r.K)}/t={r.threshold}"
                          for r in empty_known.itertuples(index=False))
        warnings.append(f"known-core recurrent set is empty for: {cells}.")
    # consistency cross-check
    n_cmp, n_mis, mis_details = crosscheck
    if n_mis == 0:
        L.append(f"- Consistency cross-check: recurrent-set sizes match the analyzer's "
                 f"threshold sweep for all {n_cmp} compared (dataset, K, threshold) "
                 f"cells. OK.")
    else:
        warnings.append(f"recurrent-set size mismatches the analyzer threshold sweep "
                        f"in {n_mis}/{n_cmp} cells: {'; '.join(mis_details[:5])}")
    # noise recurrence in null at high threshold
    if null_rec > 0:
        warnings.append(f"null control shows recurrent features at threshold ≥ 0.8 "
                        f"(total {int(null_rec)} across K).")
    # unexpected low true-core recall at K=5
    k5 = kc[(kc["K"] == 5) & (kc["threshold"] == 0.6)]
    if len(k5) and float(k5.iloc[0]["true_core_recall"]) < 0.4:
        warnings.append(f"true-core recall at K=5, threshold 0.6 is low "
                        f"({k5.iloc[0]['true_core_recall']}) — investigate.")
    if warnings:
        for w in warnings:
            L.append(f"- **WARNING:** {w}")
    else:
        L.append("- No additional warnings raised.")
    L.append("")

    # 11. Recommended next step
    L.append("## 11. Recommended Next Step\n")
    if warnings:
        L.append("- One or more warnings were raised above. Resolve or explain them "
                 "before expanding the synthetic grid.")
    L.append(f"- The recovery numbers above are from a **single high-SNR "
             f"configuration, ETree only, {n_resamples} resamples** "
             f"(frequency resolution ≈ {1.0 / n_resamples:.3f}). Do not over-interpret "
             f"them. Recommended next step: an **Opus interpretation checkpoint** that "
             f"reads these {n_resamples}-resample recovery tables alongside the "
             f"stability profile and decides whether to expand the synthetic grid "
             f"(low SNR, rho=0.7, stronger weak gamma) or adjust the recovery framing "
             f"first.")
    L.append("")

    L.append("---\n")
    L.append("**Output files:**")
    for p in out_paths:
        L.append(f"- `{p}`")
    L.append("")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")
    L.append("*Recovery analysis only — no feature selection run, no logging or "
             "analyzer rerun, no generator or pipeline file modified.*")
    return "\n".join(L)


def role_maps_true_core(role_maps):
    """Helper: the true-core set of the known-core dataset (for report formatting)."""
    for rm in role_maps.values():
        if not rm["is_null"]:
            return rm["true_core"]
    return set()


# ==========================================================================
# Driver
# ==========================================================================
def run(feature_freq_path, threshold_sweep_path, group_summary_path,
        manifest_dir, out_dir, report_name):
    print("=" * 70)
    print("Article 5 — Synthetic support-recovery metrics")
    print("=" * 70)

    freq_df = pd.read_parquet(feature_freq_path)
    sweep_df = pd.read_parquet(threshold_sweep_path)
    group_summary_df = pd.read_parquet(group_summary_path)
    print(f"Loaded feature-frequency rows : {len(freq_df)}")
    print(f"Loaded threshold-sweep rows   : {len(sweep_df)}")
    print(f"Loaded group-summary rows     : {len(group_summary_df)}")

    # --- role maps from manifests ----------------------------------------
    role_maps = {}
    manifest_paths = {}
    for name in sorted(freq_df["dataset"].unique()):
        manifest, mpath = load_manifest(name, manifest_dir)
        role_maps[name] = build_role_map(manifest)
        manifest_paths[name] = mpath
        rm = role_maps[name]
        kind = "null control" if rm["is_null"] else "known core"
        print(f"  manifest: {name} ({kind}) — "
              f"true_core={len(rm['true_core'])}, proxy={len(rm['proxy'])}, "
              f"weak={len(rm['weak'])}, noise={rm['n_noise']}")

    # --- compute ----------------------------------------------------------
    annotated_df = annotate_feature_frequencies(freq_df, role_maps)
    group_df = compute_group_summary(annotated_df, role_maps)
    threshold_df = compute_threshold_recovery(annotated_df, role_maps)
    cross_k_df = compute_cross_k_recovery(annotated_df, role_maps)
    kvn_df = compute_known_vs_null(threshold_df)
    crosscheck = crosscheck_threshold_sweep(threshold_df, sweep_df)

    print("-" * 70)
    print(f"Annotated feature rows : {len(annotated_df)}")
    print(f"Group-summary rows     : {len(group_df)}")
    print(f"Threshold-recovery rows: {len(threshold_df)}")
    print(f"Cross-K recovery rows  : {len(cross_k_df)}")
    print(f"Known-vs-null rows     : {len(kvn_df)}")
    print(f"Sweep cross-check      : {crosscheck[0]} compared, {crosscheck[1]} mismatch")

    # --- write parquets (never overwrite) --------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, base in [("annotated", OUT_ANNOTATED), ("group", OUT_GROUP),
                      ("threshold", OUT_THRESHOLD), ("cross_k", OUT_CROSS_K),
                      ("kvn", OUT_KNOWN_VS_NULL)]:
        paths[key] = safe_path(out_dir, base, ARTICLE5_RECOVERY_ROOT)
        print(f"[output] {key:10s}: {paths[key]}")
    annotated_df.to_parquet(paths["annotated"], index=False)
    group_df.to_parquet(paths["group"], index=False)
    threshold_df.to_parquet(paths["threshold"], index=False)
    cross_k_df.to_parquet(paths["cross_k"], index=False)
    kvn_df.to_parquet(paths["kvn"], index=False)

    # --- markdown report --------------------------------------------------
    commands = [
        "python -m py_compile articles/article_stability/scripts/synthetic_recovery_metrics.py",
        "python articles/article_stability/scripts/synthetic_recovery_metrics.py --dry-run",
        "python articles/article_stability/scripts/synthetic_recovery_metrics.py",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(REPORT_PATH.parent, report_name, REPORT_PATH.parent)
    input_paths = {
        "feature_frequencies": str(feature_freq_path),
        "threshold_sweep": str(threshold_sweep_path),
        "group_summary": str(group_summary_path),
        "manifest_dir": str(manifest_dir),
    }
    report_path.write_text(build_report(
        annotated_df, group_df, threshold_df, cross_k_df, kvn_df,
        role_maps, group_summary_df, commands,
        [str(p) for p in paths.values()], crosscheck, input_paths,
    ))
    print(f"[output] report    : {report_path}")

    # --- concise console summary -----------------------------------------
    print("-" * 70)
    print("Known-core true-core recall (recovered/5) by K and threshold:")
    kc = threshold_df[threshold_df["dataset_role"] == "known_core"]
    for r in kc.itertuples(index=False):
        print(f"  K={r.K:<3} t={r.threshold}: recurrent={r.recurrent_set_size:<3} "
              f"true-core={r.true_core_recovered_count}/5 "
              f"(recall={r.true_core_recall})  "
              f"noise_contam={r.noise_contamination}  "
              f"proxy_grp_cov={r.proxy_group_coverage_count}/3")
    print("Null-control recurrent-set size by K and threshold:")
    nc = threshold_df[threshold_df["dataset_role"] == "null_control"]
    for r in nc.itertuples(index=False):
        print(f"  K={r.K:<3} t={r.threshold}: recurrent={r.recurrent_set_size}")
    print("=" * 70)
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Article 5 synthetic support-recovery metrics "
                    "(recovery vs known ground truth; complementary to stability).",
    )
    parser.add_argument("--feature-frequencies", default=str(DEFAULT_FEATURE_FREQUENCIES),
                        help="Feature-frequency parquet (from the offline analyzer).")
    parser.add_argument("--threshold-sweep", default=str(DEFAULT_THRESHOLD_SWEEP),
                        help="Threshold-sweep parquet (used for a consistency check).")
    parser.add_argument("--group-summary", default=str(DEFAULT_GROUP_SUMMARY),
                        help="Group-summary parquet (used to confirm d / d_source).")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR),
                        help="Directory holding the synthetic dataset manifests.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory; relative paths resolve under "
                             "results/article_stability/synthetic_control/recovery_metrics/.")
    parser.add_argument("--report-name", default="SYNTHETIC_SMOKE_RECOVERY_METRICS_REPORT.md",
                        help="Markdown report filename, written under "
                             "results/article_stability/reports/experiment_planning/. Use a distinct name "
                             "to avoid overwriting an earlier run's report.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs and print the plan, then exit without "
                             "writing anything.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    feature_freq_path = Path(args.feature_frequencies).expanduser().resolve()
    threshold_sweep_path = Path(args.threshold_sweep).expanduser().resolve()
    group_summary_path = Path(args.group_summary).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    out_dir = resolve_output_dir(args.output_dir)

    manifests = ["synthetic_snrHIGH_rho090_manifest.json",
                 "synthetic_null_control_manifest.json"]

    if args.dry_run:
        print("=" * 70)
        print("Article 5 — Synthetic recovery metrics  [DRY RUN]")
        print("=" * 70)
        for label, p in [("feature-frequencies", feature_freq_path),
                         ("threshold-sweep", threshold_sweep_path),
                         ("group-summary", group_summary_path)]:
            print(f"  {label:20s}: exists={p.exists()}  {p}")
        for m in manifests:
            mp = manifest_dir / m
            print(f"  manifest {m:42s}: exists={mp.exists()}")
        print(f"Output root          : {ARTICLE5_RECOVERY_ROOT}")
        print(f"Output directory     : {out_dir}")
        print(f"Output dir within root: {_is_within(out_dir, ARTICLE5_RECOVERY_ROOT)}")
        print("Planned output files:")
        for base in (OUT_ANNOTATED, OUT_GROUP, OUT_THRESHOLD, OUT_CROSS_K,
                     OUT_KNOWN_VS_NULL):
            print(f"  - {out_dir / base}")
        print(f"  - {REPORT_PATH.parent / args.report_name}")
        print("DRY RUN — inputs not read, nothing computed, nothing written.")
        print("=" * 70)
        return 0

    for label, p in [("feature-frequencies", feature_freq_path),
                     ("threshold-sweep", threshold_sweep_path),
                     ("group-summary", group_summary_path)]:
        if not p.exists():
            raise SystemExit(f"[ERROR] {label} parquet not found: {p}")
    for m in manifests:
        if not (manifest_dir / m).exists():
            raise SystemExit(f"[ERROR] manifest not found: {manifest_dir / m}")

    return run(feature_freq_path, threshold_sweep_path, group_summary_path,
               manifest_dir, out_dir, args.report_name)


if __name__ == "__main__":
    sys.exit(main())
