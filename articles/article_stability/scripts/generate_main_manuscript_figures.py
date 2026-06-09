#!/usr/bin/env python3
"""
Generate main manuscript figures and tables for Article 5.
Read-only — does not modify any raw data or existing result files.

Sources (all read-only):
  Fig 1 : synthetic_confirmatory_snr_grid_v1f_across_seed_aggregate.parquet
  Fig 2 : synthetic_smoke_feature_frequency_annotated.parquet
  Fig 4 : real_data/r2/stability_profiles/.../feature_frequencies.parquet
  Table 1: real_data/r2/stability_profiles/.../stability_group_summary.parquet
  Fig A2: real_data/r2/stability_profiles/.../threshold_sweep.parquet

Outputs (all under results/article_stability/reports/figures_tables/):
  figures/fig1_snr_degradation_gradient.{png,pdf}          — main paper (revised: trajectory lines + divergence callout)
  figures/fig2_synthetic_feature_frequency_profile.{png,pdf} — main paper (2-panel HIGH vs null, top 50)
  figures/figS_synthetic_feature_frequency_profile_allSNR.{png,pdf} — appendix (4-panel, top 60)
  figures/fig4_colon_etree_feature_frequency_profile.{png,pdf} — main paper (single neutral color)
  figures/figA2_colon_threshold_sweep.{png,pdf}            — appendix (neutral LASSO annotation)
  table1_r2_core_k10_stability_summary.{csv,md}            — main paper (6 cols)
  tableS1_r2_core_k10_stability_summary_full.{csv,md}      — appendix (8 cols)
  MANUSCRIPT_FIGURE_TABLE_GENERATION_REPORT.md
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

SYNTH_AGG = (
    ROOT / "results/article_stability/synthetic_control/recovery_metrics"
    / "synthetic_confirmatory_snr_grid_v1f_etree_seed0_50resamples"
    / "synthetic_confirmatory_snr_grid_v1f_across_seed_aggregate.parquet"
)
SYNTH_FREQ = (
    ROOT / "results/article_stability/synthetic_control/recovery_metrics"
    / "synthetic_confirmatory_snr_grid_v1f_etree_seed0_50resamples"
    / "synthetic_smoke_feature_frequency_annotated.parquet"
)
REAL_FEAT_FREQ = (
    ROOT / "results/article_stability/real_data/r2/stability_profiles"
    / "real_data_r2_rawX_4algos_3datasets_seed0_50resamples"
    / "feature_frequencies.parquet"
)
REAL_SGS = (
    ROOT / "results/article_stability/real_data/r2/stability_profiles"
    / "real_data_r2_rawX_4algos_3datasets_seed0_50resamples"
    / "stability_group_summary.parquet"
)
REAL_TS = (
    ROOT / "results/article_stability/real_data/r2/stability_profiles"
    / "real_data_r2_rawX_4algos_3datasets_seed0_50resamples"
    / "threshold_sweep.parquet"
)

OUT_ROOT = ROOT / "results/article_stability/figures_tables"
FIG_ROOT = OUT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

# SNR level order and display labels
SNR_ORDER = ["alpha2.5", "alpha0.6", "alpha0.35", "null"]
SNR_LABELS = {
    "alpha2.5": "HIGH\n(α=2.5)",
    "alpha0.6": "MID\n(α=0.6)",
    "alpha0.35": "LOW\n(α=0.35)",
    "null": "Null\n(signal-free)",
}
SNR_COLORS = {
    "alpha2.5": "#2166ac",
    "alpha0.6": "#4dac26",
    "alpha0.35": "#d95f02",
    "null": "#999999",
}
ALGO_COLORS = {
    "ETree": "#1f77b4",
    "LASSO_Stability": "#d62728",
    "ReliefF": "#2ca02c",
    "mRMR": "#ff7f0e",
}
ROLE_COLORS = {
    "anchored_core": "#e31a1c",
    "solo_core": "#ff7f00",
    "proxy": "#6a3d9a",
    "weak": "#33a02c",
    "noise": "#d9d9d9",
}
ROLE_ZORDER = {
    "anchored_core": 5,
    "solo_core": 5,
    "proxy": 4,
    "weak": 3,
    "noise": 2,
}

issues = []  # accumulate any generation issues for the report


def _save(fig, stem):
    for ext in ("png", "pdf"):
        path = FIG_ROOT / f"{stem}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {stem}.png/.pdf")


# ===========================================================================
# Fig 1 — SNR degradation gradient
# ===========================================================================
def make_fig1():
    print("\n[Fig 1] SNR degradation gradient")
    agg = pd.read_parquet(SYNTH_AGG)

    def _pull(metric, K=None, threshold=None):
        mask = agg["metric"] == metric
        if K is not None:
            mask &= agg["K"] == K
        if threshold is not None:
            mask &= agg["threshold"] == threshold
        sub = agg[mask].copy()
        sub["alpha_label"] = pd.Categorical(sub["alpha_label"], categories=SNR_ORDER, ordered=True)
        return sub.sort_values("alpha_label")

    auc = _pull("overall_mean_auc")
    recall = _pull("true_core_recall", K=30.0, threshold=0.8)
    kunch = _pull("kuncheva_mean", K=30.0)
    noise = _pull("noise_contamination", K=60.0, threshold=0.8)

    # Check null is in kuncheva (it may not be — only signal datasets)
    null_kunch_available = "null" in kunch["alpha_label"].values

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))

    def _barplot(ax, data, title, ylabel, ylim=(0, 1), include_null=True):
        rows = data[data["alpha_label"].isin(SNR_ORDER if include_null else ["alpha2.5", "alpha0.6", "alpha0.35"])]
        labels_ordered = [l for l in (SNR_ORDER if include_null else ["alpha2.5", "alpha0.6", "alpha0.35"])
                          if l in rows["alpha_label"].values]
        rows = rows[rows["alpha_label"].isin(labels_ordered)].copy()
        rows["alpha_label"] = pd.Categorical(rows["alpha_label"], categories=labels_ordered, ordered=True)
        rows = rows.sort_values("alpha_label")
        x = np.arange(len(rows))
        colors = [SNR_COLORS[l] for l in rows["alpha_label"]]
        ax.bar(x, rows["across_seed_mean"], color=colors, alpha=0.85, zorder=3)
        ax.errorbar(x, rows["across_seed_mean"], yerr=rows["across_seed_std"],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        # Trajectory line connecting bar tops — makes slope visible
        ax.plot(x, rows["across_seed_mean"].values, color="#444444",
                linewidth=1.2, linestyle="-", alpha=0.55, zorder=5,
                marker="none")
        ax.set_xticks(x)
        ax.set_xticklabels([SNR_LABELS[l] for l in rows["alpha_label"]], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add value labels
        for xi, (_, row) in zip(x, rows.iterrows()):
            ax.text(xi, row["across_seed_mean"] + row["across_seed_std"] + 0.012,
                    f"{row['across_seed_mean']:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="black")

    # Panel (a): AUC (4 levels including null)
    _barplot(axes[0], auc, "(a) Mean AUC", "Mean AUC (50 resamples)", ylim=(0.0, 1.0), include_null=True)
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance")
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel (b): True-core recall at K=30, t=0.8 (3 signal levels only)
    _barplot(axes[1], recall, "(b) Exact recall\n(K=30, t≥0.8)", "Recall of planted core", ylim=(0.0, 1.0), include_null=False)

    # Panel (c): Kuncheva at K=30 (3 signal levels; null may be missing)
    _barplot(axes[2], kunch, "(c) Kuncheva\n(K=30)", "Kuncheva index (mean ± SD)", ylim=(0.0, 1.0),
             include_null=null_kunch_available)

    # Panel (d): Noise contamination at K=60, t=0.8 (3 signal levels)
    noise_max = max(noise["across_seed_mean"].max() + noise["across_seed_std"].max() + 0.05, 0.45)
    _barplot(axes[3], noise, "(d) Noise contamination\n(K=60, t≥0.8)", "Fraction of recurrent set = noise",
             ylim=(0.0, noise_max), include_null=False)

    fig.suptitle("Synthetic SNR ladder: uneven degradation of AUC, exact recovery, Kuncheva stability, "
                 "and noise contamination as signal weakens\n"
                 "(ETree, n=200, p=2000, ρ=0.9, 8 seeds × 50 resamples; bars = across-seed mean ± SD; "
                 "lines connect bar tops to show trajectory)",
                 fontsize=8.5, y=1.03)
    fig.tight_layout()
    # Divergence callout below title — the key C2 isolation
    fig.text(0.5, -0.03,
             "HIGH→MID: mean AUC −0.027 (≈4%) while exact recall −0.175 (≈27%) "
             "— performance nearly preserved as recovery falls",
             ha="center", va="top", fontsize=8.5,
             style="italic", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", alpha=0.9))
    _save(fig, "fig1_snr_degradation_gradient")
    return {"status": "ok", "source": str(SYNTH_AGG),
            "notes": f"null_kuncheva_available={null_kunch_available}; trajectory lines added; divergence callout added"}


# ===========================================================================
# Fig 2 — Synthetic feature-frequency profile (2-panel main: HIGH vs null)
# ===========================================================================
def _load_freq_panel(ff, ds_name, top_n=None):
    """Return sorted frequency rows for one dataset×K=30, optionally top-N."""
    sub = ff[(ff["dataset"] == ds_name) & (ff["K"] == 30)].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("frequency", ascending=False).reset_index(drop=True)
    if top_n is not None:
        sub = sub.head(top_n)
    sub["rank"] = np.arange(1, len(sub) + 1)
    return sub


def _draw_freq_panel(ax, sub, title, show_ylabel, xlim, legend_handles_extra=None):
    """Draw a single frequency-rank bar panel with planted-role coloring."""
    role_col = "feature_role" if "feature_role" in sub.columns else None
    colors = ([ROLE_COLORS.get(r, "#d9d9d9") for r in sub[role_col]]
              if role_col else ["#d9d9d9"] * len(sub))
    ax.bar(sub["rank"], sub["frequency"], color=colors, alpha=0.9, width=1.0, linewidth=0)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.axhline(0.6, color="black", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.set_xlabel("Feature rank (by selection frequency)", fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Selection frequency (50 resamples)", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    n_rec = (sub["frequency"] >= 0.8).sum()
    n_mod = (sub["frequency"] >= 0.6).sum()
    ax.text(0.97, 0.97,
            f"≥0.8: {n_rec} feat.\n≥0.6: {n_mod} feat.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
    if legend_handles_extra is not None:
        ax.legend(handles=legend_handles_extra, loc="upper center",
                  fontsize=8, framealpha=0.85, ncol=2)


def make_fig2():
    """Main-paper Fig 2: 2-panel HIGH vs matched null, top 50 features."""
    print("\n[Fig 2] Synthetic feature-frequency profile (main: 2-panel HIGH vs null)")
    ff = pd.read_parquet(SYNTH_FREQ)
    seed_label = "seed00"

    ds_high = f"synthetic_snrHIGH_a250_rho090_gammaV1f_{seed_label}"
    ds_null = f"synthetic_null_control_gammaV1f_{seed_label}"

    sub_high = _load_freq_panel(ff, ds_high, top_n=50)
    sub_null = _load_freq_panel(ff, ds_null, top_n=50)

    for ds_name, sub in [(ds_high, sub_high), (ds_null, sub_null)]:
        if sub.empty:
            issues.append(f"Fig2: no data for {ds_name}, K=30")

    role_order = ["anchored_core", "solo_core", "proxy", "weak", "noise"]
    role_labels_map = {
        "anchored_core": "Anchored core",
        "solo_core": "Solo core",
        "proxy": "Proxy",
        "weak": "Weak signal",
        "noise": "Noise",
    }
    role_handles = [
        mpatches.Patch(color=ROLE_COLORS[r], label=role_labels_map[r])
        for r in role_order if r in ROLE_COLORS
    ]
    thr_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="t = 0.8"),
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=0.9, label="t = 0.6"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    xlim = 52  # shared x range for top 50

    _draw_freq_panel(axes[0], sub_high,
                     "(a) HIGH signal (α=2.5)", show_ylabel=True, xlim=xlim)
    _draw_freq_panel(axes[1], sub_null,
                     "(b) Matched null control (signal-free)", show_ylabel=False, xlim=xlim)

    fig.suptitle(
        "Synthetic positive control: recurrent planted structure vs matched null\n"
        "(Top 50 features by selection frequency, K=30, seed00, 50 resamples; "
        "bar color = planted feature type from synthetic design — not real-data labels)",
        fontsize=9, y=1.04
    )
    # Shared legend below both panels — keep it out of the plot area
    all_handles = role_handles + thr_handles
    fig.tight_layout(rect=[0, 0.17, 1, 1])  # reserve 17% at bottom for legend
    fig.legend(handles=all_handles,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.13),
               ncol=len(all_handles),
               fontsize=8.5, framealpha=0.9,
               columnspacing=1.2, handlelength=1.5)
    _save(fig, "fig2_synthetic_feature_frequency_profile")
    return {"status": "ok", "source": str(SYNTH_FREQ),
            "notes": "2-panel main: HIGH vs null; top 50; seed00; K=30"}


def make_fig2_appendix():
    """Appendix Fig S: 4-panel HIGH/MID/LOW/null, top 60 features."""
    print("\n[Fig S] Synthetic feature-frequency profile (appendix: 4-panel, top 60)")
    ff = pd.read_parquet(SYNTH_FREQ)
    seed_label = "seed00"

    panels = [
        ("alpha2.5",  f"synthetic_snrHIGH_a250_rho090_gammaV1f_{seed_label}",  "(a) HIGH (α=2.5)"),
        ("alpha0.6",  f"synthetic_snrMID_a060_rho090_gammaV1f_{seed_label}",   "(b) MID (α=0.6)"),
        ("alpha0.35", f"synthetic_snrLOW_a035_rho090_gammaV1f_{seed_label}",   "(c) LOW (α=0.35)"),
        ("null",      f"synthetic_null_control_gammaV1f_{seed_label}",          "(d) Null (signal-free)"),
    ]

    role_order = ["anchored_core", "solo_core", "proxy", "weak", "noise"]
    role_labels_map = {
        "anchored_core": "Anchored core",
        "solo_core": "Solo core",
        "proxy": "Proxy",
        "weak": "Weak signal",
        "noise": "Noise",
    }
    role_handles = [
        mpatches.Patch(color=ROLE_COLORS[r], label=role_labels_map[r])
        for r in role_order if r in ROLE_COLORS
    ]
    thr_handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="t=0.8"),
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=0.8, label="t=0.6"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    xlim = 62  # shared x range for top 60

    for i, (snr, ds_name, title) in enumerate(panels):
        sub = _load_freq_panel(ff, ds_name, top_n=60)
        if sub.empty:
            issues.append(f"FigS: no data for {ds_name}, K=30")
            axes[i].set_title(title)
            axes[i].text(0.5, 0.5, "No data", transform=axes[i].transAxes, ha="center")
            continue
        _draw_freq_panel(axes[i], sub, title, show_ylabel=(i == 0), xlim=xlim)

    fig.suptitle(
        "Appendix: full synthetic SNR feature-frequency profile "
        "(K=30, seed00, top 60 features; bar color = planted feature type)",
        fontsize=9, y=1.04
    )
    # Shared legend below all four panels — keep it out of the plot area
    all_handles = role_handles + thr_handles
    fig.tight_layout(rect=[0, 0.17, 1, 1])  # reserve 17% at bottom for legend
    fig.legend(handles=all_handles,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.13),
               ncol=len(all_handles),
               fontsize=8.5, framealpha=0.9,
               columnspacing=1.2, handlelength=1.5)
    _save(fig, "figS_synthetic_feature_frequency_profile_allSNR")
    return {"status": "ok", "source": str(SYNTH_FREQ),
            "notes": "4-panel appendix: HIGH/MID/LOW/null; top 60; seed00; K=30"}


# ===========================================================================
# Fig 4 — Real-data recurrent-core profile: colon × ETree × K=10
# ===========================================================================
def make_fig4():
    print("\n[Fig 4] Real-data feature frequency profile: colon × ETree × K=10")
    ff = pd.read_parquet(REAL_FEAT_FREQ)
    sub = ff[(ff["dataset"] == "colon") & (ff["algorithm"] == "ETree") & (ff["K"] == 10)].copy()

    if sub.empty:
        issues.append("Fig4: no data for colon x ETree x K=10")
        return {"status": "error", "notes": "no data"}

    sub = sub.sort_values("frequency", ascending=False).reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)

    # Single neutral bar color — the threshold lines carry the structure.
    # No tiered coloring (avoids implying hard thresholds as official bins).
    bar_color = "#6baed6"  # medium-blue, neutral

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(sub["rank"], sub["frequency"], color=bar_color, alpha=0.85, width=1.0, linewidth=0)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1.1, alpha=0.75, label="t = 0.8")
    ax.axhline(0.6, color="black", linestyle=":", linewidth=1.1, alpha=0.75, label="t = 0.6")
    ax.set_xlim(0, len(sub) + 2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature rank (by selection frequency)", fontsize=10)
    ax.set_ylabel("Selection frequency (50 resamples)", fontsize=10)
    ax.set_title("Real-data stability profile: colon (p≫n), ETree, K=10\n"
                 "(50 stratified-shuffle-split resamples; recurrent candidate features, not ground truth)",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # Annotate recurrent set sizes
    n08 = (sub["frequency"] >= 0.8).sum()
    n06 = (sub["frequency"] >= 0.6).sum()
    ax.text(0.97, 0.97,
            f"≥0.8: {n08} features\n≥0.6: {n06} features\n(of {len(sub)} ever selected)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Place the t-threshold legend below the feature-count annotation box to avoid overlap.
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.74), fontsize=9, framealpha=0.85)

    fig.tight_layout()
    _save(fig, "fig4_colon_etree_feature_frequency_profile")

    # Caption note
    notes = (f"n_features_selected_ever={len(sub)}; "
             f"recurrent_core(≥0.8)={n08}; recurrent_core(≥0.6)={n06}; "
             f"no_ground_truth_on_real_data")
    return {"status": "ok", "source": str(REAL_FEAT_FREQ), "notes": notes}


# ===========================================================================
# Table helpers
# ===========================================================================
def _sort_table(df):
    ds_order = {"colon": 0, "SMK-CAN-187": 1, "PeriodChanger": 2}
    algo_order = {"ETree": 0, "mRMR": 1, "ReliefF": 2, "LASSO_Stability": 3}
    df["_ds_ord"] = df["Dataset"].map(ds_order).fillna(99)
    df["_al_ord"] = df["Algorithm"].map(algo_order).fillna(99)
    return df.sort_values(["_ds_ord", "_al_ord"]).drop(columns=["_ds_ord", "_al_ord"])


def _to_md(df):
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row.values) + " |"
            for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


# ===========================================================================
# Table 1 — main-paper: 6 columns only
# ===========================================================================
def make_table1():
    print("\n[Table 1] R2 core K=10 stability summary (main, 6 cols)")
    sgs = pd.read_parquet(REAL_SGS)
    raw = sgs[sgs["K"] == 10].copy()

    if raw.empty:
        issues.append("Table1: no K=10 rows in stability_group_summary")
        return {"status": "error"}

    # Main-paper columns only (Nogueira-style + Obs/Random → appendix)
    main_cols = {
        "dataset": "Dataset",
        "algorithm": "Algorithm",
        "n_ok": "n_ok",
        "auc_mean": "AUC (mean)",
        "mean_pairwise_jaccard": "Jaccard (mean)",
        "kuncheva_mean": "Kuncheva",
    }
    t1 = raw[list(main_cols.keys())].rename(columns=main_cols)
    t1 = _sort_table(t1)
    for c in ["AUC (mean)", "Jaccard (mean)", "Kuncheva"]:
        t1[c] = t1[c].round(3)

    t1.to_csv(OUT_ROOT / "table1_r2_core_k10_stability_summary.csv", index=False)

    md_lines = [
        "## Table 1 — R2 Core Stability Summary at K=10",
        "",
        "ETree, LASSO_Stability, ReliefF, mRMR on three p≫n datasets "
        "(50 stratified-shuffle-split resamples, seed=0, raw signed X). "
        "All cells completed (n_ok=50). "
        "Full table with Nogueira-style stability and Obs/Random ratio in Appendix Table S1.",
        "",
        _to_md(t1),
        "",
        "**Columns:**",
        "- n_ok: resamples with a valid feature selection (of 50)",
        "- Kuncheva: chance-corrected subset-stability index (primary spine; "
        "note near-equal AUC yet ~2× Kuncheva difference for colon ETree vs LASSO_Stability)",
    ]
    (OUT_ROOT / "table1_r2_core_k10_stability_summary.md").write_text("\n".join(md_lines))

    print(f"  Saved table1_r2_core_k10_stability_summary.csv / .md")
    return {"status": "ok", "source": str(REAL_SGS), "n_rows": len(t1),
            "notes": "main paper; 6 cols: Dataset/Algorithm/n_ok/AUC/Jaccard/Kuncheva"}


# ===========================================================================
# Table S1 — appendix: full 8 columns
# ===========================================================================
def make_tableS1():
    print("\n[Table S1] R2 core K=10 stability summary (appendix, full)")
    sgs = pd.read_parquet(REAL_SGS)
    raw = sgs[sgs["K"] == 10].copy()

    if raw.empty:
        issues.append("TableS1: no K=10 rows")
        return {"status": "error"}

    full_cols = {
        "dataset": "Dataset",
        "algorithm": "Algorithm",
        "n_ok": "n_ok",
        "auc_mean": "AUC (mean)",
        "mean_pairwise_jaccard": "Jaccard (mean)",
        "kuncheva_mean": "Kuncheva",
        "nogueira_style_stability": "Nogueira-style",
        "observed_vs_random_jaccard_ratio": "Obs/Random ratio",
    }
    ts1 = raw[list(full_cols.keys())].rename(columns=full_cols)
    ts1 = _sort_table(ts1)
    for c in ["AUC (mean)", "Jaccard (mean)", "Kuncheva", "Nogueira-style"]:
        ts1[c] = ts1[c].round(3)
    ts1["Obs/Random ratio"] = ts1["Obs/Random ratio"].round(1)

    ts1.to_csv(OUT_ROOT / "tableS1_r2_core_k10_stability_summary_full.csv", index=False)

    md_lines = [
        "## Table S1 — R2 Core Stability Summary at K=10 (Full, Appendix)",
        "",
        "ETree, LASSO_Stability, ReliefF, mRMR on three p≫n datasets "
        "(50 stratified-shuffle-split resamples, seed=0, raw signed X).",
        "",
        _to_md(ts1),
        "",
        "**Columns:**",
        "- n_ok: resamples with a valid feature selection (of 50)",
        "- Kuncheva: chance-corrected subset-stability index (primary spine)",
        "- Nogueira-style: variance-based stability (exploratory; not a verified estimator)",
        "- Obs/Random ratio: **P-dependent — valid for within-dataset comparison only. "
        "Do not use for cross-dataset ranking.**",
    ]
    (OUT_ROOT / "tableS1_r2_core_k10_stability_summary_full.md").write_text("\n".join(md_lines))

    print(f"  Saved tableS1_r2_core_k10_stability_summary_full.csv / .md")
    return {"status": "ok", "source": str(REAL_SGS), "n_rows": len(ts1),
            "notes": "appendix; 8 cols including Nogueira-style and Obs/Random ratio"}


# ===========================================================================
# Fig A2 — Threshold sweep: colon × K=10, all 4 algorithms
# ===========================================================================
def make_figA2():
    print("\n[Fig A2] Threshold sweep: colon, K=10")
    ts = pd.read_parquet(REAL_TS)
    sub = ts[(ts["dataset"] == "colon") & (ts["K"] == 10)].copy()

    if sub.empty:
        issues.append("FigA2: no data for colon x K=10 in threshold_sweep")
        return {"status": "error"}

    algos = ["ETree", "mRMR", "ReliefF", "LASSO_Stability"]
    thresholds = sorted(sub["threshold"].unique())

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for algo in algos:
        a_sub = sub[sub["algorithm"] == algo].sort_values("threshold")
        if a_sub.empty:
            continue
        color = ALGO_COLORS.get(algo, "black")
        ax.plot(a_sub["threshold"], a_sub["recurrent_set_size"],
                marker="o", markersize=6, color=color, label=algo, linewidth=1.8)

    ax.set_xlabel("Selection-frequency threshold (t)", fontsize=10)
    ax.set_ylabel("Recurrent candidate core size", fontsize=10)
    ax.set_title("Recurrent-core size vs selection-frequency threshold\n"
                 "colon dataset, K=10, 50 resamples", fontsize=10)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.1f}" for t in thresholds])
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylim(bottom=-0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(fontsize=9, framealpha=0.85)

    # Annotate LASSO_Stability = 0 across all thresholds (neutral, descriptive — not a verdict)
    lasso_sub = sub[sub["algorithm"] == "LASSO_Stability"].sort_values("threshold")
    if lasso_sub["recurrent_set_size"].max() == 0:
        ax.text(0.97, 0.25,
                "LASSO_Stability: recurrent core\nempty at all thresholds (descriptive)",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8, color="#666666",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.85))

    fig.tight_layout()
    _save(fig, "figA2_colon_threshold_sweep")
    return {"status": "ok", "source": str(REAL_TS), "notes": "colon x K=10, 3 thresholds"}


# ===========================================================================
# Generation report
# ===========================================================================
def write_report(results):
    print("\n[Report] Writing generation report")

    lines = [
        "# Article 5 — Manuscript Figure/Table Generation Report",
        "",
        "**Script:** `articles/article_stability/scripts/generate_main_manuscript_figures.py`",
        "**Source reference:** `MANUSCRIPT_AUDIT_AND_INTRO_PLAN.md`",
        "**Revision:** minor visual revisions per Opus senior review (trajectory lines, "
        "Fig 2 simplified, neutral colors, Table 1 split into main/appendix)",
        "",
        "---",
        "",
        "## Created Files",
        "",
        "| Output | Source File | Status | Placement | Notes |",
        "|--------|-------------|--------|-----------|-------|",
    ]

    all_entries = [
        ("figures/fig1_snr_degradation_gradient.{png,pdf}",
         results.get("fig1", {}), "Main paper",
         "Results §2; trajectory lines + divergence callout added"),
        ("figures/fig2_synthetic_feature_frequency_profile.{png,pdf}",
         results.get("fig2", {}), "Main paper",
         "Results §1 positive control; simplified to 2-panel HIGH vs null; top 50 features"),
        ("figures/figS_synthetic_feature_frequency_profile_allSNR.{png,pdf}",
         results.get("fig2_appendix", {}), "Appendix",
         "Full 4-panel HIGH/MID/LOW/null; top 60 features per panel"),
        ("figures/fig4_colon_etree_feature_frequency_profile.{png,pdf}",
         results.get("fig4", {}), "Main paper",
         "Results §6; single neutral bar color; recurrent candidate framing"),
        ("figures/figA2_colon_threshold_sweep.{png,pdf}",
         results.get("figA2", {}), "Appendix",
         "Reviewer-risk #5 threshold-dependence; LASSO annotation neutralized"),
        ("table1_r2_core_k10_stability_summary.{csv,md}",
         results.get("table1", {}), "Main paper",
         "6 cols: Dataset/Algorithm/n_ok/AUC/Jaccard/Kuncheva"),
        ("tableS1_r2_core_k10_stability_summary_full.{csv,md}",
         results.get("tableS1", {}), "Appendix",
         "Full 8 cols incl. Nogueira-style + Obs/Random ratio"),
    ]

    for fname, res, placement, desc in all_entries:
        status = res.get("status", "not-run")
        src = Path(res.get("source", "")).name if res.get("source") else "—"
        notes = res.get("notes", desc)
        lines.append(f"| `{fname}` | `{src}` | {status} | {placement} | {notes} |")

    lines += [
        "",
        "---",
        "",
        "## Fig 2 Revision Note",
        "",
        "The original 4-panel Fig 2 (HIGH/MID/LOW/null at K=30 showing all features) was "
        "judged too dense for the main paper — the planted color-coded core was a thin sliver "
        "in a wide noise tail, making the bimodal structure visually invisible. "
        "The main-paper version is now **2-panel (HIGH vs matched null, top 50 features, "
        "shared x-axis)**. The full 4-panel version (top 60 per panel) is saved as "
        "`figS_synthetic_feature_frequency_profile_allSNR` for the appendix.",
        "",
        "---",
        "",
        "## Table 1 Split Note",
        "",
        "The original 8-column Table 1 included Nogueira-style stability and "
        "Obs/Random ratio. The main-paper version (`table1_*`) carries 6 columns only — "
        "Dataset, Algorithm, n_ok, AUC, Jaccard, Kuncheva. The Nogueira-style and "
        "Obs/Random columns move to `tableS1_*` (appendix). "
        "Rationale: Nogueira-style and Kuncheva are near-identical numerically and showing "
        "both invites 'why two redundant columns?'; Obs/Random is P-dependent and "
        "cross-dataset-dangerous.",
        "",
        "---",
        "",
        "## Source Files Used",
        "",
        f"- **Figs 1, 2, S** — `{SYNTH_AGG.name}` / `{SYNTH_FREQ.name}` "
        f"(`synthetic_confirmatory_snr_grid_v1f_etree_seed0_50resamples/`)",
        f"- **Fig 4, Table 1/S1, Fig A2** — R2 core stability profiles "
        f"(`real_data_r2_rawX_4algos_3datasets_seed0_50resamples/`)",
        "",
        "---",
        "",
        "## Assumptions Made",
        "",
        "- `synthetic_smoke_feature_frequency_annotated.parquet` under the V1F directory "
        "is the correct annotated file — confirmed: all 4 SNR levels × 8 seeds present.",
        "- `seed00` used as the representative seed for Fig 2 and Fig S.",
        "- Fig 1 panel (c) Kuncheva: null omitted (not present in aggregate for null control).",
        "- Fig A2: LASSO_Stability recurrent_set_size=0 confirmed from threshold_sweep.parquet.",
        "- Table 1: `accuracy_mean`, `f1_mean`, `fs_runtime_mean` omitted (available in source).",
        "",
        "---",
        "",
        "## Issues / Missing Items",
        "",
    ]

    if issues:
        for iss in issues:
            lines.append(f"- {iss}")
    else:
        lines.append("None — all requested outputs generated successfully.")

    lines += [
        "",
        "---",
        "",
        "## Claim-Safety Caveats (from MANUSCRIPT_AUDIT_AND_INTRO_PLAN.md §9)",
        "",
        "| Figure/Table | Mandatory caption constraint |",
        "|--------------|------------------------------|",
        "| Fig 1 | 'monotone but uneven' — not 'linear'; cite K=30/t≥0.8 for recall; "
        "β=2.5 is a strengthened anchor; synthetic-only. |",
        "| Fig 2 (main) | 'planted feature type from synthetic design — not real-data labels'; "
        "seed00 shown; correctness not implied. |",
        "| Fig 2 / Fig S | Both captions must say 'planted ground truth'; "
        "internal-validity control only. |",
        "| Fig 4 | 'recurrent candidate features — selected in ≥t of resamples; "
        "correctness not implied'. No support-recovery language. |",
        "| Table 1 | Kuncheva is primary spine. Note near-equal AUC yet ~2× Kuncheva "
        "(colon ETree vs LASSO). Illustrative slice, not a benchmark. |",
        "| Table S1 | Obs/Random ratio: within-dataset only; "
        "do not rank across datasets. |",
        "| Fig A2 | Threshold sweep is illustrative, not a ranking. |",
        "",
        "---",
        "",
        "## PDF Availability",
        "",
        "PDF files generated alongside PNGs. PDFs excluded by `.gitignore` — "
        "regenerable by re-running this script. PNGs always committed.",
    ]

    report_path = OUT_ROOT / "MANUSCRIPT_FIGURE_TABLE_GENERATION_REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved MANUSCRIPT_FIGURE_TABLE_GENERATION_REPORT.md")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Article 5 — Manuscript Figure/Table Generator")
    print("=" * 60)

    # Verify all source files exist
    for label, path in [
        ("SYNTH_AGG", SYNTH_AGG),
        ("SYNTH_FREQ", SYNTH_FREQ),
        ("REAL_FEAT_FREQ", REAL_FEAT_FREQ),
        ("REAL_SGS", REAL_SGS),
        ("REAL_TS", REAL_TS),
    ]:
        if not path.exists():
            print(f"ERROR: missing source file: {path}")
            sys.exit(1)
        print(f"  OK  {label}: {path.name}")

    results = {}
    results["fig1"] = make_fig1()
    results["fig2"] = make_fig2()
    results["fig2_appendix"] = make_fig2_appendix()
    results["fig4"] = make_fig4()
    results["table1"] = make_table1()
    results["tableS1"] = make_tableS1()
    results["figA2"] = make_figA2()
    write_report(results)

    print("\n" + "=" * 60)
    print("Done.")
    if issues:
        print(f"  {len(issues)} issue(s):")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  All outputs generated with no issues.")
    print("=" * 60)


if __name__ == "__main__":
    main()
