"""
make_synthetic_control.py — Article 5, synthetic known-core control generator
=============================================================================

Generates synthetic datasets with a *planted, known* feature structure, so that
the Article 5 stability profile can later be tested against ground truth.

This script is a data generator only. It does NOT run feature selection, does
NOT build the resampling logging driver, and does NOT touch the benchmark
pipeline or any real dataset.

Design spec: results/article_stability/reports/experiment_planning/SYNTHETIC_CONTROL_DESIGN.md
Grid v1 plan: results/article_stability/reports/experiment_planning/SYNTHETIC_50RESAMPLE_INTERPRETATION_CHECKPOINT.md

Three generation modes:
  * default  — the original single high-SNR / rho=0.9 smoke dataset + null
               control (weak gamma = 0.2). Preserved for backward compatibility.
  * --grid-v1 — Synthetic Grid Expansion v1: the 2x2 SNR x rho grid
               (high/low SNR x rho 0.9/0.7) = 4 known-core datasets + 1 null,
               all with a recalibrated weak gamma and a `_gammaV1` name suffix.
  * --grid-v1-controlled — the controlled retune of the v1 grid: the same 2x2
               grid, but all 4 known-core cells share one base feature matrix,
               one planted-index layout, one eta, one label-uniform vector, and
               one set of proxy epsilons — only alpha (SNR) and rho vary. Uses a
               lower auto-selected alpha_low and a `_gammaV1c` name suffix.

Each dataset is saved as a `.mat` (X, Y keys — compatible with scipy.io.loadmat)
plus a JSON manifest fully describing the planted structure and the structural
verification summary.

Isolation contract:
  * Modifies no other script, no benchmark file, no notebook, no real dataset.
  * Writes ONLY under results/article_stability/synthetic_control/datasets/.
  * Never overwrites an existing file (timestamps a new name instead).

Usage:
    python articles/article_stability/scripts/make_synthetic_control.py --dry-run            # original-mode plan
    python articles/article_stability/scripts/make_synthetic_control.py                      # original smoke datasets
    python articles/article_stability/scripts/make_synthetic_control.py --dry-run --grid-v1  # grid-v1 plan
    python articles/article_stability/scripts/make_synthetic_control.py --grid-v1            # generate the v1 grid
"""

import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io

# --------------------------------------------------------------------------
# Path setup. This script lives at articles/article_stability/scripts/make_synthetic_control.py
# so parents[3] == project root.
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]

# Safety root — all generated datasets MUST stay inside this directory.
ARTICLE5_SYNTH_DATASETS_ROOT = (
    PROJECT_ROOT / "results" / "article_stability" / "synthetic_control" / "datasets"
).resolve()

REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_CONTROL_GENERATION_REPORT.md"
).resolve()

GRID_V1_REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_GRID_V1_GENERATION_REPORT.md"
).resolve()

GRID_V1C_REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_GRID_V1_CONTROLLED_GENERATION_REPORT.md"
).resolve()

GRID_V1D_GEN_REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_INSTANCE_DIAGNOSTIC_V1D_GENERATION_REPORT.md"
).resolve()

GRID_V1E_GEN_REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_POSITIVE_CONTROL_REDESIGN_V1E_GENERATION_REPORT.md"
).resolve()

GRID_V1F_GEN_REPORT_PATH = (
    PROJECT_ROOT / "results" / "article_stability" / "reports"
    / "SYNTHETIC_MIDLOW_CALIBRATION_V1F_GENERATION_REPORT.md"
).resolve()

# ==========================================================================
# Dataset specification (from SYNTHETIC_CONTROL_DESIGN.md sections 2-3)
# ==========================================================================
N_SAMPLES = 200
N_TOTAL_FEATURES = 2000
N_SOLO_CORE = 2
N_ANCHORED_CORE = 3
N_PROXY_GROUPS = 3
PROXIES_PER_GROUP = 4
N_PROXY = N_PROXY_GROUPS * PROXIES_PER_GROUP          # 12
N_WEAK = 15
N_CORE = N_SOLO_CORE + N_ANCHORED_CORE                # 5
N_PLANTED = N_CORE + N_PROXY + N_WEAK                  # 32
N_NOISE = N_TOTAL_FEATURES - N_PLANTED                 # 1968

# Signal-construction parameters.
CORE_BETA = 1.5          # coefficient for every true-core feature (strong, unique)
WEAK_GAMMA = 0.2         # ORIGINAL weak coefficient (under-calibrated — smoke mode only)
WEAK_GAMMA_V1 = 0.8      # GRID v1 weak coefficient. Raised from 0.2 so that weak
                         # features have a mean |corr with y| clearly above the
                         # pure-noise floor (~0.057) and clearly below the core,
                         # i.e. the weak category is structurally distinguishable.

# SNR knob: multiplies the linear predictor before the sigmoid.
#   high = 2.5 (unchanged — strongly saturated, easy regime)
#   low  = 0.5 (a transparent 5x reduction — the linear predictor is far less
#               saturated, so labels are genuinely noisier and the core signal
#               is weaker, while still learnable and visible above noise).
SNR_ALPHA = {"high": 2.5, "low": 0.5}

DEFAULT_RHO = 0.9        # default proxy<->parent Pearson correlation (smoke mode)

# Null control uses a generation seed offset from the known-core seed.
NULL_SEED_OFFSET = 101

# Grid v1: the 2x2 SNR x rho cells, and the dataset-name suffix.
GRID_V1_CELLS = [("high", 0.9), ("high", 0.7), ("low", 0.9), ("low", 0.7)]
GRID_V1_SUFFIX = "_gammaV1"

# Grid v1 CONTROLLED retune: a distinct suffix so the controlled grid never
# overwrites or is confused with the earlier (confounded) `_gammaV1` grid.
GRID_V1C_SUFFIX = "_gammaV1c"
# Controlled alpha_low candidates, tried in order; the lowest (most aggressive)
# candidate that passes the structural checks is auto-selected. SNR_ALPHA["low"]
# (0.5) is intentionally left unchanged so the existing --grid-v1 mode is
# preserved.
CONTROLLED_ALPHA_LOW_CANDIDATES = [0.25, 0.30, 0.35]

# Instance-variance diagnostic (gammaV1d): a reduced multi-seed probe. For each
# of N generation seeds the generator builds a within-seed-controlled pair of
# known-core cells — high SNR + a candidate mid alpha — plus a null control.
# Across seeds the base matrix / planted layout are freshly drawn; this is what
# lets the diagnostic quantify how much recovery varies with the instance.
GRID_V1D_SUFFIX = "_gammaV1d"
DEFAULT_N_GENERATION_SEEDS = 5
DIAGNOSTIC_RHO = 0.9                         # rho fixed for the whole diagnostic
DIAGNOSTIC_ALPHA_HIGH = SNR_ALPHA["high"]    # 2.5 — the easy/saturated regime
DIAGNOSTIC_ALPHA_MID = 0.35                  # candidate hard-but-learnable alpha

# Positive-control redesign (gammaV1e): a targeted, bounded redesign of the
# high-SNR positive-control regime. v1d showed alpha=2.5 at core beta 1.5 is
# instance-variable (held-out AUC 0.54-0.73, one near-failure). The redesign
# raises the core EFFECT SIZE (core beta) so the planted core reliably clears
# the noise upper tail across seeds, and promotes the operational-learnability
# checks (core-vs-noise-p99 margin, oracle true-core AUC) to HARD acceptance
# gates. alpha is deliberately NOT the lever (the v1d checkpoint showed the
# bottleneck is core-vs-noise tail, not label noise); n and p are held at the
# v1d values to preserve the p>>n regime.
GRID_V1E_SUFFIX = "_gammaV1e"
DEFAULT_N_POSCTRL_SEEDS = 8
POSCTRL_ALPHA = SNR_ALPHA["high"]                 # 2.5 — unchanged; not the lever
POSCTRL_RHO = 0.9
POSCTRL_CORE_BETA_CANDIDATES = [2.0, 2.5]         # try 2.0 first, then 2.5
POSCTRL_GATE_P99_MARGIN_MIN = 0.05                # core must clear the noise p99
POSCTRL_GATE_ORACLE_AUC_MEAN_MIN = 0.85           # across-seed oracle-AUC mean
POSCTRL_GATE_ORACLE_AUC_SEED_MIN = 0.75           # per-seed oracle-AUC floor

# Mid/low SNR calibration (gammaV1f): a cheap alpha-sweep at the redesigned
# beta=2.5 positive-control regime, to calibrate a hard-but-learnable mid/low
# alpha cell. The beta=2.5 / alpha=2.5 reference is the established v1e
# positive-control anchor (the hard operational gates still apply to it); the
# mid/low alpha candidates are MEANT to be harder, so their operational checks
# are recorded as informational diagnostics, not hard-fail gates. beta, rho, n,
# p, and weak gamma are all held fixed; only alpha is swept.
GRID_V1F_SUFFIX = "_gammaV1f"
DEFAULT_N_CALIBRATION_SEEDS = 8
CALIBRATION_BETA = 2.5                            # fixed — the v1e anchor beta
CALIBRATION_RHO = 0.9
# (snr_label, alpha, cell_role) — alpha=2.5 is the positive-control reference;
# the rest are mid/low candidates, swept downward in alpha only.
CALIBRATION_ALPHA_CELLS = [
    ("HIGH", 2.5, "positive_control_reference"),
    ("MID", 1.0, "midlow_candidate"),
    ("MID", 0.6, "midlow_candidate"),
    ("LOW", 0.35, "midlow_candidate"),
]


# ==========================================================================
# Output-path safety
# ==========================================================================
def _is_within(child, parent):
    child = Path(child).resolve()
    parent = Path(parent).resolve()
    return child == parent or parent in child.parents


def resolve_output_dir(output_dir_arg):
    """Resolve + validate the dataset output directory (must stay inside the
    synthetic-control datasets safety root)."""
    if output_dir_arg is None:
        out_dir = ARTICLE5_SYNTH_DATASETS_ROOT
    else:
        candidate = Path(output_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = ARTICLE5_SYNTH_DATASETS_ROOT / candidate
        out_dir = candidate.resolve()
    if not _is_within(out_dir, ARTICLE5_SYNTH_DATASETS_ROOT):
        raise SystemExit(
            f"[SAFETY] Refusing to write outside the synthetic-control datasets root.\n"
            f"         requested : {out_dir}\n"
            f"         allowed   : {ARTICLE5_SYNTH_DATASETS_ROOT}"
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
# Numeric helpers
# ==========================================================================
def sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_columns(X, idx):
    """Z-score the given columns of X in place (mean 0, std 1)."""
    cols = X[:, idx]
    mu = cols.mean(axis=0)
    sd = cols.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    X[:, idx] = (cols - mu) / sd


def place_planted_indices(rng, n_total, n_planted):
    """Pick `n_planted` feature positions spread across 0..n_total-1.

    Partition the index range into `n_planted` contiguous bins and draw one
    position from each bin. This guarantees the planted indices are spread
    across the whole range and never clustered at the start.
    """
    edges = np.linspace(0, n_total, n_planted + 1, dtype=int)
    picks = []
    for b in range(n_planted):
        lo, hi = edges[b], edges[b + 1]
        picks.append(int(rng.integers(lo, hi)))
    picks = np.array(picks, dtype=int)
    rng.shuffle(picks)   # randomize which planted role each spread position gets
    return picks


def abs_corr_with_y(col, y):
    """Absolute Pearson correlation between a feature column and binary y."""
    col = np.asarray(col, dtype=np.float64)
    if col.std() == 0.0:
        return 0.0
    c = np.corrcoef(col, y.astype(np.float64))[0, 1]
    return float(abs(c)) if np.isfinite(c) else 0.0


# ==========================================================================
# Dataset generation
# ==========================================================================
def generate_known_core(seed, snr_label, rho, weak_gamma=WEAK_GAMMA, name_suffix=""):
    """Generate one synthetic known-core dataset.

    Construction order (per SYNTHETIC_CONTROL_DESIGN.md section 2):
      1. Draw all p columns i.i.d. N(0,1).
      2. Standardize the 5 core and 15 weak columns.
      3. Linear predictor eta = sum(beta * core) + sum(gamma * weak).
      4. Labels y ~ Bernoulli(sigmoid(alpha * eta)).
      5. Proxies generated LAST, after y: proxy = rho*parent + sqrt(1-rho^2)*eps,
         so each proxy is correlated with y ONLY through its parent core feature
         (true redundancy, no independent signal).
      6. Noise columns keep their i.i.d. draw.

    `weak_gamma` is the per-weak-feature coefficient; `name_suffix` is appended
    to the dataset name (e.g. "_gammaV1" for the grid).
    """
    rng = np.random.default_rng(seed)
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    alpha = SNR_ALPHA[snr_label]

    # 1. base matrix
    X = rng.standard_normal((n, p))

    # planted index placement (spread across 0..p-1)
    planted = place_planted_indices(rng, p, N_PLANTED)
    solo_core = sorted(int(i) for i in planted[:N_SOLO_CORE])
    anchored_core = sorted(int(i) for i in planted[N_SOLO_CORE:N_CORE])
    proxy_block = [int(i) for i in planted[N_CORE:N_CORE + N_PROXY]]
    weak_idx = sorted(int(i) for i in planted[N_CORE + N_PROXY:])
    core_idx = sorted(solo_core + anchored_core)
    planted_set = set(core_idx) | set(proxy_block) | set(weak_idx)
    noise_idx = [i for i in range(p) if i not in planted_set]

    # proxy groups: each anchored core feature gets one 4-proxy group
    proxy_groups = {}
    for g, parent in enumerate(anchored_core):
        proxy_groups[parent] = sorted(
            proxy_block[g * PROXIES_PER_GROUP:(g + 1) * PROXIES_PER_GROUP]
        )

    # 2. standardize core + weak columns
    standardize_columns(X, core_idx)
    standardize_columns(X, weak_idx)

    # 3. linear predictor (core mutually near-orthogonal by construction)
    beta_core = {i: CORE_BETA for i in core_idx}
    gamma_weak = {i: weak_gamma for i in weak_idx}
    eta = np.zeros(n, dtype=np.float64)
    for i in core_idx:
        eta += beta_core[i] * X[:, i]
    for i in weak_idx:
        eta += gamma_weak[i] * X[:, i]

    # 4. labels
    prob = sigmoid(alpha * eta)
    y = rng.binomial(1, prob).astype(np.int64)

    # 5. proxies — generated AFTER y, correlated with the parent core column only
    sqrt_term = math.sqrt(max(0.0, 1.0 - rho ** 2))
    for parent, proxies in proxy_groups.items():
        for pj in proxies:
            eps = rng.standard_normal(n)
            X[:, pj] = rho * X[:, parent] + sqrt_term * eps

    # 6. noise columns: unchanged i.i.d. draw

    rho_code = int(round(rho * 100))
    manifest = {
        "dataset_name": f"synthetic_snr{snr_label.upper()}_rho{rho_code:03d}{name_suffix}",
        "generation_seed": int(seed),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": snr_label,
        "alpha_value": float(alpha),
        "rho": float(rho),
        "weak_gamma": float(weak_gamma),
        "core_beta": float(CORE_BETA),
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": solo_core,
        "true_core_anchored_indices": anchored_core,
        "proxy_groups": {str(parent): proxies for parent, proxies in proxy_groups.items()},
        "weak_signal_indices": weak_idx,
        "pure_noise_indices": {
            "count": len(noise_idx),
            "sample_first10": noise_idx[:10],
            "sample_last10": noise_idx[-10:],
            "definition": "all indices in 0..1999 not assigned to core/proxy/weak",
        },
        "core_beta_per_feature": {str(i): float(b) for i, b in beta_core.items()},
        "weak_gamma_per_feature": {str(i): float(g) for i, g in gamma_weak.items()},
        "signal_construction_notes": (
            "All 2000 columns drawn i.i.d. N(0,1). The 5 core and 15 weak columns "
            "were z-scored. Linear predictor eta = sum(beta*core) + sum(gamma*weak) "
            f"with beta={CORE_BETA} per core feature and gamma={weak_gamma} per weak "
            "feature. Core features are mutually near-orthogonal by construction "
            "(i.i.d. draws, n<<p). The 12 proxy columns were generated LAST, after y, "
            f"as proxy = rho*parent + sqrt(1-rho^2)*eps with rho={rho}; each proxy is "
            "correlated with y only through its parent core feature (true redundancy, "
            "no independent signal). The remaining 1968 noise columns keep their "
            "i.i.d. N(0,1) draw."
        ),
        "label_construction_notes": (
            f"y ~ Bernoulli(sigmoid(alpha*eta)) with alpha={alpha} (the SNR knob). "
            "Labels are in {0,1}. eta is symmetric about 0 so the class balance is "
            "expected to be approximately even."
        ),
        # structural_verification_summary is added by the generation driver
        # after verify_known_core() runs.
    }
    return X.astype(np.float64), y, manifest


def generate_null_control(seed, name_suffix=""):
    """Generate the null-control dataset: all 2000 features pure noise, random y.

    The stability profile must show NO recurrent core here — a decisive sanity
    check that the profile does not manufacture structure from noise.
    """
    rng = np.random.default_rng(seed)
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    X = rng.standard_normal((n, p)).astype(np.float64)
    y = rng.binomial(1, 0.5, size=n).astype(np.int64)

    manifest = {
        "dataset_name": f"synthetic_null_control{name_suffix}",
        "generation_seed": int(seed),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": "null",
        "alpha_value": None,
        "rho": None,
        "weak_gamma": None,
        "core_beta": None,
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": [],
        "true_core_anchored_indices": [],
        "proxy_groups": {},
        "weak_signal_indices": [],
        "pure_noise_indices": {
            "count": int(p),
            "sample_first10": list(range(10)),
            "sample_last10": list(range(p - 10, p)),
            "definition": "ALL 2000 features are pure noise; there is no planted signal",
        },
        "core_beta_per_feature": {},
        "weak_gamma_per_feature": {},
        "all_features_are_noise": True,
        "null_control_notes": (
            "Null-control dataset: every one of the 2000 features is pure noise; "
            "there is no planted true core, no proxy group, and no weak periphery. "
            "Any recurrent feature found downstream on this dataset must be treated "
            "as noise."
        ),
        "signal_construction_notes": (
            "All 2000 columns drawn i.i.d. N(0,1). No feature carries any signal."
        ),
        "label_construction_notes": (
            "y ~ Bernoulli(0.5), independent of every feature. Labels are in {0,1}. "
            "By construction no feature is associated with y beyond chance."
        ),
        # structural_verification_summary is added by the generation driver.
    }
    return X, y, manifest


# ==========================================================================
# Structural verification
# ==========================================================================
def _x_is_finite(X):
    return bool(np.isfinite(np.asarray(X)).all())


def _y_is_binary(y):
    return set(int(v) for v in np.unique(y)).issubset({0, 1})


def verify_known_core(X, y, manifest, rho):
    """Structural verification for a known-core dataset.

    Hard gate (`passed`): shape, Y binary, X finite, class balance reasonable,
    proxy-parent correlations close to rho, the core > weak > noise association
    ordering, and the core visible above noise.
    """
    v = {"dataset_name": manifest["dataset_name"], "checks": {}, "warnings": []}
    c = v["checks"]

    # --- shape / dtype integrity --------------------------------------------
    c["X_shape"] = list(X.shape)
    c["Y_len"] = int(len(y))
    c["shape_ok"] = bool(X.shape == (N_SAMPLES, N_TOTAL_FEATURES) and len(y) == N_SAMPLES)
    c["Y_is_binary"] = bool(_y_is_binary(y))
    c["X_finite_no_nan_inf"] = bool(_x_is_finite(X))
    if not c["Y_is_binary"]:
        v["warnings"].append("Y is not binary {0,1}")
    if not c["X_finite_no_nan_inf"]:
        v["warnings"].append("X contains NaN or Inf")

    # --- class balance ------------------------------------------------------
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    minf = min(n0, n1) / len(y)
    c["class_balance"] = {"n_class_0": n0, "n_class_1": n1,
                          "minority_fraction": round(minf, 4)}
    c["class_balance_reasonable"] = bool(minf >= 0.25)
    if not c["class_balance_reasonable"]:
        v["warnings"].append(f"class balance skewed (minority fraction {minf:.3f} < 0.25)")

    # --- proxy <-> parent empirical Pearson correlations --------------------
    proxy_corrs = []
    for parent_s, proxies in manifest["proxy_groups"].items():
        parent = int(parent_s)
        for pj in proxies:
            cc = np.corrcoef(X[:, pj], X[:, parent])[0, 1]
            proxy_corrs.append(float(cc))
    c["proxy_parent_corr"] = {
        "target_rho": rho,
        "mean": round(float(np.mean(proxy_corrs)), 4),
        "min": round(float(np.min(proxy_corrs)), 4),
        "max": round(float(np.max(proxy_corrs)), 4),
        "n_proxies": len(proxy_corrs),
    }
    c["proxy_parent_corr_close_to_rho"] = bool(
        all(abs(cc - rho) <= 0.10 for cc in proxy_corrs)
    )
    if not c["proxy_parent_corr_close_to_rho"]:
        v["warnings"].append(
            f"at least one proxy-parent correlation deviates >0.10 from rho={rho}")

    # --- univariate association with y --------------------------------------
    solo = manifest["true_core_solo_indices"]
    anchored = manifest["true_core_anchored_indices"]
    weak = manifest["weak_signal_indices"]
    rng = np.random.default_rng(12345)
    proxy_all = set()
    for pl in manifest["proxy_groups"].values():
        proxy_all.update(int(x) for x in pl)
    noise_all = [i for i in range(N_TOTAL_FEATURES)
                 if i not in set(solo) | set(anchored) | set(weak) | proxy_all]
    noise_sample = sorted(rng.choice(noise_all, size=min(50, len(noise_all)),
                                     replace=False).tolist())

    solo_corr = [abs_corr_with_y(X[:, i], y) for i in solo]
    anch_corr = [abs_corr_with_y(X[:, i], y) for i in anchored]
    weak_corr = [abs_corr_with_y(X[:, i], y) for i in weak]
    noise_corr = [abs_corr_with_y(X[:, i], y) for i in noise_sample]

    core_mean = float(np.mean(solo_corr + anch_corr))
    weak_mean = float(np.mean(weak_corr))
    noise_mean = float(np.mean(noise_corr))
    noise_max = float(np.max(noise_corr))
    c["mean_abs_corr_with_y"] = {
        "solo_core": round(float(np.mean(solo_corr)), 4),
        "anchored_core": round(float(np.mean(anch_corr)), 4),
        "core_all": round(core_mean, 4),
        "weak": round(weak_mean, 4),
        "noise_sample_mean": round(noise_mean, 4),
        "noise_sample_max": round(noise_max, 4),
        "noise_sample_size": len(noise_sample),
    }

    # tightened ordering checks (the previous version allowed weak >= noise-0.05)
    c["core_stronger_than_weak"] = bool(core_mean > weak_mean)
    c["weak_stronger_than_noise"] = bool(weak_mean > noise_mean)
    c["ordering_core_gt_weak_gt_noise"] = bool(core_mean > weak_mean > noise_mean)
    c["core_visible_above_noise"] = bool(core_mean > 2.0 * noise_mean)
    c["noise_low_association"] = bool(noise_mean < 0.15)
    # informational (not part of the hard gate): weak above the luckiest sampled
    # noise feature. Expected to hold at high SNR; may fail at low SNR.
    c["weak_above_noise_sample_max"] = bool(weak_mean > noise_max)

    if not c["core_stronger_than_weak"]:
        v["warnings"].append(
            f"core mean |corr| ({core_mean:.3f}) is NOT > weak mean ({weak_mean:.3f})")
    if not c["weak_stronger_than_noise"]:
        v["warnings"].append(
            f"weak mean |corr| ({weak_mean:.3f}) is NOT > noise sample mean "
            f"({noise_mean:.3f}) — the weak > noise ordering FAILED")
    if not c["core_visible_above_noise"]:
        v["warnings"].append(
            f"core mean |corr| ({core_mean:.3f}) is not clearly above the noise floor")
    if not c["weak_above_noise_sample_max"]:
        v["warnings"].append(
            f"weak mean |corr| ({weak_mean:.3f}) does not exceed the sampled-noise "
            f"maximum ({noise_max:.3f}) — informational; expected to relax at low SNR")

    v["passed"] = bool(
        c["shape_ok"] and c["Y_is_binary"] and c["X_finite_no_nan_inf"]
        and c["class_balance_reasonable"] and c["proxy_parent_corr_close_to_rho"]
        and c["ordering_core_gt_weak_gt_noise"] and c["core_visible_above_noise"]
    )
    return v


def verify_null_control(X, y, manifest):
    """Structural verification for the null-control dataset."""
    v = {"dataset_name": manifest["dataset_name"], "checks": {}, "warnings": []}
    c = v["checks"]

    c["X_shape"] = list(X.shape)
    c["Y_len"] = int(len(y))
    c["shape_ok"] = bool(X.shape == (N_SAMPLES, N_TOTAL_FEATURES) and len(y) == N_SAMPLES)
    c["Y_is_binary"] = bool(_y_is_binary(y))
    c["X_finite_no_nan_inf"] = bool(_x_is_finite(X))
    if not c["Y_is_binary"]:
        v["warnings"].append("Y is not binary {0,1}")
    if not c["X_finite_no_nan_inf"]:
        v["warnings"].append("X contains NaN or Inf")

    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    minf = min(n0, n1) / len(y)
    c["class_balance"] = {"n_class_0": n0, "n_class_1": n1,
                          "minority_fraction": round(minf, 4)}
    c["class_balance_reasonable"] = bool(minf >= 0.25)
    if not c["class_balance_reasonable"]:
        v["warnings"].append(f"class balance skewed (minority fraction {minf:.3f} < 0.25)")

    # No planted signal: scan all features' |corr with y|.
    all_corr = np.array([abs_corr_with_y(X[:, i], y) for i in range(N_TOTAL_FEATURES)])
    c["abs_corr_with_y"] = {
        "mean": round(float(all_corr.mean()), 4),
        "max": round(float(all_corr.max()), 4),
        "p99": round(float(np.percentile(all_corr, 99)), 4),
    }
    # With p=2000, n=200 the max of ~2000 chance correlations is expected around
    # ~0.25-0.30; "no planted signal" means no large outlier.
    c["no_strong_planted_signal"] = bool(all_corr.max() < 0.40)
    if not c["no_strong_planted_signal"]:
        v["warnings"].append(
            f"null-control max |corr with y| ({all_corr.max():.3f}) is unexpectedly high")

    v["passed"] = bool(
        c["shape_ok"] and c["Y_is_binary"] and c["X_finite_no_nan_inf"]
        and c["class_balance_reasonable"] and c["no_strong_planted_signal"]
    )
    return v


def embed_verification(manifest, v):
    """Embed a compact structural-verification summary into the manifest."""
    manifest["structural_verification_summary"] = {
        "passed": bool(v["passed"]),
        "checks": v["checks"],
        "warnings": list(v["warnings"]),
    }
    return manifest


# ==========================================================================
# Writing
# ==========================================================================
def write_dataset(out_dir, manifest, X, y):
    """Write one dataset's .mat and manifest.json (non-overwriting). Returns paths."""
    name = manifest["dataset_name"]
    mat_path = safe_path(out_dir, f"{name}.mat", ARTICLE5_SYNTH_DATASETS_ROOT)
    man_path = safe_path(out_dir, f"{name}_manifest.json", ARTICLE5_SYNTH_DATASETS_ROOT)
    print(f"[output] dataset  : {mat_path}")
    print(f"[output] manifest : {man_path}")
    scipy.io.savemat(str(mat_path), {"X": X, "Y": y.reshape(-1, 1).astype(np.int64)})
    man_path.write_text(json.dumps(manifest, indent=2))
    return mat_path, man_path


def reload_check(mat_path):
    """Reload a .mat to confirm it has X / Y keys and report shapes."""
    D = scipy.io.loadmat(str(mat_path))
    has_X, has_Y = "X" in D, "Y" in D
    xs = list(D["X"].shape) if has_X else None
    ys = list(D["Y"].shape) if has_Y else None
    return {"loads_ok": True, "has_X": has_X, "has_Y": has_Y,
            "X_shape": xs, "Y_shape": ys}


# ==========================================================================
# Markdown report — original smoke mode
# ==========================================================================
def build_report(commands, written, manifests, verifications, reloads):
    L = []
    L.append("# Synthetic Known-Core Control — Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Data generator only. No feature selection, no logging driver, no real "
             "dataset.\n")
    L.append("---\n")

    L.append("## 1. Commands Run\n")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")

    L.append("## 2. Output Files Created\n")
    for label, path in written:
        L.append(f"- **{label}:** `{path}`")
    L.append(f"- **This report:** `{REPORT_PATH}`\n")

    L.append("## 3. Manifest Summary\n")
    for m in manifests:
        L.append(f"### {m['dataset_name']}\n")
        L.append(f"- generation_seed: {m['generation_seed']}")
        L.append(f"- snr_label: {m['snr_label']} | alpha_value: {m['alpha_value']} | "
                 f"rho: {m['rho']} | weak_gamma: {m['weak_gamma']}")
        L.append(f"- class_balance: class0={m['class_balance']['n_class_0']}, "
                 f"class1={m['class_balance']['n_class_1']}, "
                 f"minority_fraction={m['class_balance']['minority_fraction']:.3f}")
        L.append(f"- true_core_solo_indices: {m['true_core_solo_indices']}")
        L.append(f"- true_core_anchored_indices: {m['true_core_anchored_indices']}")
        if m["proxy_groups"]:
            L.append("- proxy_groups (parent -> proxies):")
            for parent, proxies in m["proxy_groups"].items():
                L.append(f"  - {parent} -> {proxies}")
        else:
            L.append("- proxy_groups: none")
        L.append(f"- weak_signal_indices: {m['weak_signal_indices']}")
        L.append(f"- pure_noise_indices: count={m['pure_noise_indices']['count']}")
        L.append("")

    L.append("## 4. Structural Verification Results\n")
    for v in verifications:
        L.append(f"### {v['dataset_name']}  —  "
                 f"{'PASS' if v['passed'] else 'CHECK WARNINGS'}\n")
        for key, val in v["checks"].items():
            L.append(f"- `{key}`: {val}")
        if v["warnings"]:
            L.append("- **warnings:**")
            for w in v["warnings"]:
                L.append(f"  - {w}")
        L.append("")
    L.append("**Reload checks (`.mat` round-trip):**\n")
    for name, rc in reloads.items():
        L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                 f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, Y_shape={rc['Y_shape']}")
    L.append("")

    L.append("---\n")
    L.append("*Generator step only — no feature selection run, no logging driver built, "
             "no real dataset used, no benchmark file modified.*")
    return "\n".join(L)


# ==========================================================================
# Markdown report — grid v1 mode
# ==========================================================================
def build_grid_v1_report(commands, written, manifests, verifications, reloads,
                          all_passed):
    L = []
    L.append("# Synthetic Grid Expansion v1 — Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Synthetic Grid Expansion v1 — generator update + 2x2 grid generation + "
             "structural verification only. No feature selection, no repeated "
             "resampling, no analyzer, no recovery metrics, no real dataset.\n")
    L.append("---\n")

    # 1. commands
    L.append("## 1. Commands Run\n")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")

    # 2. generator changes
    L.append("## 2. Generator Changes Made\n")
    L.append(f"- **Weak gamma recalibrated:** `WEAK_GAMMA_V1 = {WEAK_GAMMA_V1}` "
             f"(raised from the original {WEAK_GAMMA}). Goal: weak features have a "
             f"mean |corr with y| clearly above the pure-noise floor (~0.057) and "
             f"clearly below the core, so the weak category is structurally "
             f"distinguishable from noise. The original {WEAK_GAMMA} value is kept "
             f"for the (unchanged) original smoke-dataset mode.")
    L.append(f"- **Low-SNR setting added:** `SNR_ALPHA = {SNR_ALPHA}`. `high = 2.5` "
             f"(unchanged, strongly saturated/easy); `low = 0.5` is a transparent 5x "
             f"reduction — the linear predictor is far less saturated, so labels are "
             f"genuinely noisier and the core signal weaker, while still learnable "
             f"and visible above noise. The generator empirically verifies the "
             f"core > weak > noise ordering for every cell.")
    L.append("- **Tightened structural verification:** the core>weak>noise ordering "
             "check is now strict (`core_mean > weak_mean > noise_mean`; the previous "
             "version permitted `weak >= noise - 0.05`); added explicit Y-binary and "
             "X-finite (no NaN/Inf) checks; the structural-verification summary is "
             "embedded into every manifest.")
    L.append("- **Grid mode added:** `--grid-v1` generates the 2x2 SNR x rho grid "
             "(4 known-core datasets) plus one null control, all with a `_gammaV1` "
             "name suffix. The original single-dataset smoke mode is preserved as the "
             "default (no `--grid-v1`).\n")

    # 3. grid cells
    L.append("## 3. Grid Cells Generated\n")
    L.append("| Cell | SNR | alpha | rho | weak gamma | seed |")
    L.append("|------|-----|-------|-----|------------|------|")
    for m in manifests:
        if m["snr_label"] == "null":
            continue
        L.append(f"| {m['dataset_name']} | {m['snr_label']} | {m['alpha_value']} | "
                 f"{m['rho']} | {m['weak_gamma']} | {m['generation_seed']} |")
    null_m = next((m for m in manifests if m["snr_label"] == "null"), None)
    if null_m:
        L.append(f"| {null_m['dataset_name']} | null | — | — | — | "
                 f"{null_m['generation_seed']} |")
    L.append("")
    L.append("Fixed across all cells: n_samples=200, n_total_features=2000, "
             "5 planted true-core features (2 solo + 3 anchored), 12 proxy features "
             "(3 uniform groups x 4 proxies), 15 weak signal features, 1968 pure "
             "noise. No p=10000, no variable proxy cardinality, no real datasets.\n")

    # 4. output files
    L.append("## 4. Output Files\n")
    for label, path in written:
        L.append(f"- **{label}:** `{path}`")
    L.append(f"- **This report:** `{GRID_V1_REPORT_PATH}`\n")

    # 5. class balance per cell
    L.append("## 5. Class Balance per Cell\n")
    L.append("| Dataset | class 0 | class 1 | minority fraction |")
    L.append("|---------|---------|---------|-------------------|")
    for m in manifests:
        cb = m["class_balance"]
        L.append(f"| {m['dataset_name']} | {cb['n_class_0']} | {cb['n_class_1']} | "
                 f"{cb['minority_fraction']:.3f} |")
    L.append("")

    # 6. proxy-parent correlation per cell
    L.append("## 6. Proxy-Parent Correlation per Cell\n")
    L.append("| Dataset | target rho | mean | min | max | close to rho |")
    L.append("|---------|------------|------|-----|-----|--------------|")
    for v in verifications:
        c = v["checks"]
        if "proxy_parent_corr" not in c:
            continue
        pc = c["proxy_parent_corr"]
        L.append(f"| {v['dataset_name']} | {pc['target_rho']} | {pc['mean']} | "
                 f"{pc['min']} | {pc['max']} | {c['proxy_parent_corr_close_to_rho']} |")
    L.append("")

    # 7. core / weak / noise association per cell
    L.append("## 7. Core / Weak / Noise Association per Cell\n")
    L.append("Mean absolute Pearson correlation with y. The hard gate is the strict "
             "ordering `core > weak > noise`.\n")
    L.append("| Dataset | core | weak | noise mean | noise max | "
             "core>weak>noise | weak>noise |")
    L.append("|---------|------|------|------------|-----------|"
             "-----------------|------------|")
    for v in verifications:
        c = v["checks"]
        if "mean_abs_corr_with_y" not in c or "ordering_core_gt_weak_gt_noise" not in c:
            continue
        mc = c["mean_abs_corr_with_y"]
        L.append(f"| {v['dataset_name']} | {mc['core_all']} | {mc['weak']} | "
                 f"{mc['noise_sample_mean']} | {mc['noise_sample_max']} | "
                 f"{c['ordering_core_gt_weak_gt_noise']} | "
                 f"{c['weak_stronger_than_noise']} |")
    L.append("")

    # 8. null-control verification
    L.append("## 8. Null-Control Verification\n")
    null_v = next((v for v in verifications
                   if v["dataset_name"].startswith("synthetic_null")), None)
    if null_v:
        c = null_v["checks"]
        L.append(f"- dataset: `{null_v['dataset_name']}` — "
                 f"{'PASS' if null_v['passed'] else 'CHECK WARNINGS'}")
        L.append(f"- shape_ok: {c['shape_ok']}, Y_is_binary: {c['Y_is_binary']}, "
                 f"X_finite: {c['X_finite_no_nan_inf']}")
        L.append(f"- class_balance: {c['class_balance']}")
        L.append(f"- abs_corr_with_y: {c['abs_corr_with_y']}")
        L.append(f"- no_strong_planted_signal: {c['no_strong_planted_signal']}")
    L.append("")

    # 9. reload checks
    L.append("## 9. Reload Checks (`.mat` round-trip)\n")
    for name, rc in reloads.items():
        L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                 f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, Y_shape={rc['Y_shape']}")
    L.append("")

    # 10. warnings
    L.append("## 10. Warnings\n")
    any_warn = any(v["warnings"] for v in verifications)
    ordering_failed = any(
        not v["checks"].get("ordering_core_gt_weak_gt_noise", True)
        for v in verifications if "ordering_core_gt_weak_gt_noise" in v["checks"]
    )
    if any_warn:
        for v in verifications:
            for w in v["warnings"]:
                L.append(f"- [{v['dataset_name']}] {w}")
    else:
        L.append("- No structural warnings — all verification checks passed.")
    if ordering_failed:
        L.append("- **CRITICAL:** the core > weak > noise ordering FAILED for at "
                 "least one cell. Do NOT proceed to 50-resample logging until this "
                 "is resolved.")
    L.append("- *Note:* `weak_above_noise_sample_max` is informational, not part of "
             "the hard gate; it is expected to hold at high SNR and may relax at low "
             "SNR (the weak mean can fall below the luckiest single sampled-noise "
             "feature while still clearly exceeding the noise mean).")
    L.append("- These datasets are **synthetic**; \"planted core\" wording is licensed "
             "only for synthetic data.\n")

    # 11. ready for logging?
    L.append("## 11. Is the Grid Ready for 50-Resample Logging?\n")
    if all_passed and not ordering_failed:
        L.append("- **Yes.** All 4 known-core cells and the null control passed "
                 "structural verification; the core > weak > noise ordering holds "
                 "strictly for every known-core cell. The grid is ready for the "
                 "50-resample logging pipeline.")
    else:
        L.append("- **No — stop before logging.** At least one cell failed structural "
                 "verification (see section 10). Resolve the failure — in particular "
                 "any core > weak > noise ordering failure — before running the "
                 "50-resample logging pipeline.")
    L.append("")

    # 12. recommended next step
    L.append("## 12. Recommended Next Step\n")
    if all_passed and not ordering_failed:
        L.append("- Proceed to the 50-resample logging pipeline per grid cell: for "
                 "each of the 4 known-core datasets and the null, run "
                 "`run_synthetic_repeated_resampling_logging.py` (50 resamples), then "
                 "`analyze_per_fold_stability.py`, then `synthetic_recovery_metrics.py` "
                 "— reusing those scripts unmodified. This generation task does not "
                 "run any of them.")
    else:
        L.append("- Do not proceed. Investigate and resolve the structural-"
                 "verification failure(s) in section 10, regenerate, and re-verify "
                 "before any logging.")
    L.append("")

    L.append("---\n")
    L.append("*Generator step only — generator updated and the 2x2 grid + null "
             "generated and structurally verified. No feature selection, no logging, "
             "no analyzer, no recovery metrics, no real dataset, no benchmark file "
             "modified.*")
    return "\n".join(L)


# ==========================================================================
# Driver — original smoke mode
# ==========================================================================
def run_generation(seed, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic known-core control generator (original smoke mode)")
    print("=" * 70)
    print(f"Seed (known-core)   : {seed}")
    print(f"Seed (null control) : {seed + NULL_SEED_OFFSET}")
    print(f"Output directory    : {out_dir}")

    Xk, yk, man_known = generate_known_core(seed, snr_label="high", rho=DEFAULT_RHO)
    Xn, yn, man_null = generate_null_control(seed + NULL_SEED_OFFSET)

    ver_known = verify_known_core(Xk, yk, man_known, rho=DEFAULT_RHO)
    ver_null = verify_null_control(Xn, yn, man_null)
    embed_verification(man_known, ver_known)
    embed_verification(man_null, ver_null)

    print("-" * 70)
    mat_k, man_k = write_dataset(out_dir, man_known, Xk, yk)
    mat_n, man_n = write_dataset(out_dir, man_null, Xn, yn)

    reloads = {
        man_known["dataset_name"]: reload_check(mat_k),
        man_null["dataset_name"]: reload_check(mat_n),
    }
    for v in (ver_known, ver_null):
        print(f"Verification {v['dataset_name']}: "
              f"{'PASS' if v['passed'] else 'CHECK WARNINGS'}")
        for w in v["warnings"]:
            print(f"  [warn] {w}")

    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run",
        "python articles/article_stability/scripts/make_synthetic_control.py",
    ]
    written = [
        ("known-core dataset", mat_k), ("known-core manifest", man_k),
        ("null-control dataset", mat_n), ("null-control manifest", man_n),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(REPORT_PATH.parent, REPORT_PATH.name, REPORT_PATH.parent)
    report_path.write_text(build_report(
        commands, written, [man_known, man_null], [ver_known, ver_null], reloads
    ))
    print("-" * 70)
    print(f"Markdown report     : {report_path}")
    print("=" * 70)
    return 0 if (ver_known["passed"] and ver_null["passed"]) else 1


# ==========================================================================
# Driver — grid v1 mode
# ==========================================================================
def run_grid_v1_generation(seed, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic Grid Expansion v1 generator")
    print("=" * 70)
    print(f"Base seed         : {seed}")
    print(f"Grid cells        : {GRID_V1_CELLS}")
    print(f"Weak gamma (v1)   : {WEAK_GAMMA_V1}")
    print(f"SNR alpha         : {SNR_ALPHA}")
    print(f"Output directory  : {out_dir}")
    print("-" * 70)

    manifests, verifications, written, reloads = [], [], [], {}
    datasets = []   # (manifest, X, y, verification)

    # 4 known-core cells — each cell gets a distinct seed (base + cell index).
    for i, (snr, rho) in enumerate(GRID_V1_CELLS):
        cell_seed = seed + i
        X, y, manifest = generate_known_core(
            cell_seed, snr_label=snr, rho=rho,
            weak_gamma=WEAK_GAMMA_V1, name_suffix=GRID_V1_SUFFIX,
        )
        v = verify_known_core(X, y, manifest, rho=rho)
        embed_verification(manifest, v)
        datasets.append((manifest, X, y, v))
        print(f"Generated {manifest['dataset_name']:38s} seed={cell_seed}  "
              f"verify={'PASS' if v['passed'] else 'CHECK'}")

    # null control — distinct seed (base + NULL_SEED_OFFSET).
    null_seed = seed + NULL_SEED_OFFSET
    Xn, yn, man_null = generate_null_control(null_seed, name_suffix=GRID_V1_SUFFIX)
    vn = verify_null_control(Xn, yn, man_null)
    embed_verification(man_null, vn)
    datasets.append((man_null, Xn, yn, vn))
    print(f"Generated {man_null['dataset_name']:38s} seed={null_seed}  "
          f"verify={'PASS' if vn['passed'] else 'CHECK'}")

    # write all datasets
    print("-" * 70)
    for manifest, X, y, v in datasets:
        mat_p, man_p = write_dataset(out_dir, manifest, X, y)
        is_null = manifest["snr_label"] == "null"
        label = "null-control" if is_null else "known-core"
        written.append((f"{label} dataset", mat_p))
        written.append((f"{label} manifest", man_p))
        reloads[manifest["dataset_name"]] = reload_check(mat_p)
        manifests.append(manifest)
        verifications.append(v)

    # verification summary to console
    print("-" * 70)
    all_passed = True
    for v in verifications:
        ok = v["passed"]
        all_passed = all_passed and ok
        print(f"Verification {v['dataset_name']:38s}: "
              f"{'PASS' if ok else 'CHECK WARNINGS'}")
        for w in v["warnings"]:
            print(f"  [warn] {w}")

    # report
    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run --grid-v1",
        "python articles/article_stability/scripts/make_synthetic_control.py --grid-v1",
    ]
    GRID_V1_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(GRID_V1_REPORT_PATH.parent, GRID_V1_REPORT_PATH.name,
                            GRID_V1_REPORT_PATH.parent)
    report_path.write_text(build_grid_v1_report(
        commands, written, manifests, verifications, reloads, all_passed
    ))
    print("-" * 70)
    print(f"Markdown report   : {report_path}")
    print(f"Grid verification : {'ALL PASS' if all_passed else 'CHECK WARNINGS'}")
    print("=" * 70)
    return 0 if all_passed else 1


# ==========================================================================
# Controlled grid v1 (gammaV1c) — shared planted indices / shared base matrix
# ==========================================================================
def _core_weak_noise_corr(X, y, core_idx, weak_idx, proxy_block):
    """Mean |corr with y| for core, weak, and a fixed 50-feature noise sample.

    Uses the same rng(12345) seed and exclusion rule as `verify_known_core`, so
    candidate-selection numbers are consistent with the later verification.
    """
    rng = np.random.default_rng(12345)
    excl = (set(int(i) for i in core_idx) | set(int(i) for i in weak_idx)
            | set(int(i) for i in proxy_block))
    noise_all = [i for i in range(N_TOTAL_FEATURES) if i not in excl]
    noise_sample = sorted(rng.choice(noise_all, size=min(50, len(noise_all)),
                                     replace=False).tolist())
    core_m = float(np.mean([abs_corr_with_y(X[:, i], y) for i in core_idx]))
    weak_m = float(np.mean([abs_corr_with_y(X[:, i], y) for i in weak_idx]))
    noise_m = float(np.mean([abs_corr_with_y(X[:, i], y) for i in noise_sample]))
    return core_m, weak_m, noise_m


def _controlled_cell(snr_label, alpha, rho, shared, chosen_alpha_low):
    """Build one controlled-grid known-core cell from the shared layout, base
    matrix, eta, label-uniform vector U, and proxy epsilons.

    Only `alpha` (SNR, via the label draw) and `rho` (via the proxy columns)
    differ across cells; the core / weak / noise columns and the planted layout
    are byte-identical to every other controlled cell.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    X = shared["X_base"].copy()

    # proxy columns: same parent core column + same epsilon array; only rho differs
    sqrt_term = math.sqrt(max(0.0, 1.0 - rho ** 2))
    for parent, proxies in shared["proxy_groups"].items():
        for pj in proxies:
            X[:, pj] = (rho * shared["X_base"][:, parent]
                        + sqrt_term * shared["eps_by_proxy"][int(pj)])

    # labels: shared uniform vector U; only alpha differs across SNR cells
    prob = sigmoid(alpha * shared["eta"])
    y = (shared["U"] < prob).astype(np.int64)

    core_idx = shared["core_idx"]
    weak_idx = shared["weak_idx"]
    noise_idx = shared["noise_idx"]
    rho_code = int(round(rho * 100))
    manifest = {
        "dataset_name": (f"synthetic_snr{snr_label.upper()}_rho{rho_code:03d}"
                         f"{GRID_V1C_SUFFIX}"),
        "generation_seed": int(shared["base_seed"]),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": snr_label,
        "alpha_value": float(alpha),
        "rho": float(rho),
        "weak_gamma": float(WEAK_GAMMA_V1),
        "core_beta": float(CORE_BETA),
        "controlled_grid_version": "gammaV1c",
        "controlled_base_seed": int(shared["base_seed"]),
        "alpha_low_chosen": float(chosen_alpha_low),
        "label_uniform_note": (
            "Labels use a single shared uniform vector U drawn from a dedicated "
            "RNG stream (stream 3 of 3 spawned from the base seed via "
            "numpy SeedSequence.spawn). y = (U < sigmoid(alpha*eta)); the SAME U "
            "is used for the high- and low-SNR cells, so only alpha differs."
        ),
        "proxy_epsilon_note": (
            "Proxy epsilon noise fields are drawn from a dedicated RNG stream "
            "(stream 2 of 3 spawned from the base seed). The SAME epsilon arrays "
            "are used for the rho=0.9 and rho=0.7 cells, so only rho differs."
        ),
        "shared_planted_indices_across_grid": True,
        "shared_base_matrix_across_grid": True,
        "shared_label_uniforms_across_snr": True,
        "shared_proxy_epsilons_across_rho": True,
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": list(shared["solo_core"]),
        "true_core_anchored_indices": list(shared["anchored_core"]),
        "proxy_groups": {str(par): list(px)
                         for par, px in shared["proxy_groups"].items()},
        "weak_signal_indices": list(weak_idx),
        "pure_noise_indices": {
            "count": len(noise_idx),
            "sample_first10": noise_idx[:10],
            "sample_last10": noise_idx[-10:],
            "definition": "all indices in 0..1999 not assigned to core/proxy/weak",
        },
        "core_beta_per_feature": {str(i): float(CORE_BETA) for i in core_idx},
        "weak_gamma_per_feature": {str(i): float(WEAK_GAMMA_V1) for i in weak_idx},
        "signal_construction_notes": (
            "CONTROLLED grid cell. A single base feature matrix X (all 2000 "
            "columns i.i.d. N(0,1), with the 5 core and 15 weak columns z-scored) "
            "and a single planted-index layout are shared by all 4 known-core "
            "cells. The linear predictor eta = sum(beta*core) + sum(gamma*weak) "
            f"(beta={CORE_BETA}, gamma={WEAK_GAMMA_V1}) is computed once and "
            "shared. Only the proxy columns differ across cells: "
            f"proxy = rho*parent + sqrt(1-rho^2)*eps with rho={rho}, using "
            "epsilon arrays shared across the rho levels. The core, weak, and "
            "noise columns are identical across all 4 cells."
        ),
        "label_construction_notes": (
            f"y = (U < sigmoid(alpha*eta)) with alpha={alpha}. U is a shared "
            "uniform vector; the high- and low-SNR cells differ only in alpha, "
            "so the SNR axis is cleanly isolated."
        ),
    }
    return X.astype(np.float64), y, manifest


def generate_controlled_grid(base_seed):
    """Generate the controlled gammaV1c grid: 4 known-core cells sharing one
    base matrix, planted layout, eta, label-uniform vector, and proxy epsilons.

    Returns (known_cells, selection_info) where known_cells is a list of
    (X, y, manifest) and selection_info records the alpha_low choice.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES

    # three independent RNG streams spawned from one base seed
    s_layout, s_eps, s_label = np.random.SeedSequence(base_seed).spawn(3)
    rng_layout = np.random.default_rng(s_layout)
    rng_eps = np.random.default_rng(s_eps)
    rng_label = np.random.default_rng(s_label)

    # base matrix + planted layout — generated ONCE, shared by all 4 cells
    X_base = rng_layout.standard_normal((n, p))
    planted = place_planted_indices(rng_layout, p, N_PLANTED)
    solo_core = sorted(int(i) for i in planted[:N_SOLO_CORE])
    anchored_core = sorted(int(i) for i in planted[N_SOLO_CORE:N_CORE])
    proxy_block = [int(i) for i in planted[N_CORE:N_CORE + N_PROXY]]
    weak_idx = sorted(int(i) for i in planted[N_CORE + N_PROXY:])
    core_idx = sorted(solo_core + anchored_core)
    planted_set = set(core_idx) | set(proxy_block) | set(weak_idx)
    noise_idx = [i for i in range(p) if i not in planted_set]
    proxy_groups = {}
    for g, parent in enumerate(anchored_core):
        proxy_groups[parent] = sorted(
            proxy_block[g * PROXIES_PER_GROUP:(g + 1) * PROXIES_PER_GROUP]
        )

    # standardize core + weak columns ONCE
    standardize_columns(X_base, core_idx)
    standardize_columns(X_base, weak_idx)

    # eta — computed ONCE, shared by all cells
    eta = np.zeros(n, dtype=np.float64)
    for i in core_idx:
        eta += CORE_BETA * X_base[:, i]
    for i in weak_idx:
        eta += WEAK_GAMMA_V1 * X_base[:, i]

    # shared label-uniform vector and shared per-proxy epsilon arrays
    U = rng_label.random(n)
    eps_by_proxy = {int(pj): rng_eps.standard_normal(n) for pj in proxy_block}

    # --- choose alpha_low: try candidates, accept the first that passes -----
    alpha_high = SNR_ALPHA["high"]
    trials = []
    chosen_alpha_low = None
    for cand in CONTROLLED_ALPHA_LOW_CANDIDATES:
        y_cand = (U < sigmoid(cand * eta)).astype(np.int64)
        core_m, weak_m, noise_m = _core_weak_noise_corr(
            X_base, y_cand, core_idx, weak_idx, proxy_block)
        minf = float(min((y_cand == 0).mean(), (y_cand == 1).mean()))
        ordering_ok = bool(core_m > weak_m > noise_m)
        core_visible = bool(core_m > 2.0 * noise_m)
        balance_ok = bool(minf >= 0.25)
        accepted = bool(ordering_ok and core_visible and balance_ok)
        trials.append({
            "alpha_low": cand,
            "core_mean_abs_corr": round(core_m, 4),
            "weak_mean_abs_corr": round(weak_m, 4),
            "noise_mean_abs_corr": round(noise_m, 4),
            "minority_fraction": round(minf, 4),
            "ordering_ok": ordering_ok,
            "core_visible_above_noise": core_visible,
            "balance_ok": balance_ok,
            "accepted": accepted,
        })
        if accepted and chosen_alpha_low is None:
            chosen_alpha_low = cand
    selection_passed = chosen_alpha_low is not None
    if not selection_passed:
        chosen_alpha_low = CONTROLLED_ALPHA_LOW_CANDIDATES[-1]

    selection_info = {
        "alpha_high": float(alpha_high),
        "alpha_low_candidates": list(CONTROLLED_ALPHA_LOW_CANDIDATES),
        "alpha_low_trials": trials,
        "chosen_alpha_low": float(chosen_alpha_low),
        "selection_passed": bool(selection_passed),
    }

    shared = {
        "base_seed": int(base_seed),
        "X_base": X_base,
        "eta": eta,
        "U": U,
        "eps_by_proxy": eps_by_proxy,
        "solo_core": solo_core,
        "anchored_core": anchored_core,
        "core_idx": core_idx,
        "proxy_block": proxy_block,
        "proxy_groups": proxy_groups,
        "weak_idx": weak_idx,
        "noise_idx": noise_idx,
    }

    # --- build the 4 known-core cells --------------------------------------
    known_cells = []
    for snr, rho in GRID_V1_CELLS:
        alpha = alpha_high if snr == "high" else chosen_alpha_low
        known_cells.append(_controlled_cell(snr, alpha, rho, shared, chosen_alpha_low))

    return known_cells, selection_info


def verify_shared_planted_indices(known_manifests):
    """Grid-level check: all 4 known-core cells must share identical planted
    indices (the controlled-design requirement)."""
    ref = known_manifests[0]
    checks = {
        "identical_solo": bool(all(
            m["true_core_solo_indices"] == ref["true_core_solo_indices"]
            for m in known_manifests)),
        "identical_anchored": bool(all(
            m["true_core_anchored_indices"] == ref["true_core_anchored_indices"]
            for m in known_manifests)),
        "identical_proxy_groups": bool(all(
            m["proxy_groups"] == ref["proxy_groups"] for m in known_manifests)),
        "identical_weak": bool(all(
            m["weak_signal_indices"] == ref["weak_signal_indices"]
            for m in known_manifests)),
    }
    checks["all_identical"] = bool(all(checks.values()))
    checks["reference_solo"] = ref["true_core_solo_indices"]
    checks["reference_anchored"] = ref["true_core_anchored_indices"]
    checks["reference_weak"] = ref["weak_signal_indices"]
    checks["reference_proxy_group_keys"] = sorted(ref["proxy_groups"].keys())
    return checks


def build_grid_v1_controlled_report(commands, written, manifests, verifications,
                                     reloads, selection_info, shared_index_check,
                                     gap_info, ready):
    L = []
    L.append("# Synthetic Grid v1 — Controlled Retune Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Controlled regeneration of the v1 grid — all 4 known-core cells share "
             "the planted indices, base matrix, eta, label uniforms, and proxy "
             "epsilons; only alpha (SNR) and rho vary. Generator update + 2x2 grid "
             "generation + structural verification only. No feature selection, no "
             "resampling, no analyzer, no recovery metrics, no real dataset.\n")
    L.append("---\n")

    L.append("## 1. Commands Run\n")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")

    L.append("## 2. Generator Changes Made\n")
    L.append("- **Controlled grid mode added (`--grid-v1-controlled`).** All 4 "
             "known-core cells share one base feature matrix, one planted-index "
             "layout, one linear predictor eta, one label-uniform vector U, and one "
             "set of proxy epsilon arrays. Only **alpha** (SNR, via the label draw) "
             "and **rho** (proxy correlation) differ across cells — removing the "
             "per-cell-seed confound of the earlier `--grid-v1`.")
    L.append("- **Three separated RNG streams** are spawned from one base seed via "
             "`numpy.random.SeedSequence(base_seed).spawn(3)`: stream 1 = layout + "
             "base matrix; stream 2 = proxy epsilons; stream 3 = label uniforms.")
    L.append(f"- **Lower alpha_low.** The controlled mode selects alpha_low from "
             f"candidates {selection_info['alpha_low_candidates']} "
             f"(alpha_high = {selection_info['alpha_high']}), auto-accepting the "
             f"lowest candidate that passes the structural checks. "
             f"`SNR_ALPHA['low']` (0.5) is left unchanged so the existing "
             f"`--grid-v1` mode is preserved.")
    L.append(f"- **Weak gamma unchanged** at `WEAK_GAMMA_V1 = {WEAK_GAMMA_V1}` — a "
             f"structural ceiling per the pre-logging checkpoint; weak-signal "
             f"recovery remains a cautious, secondary outcome.\n")

    L.append("## 3. alpha_low Selection\n")
    L.append("Each candidate is evaluated on the shared eta and label uniforms; the "
             "lowest candidate yielding strict `core > weak > noise`, `core > "
             "2*noise`, and reasonable class balance is selected.\n")
    L.append("| alpha_low | core | weak | noise | minority frac | ordering | "
             "core>2·noise | balance | accepted |")
    L.append("|-----------|------|------|-------|---------------|----------|"
             "--------------|---------|----------|")
    for t in selection_info["alpha_low_trials"]:
        L.append(f"| {t['alpha_low']} | {t['core_mean_abs_corr']} | "
                 f"{t['weak_mean_abs_corr']} | {t['noise_mean_abs_corr']} | "
                 f"{t['minority_fraction']} | {t['ordering_ok']} | "
                 f"{t['core_visible_above_noise']} | {t['balance_ok']} | "
                 f"{t['accepted']} |")
    L.append("")
    chosen = selection_info["chosen_alpha_low"]
    first = selection_info["alpha_low_candidates"][0]
    if selection_info["selection_passed"] and chosen == first:
        L.append(f"- **Chosen alpha_low = {chosen}.** The primary candidate ({first}) "
                 f"passed all structural checks — no fallback was needed.")
    elif selection_info["selection_passed"]:
        L.append(f"- **Chosen alpha_low = {chosen}.** The primary candidate ({first}) "
                 f"did not pass the structural checks (low-SNR core too close to "
                 f"weak/noise); the generator stepped up to the first candidate that "
                 f"did. This fallback is documented here per the task instruction.")
    else:
        L.append(f"- **No candidate passed.** alpha_low fell back to the "
                 f"least-aggressive candidate {chosen}; the grid is NOT ready (see "
                 f"section 14).")
    L.append("")

    L.append("## 4. Controlled-Grid Design\n")
    L.append("| Property | Shared across the 4 known-core cells? |")
    L.append("|----------|----------------------------------------|")
    L.append("| planted feature indices (core/proxy/weak/noise) | yes |")
    L.append("| base feature matrix (core / weak / noise columns) | yes |")
    L.append("| linear predictor eta | yes |")
    L.append("| label-uniform vector U | yes — high & low SNR use the same U |")
    L.append("| proxy epsilon arrays | yes — rho=0.9 & rho=0.7 use the same eps |")
    L.append("| alpha (SNR level) | **varies** — high vs low |")
    L.append("| rho (proxy correlation) | **varies** — 0.9 vs 0.7 |")
    L.append("")

    L.append("## 5. Output Files\n")
    for label, path in written:
        L.append(f"- **{label}:** `{path}`")
    L.append(f"- **This report:** `{GRID_V1C_REPORT_PATH}`\n")

    L.append("## 6. Class Balance per Cell\n")
    L.append("| Dataset | class 0 | class 1 | minority fraction |")
    L.append("|---------|---------|---------|-------------------|")
    for m in manifests:
        cb = m["class_balance"]
        L.append(f"| {m['dataset_name']} | {cb['n_class_0']} | {cb['n_class_1']} | "
                 f"{cb['minority_fraction']:.3f} |")
    L.append("")

    L.append("## 7. Proxy-Parent Correlation per Cell\n")
    L.append("| Dataset | target rho | mean | min | max | close to rho |")
    L.append("|---------|------------|------|-----|-----|--------------|")
    for v in verifications:
        c = v["checks"]
        if "proxy_parent_corr" not in c:
            continue
        pc = c["proxy_parent_corr"]
        L.append(f"| {v['dataset_name']} | {pc['target_rho']} | {pc['mean']} | "
                 f"{pc['min']} | {pc['max']} | {c['proxy_parent_corr_close_to_rho']} |")
    L.append("")

    L.append("## 8. Core / Weak / Noise Association per Cell\n")
    L.append("Mean absolute Pearson correlation with y. The hard gate is the strict "
             "ordering `core > weak > noise`.\n")
    L.append("| Dataset | core | weak | noise mean | noise max | "
             "core>weak>noise | weak>noise |")
    L.append("|---------|------|------|------------|-----------|"
             "-----------------|------------|")
    for v in verifications:
        c = v["checks"]
        if "mean_abs_corr_with_y" not in c:
            continue
        mc = c["mean_abs_corr_with_y"]
        L.append(f"| {v['dataset_name']} | {mc['core_all']} | {mc['weak']} | "
                 f"{mc['noise_sample_mean']} | {mc['noise_sample_max']} | "
                 f"{c['ordering_core_gt_weak_gt_noise']} | "
                 f"{c['weak_stronger_than_noise']} |")
    L.append("")

    L.append("## 9. High-vs-Low SNR Core Gap, by rho\n")
    L.append("| rho | high-SNR core | low-SNR core | gap |")
    L.append("|-----|---------------|--------------|-----|")
    L.append(f"| 0.9 | {gap_info['high_core_rho090']} | {gap_info['low_core_rho090']} "
             f"| {gap_info['gap_rho090']} |")
    L.append(f"| 0.7 | {gap_info['high_core_rho070']} | {gap_info['low_core_rho070']} "
             f"| {gap_info['gap_rho070']} |")
    L.append("")
    L.append(f"- The earlier uncontrolled v1 grid had a high-vs-low core gap of "
             f"≈ 0.015 at rho=0.9. The controlled gap is **{gap_info['gap_rho090']}** "
             f"(rho=0.9) / **{gap_info['gap_rho070']}** (rho=0.7) — a material "
             f"improvement.")
    L.append("- In the controlled design rho does not touch the core columns, so the "
             "high-vs-low core gap is the same at both rho levels — a confirmation "
             "that the SNR and rho axes are cleanly separated.\n")

    L.append("## 10. Shared Planted-Index Verification (grid-level)\n")
    sc = shared_index_check
    L.append(f"- identical solo-core indices across all 4 cells: **{sc['identical_solo']}**")
    L.append(f"- identical anchored-core indices: **{sc['identical_anchored']}**")
    L.append(f"- identical proxy groups: **{sc['identical_proxy_groups']}**")
    L.append(f"- identical weak indices: **{sc['identical_weak']}**")
    L.append(f"- **all planted indices identical across the 4 known-core cells: "
             f"{sc['all_identical']}**")
    L.append(f"- reference solo={sc['reference_solo']}, "
             f"anchored={sc['reference_anchored']}")
    L.append("")

    L.append("## 11. Null-Control Verification\n")
    null_v = next((v for v in verifications
                   if v["dataset_name"].startswith("synthetic_null")), None)
    if null_v:
        c = null_v["checks"]
        L.append(f"- `{null_v['dataset_name']}` — "
                 f"{'PASS' if null_v['passed'] else 'CHECK WARNINGS'}")
        L.append(f"- shape_ok={c['shape_ok']}, Y_is_binary={c['Y_is_binary']}, "
                 f"X_finite={c['X_finite_no_nan_inf']}")
        L.append(f"- class_balance: {c['class_balance']}")
        L.append(f"- abs_corr_with_y: {c['abs_corr_with_y']}")
        L.append(f"- no_strong_planted_signal: {c['no_strong_planted_signal']}")
    L.append("")

    L.append("## 12. Reload Checks (`.mat` round-trip)\n")
    for name, rc in reloads.items():
        L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                 f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, Y_shape={rc['Y_shape']}")
    L.append("")

    L.append("## 13. Warnings / Caveats\n")
    any_warn = any(v["warnings"] for v in verifications)
    if any_warn:
        for v in verifications:
            for w in v["warnings"]:
                L.append(f"- [{v['dataset_name']}] {w}")
    else:
        L.append("- No structural warnings.")
    L.append("- **Weak magnitude remains modest** (~0.13-0.15) — weak γ is at its "
             "structural ceiling (0.8); weak features sit clearly above the noise "
             "*mean* but may sit below the noise *sample maximum*. Weak-signal "
             "recovery stays a cautious, secondary outcome (not retuned here).")
    L.append("- `weak_above_noise_sample_max` is informational only, not part of the "
             "hard gate.")
    L.append("- These datasets are **synthetic**; \"planted core\" wording is licensed "
             "only for synthetic data.\n")

    L.append("## 14. Is the Controlled Grid Ready for 50-Resample Logging?\n")
    if ready:
        L.append("- **Yes.** Acceptance gate met: (a) strict `core > weak > noise` "
                 "with `core > 2·noise` in every known-core cell; (b) the high-vs-low "
                 f"core gap (rho=0.9: {gap_info['gap_rho090']}, rho=0.7: "
                 f"{gap_info['gap_rho070']}) is materially larger than the prior "
                 "≈0.015; (c) all 4 known-core cells share identical planted indices; "
                 "(d) class balance reasonable; (e) proxy-parent rho on target.")
    else:
        L.append("- **No — stop before logging.** The acceptance gate is not fully "
                 "met. Review sections 3, 8, 9, 10, 13 — in particular any "
                 "`core > weak > noise` failure, an insufficient high-vs-low gap, or "
                 "a shared-index mismatch — and resolve before any logging.")
    L.append("")

    L.append("## 15. Recommended Next Step\n")
    if ready:
        L.append("- Proceed to the 50-resample logging pipeline per controlled grid "
                 "cell: for each of the 4 known-core `_gammaV1c` datasets and the "
                 "`_gammaV1c` null, run `run_synthetic_repeated_resampling_logging.py` "
                 "(50 resamples), then `analyze_per_fold_stability.py`, then "
                 "`synthetic_recovery_metrics.py` — reusing those scripts unmodified. "
                 "This generation task runs none of them.")
    else:
        L.append("- Do not proceed to logging. Resolve the acceptance-gate failure(s) "
                 "in section 14, regenerate, and re-verify.")
    L.append("")

    L.append("---\n")
    L.append("*Generator step only — controlled grid generated and structurally "
             "verified. No feature selection, no logging, no analyzer, no recovery "
             "metrics, no real dataset, no benchmark file modified.*")
    return "\n".join(L)


def run_grid_v1_controlled_generation(seed, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic Grid v1 CONTROLLED retune generator")
    print("=" * 70)
    print(f"Base seed         : {seed}")
    print(f"Grid cells        : {GRID_V1_CELLS}")
    print(f"Weak gamma (v1)   : {WEAK_GAMMA_V1}  (unchanged)")
    print(f"alpha_high        : {SNR_ALPHA['high']}")
    print(f"alpha_low cands   : {CONTROLLED_ALPHA_LOW_CANDIDATES}")
    print(f"Suffix            : {GRID_V1C_SUFFIX}")
    print(f"Output directory  : {out_dir}")
    print("-" * 70)

    known_cells, selection_info = generate_controlled_grid(seed)
    print(f"alpha_low selection : chosen={selection_info['chosen_alpha_low']} "
          f"(passed={selection_info['selection_passed']})")
    for t in selection_info["alpha_low_trials"]:
        print(f"  candidate {t['alpha_low']}: core={t['core_mean_abs_corr']} "
              f"weak={t['weak_mean_abs_corr']} noise={t['noise_mean_abs_corr']} "
              f"accepted={t['accepted']}")
    print("-" * 70)

    null_seed = seed + NULL_SEED_OFFSET
    Xn, yn, man_null = generate_null_control(null_seed, name_suffix=GRID_V1C_SUFFIX)

    verifications, known_manifests, datasets = [], [], []
    for X, y, manifest in known_cells:
        v = verify_known_core(X, y, manifest, rho=manifest["rho"])
        embed_verification(manifest, v)
        verifications.append(v)
        known_manifests.append(manifest)
        datasets.append((manifest, X, y))
        print(f"Generated {manifest['dataset_name']:42s} "
              f"verify={'PASS' if v['passed'] else 'CHECK'}")
    vn = verify_null_control(Xn, yn, man_null)
    embed_verification(man_null, vn)
    verifications.append(vn)
    datasets.append((man_null, Xn, yn))
    print(f"Generated {man_null['dataset_name']:42s} "
          f"verify={'PASS' if vn['passed'] else 'CHECK'}")

    # grid-level checks
    shared_index_check = verify_shared_planted_indices(known_manifests)
    core_by = {}
    for v, m in zip(verifications[:len(known_manifests)], known_manifests):
        core_by[(m["snr_label"], float(m["rho"]))] = \
            v["checks"]["mean_abs_corr_with_y"]["core_all"]
    gap_info = {
        "high_core_rho090": core_by[("high", 0.9)],
        "low_core_rho090": core_by[("low", 0.9)],
        "high_core_rho070": core_by[("high", 0.7)],
        "low_core_rho070": core_by[("low", 0.7)],
        "gap_rho090": round(core_by[("high", 0.9)] - core_by[("low", 0.9)], 4),
        "gap_rho070": round(core_by[("high", 0.7)] - core_by[("low", 0.7)], 4),
    }

    # write all datasets
    print("-" * 70)
    written, reloads = [], {}
    for manifest, X, y in datasets:
        mat_p, man_p = write_dataset(out_dir, manifest, X, y)
        label = "null-control" if manifest["snr_label"] == "null" else "known-core"
        written.append((f"{label} dataset", mat_p))
        written.append((f"{label} manifest", man_p))
        reloads[manifest["dataset_name"]] = reload_check(mat_p)

    all_passed = all(v["passed"] for v in verifications)
    gap_material = bool(gap_info["gap_rho090"] > 0.04 and gap_info["gap_rho070"] > 0.04)
    ready = bool(all_passed and shared_index_check["all_identical"]
                 and gap_material and selection_info["selection_passed"])

    manifests = known_manifests + [man_null]
    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run --grid-v1-controlled",
        "python articles/article_stability/scripts/make_synthetic_control.py --grid-v1-controlled",
    ]
    GRID_V1C_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(GRID_V1C_REPORT_PATH.parent, GRID_V1C_REPORT_PATH.name,
                            GRID_V1C_REPORT_PATH.parent)
    report_path.write_text(build_grid_v1_controlled_report(
        commands, written, manifests, verifications, reloads,
        selection_info, shared_index_check, gap_info, ready,
    ))

    print("-" * 70)
    for v in verifications:
        for w in v["warnings"]:
            print(f"  [warn] {v['dataset_name']}: {w}")
    print(f"Shared planted indices identical : {shared_index_check['all_identical']}")
    print(f"High-vs-low core gap : rho=0.9={gap_info['gap_rho090']}  "
          f"rho=0.7={gap_info['gap_rho070']}")
    print(f"Markdown report   : {report_path}")
    print(f"Controlled grid ready for logging: {ready}")
    print("=" * 70)
    return 0 if ready else 1


def _dry_run_grid_v1_controlled(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic Grid v1 CONTROLLED retune generator  [DRY RUN]")
    print("=" * 70)
    print(f"Base seed             : {args.seed}")
    print(f"Datasets output root  : {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory      : {out_dir}")
    print(f"Weak gamma (unchanged): {WEAK_GAMMA_V1}")
    print(f"alpha_high            : {SNR_ALPHA['high']}")
    print(f"alpha_low candidates  : {CONTROLLED_ALPHA_LOW_CANDIDATES} "
          f"(lowest passing structural checks is auto-selected)")
    print(f"Dataset-name suffix   : {GRID_V1C_SUFFIX}")
    print("Planned controlled grid cells (all 4 share planted indices + base matrix):")
    planned = []
    for snr, rho in GRID_V1_CELLS:
        name = f"synthetic_snr{snr.upper()}_rho{int(round(rho*100)):03d}{GRID_V1C_SUFFIX}"
        print(f"  - SNR={snr} (alpha_high={SNR_ALPHA['high']} / alpha_low=candidate), "
              f"rho={rho}  ->  {name}")
        planned += [f"{name}.mat", f"{name}_manifest.json"]
    null_name = f"synthetic_null_control{GRID_V1C_SUFFIX}"
    print(f"  - null control (seed {args.seed + NULL_SEED_OFFSET})  ->  {null_name}")
    planned += [f"{null_name}.mat", f"{null_name}_manifest.json"]
    print("Planned output files:")
    for f in planned:
        print(f"  - {out_dir / f}")
    print(f"Planned markdown report: {GRID_V1C_REPORT_PATH}")
    print("Controlled design: the 4 known-core cells share identical planted indices, "
          "base feature matrix, eta, label uniforms, and proxy epsilons; only alpha "
          "(SNR) and rho vary across cells.")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


# ==========================================================================
# Instance-variance diagnostic (gammaV1d) — multi-seed reduced probe
# ==========================================================================
def _oracle_core_auc(X, y, core_idx, n_splits=5, test_size=0.2, seed=20260521):
    """Quick held-out ExtraTrees AUC trained on the KNOWN true-core columns only.

    This is an ORACLE probe, NOT feature selection: the true-core indices come
    straight from the planted ground truth. It quantifies whether the planted
    signal is operationally learnable when the correct features are handed in —
    the question the alpha=0.25 collapse exposed (a core that beats the noise
    *mean* can still be operationally unrecoverable). A plain ExtraTrees is used
    here, separate from the pipeline's evaluation classifier; this keeps the
    generator isolated from the benchmark engine.
    """
    cols = sorted(int(i) for i in core_idx)
    if not cols:
        return {"mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_splits": 0}
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit

    Xc = np.asarray(X, dtype=np.float64)[:, cols]
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    aucs = []
    for tr, te in sss.split(Xc, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = ExtraTreesClassifier(n_estimators=200, random_state=seed)
        clf.fit(Xc[tr], y[tr])
        proba = clf.predict_proba(Xc[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], proba)))
    if not aucs:
        return {"mean": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_splits": 0}
    return {"mean": float(np.mean(aucs)), "min": float(np.min(aucs)),
            "max": float(np.max(aucs)), "n_splits": len(aucs)}


def compute_operational_diagnostic(X, y, manifest):
    """Generation-time operational-learnability diagnostic for one known-core
    dataset.

    Reports the core mean |corr with y| against the noise UPPER TAIL (p95 / p99
    / max), not just the noise mean — the alpha=0.25 collapse showed that
    `core > noise_mean` does not imply operational recoverability against ~2000
    distractors at n=200. Also runs a quick oracle held-out AUC (true-core
    columns only). This does NOT run feature selection.
    """
    solo = [int(i) for i in manifest.get("true_core_solo_indices", [])]
    anchored = [int(i) for i in manifest.get("true_core_anchored_indices", [])]
    core_idx = sorted(solo + anchored)
    weak_idx = sorted(int(i) for i in manifest.get("weak_signal_indices", []))
    proxy_all = set()
    for plist in manifest.get("proxy_groups", {}).values():
        proxy_all.update(int(x) for x in plist)
    planted = set(core_idx) | set(weak_idx) | proxy_all
    noise_idx = [i for i in range(N_TOTAL_FEATURES) if i not in planted]

    core_corr = np.array([abs_corr_with_y(X[:, i], y) for i in core_idx])
    weak_corr = np.array([abs_corr_with_y(X[:, i], y) for i in weak_idx])
    noise_corr = np.array([abs_corr_with_y(X[:, i], y) for i in noise_idx])

    core_mean = float(core_corr.mean())
    weak_mean = float(weak_corr.mean())
    noise_mean = float(noise_corr.mean())
    noise_p95 = float(np.percentile(noise_corr, 95))
    noise_p99 = float(np.percentile(noise_corr, 99))
    noise_max = float(noise_corr.max())

    oracle = _oracle_core_auc(X, y, core_idx)

    return {
        "core_mean_abs_corr": round(core_mean, 4),
        "weak_mean_abs_corr": round(weak_mean, 4),
        "noise_mean_abs_corr": round(noise_mean, 4),
        "noise_p95_abs_corr": round(noise_p95, 4),
        "noise_p99_abs_corr": round(noise_p99, 4),
        "noise_max_abs_corr": round(noise_max, 4),
        "core_vs_noise_mean_margin": round(core_mean - noise_mean, 4),
        "core_vs_noise_p99_margin": round(core_mean - noise_p99, 4),
        "core_vs_noise_max_margin": round(core_mean - noise_max, 4),
        "core_above_noise_p99": bool(core_mean > noise_p99),
        "core_above_noise_max": bool(core_mean > noise_max),
        "noise_features_scanned": int(len(noise_idx)),
        "oracle_true_core_auc_mean": round(oracle["mean"], 4),
        "oracle_true_core_auc_min": round(oracle["min"], 4),
        "oracle_true_core_auc_max": round(oracle["max"], 4),
        "oracle_true_core_auc_n_splits": int(oracle["n_splits"]),
        "notes": (
            "core / weak / noise mean |corr with y| over ALL planted-role and ALL "
            "pure-noise features. core_vs_noise_p99_margin and "
            "core_vs_noise_max_margin compare the core mean to the noise UPPER "
            "TAIL; the alpha=0.25 collapse showed core > noise_mean is not "
            "sufficient for operational recoverability. oracle_true_core_auc is a "
            "quick held-out ExtraTrees AUC trained on the KNOWN true-core columns "
            "only (an oracle probe; NOT feature selection) — it isolates whether "
            "the planted signal is learnable when the true features are supplied."
        ),
    }


def _build_diagnostic_shared(base_seed, core_beta=CORE_BETA):
    """Shared within-seed structure for one diagnostic instance: base feature
    matrix, planted layout, linear predictor eta, label-uniform vector U, and
    proxy epsilon arrays.

    Mirrors the controlled-grid shared construction (three RNG streams spawned
    from one base seed); the high-SNR and mid035 cells built from it differ
    ONLY in alpha. Across diagnostic seeds a fresh base_seed yields a fresh
    layout and base matrix.

    `core_beta` is the per-core-feature coefficient in the linear predictor.
    It defaults to CORE_BETA (the v1c/v1d value) so existing callers are
    unchanged; the positive-control redesign passes a stronger value.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    s_layout, s_eps, s_label = np.random.SeedSequence(base_seed).spawn(3)
    rng_layout = np.random.default_rng(s_layout)
    rng_eps = np.random.default_rng(s_eps)
    rng_label = np.random.default_rng(s_label)

    X_base = rng_layout.standard_normal((n, p))
    planted = place_planted_indices(rng_layout, p, N_PLANTED)
    solo_core = sorted(int(i) for i in planted[:N_SOLO_CORE])
    anchored_core = sorted(int(i) for i in planted[N_SOLO_CORE:N_CORE])
    proxy_block = [int(i) for i in planted[N_CORE:N_CORE + N_PROXY]]
    weak_idx = sorted(int(i) for i in planted[N_CORE + N_PROXY:])
    core_idx = sorted(solo_core + anchored_core)
    planted_set = set(core_idx) | set(proxy_block) | set(weak_idx)
    noise_idx = [i for i in range(p) if i not in planted_set]
    proxy_groups = {}
    for g, parent in enumerate(anchored_core):
        proxy_groups[parent] = sorted(
            proxy_block[g * PROXIES_PER_GROUP:(g + 1) * PROXIES_PER_GROUP]
        )

    standardize_columns(X_base, core_idx)
    standardize_columns(X_base, weak_idx)

    eta = np.zeros(n, dtype=np.float64)
    for i in core_idx:
        eta += core_beta * X_base[:, i]
    for i in weak_idx:
        eta += WEAK_GAMMA_V1 * X_base[:, i]

    U = rng_label.random(n)
    eps_by_proxy = {int(pj): rng_eps.standard_normal(n) for pj in proxy_block}

    return {
        "base_seed": int(base_seed),
        "core_beta": float(core_beta),
        "X_base": X_base, "eta": eta, "U": U, "eps_by_proxy": eps_by_proxy,
        "solo_core": solo_core, "anchored_core": anchored_core,
        "core_idx": core_idx, "proxy_block": proxy_block,
        "proxy_groups": proxy_groups, "weak_idx": weak_idx, "noise_idx": noise_idx,
    }


def _diagnostic_cell(snr_label, alpha, rho, shared, seed_index, suffix):
    """Build one diagnostic known-core cell (high SNR or mid035) from the shared
    within-seed structure. Only `alpha` differs between the two cells of a seed.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    X = shared["X_base"].copy()

    sqrt_term = math.sqrt(max(0.0, 1.0 - rho ** 2))
    for parent, proxies in shared["proxy_groups"].items():
        for pj in proxies:
            X[:, pj] = (rho * shared["X_base"][:, parent]
                        + sqrt_term * shared["eps_by_proxy"][int(pj)])

    prob = sigmoid(alpha * shared["eta"])
    y = (shared["U"] < prob).astype(np.int64)

    core_idx = shared["core_idx"]
    weak_idx = shared["weak_idx"]
    noise_idx = shared["noise_idx"]
    rho_code = int(round(rho * 100))
    manifest = {
        "dataset_name": (f"synthetic_snr{snr_label.upper()}_rho{rho_code:03d}"
                         f"{suffix}"),
        "generation_seed": int(shared["base_seed"]),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": snr_label,
        "alpha_value": float(alpha),
        "rho": float(rho),
        "weak_gamma": float(WEAK_GAMMA_V1),
        "core_beta": float(CORE_BETA),
        "instance_diagnostic_version": "gammaV1d",
        "diagnostic_seed_index": int(seed_index),
        "base_generation_seed": int(shared["base_seed"]),
        "shared_within_seed_planted_indices": True,
        "shared_within_seed_base_matrix": True,
        "shared_within_seed_label_uniforms": True,
        "shared_within_seed_proxy_epsilons": True,
        "label_uniform_note": (
            "Labels use a single shared uniform vector U drawn from a dedicated "
            "RNG stream (stream 3 of 3 spawned from the base seed via "
            "numpy SeedSequence.spawn). y = (U < sigmoid(alpha*eta)); the SAME U "
            "is used for the high-SNR and mid035 cells of this seed, so within a "
            "seed only alpha differs."
        ),
        "proxy_epsilon_note": (
            "Proxy epsilon noise fields are drawn from a dedicated RNG stream "
            "(stream 2 of 3 spawned from the base seed) and are shared by the "
            "high-SNR and mid035 cells of this seed."
        ),
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": list(shared["solo_core"]),
        "true_core_anchored_indices": list(shared["anchored_core"]),
        "proxy_groups": {str(par): list(px)
                         for par, px in shared["proxy_groups"].items()},
        "weak_signal_indices": list(weak_idx),
        "pure_noise_indices": {
            "count": len(noise_idx),
            "sample_first10": noise_idx[:10],
            "sample_last10": noise_idx[-10:],
            "definition": "all indices in 0..1999 not assigned to core/proxy/weak",
        },
        "core_beta_per_feature": {str(i): float(CORE_BETA) for i in core_idx},
        "weak_gamma_per_feature": {str(i): float(WEAK_GAMMA_V1) for i in weak_idx},
        "signal_construction_notes": (
            "Instance-diagnostic (gammaV1d) known-core cell. A single base "
            "feature matrix (all 2000 columns i.i.d. N(0,1), the 5 core and 15 "
            "weak columns z-scored) and a single planted-index layout are shared "
            "within the generation seed by the high-SNR and mid035 cells. The "
            f"linear predictor eta = sum(beta*core)+sum(gamma*weak) (beta="
            f"{CORE_BETA}, gamma={WEAK_GAMMA_V1}) is computed once and shared. "
            f"Proxy columns proxy = rho*parent + sqrt(1-rho^2)*eps with rho={rho}."
        ),
        "label_construction_notes": (
            f"y = (U < sigmoid(alpha*eta)) with alpha={alpha}. U is the shared "
            "within-seed uniform vector; the high-SNR and mid035 cells of this "
            "seed differ only in alpha, isolating the SNR axis."
        ),
    }
    return X.astype(np.float64), y, manifest


def generate_diagnostic_instance(base_seed, seed_index):
    """Generate one diagnostic instance: a within-seed-controlled (high SNR,
    mid035) known-core pair plus a null control. Returns three (X, y, manifest)
    tuples (high, mid, null).
    """
    suffix = f"{GRID_V1D_SUFFIX}_seed{seed_index:02d}"
    shared = _build_diagnostic_shared(base_seed)
    high = _diagnostic_cell("high", DIAGNOSTIC_ALPHA_HIGH, DIAGNOSTIC_RHO,
                            shared, seed_index, suffix)
    mid = _diagnostic_cell("mid035", DIAGNOSTIC_ALPHA_MID, DIAGNOSTIC_RHO,
                           shared, seed_index, suffix)
    Xn, yn, man_null = generate_null_control(base_seed + NULL_SEED_OFFSET,
                                             name_suffix=suffix)
    man_null["instance_diagnostic_version"] = "gammaV1d"
    man_null["diagnostic_seed_index"] = int(seed_index)
    man_null["base_generation_seed"] = int(base_seed)
    return high, mid, (Xn, yn, man_null)


def verify_diagnostic_within_seed(high_manifest, mid_manifest):
    """Within-seed control check: the high-SNR and mid035 known-core cells of a
    seed must share identical planted indices (only alpha may differ)."""
    checks = {
        "identical_solo": bool(
            high_manifest["true_core_solo_indices"]
            == mid_manifest["true_core_solo_indices"]),
        "identical_anchored": bool(
            high_manifest["true_core_anchored_indices"]
            == mid_manifest["true_core_anchored_indices"]),
        "identical_proxy_groups": bool(
            high_manifest["proxy_groups"] == mid_manifest["proxy_groups"]),
        "identical_weak": bool(
            high_manifest["weak_signal_indices"]
            == mid_manifest["weak_signal_indices"]),
    }
    checks["all_identical"] = bool(all(checks.values()))
    return checks


def build_instance_diagnostic_report(commands, written, instances, reloads,
                                     all_struct_ok, base_seed, n_seeds):
    L = []
    L.append("# Synthetic Instance-Variance Diagnostic (gammaV1d) — "
             "Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Generator step only — multi-seed diagnostic datasets generated and "
             "structurally verified, with an added operational-learnability "
             "diagnostic. No feature selection, no resampling, no analyzer, no "
             "recovery metrics, no real dataset.\n")
    L.append("---\n")

    L.append("## 1. Commands Run\n")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")

    L.append("## 2. Generator Changes Made\n")
    L.append(f"- **Instance-diagnostic mode added (`--instance-diagnostic-v1`).** "
             f"Generates N={n_seeds} synthetic instances (one per generation seed). "
             f"For each seed it builds a within-seed-controlled pair of known-core "
             f"cells — high SNR (alpha={DIAGNOSTIC_ALPHA_HIGH}) and mid035 "
             f"(alpha={DIAGNOSTIC_ALPHA_MID}) — plus a null control. rho is fixed at "
             f"{DIAGNOSTIC_RHO}; weak gamma stays {WEAK_GAMMA_V1}.")
    L.append("- **Within-seed control.** The high-SNR and mid035 cells of a seed "
             "share one base feature matrix, one planted-index layout, one eta, one "
             "label-uniform vector U, and one set of proxy epsilons (three RNG "
             "streams spawned from the seed via `SeedSequence.spawn(3)`); only alpha "
             "differs. Across seeds the base matrix and layout are freshly drawn — "
             "this is what lets the diagnostic quantify instance variance.")
    L.append("- **Operational-learnability diagnostic added.** For each known-core "
             "dataset the generator now reports the core mean |corr with y| against "
             "the noise UPPER TAIL (p95 / p99 / max), not just the noise mean, plus "
             "a quick oracle held-out ExtraTrees AUC trained on the KNOWN true-core "
             "columns only. The alpha=0.25 collapse showed `core > noise_mean` is "
             "not sufficient for operational recoverability; this check is meant to "
             "expose that at generation time.")
    L.append(f"- **New name suffix `{GRID_V1D_SUFFIX}_seed<NN>`** — the earlier "
             "`_gammaV1`, `_gammaV1c`, and smoke outputs are not touched, "
             "overwritten, or renamed.\n")

    L.append("## 3. Datasets Generated\n")
    L.append(f"{n_seeds} generation seeds x 3 datasets = {n_seeds * 3} datasets. "
             f"Instance i uses base generation seed {base_seed}+i; the null control "
             f"of instance i uses base seed + {NULL_SEED_OFFSET}.\n")
    L.append("| Seed idx | base seed | dataset | setting | alpha | rho | suffix |")
    L.append("|----------|-----------|---------|---------|-------|-----|--------|")
    for ins in instances:
        i = ins["seed_index"]
        for key, setting in (("high", "high SNR"), ("mid", "mid035"),
                             ("null", "null")):
            m = ins[key]
            L.append(f"| {i:02d} | {ins['base_seed']} | {m['dataset_name']} | "
                     f"{setting} | {m.get('alpha_value')} | {m.get('rho')} | "
                     f"{GRID_V1D_SUFFIX}_seed{i:02d} |")
    L.append("")
    L.append("- *Old outputs untouched:* this run wrote only `_gammaV1d_seed<NN>` "
             "files; no `_gammaV1`, `_gammaV1c`, smoke, or controlled-grid dataset, "
             "manifest, or report was deleted, overwritten, or renamed.")
    L.append("- *Note:* with base seed 0, instance 00's high-SNR cell shares its "
             "base matrix / layout / eta / U with the earlier `_gammaV1c` high-SNR "
             "rho=0.9 cell (same `SeedSequence(0).spawn(3)`); this is a deliberate, "
             "harmless tie-in — the file is written under the new suffix and "
             "overwrites nothing.\n")

    L.append("## 4. Structural Verification by Seed and Setting\n")
    L.append("Mean absolute Pearson correlation with y; the hard gate is strict "
             "`core > weak > noise` with `core > 2*noise_mean`, plus reasonable "
             "class balance and proxy-parent rho on target.\n")
    L.append("| Seed | setting | core | weak | noise mean | noise sample max | "
             "class 0/1 | proxy-parent mean | verify |")
    L.append("|------|---------|------|------|------------|------------------|"
             "-----------|-------------------|--------|")
    for ins in instances:
        i = ins["seed_index"]
        for key, setting in (("high", "high"), ("mid", "mid035")):
            v = ins[f"ver_{key}"]
            c = v["checks"]
            mc = c["mean_abs_corr_with_y"]
            cb = c["class_balance"]
            pc = c.get("proxy_parent_corr", {})
            L.append(f"| {i:02d} | {setting} | {mc['core_all']} | {mc['weak']} | "
                     f"{mc['noise_sample_mean']} | {mc['noise_sample_max']} | "
                     f"{cb['n_class_0']}/{cb['n_class_1']} | {pc.get('mean')} | "
                     f"{'PASS' if v['passed'] else 'CHECK'} |")
    L.append("")

    L.append("## 5. Operational Learnability Diagnostic\n")
    L.append("The structural gate compares the core to the noise *mean*. "
             "Operational recoverability against ~2000 distractors at n=200 depends "
             "on the core versus the noise *upper tail*. Below: core mean |corr|, "
             "noise p95 / p99 / max |corr| (scanned over ALL ~1968 noise features), "
             "the core-vs-noise-p99 and core-vs-noise-max margins (a negative margin "
             "means the core sits inside the noise upper tail), and an oracle "
             "held-out ExtraTrees AUC trained on the KNOWN true-core columns only "
             "(NOT feature selection).\n")
    L.append("| Seed | setting | core mean | noise p95 | noise p99 | noise max | "
             "core-p99 margin | core-max margin | oracle AUC mean/min/max |")
    L.append("|------|---------|-----------|-----------|-----------|-----------|"
             "-----------------|-----------------|--------------------------|")
    for ins in instances:
        i = ins["seed_index"]
        for key, setting in (("high", "high"), ("mid", "mid035")):
            od = ins[key]["operational_learnability_diagnostic"]
            L.append(f"| {i:02d} | {setting} | {od['core_mean_abs_corr']} | "
                     f"{od['noise_p95_abs_corr']} | {od['noise_p99_abs_corr']} | "
                     f"{od['noise_max_abs_corr']} | "
                     f"{od['core_vs_noise_p99_margin']} | "
                     f"{od['core_vs_noise_max_margin']} | "
                     f"{od['oracle_true_core_auc_mean']}/"
                     f"{od['oracle_true_core_auc_min']}/"
                     f"{od['oracle_true_core_auc_max']} |")
    L.append("")
    L.append("**Reading this table.**")
    L.append("- The **oracle true-core AUC** is the key operational number: the "
             "held-out AUC a classifier reaches when *handed the true core "
             "features*. If it is near 0.5 the planted signal is not learnable even "
             "with the correct features, and no feature-selection method could "
             "recover it — that is the alpha=0.25 failure mode.")
    L.append("- A **negative core-vs-noise-p99 / -max margin** means the core mean "
             "sits inside the noise upper tail; against ~2000 distractors a selector "
             "cannot reliably separate the core from the luckiest noise features.")
    L.append("- These numbers are **not a hard gate** for the diagnostic — the whole "
             "point of the reduced-resample pipeline is to test alpha=0.35 "
             "operationally. They are reported so the pipeline result can be read "
             "against generation-time expectation.\n")

    L.append("## 6. Within-Seed Shared-Index Verification\n")
    L.append("| Seed | identical solo | identical anchored | identical proxy "
             "groups | identical weak | all identical |")
    L.append("|------|----------------|--------------------|--------------------|"
             "----------------|---------------|")
    for ins in instances:
        w = ins["within"]
        L.append(f"| {ins['seed_index']:02d} | {w['identical_solo']} | "
                 f"{w['identical_anchored']} | {w['identical_proxy_groups']} | "
                 f"{w['identical_weak']} | **{w['all_identical']}** |")
    L.append("")
    L.append("Within each seed the high-SNR and mid035 cells must share identical "
             "planted indices — only alpha may differ. Across seeds the layouts "
             "differ by design.\n")

    L.append("## 7. Null-Control Verification\n")
    L.append("| Seed | dataset | max abs corr with y | p99 | "
             "no strong planted signal | verify |")
    L.append("|------|---------|---------------------|-----|"
             "--------------------------|--------|")
    for ins in instances:
        m = ins["null"]
        v = ins["ver_null"]
        c = v["checks"]
        ac = c.get("abs_corr_with_y", {})
        L.append(f"| {ins['seed_index']:02d} | {m['dataset_name']} | "
                 f"{ac.get('max')} | {ac.get('p99')} | "
                 f"{c.get('no_strong_planted_signal')} | "
                 f"{'PASS' if v['passed'] else 'CHECK'} |")
    L.append("")

    L.append("## 8. Reload Checks (`.mat` round-trip)\n")
    for name, rc in reloads.items():
        L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                 f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, "
                 f"Y_shape={rc['Y_shape']}")
    L.append("")

    L.append("## 9. Warnings / Caveats\n")
    warnings = []
    for ins in instances:
        i = ins["seed_index"]
        for key, setting in (("high", "high"), ("mid", "mid035")):
            v = ins[f"ver_{key}"]
            if not v["passed"]:
                warnings.append(f"seed {i:02d} {setting}: structural verification "
                                f"did not pass — {v['warnings']}")
            od = ins[key]["operational_learnability_diagnostic"]
            if (od["oracle_true_core_auc_n_splits"] > 0
                    and od["oracle_true_core_auc_mean"] <= 0.60):
                warnings.append(
                    f"seed {i:02d} {setting}: oracle true-core AUC "
                    f"{od['oracle_true_core_auc_mean']} <= 0.60 — the planted "
                    f"signal is weakly learnable even with the true features.")
            if not od["core_above_noise_p99"]:
                warnings.append(
                    f"seed {i:02d} {setting}: core mean |corr| is below the noise "
                    f"p99 (core-p99 margin {od['core_vs_noise_p99_margin']}) — "
                    f"the core sits inside the noise upper tail.")
        if not ins["within"]["all_identical"]:
            warnings.append(f"seed {i:02d}: within-seed planted indices are NOT "
                            f"identical between the high and mid035 cells.")
    if warnings:
        for w in warnings:
            L.append(f"- **WARNING:** {w}")
    else:
        L.append("- No warnings raised by the automated checks.")
    L.append("- These datasets are **synthetic**; \"planted core\" wording is "
             "licensed only for synthetic data.")
    L.append("- The mid035 setting (alpha=0.35) is an explicitly *untested "
             "candidate*; operational flags above are expected and are exactly what "
             "the reduced-resample pipeline must resolve.\n")

    L.append("## 10. Readiness and Recommended Next Step\n")
    if all_struct_ok:
        L.append("- **Structurally ready.** Every known-core cell passed the strict "
                 "`core > weak > noise` gate, every within-seed shared-index check "
                 "holds, and every null control passed.")
    else:
        L.append("- **At least one structural / within-seed check did not pass** "
                 "(see sections 4, 6, 9). The datasets were still written; review "
                 "the flagged cell before relying on it.")
    L.append("- Recommended next step: run the reduced multi-seed diagnostic "
             "pipeline `run_instance_diagnostic_v1_pipeline.py` (ETree only, 10 "
             "resamples per dataset) — per-resample logging, offline stability "
             "analysis, recovery metrics, and an across-seed diagnostic summary — to "
             "answer the two go/no-go questions: (a) is high-SNR recovery reliably "
             "non-trivial across seeds, and (b) is alpha=0.35 hard-but-learnable "
             "rather than a repeat of the alpha=0.25 collapse. This generator step "
             "runs none of that.\n")

    L.append("---\n")
    L.append("*Generator step only — multi-seed diagnostic datasets generated, "
             "structurally verified, and given an operational-learnability "
             "diagnostic. No feature selection, no logging, no analyzer, no recovery "
             "metrics, no real dataset, no benchmark file modified.*")
    return "\n".join(L)


def run_instance_diagnostic_generation(seed, n_seeds, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic instance-variance diagnostic (gammaV1d) generator")
    print("=" * 70)
    print(f"Base seed             : {seed}")
    print(f"Generation seeds      : {n_seeds}  (instance i uses base seed {seed}+i)")
    print(f"rho (fixed)           : {DIAGNOSTIC_RHO}")
    print(f"alpha high / mid035   : {DIAGNOSTIC_ALPHA_HIGH} / {DIAGNOSTIC_ALPHA_MID}")
    print(f"weak gamma            : {WEAK_GAMMA_V1}")
    print(f"Suffix                : {GRID_V1D_SUFFIX}_seed<NN>")
    print(f"Output directory      : {out_dir}")
    print("-" * 70)

    instances, written, reloads = [], [], {}
    for i in range(n_seeds):
        base_seed = seed + i
        high, mid, null = generate_diagnostic_instance(base_seed, i)
        (Xh, yh, mh), (Xm, ym, mm), (Xn, yn, mn) = high, mid, null

        # operational-learnability diagnostic (added before the manifest write)
        mh["operational_learnability_diagnostic"] = \
            compute_operational_diagnostic(Xh, yh, mh)
        mm["operational_learnability_diagnostic"] = \
            compute_operational_diagnostic(Xm, ym, mm)

        vh = verify_known_core(Xh, yh, mh, rho=mh["rho"])
        vm = verify_known_core(Xm, ym, mm, rho=mm["rho"])
        vn = verify_null_control(Xn, yn, mn)
        embed_verification(mh, vh)
        embed_verification(mm, vm)
        embed_verification(mn, vn)

        within = verify_diagnostic_within_seed(mh, mm)
        mh["within_seed_shared_index_check"] = within
        mm["within_seed_shared_index_check"] = within

        for man, X, y in ((mh, Xh, yh), (mm, Xm, ym), (mn, Xn, yn)):
            mat_p, man_p = write_dataset(out_dir, man, X, y)
            role = "null-control" if man["snr_label"] == "null" else "known-core"
            written.append((f"seed{i:02d} {role} dataset", mat_p))
            written.append((f"seed{i:02d} {role} manifest", man_p))
            reloads[man["dataset_name"]] = reload_check(mat_p)

        instances.append({
            "seed_index": i, "base_seed": base_seed,
            "high": mh, "mid": mm, "null": mn,
            "ver_high": vh, "ver_mid": vm, "ver_null": vn, "within": within,
        })
        oh = mh["operational_learnability_diagnostic"]
        om = mm["operational_learnability_diagnostic"]
        print(f"  seed {i:02d} (base {base_seed}): "
              f"high verify={'PASS' if vh['passed'] else 'CHECK'} "
              f"mid035 verify={'PASS' if vm['passed'] else 'CHECK'} "
              f"null verify={'PASS' if vn['passed'] else 'CHECK'} "
              f"within-seed-shared={within['all_identical']}")
        print(f"            oracle true-core AUC: "
              f"high={oh['oracle_true_core_auc_mean']}  "
              f"mid035={om['oracle_true_core_auc_mean']}")

    all_struct_ok = all(
        ins["ver_high"]["passed"] and ins["ver_mid"]["passed"]
        and ins["ver_null"]["passed"] and ins["within"]["all_identical"]
        for ins in instances
    )

    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run "
        "--instance-diagnostic-v1",
        "python articles/article_stability/scripts/make_synthetic_control.py --instance-diagnostic-v1",
    ]
    GRID_V1D_GEN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(GRID_V1D_GEN_REPORT_PATH.parent,
                            GRID_V1D_GEN_REPORT_PATH.name,
                            GRID_V1D_GEN_REPORT_PATH.parent)
    report_path.write_text(build_instance_diagnostic_report(
        commands, written, instances, reloads, all_struct_ok, seed, n_seeds))

    print("-" * 70)
    for ins in instances:
        for key in ("ver_high", "ver_mid", "ver_null"):
            for w in ins[key]["warnings"]:
                print(f"  [warn] seed{ins['seed_index']:02d} {key}: {w}")
    print(f"Datasets generated : {len(instances) * 3}  ({n_seeds} seeds x 3)")
    print(f"Markdown report    : {report_path}")
    print(f"All structural + within-seed checks pass: {all_struct_ok}")
    print("=" * 70)
    return 0 if all_struct_ok else 1


def _dry_run_instance_diagnostic(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic instance-variance diagnostic (gammaV1d)  [DRY RUN]")
    print("=" * 70)
    n_seeds = args.n_generation_seeds
    print(f"Base seed             : {args.seed}")
    print(f"Generation seeds      : {n_seeds}  "
          f"(instance i uses base seed {args.seed}+i)")
    print(f"Datasets output root  : {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory      : {out_dir}")
    print(f"rho (fixed)           : {DIAGNOSTIC_RHO}")
    print(f"alpha high / mid035   : {DIAGNOSTIC_ALPHA_HIGH} / {DIAGNOSTIC_ALPHA_MID}")
    print(f"weak gamma (unchanged): {WEAK_GAMMA_V1}")
    print(f"Suffix                : {GRID_V1D_SUFFIX}_seed<NN>")
    print(f"Planned datasets      : {n_seeds} seeds x 3 = {n_seeds * 3}")
    planned = []
    for i in range(n_seeds):
        sfx = f"{GRID_V1D_SUFFIX}_seed{i:02d}"
        for nm in (f"synthetic_snrHIGH_rho090{sfx}",
                   f"synthetic_snrMID035_rho090{sfx}",
                   f"synthetic_null_control{sfx}"):
            planned += [f"{nm}.mat", f"{nm}_manifest.json"]
    print("Planned output files:")
    for f in planned:
        print(f"  - {out_dir / f}")
    print(f"Planned markdown report: {GRID_V1D_GEN_REPORT_PATH}")
    print("Within-seed control: the high-SNR and mid035 cells of each seed share "
          "identical planted indices, base matrix, eta, label uniforms, and proxy "
          "epsilons; only alpha differs. Across seeds the layout is freshly drawn.")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


# ==========================================================================
# Positive-control redesign (gammaV1e) — targeted, bounded high-SNR redesign
# ==========================================================================
def _posctrl_known_cell(alpha, rho, shared, seed_index, suffix, core_beta):
    """Build one redesigned high-SNR positive-control known-core cell from the
    shared structure. The redesign lever is `core_beta` (a stronger core effect
    size); n, p, alpha, weak gamma, and rho are held at the v1d values.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    X = shared["X_base"].copy()

    sqrt_term = math.sqrt(max(0.0, 1.0 - rho ** 2))
    for parent, proxies in shared["proxy_groups"].items():
        for pj in proxies:
            X[:, pj] = (rho * shared["X_base"][:, parent]
                        + sqrt_term * shared["eps_by_proxy"][int(pj)])

    prob = sigmoid(alpha * shared["eta"])
    y = (shared["U"] < prob).astype(np.int64)

    core_idx = shared["core_idx"]
    weak_idx = shared["weak_idx"]
    noise_idx = shared["noise_idx"]
    rho_code = int(round(rho * 100))
    manifest = {
        "dataset_name": f"synthetic_snrHIGH_rho{rho_code:03d}{suffix}",
        "generation_seed": int(shared["base_seed"]),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": "high",
        "alpha_value": float(alpha),
        "rho": float(rho),
        "weak_gamma": float(WEAK_GAMMA_V1),
        "core_beta": float(core_beta),
        "positive_control_redesign_version": "gammaV1e",
        "redesign_seed_index": int(seed_index),
        "base_generation_seed": int(shared["base_seed"]),
        "redesign_core_beta": float(core_beta),
        "redesign_core_beta_baseline": float(CORE_BETA),
        "redesign_lever": (
            "stronger core beta (core effect size). v1d showed alpha=2.5 at "
            f"core beta {CORE_BETA} is instance-variable; this redesign raises "
            f"core beta to {core_beta} so the planted core reliably clears the "
            "noise upper tail. n_samples, n_total_features, alpha, weak gamma, "
            "and rho are held at the v1d values to preserve the p>>n regime."
        ),
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": list(shared["solo_core"]),
        "true_core_anchored_indices": list(shared["anchored_core"]),
        "proxy_groups": {str(par): list(px)
                         for par, px in shared["proxy_groups"].items()},
        "weak_signal_indices": list(weak_idx),
        "pure_noise_indices": {
            "count": len(noise_idx),
            "sample_first10": noise_idx[:10],
            "sample_last10": noise_idx[-10:],
            "definition": "all indices in 0..1999 not assigned to core/proxy/weak",
        },
        "core_beta_per_feature": {str(i): float(core_beta) for i in core_idx},
        "weak_gamma_per_feature": {str(i): float(WEAK_GAMMA_V1) for i in weak_idx},
        "signal_construction_notes": (
            "Positive-control redesign (gammaV1e) high-SNR known-core cell. All "
            "2000 columns drawn i.i.d. N(0,1); the 5 core and 15 weak columns "
            "z-scored. Linear predictor eta = sum(beta*core)+sum(gamma*weak) "
            f"with beta={core_beta} (raised from the v1d {CORE_BETA}) and "
            f"gamma={WEAK_GAMMA_V1}. Proxy columns proxy = rho*parent + "
            f"sqrt(1-rho^2)*eps with rho={rho}. Across generation seeds the base "
            "matrix and planted layout are freshly drawn."
        ),
        "label_construction_notes": (
            f"y = (U < sigmoid(alpha*eta)) with alpha={alpha} (unchanged from "
            "v1d — alpha is not the redesign lever). U is a per-seed uniform "
            "vector drawn from a dedicated RNG stream."
        ),
    }
    return X.astype(np.float64), y, manifest


def evaluate_posctrl_candidate(core_beta, n_seeds, base_seed):
    """Generate, structurally verify, and operationally diagnose every seed of
    one positive-control candidate (a single core_beta). Returns a dict with
    the per-seed arrays/manifests/verifications and the candidate-level hard
    acceptance-gate results. Writes nothing — candidate selection is in-memory.
    """
    seeds_info = []
    for i in range(n_seeds):
        bs = base_seed + i
        suffix = f"{GRID_V1E_SUFFIX}_seed{i:02d}"
        shared = _build_diagnostic_shared(bs, core_beta=core_beta)
        Xh, yh, mh = _posctrl_known_cell(POSCTRL_ALPHA, POSCTRL_RHO, shared,
                                         i, suffix, core_beta)
        Xn, yn, mn = generate_null_control(bs + NULL_SEED_OFFSET,
                                           name_suffix=suffix)
        mn["positive_control_redesign_version"] = "gammaV1e"
        mn["redesign_seed_index"] = int(i)
        mn["base_generation_seed"] = int(bs)

        op = compute_operational_diagnostic(Xh, yh, mh)
        mh["operational_learnability_diagnostic"] = op
        vh = verify_known_core(Xh, yh, mh, rho=mh["rho"])
        vn = verify_null_control(Xn, yn, mn)

        gate = {
            "structural_passed": bool(vh["passed"]),
            "core_vs_noise_p99_margin": op["core_vs_noise_p99_margin"],
            "core_p99_margin_ok": bool(
                op["core_vs_noise_p99_margin"] >= POSCTRL_GATE_P99_MARGIN_MIN),
            "core_vs_noise_max_margin": op["core_vs_noise_max_margin"],
            "oracle_true_core_auc_mean": op["oracle_true_core_auc_mean"],
            "ordering_core_gt_weak_gt_noise": bool(
                vh["checks"]["ordering_core_gt_weak_gt_noise"]),
            "class_balance_reasonable": bool(
                vh["checks"]["class_balance_reasonable"]),
            "proxy_parent_corr_close_to_rho": bool(
                vh["checks"]["proxy_parent_corr_close_to_rho"]),
            "x_finite_no_nan_inf": bool(vh["checks"]["X_finite_no_nan_inf"]),
            "null_passed": bool(vn["passed"]),
        }
        seeds_info.append(dict(
            seed_index=i, base_seed=bs, suffix=suffix,
            Xh=Xh, yh=yh, mh=mh, Xn=Xn, yn=yn, mn=mn,
            vh=vh, vn=vn, op=op, gate=gate,
        ))

    oracle_means = [s["op"]["oracle_true_core_auc_mean"] for s in seeds_info]
    margins = [s["op"]["core_vs_noise_p99_margin"] for s in seeds_info]
    oracle_mean_across = float(np.mean(oracle_means)) if oracle_means else 0.0
    oracle_min_seed = float(np.min(oracle_means)) if oracle_means else 0.0

    gate_results = {
        "core_beta": float(core_beta),
        "n_seeds": int(n_seeds),
        "all_structural_passed": all(s["gate"]["structural_passed"]
                                     for s in seeds_info),
        "all_null_passed": all(s["gate"]["null_passed"] for s in seeds_info),
        "all_core_p99_margin_ge_min": all(s["gate"]["core_p99_margin_ok"]
                                          for s in seeds_info),
        "min_core_p99_margin": float(np.min(margins)) if margins else None,
        "max_core_p99_margin": float(np.max(margins)) if margins else None,
        "oracle_auc_mean_across_seeds": round(oracle_mean_across, 4),
        "oracle_auc_min_seed": round(oracle_min_seed, 4),
        "oracle_mean_gate_ok": bool(
            oracle_mean_across >= POSCTRL_GATE_ORACLE_AUC_MEAN_MIN),
        "oracle_min_gate_ok": bool(
            oracle_min_seed >= POSCTRL_GATE_ORACLE_AUC_SEED_MIN),
        "p99_margin_target": POSCTRL_GATE_P99_MARGIN_MIN,
        "oracle_mean_target": POSCTRL_GATE_ORACLE_AUC_MEAN_MIN,
        "oracle_min_target": POSCTRL_GATE_ORACLE_AUC_SEED_MIN,
    }
    gate_results["all_gates_pass"] = bool(
        gate_results["all_structural_passed"]
        and gate_results["all_null_passed"]
        and gate_results["all_core_p99_margin_ge_min"]
        and gate_results["oracle_mean_gate_ok"]
        and gate_results["oracle_min_gate_ok"]
    )
    return {"core_beta": float(core_beta), "seeds_info": seeds_info,
            "gate_results": gate_results}


def build_posctrl_redesign_report(commands, candidates, chosen, written,
                                   reloads, base_seed, n_seeds):
    L = []
    L.append("# Synthetic Positive-Control Redesign (gammaV1e) — "
             "Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Generator step only — a targeted, bounded redesign of the "
             "high-SNR positive-control regime, with the operational-learnability "
             "checks promoted to HARD acceptance gates. No feature selection, no "
             "resampling, no analyzer, no recovery metrics, no real dataset.\n")
    L.append("---\n")

    # 1. purpose
    L.append("## 1. Purpose\n")
    L.append("Create a redesigned high-SNR positive-control regime in which the "
             "planted core **reliably clears the noise upper tail across "
             "generation seeds**, so that ETree can recover the planted "
             "structure consistently and the regime can serve as a valid "
             "positive control for the Article 5 stability/recovery profile. "
             "The point is NOT an unrealistically easy toy — the p>>n regime is "
             "preserved — but a regime where recovery demonstrably works.\n")

    # 2. why v1d required redesign
    L.append("## 2. Why v1d Required a Redesign\n")
    L.append(f"- v1d high-SNR (alpha={POSCTRL_ALPHA}, core beta {CORE_BETA}) was "
             "instance-variable: per-seed mean held-out AUC swung ~0.54-0.73 "
             "across 5 seeds, with one near-failure — not a reliable positive "
             "control.")
    L.append("- alpha=0.35 was oracle-learnable but selection-hard (~0.05 AUC "
             "above null); the root cause is core-vs-noise **upper-tail** "
             "overlap in p>>n — `core > noise_mean` does not predict operational "
             "recoverability.")
    L.append("- The v1d checkpoint's decision: a targeted, bounded redesign — "
             "raise the core effect size (core beta), keep alpha/n/p, and "
             "promote the operational-learnability checks to hard gates. alpha "
             "is explicitly not the lever.\n")

    # 3. candidates considered
    L.append("## 3. Candidate Redesign Settings Considered\n")
    L.append(f"Levers tried, in the checkpoint's preferred order: stronger core "
             f"beta first (candidates {POSCTRL_CORE_BETA_CANDIDATES}), at "
             f"n_samples={N_SAMPLES}, p={N_TOTAL_FEATURES}, alpha={POSCTRL_ALPHA}, "
             f"rho={POSCTRL_RHO}, weak gamma={WEAK_GAMMA_V1}. n_samples=300 is a "
             f"documented fallback considered only if no core-beta candidate "
             f"passes the hard gates.\n")
    L.append("| Candidate (core beta) | structural all-pass | null all-pass | "
             "core-p99 margin all >= target | oracle AUC mean | oracle AUC min "
             "seed | all gates pass |")
    L.append("|-----------------------|---------------------|---------------|"
             "-------------------------------|-----------------|------------------"
             "|-----------------|")
    for cand in candidates:
        g = cand["gate_results"]
        L.append(f"| {g['core_beta']} | {g['all_structural_passed']} | "
                 f"{g['all_null_passed']} | {g['all_core_p99_margin_ge_min']} "
                 f"(min margin {round(g['min_core_p99_margin'], 4)}) | "
                 f"{g['oracle_auc_mean_across_seeds']} | "
                 f"{g['oracle_auc_min_seed']} | **{g['all_gates_pass']}** |")
    L.append("")
    L.append(f"Hard-gate targets: core-vs-noise-p99 margin >= "
             f"{POSCTRL_GATE_P99_MARGIN_MIN} for every seed; oracle true-core "
             f"AUC mean across seeds >= {POSCTRL_GATE_ORACLE_AUC_MEAN_MIN} and "
             f"per-seed minimum >= {POSCTRL_GATE_ORACLE_AUC_SEED_MIN}; strict "
             "core > weak > noise ordering; class balance reasonable; "
             "proxy-parent correlation near rho; no NaN/Inf.\n")

    # 4. selected setting + rationale
    L.append("## 4. Selected Setting and Rationale\n")
    if chosen is not None:
        cb = chosen["gate_results"]["core_beta"]
        L.append(f"- **Selected: core beta = {cb}** (alpha={POSCTRL_ALPHA}, "
                 f"n_samples={N_SAMPLES}, p={N_TOTAL_FEATURES}, rho={POSCTRL_RHO}, "
                 f"weak gamma={WEAK_GAMMA_V1}).")
        L.append(f"- It is the **lowest core-beta candidate** that passes all "
                 f"hard acceptance gates across all {n_seeds} seeds — the "
                 "minimum change that yields a reliable positive control, per "
                 "the bounded-redesign principle.")
        not_used = [c for c in POSCTRL_CORE_BETA_CANDIDATES if c > cb]
        if not_used:
            L.append(f"- More aggressive options ({not_used}, and the "
                     "n_samples=300 fallback) were **not used**: they were "
                     "unnecessary once the selected setting passed, and a "
                     "larger core effect size / larger n would move further "
                     "from the v1d regime than required.")
        else:
            L.append("- This was the strongest core-beta candidate in the "
                     "considered set; the n_samples=300 fallback was not "
                     "needed because this setting passed.")
    else:
        L.append("- **No candidate passed the hard acceptance gates.** None of "
                 f"the core-beta candidates {POSCTRL_CORE_BETA_CANDIDATES} "
                 f"cleared all gates across all {n_seeds} seeds (see section 3).")
        L.append("- **Recommended next move:** the documented fallback — try "
                 "n_samples=300 (which thins the noise correlation tail, chance "
                 "|corr| ~ 1/sqrt(n), while keeping p>>n), or a further bounded "
                 "core-beta increase. Per the v1d checkpoint stopping rule, if a "
                 "targeted redesign still cannot yield a reliable positive "
                 "control, the p>>n recovery difficulty is itself the finding "
                 "and the next step is to write it up — escalate to Opus.")
    L.append("")

    # 5. parameters
    L.append("## 5. Parameters\n")
    L.append("| Parameter | Value |")
    L.append("|-----------|-------|")
    L.append(f"| n_samples | {N_SAMPLES} |")
    L.append(f"| p (n_total_features) | {N_TOTAL_FEATURES} |")
    L.append(f"| alpha (high SNR) | {POSCTRL_ALPHA} |")
    L.append(f"| rho | {POSCTRL_RHO} |")
    if chosen is not None:
        L.append(f"| core beta (selected) | {chosen['gate_results']['core_beta']}"
                 f"  (v1d baseline {CORE_BETA}) |")
    else:
        L.append(f"| core beta | no candidate passed (baseline {CORE_BETA}) |")
    L.append(f"| weak gamma | {WEAK_GAMMA_V1} |")
    L.append(f"| generation seeds | {n_seeds}  (instance i uses base seed "
             f"{base_seed}+i) |")
    L.append("")

    # 6. operational gates (selected candidate, per seed)
    L.append("## 6. Operational Gates — Per Seed (selected candidate)\n")
    target_cand = chosen if chosen is not None else (
        candidates[-1] if candidates else None)
    if target_cand is not None:
        tag = ("selected" if chosen is not None
               else f"best-evaluated (core beta {target_cand['core_beta']}, "
                    "did NOT pass)")
        L.append(f"Candidate shown: {tag}.\n")
        L.append("| Seed | core mean | weak mean | noise mean | noise p99 | "
                 "core-p99 margin | core-max margin | oracle AUC mean/min/max | "
                 "class 0/1 | proxy rho | core>weak>noise |")
        L.append("|------|-----------|-----------|------------|-----------|"
                 "-----------------|-----------------|--------------------------|"
                 "-----------|-----------|-----------------|")
        for s in target_cand["seeds_info"]:
            op = s["op"]
            c = s["vh"]["checks"]
            cb = c["class_balance"]
            pc = c.get("proxy_parent_corr", {})
            L.append(f"| {s['seed_index']:02d} | {op['core_mean_abs_corr']} | "
                     f"{op['weak_mean_abs_corr']} | {op['noise_mean_abs_corr']} | "
                     f"{op['noise_p99_abs_corr']} | "
                     f"{op['core_vs_noise_p99_margin']} | "
                     f"{op['core_vs_noise_max_margin']} | "
                     f"{op['oracle_true_core_auc_mean']}/"
                     f"{op['oracle_true_core_auc_min']}/"
                     f"{op['oracle_true_core_auc_max']} | "
                     f"{cb['n_class_0']}/{cb['n_class_1']} | {pc.get('mean')} | "
                     f"{c['ordering_core_gt_weak_gt_noise']} |")
        L.append("")

    # 7. per-seed structural / operational diagnostics summary
    L.append("## 7. Per-Seed Structural / Operational Diagnostics\n")
    if target_cand is not None:
        for s in target_cand["seeds_info"]:
            g = s["gate"]
            L.append(f"- seed {s['seed_index']:02d} (base {s['base_seed']}): "
                     f"structural verify={'PASS' if g['structural_passed'] else 'CHECK'}, "
                     f"core-p99 margin={g['core_vs_noise_p99_margin']} "
                     f"({'ok' if g['core_p99_margin_ok'] else 'BELOW TARGET'}), "
                     f"oracle AUC mean={g['oracle_true_core_auc_mean']}, "
                     f"null verify={'PASS' if g['null_passed'] else 'CHECK'}")
    L.append("")

    # 8. acceptance gates
    L.append("## 8. Acceptance Gates\n")
    if chosen is not None:
        g = chosen["gate_results"]
        L.append(f"- **All hard acceptance gates passed** for core beta "
                 f"{g['core_beta']} across all {n_seeds} seeds:")
        L.append(f"  - structural verification: {g['all_structural_passed']}")
        L.append(f"  - null controls: {g['all_null_passed']}")
        L.append(f"  - core-vs-noise-p99 margin >= "
                 f"{POSCTRL_GATE_P99_MARGIN_MIN} every seed: "
                 f"{g['all_core_p99_margin_ge_min']} "
                 f"(min margin {round(g['min_core_p99_margin'], 4)})")
        L.append(f"  - oracle AUC mean >= {POSCTRL_GATE_ORACLE_AUC_MEAN_MIN}: "
                 f"{g['oracle_mean_gate_ok']} "
                 f"({g['oracle_auc_mean_across_seeds']})")
        L.append(f"  - oracle AUC min seed >= {POSCTRL_GATE_ORACLE_AUC_SEED_MIN}: "
                 f"{g['oracle_min_gate_ok']} ({g['oracle_auc_min_seed']})")
    else:
        L.append("- **The hard acceptance gates were NOT met by any candidate.** "
                 "See section 3 for per-candidate gate results.")
    L.append("")

    # 9. safe to run the diagnostic?
    L.append("## 9. Is It Safe to Run the 10-Resample Feature-Selection "
             "Diagnostic?\n")
    if chosen is not None:
        L.append("- **Yes.** The redesigned datasets passed every hard "
                 "acceptance gate. Proceed to the 10-resample positive-control "
                 "re-diagnostic pipeline "
                 "(`run_positive_control_redesign_v1_pipeline.py`).")
    else:
        L.append("- **No — stop.** No candidate passed the hard gates; the "
                 "feature-selection diagnostic must NOT be run. Address the "
                 "fallback in section 4 (n_samples=300 or a further bounded "
                 "core-beta increase), regenerate, and re-verify first.")
    L.append("")

    # 10. warnings / caveats
    L.append("## 10. Warnings / Caveats\n")
    warnings = []
    for cand in candidates:
        for s in cand["seeds_info"]:
            for w in s["vh"]["warnings"]:
                if "weak mean" in w and "sampled-noise maximum" in w:
                    continue   # informational, not part of the hard gate
                warnings.append(f"core beta {cand['core_beta']} seed "
                                f"{s['seed_index']:02d}: {w}")
    if warnings:
        for w in warnings[:20]:
            L.append(f"- **WARNING:** {w}")
    else:
        L.append("- No hard-gate structural warnings raised.")
    L.append("- `weak_above_noise_sample_max` is informational only and is not "
             "part of the hard gate; weak gamma is at its structural ceiling.")
    L.append("- The oracle true-core AUC is a generation-time probe (ExtraTrees "
             "trained on the KNOWN true-core columns); it is NOT feature "
             "selection and is distinct from the pipeline's evaluation.")
    L.append("- These datasets are **synthetic**; \"planted core\" wording is "
             "licensed only for synthetic data.")
    L.append("- alpha was deliberately not changed — the v1d checkpoint "
             "established the bottleneck is core-vs-noise tail, not label "
             "noise.\n")

    # output files
    L.append("---\n")
    L.append("**Output files:**")
    if written:
        for label, path in written:
            L.append(f"- {label}: `{path}`")
    else:
        L.append("- No datasets written (no candidate passed the hard gates).")
    L.append(f"- This report: `{GRID_V1E_GEN_REPORT_PATH}`")
    if reloads:
        L.append("")
        L.append("**Reload checks (`.mat` round-trip):**")
        for name, rc in reloads.items():
            L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                     f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, "
                     f"Y_shape={rc['Y_shape']}")
    L.append("")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")
    L.append("*Generator step only — positive-control redesign generated and "
             "gated. No feature selection, no logging, no analyzer, no recovery "
             "metrics, no real dataset, no benchmark file modified.*")
    return "\n".join(L)


def run_positive_control_redesign_generation(seed, n_seeds, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic positive-control redesign (gammaV1e) generator")
    print("=" * 70)
    print(f"Base seed             : {seed}")
    print(f"Generation seeds      : {n_seeds}  (instance i uses base seed "
          f"{seed}+i)")
    print(f"alpha (high SNR)      : {POSCTRL_ALPHA}  (not the redesign lever)")
    print(f"rho (fixed)           : {POSCTRL_RHO}")
    print(f"core beta candidates  : {POSCTRL_CORE_BETA_CANDIDATES}  "
          f"(v1d baseline {CORE_BETA})")
    print(f"weak gamma            : {WEAK_GAMMA_V1}")
    print(f"n_samples / p         : {N_SAMPLES} / {N_TOTAL_FEATURES}")
    print(f"Suffix                : {GRID_V1E_SUFFIX}_seed<NN>")
    print(f"Output directory      : {out_dir}")
    print("-" * 70)

    candidates, chosen = [], None
    for core_beta in POSCTRL_CORE_BETA_CANDIDATES:
        print(f"Evaluating candidate: core beta = {core_beta} ...")
        cand = evaluate_posctrl_candidate(core_beta, n_seeds, seed)
        candidates.append(cand)
        g = cand["gate_results"]
        print(f"  core-p99 margin: min={round(g['min_core_p99_margin'], 4)} "
              f"(all >= {POSCTRL_GATE_P99_MARGIN_MIN}: "
              f"{g['all_core_p99_margin_ge_min']})")
        print(f"  oracle AUC: mean={g['oracle_auc_mean_across_seeds']} "
              f"(>= {POSCTRL_GATE_ORACLE_AUC_MEAN_MIN}: {g['oracle_mean_gate_ok']}), "
              f"min seed={g['oracle_auc_min_seed']} "
              f"(>= {POSCTRL_GATE_ORACLE_AUC_SEED_MIN}: {g['oracle_min_gate_ok']})")
        print(f"  structural all-pass={g['all_structural_passed']}  "
              f"null all-pass={g['all_null_passed']}  "
              f"ALL GATES PASS={g['all_gates_pass']}")
        if g["all_gates_pass"]:
            chosen = cand
            break   # lowest passing core-beta wins (bounded redesign)

    written, reloads = [], {}
    if chosen is not None:
        print("-" * 70)
        print(f"Selected core beta = {chosen['gate_results']['core_beta']} — "
              f"writing {n_seeds * 2} datasets.")
        for s in chosen["seeds_info"]:
            embed_verification(s["mh"], s["vh"])
            embed_verification(s["mn"], s["vn"])
            s["mh"]["positive_control_gate_results"] = s["gate"]
            for man, X, y in ((s["mh"], s["Xh"], s["yh"]),
                              (s["mn"], s["Xn"], s["yn"])):
                mat_p, man_p = write_dataset(out_dir, man, X, y)
                role = ("null-control" if man["snr_label"] == "null"
                        else "known-core")
                written.append((f"seed{s['seed_index']:02d} {role} dataset",
                                mat_p))
                written.append((f"seed{s['seed_index']:02d} {role} manifest",
                                man_p))
                reloads[man["dataset_name"]] = reload_check(mat_p)
    else:
        print("-" * 70)
        print("NO candidate passed the hard acceptance gates — writing a "
              "failure report; NO datasets written; do NOT run the pipeline.")

    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run "
        "--positive-control-redesign-v1",
        "python articles/article_stability/scripts/make_synthetic_control.py "
        "--positive-control-redesign-v1",
    ]
    GRID_V1E_GEN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(GRID_V1E_GEN_REPORT_PATH.parent,
                            GRID_V1E_GEN_REPORT_PATH.name,
                            GRID_V1E_GEN_REPORT_PATH.parent)
    report_path.write_text(build_posctrl_redesign_report(
        commands, candidates, chosen, written, reloads, seed, n_seeds))

    print("-" * 70)
    print(f"Markdown report    : {report_path}")
    if chosen is not None:
        print(f"Datasets generated : {n_seeds * 2}  ({n_seeds} seeds x 2)")
        print(f"Selected core beta : {chosen['gate_results']['core_beta']}")
        print("Gate outcome       : ALL HARD GATES PASSED — safe to run the "
              "10-resample diagnostic pipeline.")
    else:
        print("Gate outcome       : NO candidate passed — do NOT run the "
              "diagnostic pipeline (see report section 4).")
    print("=" * 70)
    return 0 if chosen is not None else 1


def _dry_run_posctrl_redesign(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic positive-control redesign (gammaV1e)  [DRY RUN]")
    print("=" * 70)
    n_seeds = args.n_posctrl_seeds
    print(f"Base seed             : {args.seed}")
    print(f"Generation seeds      : {n_seeds}  "
          f"(instance i uses base seed {args.seed}+i)")
    print(f"Datasets output root  : {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory      : {out_dir}")
    print(f"alpha (high SNR)      : {POSCTRL_ALPHA}  (not the redesign lever)")
    print(f"rho (fixed)           : {POSCTRL_RHO}")
    print(f"core beta candidates  : {POSCTRL_CORE_BETA_CANDIDATES}  "
          f"(v1d baseline {CORE_BETA}; lowest passing is selected)")
    print(f"weak gamma (unchanged): {WEAK_GAMMA_V1}")
    print(f"n_samples / p         : {N_SAMPLES} / {N_TOTAL_FEATURES}  "
          f"(held at v1d values; n=300 is a documented fallback)")
    print(f"Hard gates            : core-p99 margin >= "
          f"{POSCTRL_GATE_P99_MARGIN_MIN} (every seed); oracle AUC mean >= "
          f"{POSCTRL_GATE_ORACLE_AUC_MEAN_MIN}, min seed >= "
          f"{POSCTRL_GATE_ORACLE_AUC_SEED_MIN}")
    print(f"Suffix                : {GRID_V1E_SUFFIX}_seed<NN>")
    print(f"Planned datasets      : {n_seeds} seeds x 2 (high-SNR + null) = "
          f"{n_seeds * 2}, written ONLY if a candidate passes the hard gates")
    planned = []
    for i in range(n_seeds):
        sfx = f"{GRID_V1E_SUFFIX}_seed{i:02d}"
        for nm in (f"synthetic_snrHIGH_rho090{sfx}",
                   f"synthetic_null_control{sfx}"):
            planned += [f"{nm}.mat", f"{nm}_manifest.json"]
    print("Planned output files (on gate pass):")
    for f in planned:
        print(f"  - {out_dir / f}")
    print(f"Planned markdown report: {GRID_V1E_GEN_REPORT_PATH}")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


# ==========================================================================
# Mid/low SNR calibration (gammaV1f) — alpha sweep at the beta=2.5 anchor
# ==========================================================================
def _calibration_cell(snr_label, alpha, rho, shared, seed_index, suffix,
                      beta, cell_role):
    """Build one calibration known-core cell from the shared structure. Within
    a generation seed the only thing that differs across the alpha-sweep cells
    is `alpha` (the label-SNR knob); base matrix, planted layout, eta, label
    uniforms, and proxy epsilons are shared.
    """
    n, p = N_SAMPLES, N_TOTAL_FEATURES
    X = shared["X_base"].copy()

    sqrt_term = math.sqrt(max(0.0, 1.0 - rho ** 2))
    for parent, proxies in shared["proxy_groups"].items():
        for pj in proxies:
            X[:, pj] = (rho * shared["X_base"][:, parent]
                        + sqrt_term * shared["eps_by_proxy"][int(pj)])

    prob = sigmoid(alpha * shared["eta"])
    y = (shared["U"] < prob).astype(np.int64)

    core_idx = shared["core_idx"]
    weak_idx = shared["weak_idx"]
    noise_idx = shared["noise_idx"]
    alpha_code = int(round(alpha * 100))
    rho_code = int(round(rho * 100))
    manifest = {
        "dataset_name": (f"synthetic_snr{snr_label.upper()}_a{alpha_code:03d}"
                         f"_rho{rho_code:03d}{suffix}"),
        "generation_seed": int(shared["base_seed"]),
        "n_samples": int(n),
        "n_total_features": int(p),
        "snr_label": snr_label.lower(),
        "alpha_value": float(alpha),
        "beta_value": float(beta),
        "rho": float(rho),
        "weak_gamma": float(WEAK_GAMMA_V1),
        "core_beta": float(beta),
        "calibration_version": "gammaV1f",
        "calibration_seed_index": int(seed_index),
        "base_generation_seed": int(shared["base_seed"]),
        "cell_role": cell_role,
        "shared_within_seed_planted_indices": True,
        "shared_within_seed_base_matrix": True,
        "shared_within_seed_eta": True,
        "shared_within_seed_label_uniforms": True,
        "shared_within_seed_proxy_epsilons": True,
        "label_uniform_note": (
            "Labels use a single shared uniform vector U drawn from a dedicated "
            "RNG stream. y = (U < sigmoid(alpha*eta)); the SAME U and the SAME "
            "eta are used by every alpha-sweep cell of this seed, so within a "
            "seed only alpha differs."
        ),
        "proxy_epsilon_note": (
            "Proxy epsilon noise fields are drawn from a dedicated RNG stream "
            "and are shared by all alpha-sweep cells of this seed."
        ),
        "class_balance": {
            "n_class_0": int((y == 0).sum()),
            "n_class_1": int((y == 1).sum()),
            "minority_fraction": float(min((y == 0).mean(), (y == 1).mean())),
        },
        "true_core_solo_indices": list(shared["solo_core"]),
        "true_core_anchored_indices": list(shared["anchored_core"]),
        "proxy_groups": {str(par): list(px)
                         for par, px in shared["proxy_groups"].items()},
        "weak_signal_indices": list(weak_idx),
        "pure_noise_indices": {
            "count": len(noise_idx),
            "sample_first10": noise_idx[:10],
            "sample_last10": noise_idx[-10:],
            "definition": "all indices in 0..1999 not assigned to core/proxy/weak",
        },
        "core_beta_per_feature": {str(i): float(beta) for i in core_idx},
        "weak_gamma_per_feature": {str(i): float(WEAK_GAMMA_V1) for i in weak_idx},
        "signal_construction_notes": (
            "Mid/low SNR calibration (gammaV1f) known-core cell. All 2000 "
            "columns drawn i.i.d. N(0,1); the 5 core and 15 weak columns "
            "z-scored. Linear predictor eta = sum(beta*core)+sum(gamma*weak) "
            f"with beta={beta} (the v1e positive-control anchor) and "
            f"gamma={WEAK_GAMMA_V1}. Proxy columns proxy = rho*parent + "
            f"sqrt(1-rho^2)*eps with rho={rho}. Across generation seeds the "
            "base matrix and planted layout are freshly drawn."
        ),
        "label_construction_notes": (
            f"y = (U < sigmoid(alpha*eta)) with alpha={alpha}. This cell is the "
            f"'{cell_role}' of the alpha sweep; alpha is the only axis varied "
            "across the cells of a seed."
        ),
    }
    return X.astype(np.float64), y, manifest


def generate_calibration_instance(base_seed, seed_index):
    """Generate one calibration instance: the alpha-sweep known-core cells
    (alpha=2.5 reference + the mid/low candidates) plus a null control, all
    sharing one base matrix / layout / eta / U / proxy epsilons within the seed.
    Returns (known_cells, null) where known_cells is a list of
    (X, y, manifest, cell_role).
    """
    suffix = f"{GRID_V1F_SUFFIX}_seed{seed_index:02d}"
    shared = _build_diagnostic_shared(base_seed, core_beta=CALIBRATION_BETA)
    known_cells = []
    for snr_label, alpha, cell_role in CALIBRATION_ALPHA_CELLS:
        X, y, manifest = _calibration_cell(
            snr_label, alpha, CALIBRATION_RHO, shared, seed_index, suffix,
            CALIBRATION_BETA, cell_role)
        known_cells.append((X, y, manifest, cell_role))
    Xn, yn, man_null = generate_null_control(base_seed + NULL_SEED_OFFSET,
                                             name_suffix=suffix)
    man_null["calibration_version"] = "gammaV1f"
    man_null["calibration_seed_index"] = int(seed_index)
    man_null["base_generation_seed"] = int(base_seed)
    return known_cells, (Xn, yn, man_null)


def verify_calibration_within_seed(known_manifests):
    """Within-seed control check: every alpha-sweep cell of a seed must share
    identical planted indices (only alpha may differ)."""
    ref = known_manifests[0]
    checks = {
        "identical_solo": bool(all(
            m["true_core_solo_indices"] == ref["true_core_solo_indices"]
            for m in known_manifests)),
        "identical_anchored": bool(all(
            m["true_core_anchored_indices"] == ref["true_core_anchored_indices"]
            for m in known_manifests)),
        "identical_proxy_groups": bool(all(
            m["proxy_groups"] == ref["proxy_groups"]
            for m in known_manifests)),
        "identical_weak": bool(all(
            m["weak_signal_indices"] == ref["weak_signal_indices"]
            for m in known_manifests)),
    }
    checks["all_identical"] = bool(all(checks.values()))
    return checks


def _calibration_reference_gate(reference_cells):
    """Hard acceptance gate for the alpha=2.5 positive-control reference cells
    (the established v1e regime). `reference_cells` is a list of per-seed dicts
    with keys `op` (operational diagnostic) and `vh` (structural verification).
    The mid/low candidates do NOT pass through this gate."""
    oracle_means = [c["op"]["oracle_true_core_auc_mean"] for c in reference_cells]
    margins = [c["op"]["core_vs_noise_p99_margin"] for c in reference_cells]
    oracle_mean = float(np.mean(oracle_means)) if oracle_means else 0.0
    oracle_min = float(np.min(oracle_means)) if oracle_means else 0.0
    all_struct = all(c["vh"]["passed"] for c in reference_cells)
    all_margin = all(m >= POSCTRL_GATE_P99_MARGIN_MIN for m in margins)
    gate = {
        "n_seeds": len(reference_cells),
        "all_structural_passed": bool(all_struct),
        "all_core_p99_margin_ge_min": bool(all_margin),
        "min_core_p99_margin": float(np.min(margins)) if margins else None,
        "oracle_auc_mean_across_seeds": round(oracle_mean, 4),
        "oracle_auc_min_seed": round(oracle_min, 4),
        "oracle_mean_gate_ok": bool(oracle_mean >= POSCTRL_GATE_ORACLE_AUC_MEAN_MIN),
        "oracle_min_gate_ok": bool(oracle_min >= POSCTRL_GATE_ORACLE_AUC_SEED_MIN),
        "p99_margin_target": POSCTRL_GATE_P99_MARGIN_MIN,
        "oracle_mean_target": POSCTRL_GATE_ORACLE_AUC_MEAN_MIN,
        "oracle_min_target": POSCTRL_GATE_ORACLE_AUC_SEED_MIN,
    }
    gate["all_gates_pass"] = bool(
        all_struct and all_margin
        and gate["oracle_mean_gate_ok"] and gate["oracle_min_gate_ok"])
    return gate


def build_midlow_calibration_report(commands, instances, reference_gate,
                                    all_nulls_pass, safe_to_run, written,
                                    reloads, base_seed, n_seeds):
    L = []
    L.append("# Synthetic Mid/Low SNR Calibration (gammaV1f) — "
             "Generation Report\n")
    L.append("**Article 5:** *Feature Stability as a Trust Layer for Feature "
             "Selection*  ")
    L.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("> Generator step only — a cheap alpha-sweep at the redesigned "
             "beta=2.5 positive-control regime, to calibrate a hard-but-"
             "learnable mid/low alpha cell. Hard operational gates apply ONLY "
             "to the alpha=2.5 reference; mid/low candidate checks are "
             "informational. No feature selection, no resampling, no analyzer, "
             "no recovery metrics, no real dataset.\n")
    L.append("---\n")

    L.append("## 1. Purpose\n")
    L.append("v1e established a reliable positive control (high-SNR, core "
             "beta=2.5). To make the synthetic control useful, a **degraded "
             "mid/low SNR cell** is needed as a contrast — a regime where "
             "recovery is clearly harder yet still learnable. This step runs a "
             "cheap alpha-sweep at fixed beta=2.5 so the next (pipeline) step "
             "can pick the alpha whose held-out AUC and recovery are clearly "
             "degraded versus the positive control yet clearly above the "
             "matched null.\n")

    L.append("## 2. Why v1e Enabled This Calibration\n")
    L.append("- v1e fixed the positive-control reliability problem: at core "
             "beta=2.5 the high-SNR regime gave reliably non-trivial recovery "
             "across 8 seeds (per-seed mean AUC 0.70-0.80, no collapse).")
    L.append("- The mid/low cell must be calibrated **under beta=2.5** — v1d's "
             "alpha=0.35 was measured under beta=1.5 and does not transfer "
             "(the core is intrinsically stronger now).")
    L.append("- Controlled comparison: hold beta=2.5 fixed and vary only alpha "
             "(the label-SNR axis). Within each seed the alpha-sweep cells "
             "share base matrix, layout, eta, label uniforms, and proxy "
             "epsilons — only alpha differs.\n")

    L.append("## 3. Fixed Parameters\n")
    L.append("| Parameter | Value |")
    L.append("|-----------|-------|")
    L.append(f"| core beta | {CALIBRATION_BETA}  (the v1e anchor; fixed) |")
    L.append(f"| p (n_total_features) | {N_TOTAL_FEATURES} |")
    L.append(f"| n_samples | {N_SAMPLES} |")
    L.append(f"| rho | {CALIBRATION_RHO} |")
    L.append(f"| weak gamma | {WEAK_GAMMA_V1} |")
    L.append(f"| generation seeds | {n_seeds}  (instance i uses base seed "
             f"{base_seed}+i) |")
    L.append("| algorithm (pipeline step) | ETree only |")
    L.append("")

    L.append("## 4. Alpha Candidates\n")
    L.append("| Cell | alpha | role |")
    L.append("|------|-------|------|")
    for snr_label, alpha, cell_role in CALIBRATION_ALPHA_CELLS:
        L.append(f"| snr{snr_label} | {alpha} | {cell_role} |")
    L.append("")
    L.append("alpha=2.5 reproduces the established v1e positive-control anchor "
             "(same beta, alpha, rho, n, p — and, at matching base seeds, the "
             "same base matrix); it is included as the in-run reference. The "
             "remaining alpha values are mid/low candidates swept downward.\n")

    # 5. per-seed structural / operational diagnostics
    L.append("## 5. Per-Seed Structural / Operational Diagnostics\n")
    L.append("core / weak / noise are mean |corr with y|; noise p99 is the "
             "noise upper tail; oracle AUC is a generation-time held-out "
             "ExtraTrees AUC trained on the KNOWN true-core columns only (NOT "
             "feature selection).\n")
    L.append("| Seed | cell | alpha | role | core | weak | noise mean | "
             "noise p99 | core-p99 margin | oracle AUC | class 0/1 | "
             "core>weak>noise | structural verify |")
    L.append("|------|------|-------|------|------|------|------------|"
             "-----------|-----------------|------------|-----------|"
             "-----------------|-------------------|")
    for ins in instances:
        i = ins["seed_index"]
        for cell in ins["known"]:
            m = cell["manifest"]
            op = m["operational_learnability_diagnostic"]
            c = m["structural_verification_summary"]["checks"]
            cb = c["class_balance"]
            role_short = ("ref" if cell["cell_role"] == "positive_control_reference"
                          else "cand")
            L.append(f"| {i:02d} | snr{cell['snr_label'].upper()} | "
                     f"{cell['alpha']} | {role_short} | "
                     f"{op['core_mean_abs_corr']} | {op['weak_mean_abs_corr']} | "
                     f"{op['noise_mean_abs_corr']} | {op['noise_p99_abs_corr']} | "
                     f"{op['core_vs_noise_p99_margin']} | "
                     f"{op['oracle_true_core_auc_mean']} | "
                     f"{cb['n_class_0']}/{cb['n_class_1']} | "
                     f"{c['ordering_core_gt_weak_gt_noise']} | "
                     f"{'PASS' if m['structural_verification_summary']['passed'] else 'CHECK'} |")
    L.append("")
    L.append("Within-seed shared-index checks (all alpha cells of a seed must "
             "share planted indices):\n")
    for ins in instances:
        w = ins["within"]
        L.append(f"- seed {ins['seed_index']:02d}: all planted indices "
                 f"identical across the alpha cells = **{w['all_identical']}**")
    L.append("")

    # 6. operational checks — gate policy
    L.append("## 6. Operational Checks — Gate Policy\n")
    L.append("- **alpha=2.5 reference (positive-control anchor):** the v1e hard "
             "acceptance gates apply — core-vs-noise-p99 margin >= "
             f"{POSCTRL_GATE_P99_MARGIN_MIN} every seed, oracle true-core AUC "
             f"mean >= {POSCTRL_GATE_ORACLE_AUC_MEAN_MIN} and per-seed minimum "
             f">= {POSCTRL_GATE_ORACLE_AUC_SEED_MIN}, strict core>weak>noise, "
             "class balance, proxy-parent rho on target.")
    L.append("- **mid/low alpha candidates (alpha 1.0 / 0.6 / 0.35):** the same "
             "quantities are computed and recorded, but **informationally "
             "only** — these cells are MEANT to be harder, so a low oracle AUC "
             "or a thin core-p99 margin is an expected diagnostic signal, not a "
             "generation failure.\n")

    # 7. reference gate result
    L.append("## 7. Does the alpha=2.5 Reference Still Pass the Hard Gates?\n")
    g = reference_gate
    L.append(f"- structural verification all seeds: {g['all_structural_passed']}")
    L.append(f"- core-vs-noise-p99 margin >= {POSCTRL_GATE_P99_MARGIN_MIN} every "
             f"seed: {g['all_core_p99_margin_ge_min']} "
             f"(min margin {g['min_core_p99_margin']})")
    L.append(f"- oracle AUC mean across seeds >= {POSCTRL_GATE_ORACLE_AUC_MEAN_MIN}: "
             f"{g['oracle_mean_gate_ok']} ({g['oracle_auc_mean_across_seeds']})")
    L.append(f"- oracle AUC min seed >= {POSCTRL_GATE_ORACLE_AUC_SEED_MIN}: "
             f"{g['oracle_min_gate_ok']} ({g['oracle_auc_min_seed']})")
    L.append(f"- null controls all pass structural verification: "
             f"{all_nulls_pass}")
    L.append(f"- **reference hard gates all pass: {g['all_gates_pass']}**")
    L.append("")

    # 8. candidate structural collapse
    L.append("## 8. Do Any Candidates Structurally Collapse Toward Null?\n")
    L.append("A candidate is flagged here if, averaged across seeds, its oracle "
             "true-core AUC is near chance (<= 0.55) or the core>weak>noise "
             "ordering fails in most seeds — i.e. the planted core is not even "
             "operationally distinguishable. (Selection-pipeline difficulty is "
             "assessed separately by the diagnostic pipeline.)\n")
    by_alpha = {}
    for ins in instances:
        for cell in ins["known"]:
            by_alpha.setdefault(cell["alpha"], []).append(cell)
    for alpha in sorted(by_alpha, reverse=True):
        cells = by_alpha[alpha]
        role = cells[0]["cell_role"]
        oracle_means = [c["manifest"]["operational_learnability_diagnostic"]
                        ["oracle_true_core_auc_mean"] for c in cells]
        ord_fail = sum(
            0 if c["manifest"]["structural_verification_summary"]["checks"]
            ["ordering_core_gt_weak_gt_noise"] else 1 for c in cells)
        om = float(np.mean(oracle_means))
        collapsed = bool(om <= 0.55 or ord_fail > len(cells) / 2)
        L.append(f"- alpha={alpha} ({role}): oracle true-core AUC mean across "
                 f"seeds = {round(om, 4)}, core>weak>noise ordering failed in "
                 f"{ord_fail}/{len(cells)} seeds — "
                 f"{'**STRUCTURALLY COLLAPSED toward null**' if collapsed else 'not structurally collapsed'}.")
    L.append("")

    # 9. safe to run the diagnostic?
    L.append("## 9. Is It Safe to Run the 10-Resample Feature-Selection "
             "Diagnostic?\n")
    if safe_to_run:
        L.append("- **Yes.** The alpha=2.5 positive-control reference passed all "
                 "hard acceptance gates and the null controls verified. Proceed "
                 "to the 10-resample calibration pipeline "
                 "(`run_midlow_calibration_v1_pipeline.py`). The mid/low "
                 "candidates are carried through regardless of their "
                 "(informational) operational numbers — judging their "
                 "selection-pipeline difficulty is the pipeline's job.")
    else:
        L.append("- **No — stop.** The alpha=2.5 reference did NOT pass the hard "
                 "acceptance gates (see section 7). The reference defines the "
                 "anchor the calibration is measured against; do not run the "
                 "feature-selection diagnostic until it is resolved. Escalate "
                 "to an Opus checkpoint.")
    L.append("")

    # 10. warnings / caveats
    L.append("## 10. Warnings / Caveats\n")
    warnings = []
    for ins in instances:
        for cell in ins["known"]:
            if cell["cell_role"] != "positive_control_reference":
                continue
            v = cell["manifest"]["structural_verification_summary"]
            if not v["passed"]:
                warnings.append(f"seed {ins['seed_index']:02d} alpha=2.5 "
                                f"reference structural verification did not "
                                f"pass: {v['warnings']}")
        if not ins["within"]["all_identical"]:
            warnings.append(f"seed {ins['seed_index']:02d}: alpha-sweep cells do "
                            f"NOT all share identical planted indices.")
    if warnings:
        for w in warnings:
            L.append(f"- **WARNING:** {w}")
    else:
        L.append("- No hard-gate warnings raised for the alpha=2.5 reference or "
                 "the within-seed controls.")
    L.append("- mid/low candidate operational numbers are **informational**; a "
             "low oracle AUC or thin core-p99 margin for alpha 1.0 / 0.6 / 0.35 "
             "is expected and is the calibration signal, not a failure.")
    L.append("- The oracle true-core AUC is a generation-time probe (ExtraTrees "
             "on the KNOWN true-core columns); it is NOT feature selection.")
    L.append("- beta=2.5 is a deliberately strengthened positive-control "
             "anchor, not a representative real-data difficulty; \"planted "
             "core\" wording is licensed only for synthetic data.\n")

    L.append("---\n")
    L.append("**Output files:**")
    for label, path in written:
        L.append(f"- {label}: `{path}`")
    L.append(f"- This report: `{GRID_V1F_GEN_REPORT_PATH}`")
    if reloads:
        L.append("")
        L.append("**Reload checks (`.mat` round-trip):**")
        for name, rc in reloads.items():
            L.append(f"- {name}: loads_ok={rc['loads_ok']}, has_X={rc['has_X']}, "
                     f"has_Y={rc['has_Y']}, X_shape={rc['X_shape']}, "
                     f"Y_shape={rc['Y_shape']}")
    L.append("")
    L.append("```bash")
    L.extend(commands)
    L.append("```\n")
    L.append("*Generator step only — alpha-sweep calibration datasets generated "
             "and gated. No feature selection, no logging, no analyzer, no "
             "recovery metrics, no real dataset, no benchmark file modified.*")
    return "\n".join(L)


def run_midlow_calibration_generation(seed, n_seeds, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic mid/low SNR calibration (gammaV1f) generator")
    print("=" * 70)
    print(f"Base seed             : {seed}")
    print(f"Generation seeds      : {n_seeds}  (instance i uses base seed "
          f"{seed}+i)")
    print(f"core beta (fixed)     : {CALIBRATION_BETA}  (the v1e anchor)")
    print(f"rho (fixed)           : {CALIBRATION_RHO}")
    print(f"alpha sweep           : "
          f"{[a for _, a, _ in CALIBRATION_ALPHA_CELLS]}")
    print(f"weak gamma            : {WEAK_GAMMA_V1}")
    print(f"n_samples / p         : {N_SAMPLES} / {N_TOTAL_FEATURES}")
    print(f"Suffix                : {GRID_V1F_SUFFIX}_seed<NN>")
    print(f"Output directory      : {out_dir}")
    print("-" * 70)

    instances, written, reloads = [], [], {}
    reference_cells, all_nulls_pass = [], True
    for i in range(n_seeds):
        base_seed = seed + i
        known_cells, (Xn, yn, mn) = generate_calibration_instance(base_seed, i)

        known_records = []
        for X, y, m, cell_role in known_cells:
            m["operational_learnability_diagnostic"] = \
                compute_operational_diagnostic(X, y, m)
            v = verify_known_core(X, y, m, rho=m["rho"])
            embed_verification(m, v)
            known_records.append(dict(
                snr_label=m["snr_label"], alpha=m["alpha_value"],
                cell_role=cell_role, manifest=m, X=X, y=y, vh=v,
                op=m["operational_learnability_diagnostic"]))

        vn = verify_null_control(Xn, yn, mn)
        embed_verification(mn, vn)
        all_nulls_pass = all_nulls_pass and vn["passed"]

        within = verify_calibration_within_seed(
            [r["manifest"] for r in known_records])
        for r in known_records:
            r["manifest"]["within_seed_shared_index_check"] = within

        # write all 5 datasets of this seed
        for r in known_records:
            mat_p, man_p = write_dataset(out_dir, r["manifest"], r["X"], r["y"])
            written.append((f"seed{i:02d} {r['cell_role']} "
                            f"(alpha={r['alpha']}) dataset", mat_p))
            written.append((f"seed{i:02d} {r['cell_role']} "
                            f"(alpha={r['alpha']}) manifest", man_p))
            reloads[r["manifest"]["dataset_name"]] = reload_check(mat_p)
        mat_n, man_n = write_dataset(out_dir, mn, Xn, yn)
        written.append((f"seed{i:02d} null-control dataset", mat_n))
        written.append((f"seed{i:02d} null-control manifest", man_n))
        reloads[mn["dataset_name"]] = reload_check(mat_n)

        for r in known_records:
            if r["cell_role"] == "positive_control_reference":
                reference_cells.append(dict(seed_index=i, op=r["op"],
                                            vh=r["vh"]))

        instances.append(dict(seed_index=i, base_seed=base_seed,
                               known=known_records, null_manifest=mn,
                               within=within))
        print(f"  seed {i:02d} (base {base_seed}): "
              + "  ".join(f"a{r['alpha']}="
                          f"{'PASS' if r['manifest']['structural_verification_summary']['passed'] else 'CHECK'}"
                          for r in known_records)
              + f"  null={'PASS' if vn['passed'] else 'CHECK'}  "
              f"within-shared={within['all_identical']}")

    reference_gate = _calibration_reference_gate(reference_cells)
    safe_to_run = bool(reference_gate["all_gates_pass"] and all_nulls_pass)

    commands = [
        "python -m py_compile articles/article_stability/scripts/make_synthetic_control.py",
        "python articles/article_stability/scripts/make_synthetic_control.py --dry-run "
        "--midlow-calibration-v1",
        "python articles/article_stability/scripts/make_synthetic_control.py --midlow-calibration-v1",
    ]
    GRID_V1F_GEN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(GRID_V1F_GEN_REPORT_PATH.parent,
                            GRID_V1F_GEN_REPORT_PATH.name,
                            GRID_V1F_GEN_REPORT_PATH.parent)
    report_path.write_text(build_midlow_calibration_report(
        commands, instances, reference_gate, all_nulls_pass, safe_to_run,
        written, reloads, seed, n_seeds))

    print("-" * 70)
    print(f"Datasets generated : {n_seeds * 5}  ({n_seeds} seeds x 5)")
    print(f"Markdown report    : {report_path}")
    print(f"alpha=2.5 reference hard gates pass: "
          f"{reference_gate['all_gates_pass']}  "
          f"(oracle mean {reference_gate['oracle_auc_mean_across_seeds']}, "
          f"min seed {reference_gate['oracle_auc_min_seed']}, "
          f"min core-p99 margin {reference_gate['min_core_p99_margin']})")
    print(f"Null controls all pass : {all_nulls_pass}")
    if safe_to_run:
        print("Gate outcome       : reference gates PASS — safe to run the "
              "10-resample calibration pipeline.")
    else:
        print("Gate outcome       : reference gates FAIL — do NOT run the "
              "calibration pipeline (see report section 7).")
    print("=" * 70)
    return 0 if safe_to_run else 1


def _dry_run_midlow_calibration(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic mid/low SNR calibration (gammaV1f)  [DRY RUN]")
    print("=" * 70)
    n_seeds = args.n_calibration_seeds
    print(f"Base seed             : {args.seed}")
    print(f"Generation seeds      : {n_seeds}  "
          f"(instance i uses base seed {args.seed}+i)")
    print(f"Datasets output root  : {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory      : {out_dir}")
    print(f"core beta (fixed)     : {CALIBRATION_BETA}  (the v1e anchor)")
    print(f"rho (fixed)           : {CALIBRATION_RHO}")
    print(f"weak gamma (unchanged): {WEAK_GAMMA_V1}")
    print(f"n_samples / p         : {N_SAMPLES} / {N_TOTAL_FEATURES}")
    print(f"alpha sweep           : "
          f"{[(s, a, r) for s, a, r in CALIBRATION_ALPHA_CELLS]}")
    print(f"Suffix                : {GRID_V1F_SUFFIX}_seed<NN>")
    print(f"Planned datasets      : {n_seeds} seeds x 5 (4 known-core alpha "
          f"cells + null) = {n_seeds * 5}")
    print("Gate policy           : hard gates apply ONLY to the alpha=2.5 "
          "reference; mid/low candidate checks are informational.")
    planned = []
    for i in range(n_seeds):
        sfx = f"{GRID_V1F_SUFFIX}_seed{i:02d}"
        rho_code = int(round(CALIBRATION_RHO * 100))
        for snr_label, alpha, _ in CALIBRATION_ALPHA_CELLS:
            ac = int(round(alpha * 100))
            planned.append(f"synthetic_snr{snr_label}_a{ac:03d}_"
                           f"rho{rho_code:03d}{sfx}")
        planned.append(f"synthetic_null_control{sfx}")
    print("Planned output files:")
    for nm in planned:
        print(f"  - {out_dir / (nm + '.mat')}")
        print(f"  - {out_dir / (nm + '_manifest.json')}")
    print(f"Planned markdown report: {GRID_V1F_GEN_REPORT_PATH}")
    print("Within-seed control: the alpha-sweep cells of each seed share "
          "identical planted indices, base matrix, eta, label uniforms, and "
          "proxy epsilons; only alpha differs. Across seeds the layout is "
          "freshly drawn.")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Article 5 synthetic known-core control generator "
                    "(data generator only).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and resolved output paths, then exit WITHOUT generating "
             "or writing anything.",
    )
    parser.add_argument(
        "--grid-v1", action="store_true",
        help="Generate the Synthetic Grid Expansion v1: the 2x2 SNR x rho grid "
             "(4 known-core datasets) plus one null control, with the recalibrated "
             "weak gamma. Default (without this flag): the original single-dataset "
             "smoke mode.",
    )
    parser.add_argument(
        "--grid-v1-controlled", action="store_true",
        help="Generate the CONTROLLED Synthetic Grid v1 retune: the 2x2 SNR x rho "
             "grid (4 known-core datasets) + null, with all 4 known-core cells "
             "sharing identical planted indices, base matrix, eta, label uniforms, "
             "and proxy epsilons (only alpha and rho vary), a lower auto-selected "
             "alpha_low, and a `_gammaV1c` name suffix. Takes precedence over "
             "--grid-v1.",
    )
    parser.add_argument(
        "--instance-diagnostic-v1", action="store_true",
        help="Generate the multi-seed instance-variance diagnostic (gammaV1d): "
             "N generation seeds, each with a within-seed-controlled (high SNR, "
             "mid035 alpha=0.35) known-core pair plus a null control, rho fixed at "
             "0.9, a `_gammaV1d_seed<NN>` suffix, and an operational-learnability "
             "diagnostic in each known-core manifest/report. Takes precedence over "
             "--grid-v1-controlled and --grid-v1.",
    )
    parser.add_argument(
        "--n-generation-seeds", type=int, default=DEFAULT_N_GENERATION_SEEDS,
        help=f"Instance-diagnostic mode only: number of generation seeds "
             f"(default {DEFAULT_N_GENERATION_SEEDS}). Instance i uses base "
             f"generation seed --seed + i.",
    )
    parser.add_argument(
        "--positive-control-redesign-v1", action="store_true",
        help="Generate the redesigned high-SNR positive-control regime "
             "(gammaV1e): a targeted, bounded redesign that raises the core "
             "effect size (core beta) so the planted core reliably clears the "
             "noise upper tail, with the operational-learnability checks "
             "(core-vs-noise-p99 margin, oracle true-core AUC) promoted to HARD "
             "acceptance gates. High-SNR + null per seed, rho fixed 0.9, a "
             "`_gammaV1e_seed<NN>` suffix. Takes precedence over the other "
             "modes.",
    )
    parser.add_argument(
        "--n-posctrl-seeds", type=int, default=DEFAULT_N_POSCTRL_SEEDS,
        help=f"Positive-control-redesign mode only: number of generation seeds "
             f"(default {DEFAULT_N_POSCTRL_SEEDS}). Instance i uses base "
             f"generation seed --seed + i.",
    )
    parser.add_argument(
        "--midlow-calibration-v1", action="store_true",
        help="Generate the mid/low SNR calibration alpha-sweep (gammaV1f): at "
             "fixed core beta=2.5 (the v1e positive-control anchor), rho=0.9, "
             "n=200, p=2000, emit per seed an alpha=2.5 positive-control "
             "reference plus mid/low candidates alpha 1.0 / 0.6 / 0.35 and a "
             "null control. Hard operational gates apply ONLY to the alpha=2.5 "
             "reference; the mid/low candidate checks are informational. Uses a "
             "`_gammaV1f_seed<NN>` suffix. Takes precedence over the other "
             "modes.",
    )
    parser.add_argument(
        "--n-calibration-seeds", type=int, default=DEFAULT_N_CALIBRATION_SEEDS,
        help=f"Mid/low-calibration mode only: number of generation seeds "
             f"(default {DEFAULT_N_CALIBRATION_SEEDS}). Instance i uses base "
             f"generation seed --seed + i.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for the generated datasets. Relative paths are "
             "interpreted under results/article_stability/synthetic_control/datasets/; the "
             "resolved path must stay inside it.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base generation seed (default 0). Grid v1: known-core cell i uses "
             "seed+i; the null control uses seed + "
             f"{NULL_SEED_OFFSET}.",
    )
    return parser.parse_args(argv)


def _dry_run_grid_v1(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic Grid Expansion v1 generator  [DRY RUN]")
    print("=" * 70)
    print(f"Base seed            : {args.seed}")
    print(f"Datasets output root : {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory     : {out_dir}")
    print(f"Weak gamma (v1)      : {WEAK_GAMMA_V1}  (original smoke gamma: {WEAK_GAMMA})")
    print(f"SNR alpha values     : high={SNR_ALPHA['high']}, low={SNR_ALPHA['low']}")
    print(f"Fixed spec           : n={N_SAMPLES}, p={N_TOTAL_FEATURES}, "
          f"core={N_CORE} (solo {N_SOLO_CORE} + anchored {N_ANCHORED_CORE}), "
          f"proxy={N_PROXY} ({N_PROXY_GROUPS}x{PROXIES_PER_GROUP}), "
          f"weak={N_WEAK}, noise={N_NOISE}")
    print("Planned grid cells:")
    planned_files = []
    for i, (snr, rho) in enumerate(GRID_V1_CELLS):
        name = f"synthetic_snr{snr.upper()}_rho{int(round(rho*100)):03d}{GRID_V1_SUFFIX}"
        print(f"  - cell {i}: SNR={snr} (alpha={SNR_ALPHA[snr]}), rho={rho}, "
              f"seed={args.seed + i}  ->  {name}")
        planned_files.append(f"{name}.mat")
        planned_files.append(f"{name}_manifest.json")
    null_name = f"synthetic_null_control{GRID_V1_SUFFIX}"
    print(f"  - null control: seed={args.seed + NULL_SEED_OFFSET}  ->  {null_name}")
    planned_files.append(f"{null_name}.mat")
    planned_files.append(f"{null_name}_manifest.json")
    print("Planned output files:")
    for f in planned_files:
        print(f"  - {out_dir / f}")
    print(f"Planned markdown report: {GRID_V1_REPORT_PATH}")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


def _dry_run_original(args, out_dir):
    print("=" * 70)
    print("Article 5 — Synthetic control generator  [DRY RUN]")
    print("=" * 70)
    print(f"Seed (known-core)   : {args.seed}")
    print(f"Seed (null control) : {args.seed + NULL_SEED_OFFSET}")
    print(f"Datasets output root: {ARTICLE5_SYNTH_DATASETS_ROOT}")
    print(f"Output directory    : {out_dir}")
    print(f"Spec                : n={N_SAMPLES}, p={N_TOTAL_FEATURES}, "
          f"core={N_CORE}, proxy={N_PROXY}, weak={N_WEAK}, noise={N_NOISE}, "
          f"weak_gamma={WEAK_GAMMA}")
    print("Planned output files:")
    for n in ("synthetic_snrHIGH_rho090.mat", "synthetic_snrHIGH_rho090_manifest.json",
              "synthetic_null_control.mat", "synthetic_null_control_manifest.json"):
        print(f"  - {out_dir / n}")
    print(f"Planned markdown report: {REPORT_PATH}")
    print("DRY RUN — nothing generated, nothing written.")
    print("=" * 70)
    return 0


def main(argv=None):
    args = parse_args(argv)
    out_dir = resolve_output_dir(args.output_dir)

    if args.dry_run:
        if args.midlow_calibration_v1:
            return _dry_run_midlow_calibration(args, out_dir)
        if args.positive_control_redesign_v1:
            return _dry_run_posctrl_redesign(args, out_dir)
        if args.instance_diagnostic_v1:
            return _dry_run_instance_diagnostic(args, out_dir)
        if args.grid_v1_controlled:
            return _dry_run_grid_v1_controlled(args, out_dir)
        if args.grid_v1:
            return _dry_run_grid_v1(args, out_dir)
        return _dry_run_original(args, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.midlow_calibration_v1:
        return run_midlow_calibration_generation(
            args.seed, args.n_calibration_seeds, out_dir)
    if args.positive_control_redesign_v1:
        return run_positive_control_redesign_generation(
            args.seed, args.n_posctrl_seeds, out_dir)
    if args.instance_diagnostic_v1:
        return run_instance_diagnostic_generation(
            args.seed, args.n_generation_seeds, out_dir)
    if args.grid_v1_controlled:
        return run_grid_v1_controlled_generation(args.seed, out_dir)
    if args.grid_v1:
        return run_grid_v1_generation(args.seed, out_dir)
    return run_generation(args.seed, out_dir)


if __name__ == "__main__":
    sys.exit(main())
