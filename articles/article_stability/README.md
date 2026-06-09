# Feature Stability as a Trust Layer for Feature Selection

**Article:** "Feature Stability as a Trust Layer for Feature Selection"
**Status:** Manuscript in preparation (stability-analysis companion to the FeatureSelect-Benchmark-Hub pipeline)

This directory contains the public reproducibility package for the stability manuscript.
It covers the Stage R2 real-data stability slice and the synthetic positive-control experiment.

---

## What this package contains

| Script | Purpose |
|--------|---------|
| `scripts/make_synthetic_control.py` | Generate synthetic datasets with a planted known-core structure (5 true-core features, 12 proxies, 15 weak, 1968 noise, N=200, P=2000). Multiple modes: default smoke, `--grid-v1`, `--grid-v1-controlled`, and higher variants. |
| `scripts/run_real_data_r2_stability_slice.py` | Stage R2 real-data slice: 50-resample stratified-shuffle-split stability profiling on colon, PeriodChanger, and SMK-CAN-187 with ETree, mRMR, LASSO_Stability, and ReliefF. Computes feature frequencies, pairwise Jaccard, Kuncheva index, Nogueira-style stability, and observed-vs-random Jaccard ratio inline. |
| `scripts/analyze_per_fold_stability.py` | Offline stability analyzer: reads per-fold feature-selection parquets and computes the full stability profile (feature frequencies, threshold sweep at 0.6/0.8/1.0, pairwise Jaccard, Kuncheva, Nogueira-style, observed-vs-random). |
| `scripts/synthetic_recovery_metrics.py` | Support-recovery metrics for the synthetic known-core control: measures planted-core recall, proxy-group coverage, noise contamination, and null-calibrated recurrence. Complementary to — and kept separate from — the stability profile. |
| `scripts/run_confirmatory_snr_grid_v1_pipeline.py` | **Reference only — cannot be run in this package.** Confirmatory 50-resample SNR-ladder pipeline (alpha = 2.5 / 0.6 / 0.35 / null). Requires a private logging driver (`run_synthetic_repeated_resampling_logging`) that is not included here. See *Missing dependency* below. |
| `scripts/generate_main_manuscript_figures.py` | Generate manuscript figures and tables (Fig 1 SNR ladder, Fig 2 synthetic frequency profile, Fig 4 real-data profile, Table 1 / Table S1 stability summary, Fig A2 threshold sweep). Reads pre-generated parquets from `results/article_stability/`. |

---

## Quick start

### 1. Generate the synthetic control datasets

```bash
# Dry run — print the plan, write nothing
python3 articles/article_stability/scripts/make_synthetic_control.py --dry-run

# Generate the original smoke datasets (single high-SNR + null control)
python3 articles/article_stability/scripts/make_synthetic_control.py

# Generate the full v1f SNR grid (8 seeds × 4 SNR levels)
python3 articles/article_stability/scripts/make_synthetic_control.py --grid-v1f
```

Outputs are written to `results/article_stability/synthetic_control/datasets/` at the hub root.

### 2. Run the Stage R2 real-data stability slice

Requires `data/scikit-feature/colon.mat`, `data/UCI/PeriodChanger.mat`, and
`data/scikit-feature/SMK-CAN-187.mat` (all present in the hub repo under `data/`).

```bash
# Dry run
python3 articles/article_stability/scripts/run_real_data_r2_stability_slice.py --dry-run

# Full run (~50 resamples × 3 datasets × 4 algorithms × 5 K values)
python3 articles/article_stability/scripts/run_real_data_r2_stability_slice.py
```

Outputs are written to `results/article_stability/real_data/r2/` at the hub root.

### 3. Analyze a per-fold parquet (offline stability)

```bash
python3 articles/article_stability/scripts/analyze_per_fold_stability.py \
    --input-parquet results/article_stability/... \
    --output-dir my_run_label
```

### 4. Compute support-recovery metrics (synthetic only)

```bash
python3 articles/article_stability/scripts/synthetic_recovery_metrics.py --dry-run
python3 articles/article_stability/scripts/synthetic_recovery_metrics.py
```

### 5. Generate manuscript figures

```bash
python3 articles/article_stability/scripts/generate_main_manuscript_figures.py
```

Reads from `results/article_stability/` (produced by steps 2–4).
Writes to `results/article_stability/figures_tables/`.

---

## Dependencies

This package extends the hub pipeline. All scripts import from the hub's `src/` directory.

**Required Python packages:**

```
numpy
pandas
scipy
scikit-learn
matplotlib
pyarrow          # for .parquet I/O
```

See the hub-level `requirements.txt` for the full list including FS algorithm dependencies.

The hub `src/configs/config.py` and `src/utilities.py` must be importable.
`run_real_data_r2_stability_slice.py` adds `src/` to `sys.path` automatically at run time.

---

## Missing dependency: `run_synthetic_repeated_resampling_logging`

`run_confirmatory_snr_grid_v1_pipeline.py` requires a private logging driver
(`run_synthetic_repeated_resampling_logging`) that coordinates per-resample
feature-selection logging for the synthetic grid. This driver is part of the
private research repo and is **not included** in this public package.

The script is provided for transparency and reproducibility documentation only.
Running it in this package will raise an `ImportError` with a clear message.
To reproduce the confirmatory SNR grid results, the logging driver must be
obtained from the corresponding private research repository.

---

## Output locations

All generated outputs go to `results/article_stability/` at the hub repo root.
These directories are created automatically by the scripts and are excluded from
git tracking (see `.gitignore`).

```
results/article_stability/
  synthetic_control/
    datasets/          # .mat + _manifest.json pairs (make_synthetic_control.py)
    per_fold_selections/  # per-resample FS log parquets (confirmatory pipeline)
    recovery_metrics/  # support-recovery parquets (synthetic_recovery_metrics.py)
  stability_profiles/  # offline stability parquets (analyze_per_fold_stability.py)
  real_data/r2/
    per_fold_selections/  # Stage R2 per-fold logs
    stability_profiles/   # Stage R2 stability profiles
  figures_tables/      # manuscript figures + tables (generate_main_manuscript_figures.py)
  reports/             # markdown generation reports
```

---

## Data

Real-data experiments use datasets already present in the hub's `data/` directory.
See [data/README.md](data/README.md) for dataset sources.

Synthetic datasets are generated by `make_synthetic_control.py` and are not committed to git
(they are large and fully reproducible from the generator + fixed seed).
