# Results

Generated outputs are written to `results/article_stability/` at the **hub repo root**
(not inside this `articles/article_stability/` directory).

This directory contains only this README.

## Output tree

```
<hub-root>/results/article_stability/
  synthetic_control/
    datasets/
      *.mat                  # synthetic datasets (make_synthetic_control.py)
      *_manifest.json        # ground-truth planted structure per dataset
    per_fold_selections/
      *.parquet              # per-resample FS logs (confirmatory pipeline)
    recovery_metrics/
      *.parquet              # support-recovery metrics (synthetic_recovery_metrics.py)
  stability_profiles/
    <run_label>/
      *.parquet              # stability profile outputs (analyze_per_fold_stability.py)
  real_data/r2/
    per_fold_selections/
      *.parquet              # Stage R2 per-fold FS logs
    stability_profiles/
      <run_label>/
        *.parquet            # Stage R2 stability profiles
  figures_tables/
    figures/
      fig*.png / fig*.pdf    # manuscript figures (generate_main_manuscript_figures.py)
    table*.csv / table*.md   # manuscript tables
    MANUSCRIPT_FIGURE_TABLE_GENERATION_REPORT.md
  reports/
    *.md                     # generation reports from each script
```

## What is committed to git

Only the `articles/article_stability/scripts/`, `configs/`, `data/`, and
`results/` README files are committed. All generated `.parquet`, `.mat`, `.png`,
and `.pdf` files are excluded by `.gitignore`.

## Regenerating outputs

Run the scripts in this order:

1. `make_synthetic_control.py` — generate synthetic datasets
2. `run_real_data_r2_stability_slice.py` — Stage R2 real-data slice
3. `analyze_per_fold_stability.py` — offline stability profiles (synthetic)
4. `synthetic_recovery_metrics.py` — support-recovery metrics
5. `generate_main_manuscript_figures.py` — figures and tables
