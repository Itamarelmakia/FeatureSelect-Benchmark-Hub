# Configs

The stability analysis scripts share the hub pipeline's configuration. There is no
separate config file in this directory.

**Shared config used by `run_real_data_r2_stability_slice.py`:**

- `src/configs/config.py` — feature selection mapping, hyperparameter sheets,
  K-list, CV settings, seed. Imported via `from configs.config import ...` after
  `src/` is added to `sys.path`.

- `src/configs/Algorithms_Hyperparameters_FS.xlsx` — hyperparameter sheets for
  all 27 FS algorithms.

**Missing from public repo:**

- `src/configs/algorithm_metadata.json` — maps algorithm names to family/category
  labels. Used by `run_real_data_r2_stability_slice.py` to populate the
  `algorithm_family` column. Its absence is non-fatal: the column is left `None`
  if the file is not found.

**Locked R2 configuration** (hardcoded in `run_real_data_r2_stability_slice.py`):

| Setting | Value |
|---------|-------|
| Datasets | colon, PeriodChanger, SMK-CAN-187 |
| Algorithms | ETree, mRMR, LASSO_Stability, ReliefF |
| Classifier | ETREE (held-out evaluation) |
| K values | 1, 5, 10, 30, 60 |
| Resamples | 50 (stratified shuffle-split, test_size=0.2) |
| Seed | 0 |
| Preprocessing | raw signed X — `np.abs(X)` intentionally not applied |
