# Data Directory

The full benchmark dataset archive (≈1–2 GB, 102 real-world datasets) is hosted
separately on Google Drive because of its size and is **not tracked in this
repository**. Only this `README.md` is committed under `data/`; everything else
under `data/` is intentionally excluded by `.gitignore` (`data/**`, with an
exception for `data/README.md`).

## How to get the full data

1. Download the archive from Google Drive:
   **[FeatureSelect Benchmark Datasets](https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing)**
2. Extract it locally, preserving the archive's internal directory structure
   (one subfolder per dataset source, e.g. `scikit-feature/`, `UCI/`, and the
   other source-specific folders in the archive).
3. Place the extracted folders directly under this `data/` directory, so that
   paths such as `data/scikit-feature/colon.mat` and `data/UCI/PeriodChanger.mat`
   resolve correctly.

Once populated this way, the scripts in `src/` and `articles/` that expect a
local `data/` tree will run without further changes — no download script is
required or provided.
