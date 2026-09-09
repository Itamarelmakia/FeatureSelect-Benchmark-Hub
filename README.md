# Feature Selection Benchmarking Hub
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Companion code and datasets for:

> **Itamar Elmakias, Dan Vilenchik**  
> *Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking*,  
> Knowledge-Based Systems, Volume 334, 2026, Article 115022.  
> DOI: [10.1016/j.knosys.2025.115022](https://doi.org/10.1016/j.knosys.2025.115022)

This hub also contains the public reproducibility package for a second, related paper on
feature-selection stability — see [Publications](#publications) below.

---

## Overview

This repository implements the full experimental framework behind the paper above.  
Its goal is to answer a deceptively simple question:

> **When is a dataset actually *hard* for feature selection, and how should we pick datasets for fair FS benchmarking?**

Rather than treating “high dimensional = hard”, we define **dataset hardness in terms of the *utility* of feature selection (FS)** compared to strong no-FS baselines. Using a unified pipeline, we profile:

- **27 FS algorithms** (filters, wrappers, embedded, ensembles, neural-based),
- across **102 real-world datasets** (63 binary, 39 multi-class),
- with consistent evaluation (5CV AUC, runtime, selected feature sets, etc.).

From this analysis we:

1. **Categorize dataset difficulty** into **Easy / Medium / Hard**, based on how much FS can improve over a no-FS, all-feature baseline.
2. Show that **most popular benchmarks are *not* inherently challenging**, i.e., many datasets are effectively “solved” without FS.
3. Introduce a **peeling procedure** that transforms existing datasets into **harder variants** while preserving the ground-truth relevant features.
4. Curate a **suite of 13 “challenge” datasets** (Medium + Hard) that are suitable for demonstrating meaningful progress in FS research.
5. Provide an **open, extensible platform** where you can plug in *your* datasets and FS algorithms and obtain the same profiling and hardness labels.

Use this hub to **reproduce all experiments**, **profile new datasets**, or **benchmark new FS methods** under the same hardness-aware protocol used in the paper.

---

## Highlights

- 🧩 **Hardness framework for FS datasets**  
  Data-driven criteria to classify datasets as **Easy / Medium / Hard**, based on the actual *lift* gained by FS over strong no-FS baselines.

- 📊 **Large-scale empirical study (27 FS algos × 102 datasets)**  
  Comprehensive profiling shows that **most widely used benchmarks are not intrinsically “hard”**, challenging common assumptions about the curse of dimensionality.

- 🪜 **Peeling procedure to generate hard variants**  
  A principled way to “peel off” informative samples and features, producing **harder versions of existing datasets** while **preserving ground-truth relevant features** — ideal for controlled benchmarking.

- 🎯 **Curated challenge suite (13 datasets)**  
  A recommended set of **Medium + Hard** datasets that form a **standardized benchmark suite** for future FS studies.

- ⚙️ **End-to-end benchmarking platform**  
  Queue-based pipeline where new datasets and FS algorithms are **automatically benchmarked** against:
  - 27 existing FS methods, and  
  - 102 pre-profiled datasets with established performance baselines.

- ☁️ **Ready-to-run in Colab**  
  A public Colab notebook that reproduces the core pipeline and lets you experiment without local setup.

---

## Quick Links

- ▶️ **Colab demo** (run the pipeline in the cloud):  
  https://colab.research.google.com/drive/1WAGunBduHnqhTAD-vuqm8_4MUVkFZIW-#scrollTo=Ffb5BmyQPlrq

- 📂 **[FeatureSelect Benchmark Datasets](https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing)** (Google Drive dataset archive, ≈1–2 GB):  
  https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing

- 🖥️ **[FeatureWise AI](https://featurewise.itamarelmakias.ai)** (live interactive system):  
  https://featurewise.itamarelmakias.ai

- 📄 **Dataset Hardness paper**  
  https://www.sciencedirect.com/science/article/abs/pii/S095070512502060X

---

## Publications

This hub hosts reproducibility material for **two distinct, published papers**. They
share the same underlying benchmark (datasets, FS algorithms, pipeline) but report on
different research questions and should be cited separately.

### 1. Dataset Hardness (main benchmark & hardness framework)

> **Itamar Elmakias, Dan Vilenchik**  
> *Choosing the right dataset: Hardness criteria for feature selection benchmarking*  
> Knowledge-Based Systems, Volume 334, 2026, Article 115022.  
> DOI: [10.1016/j.knosys.2025.115022](https://doi.org/10.1016/j.knosys.2025.115022)

This is the primary paper behind the hub: the 27-algorithm × 102-dataset benchmark,
the Easy/Medium/Hard hardness framework, and the peeling procedure described in
[Overview](#overview). Its reproducibility pointer lives at
[`articles/article_dataset_hardness/README.md`](articles/article_dataset_hardness/README.md).

### 2. Feature Stability as a Trust Layer

> **Itamar Elmakias, Dor Kolsky, Dan Vilenchik**  
> *Feature Stability as a Trust Layer for Feature Selection: Resampling-Based Recurrence
> Profiles Beyond Predictive Performance*  
> Mathematics, 2026, 14(13), 2372.  
> DOI: [10.3390/math14132372](https://doi.org/10.3390/math14132372)

A separate study built on top of the same hub pipeline, examining resampling-based
feature-selection **stability** (recurrence profiles) as a complement to predictive
performance. Its public reproducibility package — scripts, configs, and data/output
documentation — lives under
[`articles/article_stability/`](articles/article_stability/README.md).

---

## Repository Structure

This reflects what is actually tracked in git. The full dataset archive is **not**
part of the repository — see [Datasets](#datasets) below.

```text
.
├── articles/                    # Reproducibility packages for individual papers
│   ├── article_dataset_hardness/   # Pointer to the KBS hardness study (this hub's main pipeline)
│   └── article_stability/          # Public reproducibility package for the Mathematics stability paper
│       ├── scripts/
│       ├── configs/README.md
│       ├── data/README.md
│       └── results/README.md
├── data/
│   └── README.md                # Instructions for obtaining the external dataset archive
├── sample_data_medium/          # A handful of small .mat files for quick local sanity checks
├── src/                         # Main FS benchmarking pipeline (source code)
│   ├── Main_FS.py                  # Main FS benchmarking pipeline
│   ├── run_single_dataset.py       # Run the pipeline on a single dataset
│   ├── utilities.py                # Utilities: CV, parallelism, logging, etc.
│   ├── configs/                    # Configuration files
│   │   └── config.py
│   └── fs_algorithms/              # FS algorithm implementations
│       ├── AdaBoost.py
│       ├── CAE/
│       ├── DLFS/
│       └── ... (other methods)
├── requirements.txt
├── LICENSE
└── README.md
```

After you download the Google Drive archive (see [Datasets](#datasets)), you populate
your **local** `data/` directory with the source-specific folders it expects
(`data/scikit-feature/`, `data/UCI/`, etc.) — those folders are intentionally not
committed to git.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Itamarelmakia/FeatureSelect-Benchmark-Hub.git
cd FeatureSelect-Benchmark-Hub
```

2. **(Recommended) Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # on Linux / macOS
# .venv\Scripts\activate     # on Windows (PowerShell / CMD)
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

This project requires **Python 3.7+**.

---

## Datasets

This repository contains the **reproducibility code and experimental scripts**. The full
benchmark dataset archive is hosted separately on Google Drive because of its size and is
not tracked in this repository.

**[FeatureSelect Benchmark Datasets](https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing)** — Google Drive folder containing all 102 real-world datasets used in the study.

To set up locally:

1. Download the **[FeatureSelect Benchmark Datasets](https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing)** folder from Google Drive.
2. Extract it, preserving the internal directory structure, into your local `data/` directory
   (e.g. `data/scikit-feature/…`, `data/UCI/…`, and the other source-specific folders in the archive).

See [`data/README.md`](data/README.md) for details. Once the `data/` folder is populated, the main scripts run without further changes.

> **Platform overview**  
> - **This GitHub repository** — reproducibility code, pipeline scripts, and configuration.  
> - **[FeatureSelect Benchmark Datasets](https://drive.google.com/drive/folders/12W8qftORPvwxVmE4dPGLVVEwTHDT5Sn9?usp=sharing)** (Google Drive) — dataset archive for offline reproduction.  
> - **[FeatureWise AI](https://featurewise.itamarelmakias.ai)** — live interactive system for exploring feature-selection results and hardness labels.

---

## Usage

### 1. Reproduce the main FS benchmark

This runs the full FS benchmarking pipeline (27 algorithms × selected datasets) using the configuration in `src/configs/config.py`:

```bash
python src/Main_FS.py --config src/configs/config.py
```

Outputs are written to `results/FS/` and include:
- per-dataset, per-algorithm performance (AUC, accuracy, etc.),
- runtime statistics,
- selected feature sets,
- hardness labels derived from the FS utility criteria.

### 2. Run everything in Colab (no local setup)

Open the Colab notebook:

> https://colab.research.google.com/drive/1WAGunBduHnqhTAD-vuqm8_4MUVkFZIW-#scrollTo=Ffb5BmyQPlrq

There you can:
- run a smaller version of the benchmark,
- inspect hardness labels for example datasets,
- experiment with different FS algorithms and settings.

### 3. Add a new FS algorithm or dataset

To add your own FS method:

1. Implement it under `src/fs_algorithms/` (follow any existing file as a template).
2. Register it in `src/configs/config.py` under the relevant algorithm list.
3. Re-run `Main_FS.py` — your method will be benchmarked alongside all others.

To add a new dataset:

1. Place the raw data under `data/<Source>/<DatasetName>/`.
2. Update the config so the dataset is included in the next run.
3. Re-run the pipeline to obtain:
   - performance across 27 FS methods,
   - runtime,
   - and an **Easy / Medium / Hard** hardness label.

---

## Citation

If you use this repository, the benchmark, or the hardness methodology in your research,
please cite the Dataset Hardness paper:

```bibtex
@article{elmakias2026datasethardness,
  author  = {Elmakias, Itamar and Vilenchik, Dan},
  title   = {Choosing the right dataset: Hardness criteria for feature selection benchmarking},
  journal = {Knowledge-Based Systems},
  volume  = {334},
  year    = {2026},
  articleno = {115022},
  doi     = {10.1016/j.knosys.2025.115022}
}
```

If you use the stability-analysis reproducibility package under `articles/article_stability/`,
please cite the Feature Stability paper instead (or in addition):

```bibtex
@article{elmakias2026featurestability,
  author  = {Elmakias, Itamar and Kolsky, Dor and Vilenchik, Dan},
  title   = {Feature Stability as a Trust Layer for Feature Selection: Resampling-Based Recurrence Profiles Beyond Predictive Performance},
  journal = {Mathematics},
  volume  = {14},
  number  = {13},
  year    = {2026},
  articleno = {2372},
  doi     = {10.3390/math14132372}
}
```

---

## License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute the code and generated datasets, provided that the original copyright notice
and license terms are included in any copies or substantial portions of the software.

See the `LICENSE` file for the full text.

---

## Acknowledgements

This repository is maintained by [Itamar Elmakias](https://github.com/Itamarelmakia).  
Special thanks to the feature selection community for feedback, datasets, and inspiration for building a hardness-aware benchmarking hub.
