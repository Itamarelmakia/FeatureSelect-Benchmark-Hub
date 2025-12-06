# Feature Selection Benchmarking Hub
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Companion code and datasets for:

> **Itamar Elmakias, Dan Vilenchik**  
> *Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking*,  
> Knowledge-Based Systems, under review (Ms. Ref. No. **KNOSYS-D-25-08690R2**).

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

- 📂 **Datasets on Google Drive** (too large for direct GitHub hosting):  
  https://drive.google.com/drive/u/0/folders/1tzrw9DSBDdoqvZp8PjVpfHjQiGox7GfS  

- 📄 **Paper PDF / preprint**  
  (Add your link here once publicly available.)

---

## Repository Structure

```text
.
├── data/                        # All datasets, grouped by source/repository
│   ├── GEMS/
│   ├── GitHub/
│   ├── Haibasim/
│   ├── ... (other repositories)
├── logs/                        # Logger folder
├── notebooks/                  # Jupyter notebooks (analysis, figures, sanity checks)
├── results/                    # Results per project
│   ├── Convert Data 2 Hard/    # Peeling / hardness-conversion experiments
│   └── FS/                     # Main FS benchmarking runs
├── src/                        # Source code
│   ├── Main_FS.py              # Main FS benchmarking pipeline
│   ├── main_Include Convert Data 2 Hard.py  # Peeling / hard-dataset generator
│   ├── configs/                # Configuration files
│   │   └── config.py
│   ├── fs_algorithms/          # 27 FS algorithms
│   │   ├── AdaBoost.py
│   │   ├── CAE/
│   │   ├── DLFS/
│   │   ├── ... (other methods)
│   ├── utilities.py            # Utilities: CV, parallelism, logging, etc.
│   └── ...
└── README.md
```

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

Due to size constraints (≈1–2 GB), **datasets are hosted on Google Drive** rather than stored directly in the repo.

1. Download the datasets from:  
   https://drive.google.com/drive/u/0/folders/1tzrw9DSBDdoqvZp8PjVpfHjQiGox7GfS
2. Place all dataset folders under the local `data/` directory, preserving the internal structure:
   - `data/GEMS/…`
   - `data/GitHub/…`
   - `data/Haibasim/…`
   - etc.

Once the `data/` folder is populated, the main scripts and notebooks should run without further changes.

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

If you use this repository, datasets, or methodology in your research, please cite:

> **Itamar Elmakias, Dan Vilenchik**.  
> *Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking*.  
> Knowledge-Based Systems, Ms. Ref. No. KNOSYS-D-25-08690R2.

A BibTeX entry will be added here once the paper is formally published.

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
