# Feature Selection Benchmarking Hub
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data hub for the paper:

> Itamar Elmakias and Dan Vilenchik  
> **"Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking"**  
> *Knowledge-Based Systems*, Ms. Ref. No. **KNOSYS-D-25-08690R2** (to appear)

---

## Overview

Feature selection (FS) is often treated as *the* solution to high dimensionality, and high-dimensional (HD) datasets are frequently assumed to be “good FS benchmarks” by default.  
In our work, we show that this assumption is often wrong: many HD datasets are actually *easy* for FS and therefore provide little information about how algorithms behave on truly challenging problems.

This repository implements the **hardness framework** and **benchmark hub** described in the paper:

- We analyse **102 real-world datasets** (63 binary, 39 multi-class) and compute dataset-complexity measures.
- Each dataset is categorized into four **hardness classes**: **Easy**, **Medium**, **Hard**, and **Fragile**.
- We benchmark **27 FS algorithms** from diverse families (filter, wrapper, embedded, ensemble, neural, etc.), plus *No-FS* baselines.
- We introduce a **peeling procedure** that takes an “easy” real dataset and produces a **smaller but genuinely hard** subset while preserving:
  - the original labels, and  
  - a known ground-truth set of relevant features.

The resulting hub is a **public benchmark platform** where you can:

- Add your own datasets and obtain hardness labels.
- Plug in new FS algorithms and benchmark them systematically.
- Compare FS vs. No-FS performance under controlled hardness and cost (runtime) conditions.

For more details on the methodology and experimental results, see the paper (preprint on SSRN) and the *Knowledge-Based Systems* article once published.

---

## Quick Links

- 📄 **Paper (preprint)** – SSRN:  
  https://ssrn.com/abstract=5132451

- ▶️ **One-click Colab demo** – run the pipeline without local installation:  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WAGunBduHnqhTAD-vuqm8_4MUVkFZIW-#scrollTo=Ffb5BmyQPlrq)

- 📂 **Datasets (Google Drive folder)** – all benchmark datasets:  
  https://drive.google.com/drive/folders/1tzrw9DSBDdoqvZp8PjVpfHjQiGox7GfS

---

## Highlights

- 🧭 **Hardness-aware FS benchmarking**  
  A unified complexity framework that labels datasets as **Easy / Medium / Hard / Fragile**, instead of using “high-dimensional” as a proxy for difficulty.

- 🧪 **Rich algorithm zoo**  
  Benchmarks **27 FS algorithms** covering classic filters, wrappers, embedded methods, ensembles, DL-based FS and more, plus **No-FS** baselines.

- 📊 **Wide, realistic dataset collection**  
  **102 datasets** from multiple domains (bioinformatics, text, imaging, tabular data, etc.), with consistent preprocessing and evaluation.

- 🧩 **Peeling procedure → “Hard” real datasets**  
  A principled way to convert an easy dataset into a **genuinely hard** one, *with known ground-truth features*, providing realistic stress-tests for FS.

- ⚙️ **Extensible benchmarking hub**  
  Add new datasets or FS algorithms simply by:
  - dropping files into `data/` or `src/fs_algorithms/`
  - updating a config, and  
  - re-running the main pipeline.

- 📈 **Performance vs. cost**  
  Results are reported both in terms of predictive performance (e.g., 5-CV AUC) and **runtime** (FS + model training), enabling cost–benefit analysis of FS vs. No-FS.

---

## Repository Structure

```text
.
├── data/                        # All datasets, grouped by repository/source
│   ├── GEMS/
│   ├── GitHub/
│   ├── Haibasim/
│   ├── ... (other repositories)
├── logs/                        # Logger output
├── notebooks/                   # Jupyter notebooks (EDA, analysis, figures)
├── results/                     # Results per project
│   ├── Convert Data 2 Hard/     # Peeling / hardness-conversion outputs
│   └── FS/                      # FS benchmarking outputs
├── src/                         # Source code
│   ├── Main_FS.py               # Main FS benchmarking pipeline
│   ├── main_Include Convert Data 2 Hard.py   # Peeling (convert dataset → Hard)
│   ├── configs/
│   │   └── config.py            # Main configuration file
│   ├── fs_algorithms/           # 27 FS algorithms (filters, wrappers, embedded, etc.)
│   │   ├── AdaBoost.py
│   │   ├── CAE/
│   │   ├── DLFS/
│   │   ├── ... (other methods)
│   ├── utilities.py             # CV, parallelism, logging, helpers
└── README.md
```

---

## Installation

> 💡 If you just want to *try* the pipeline, start with the **Colab link** above – no local setup required.

For local use:

### 1. Download the datasets

Due to size constraints (~1GB+), datasets are hosted on Google Drive.  
Download them and place inside the local `data/` directory:

- 🔗 **Google Drive Dataset Folder**  
  https://drive.google.com/drive/folders/1tzrw9DSBDdoqvZp8PjVpfHjQiGox7GfS

### 2. Clone the repository and install dependencies

```bash
git clone https://github.com/Itamarelmakia/FeatureSelect-Benchmark-Hub.git
cd FeatureSelect-Benchmark-Hub

# (Recommended) create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
```

Python **3.8+** is recommended.

---

## Usage

### 1. Run the FS benchmark pipeline

This runs the configured FS algorithms on all datasets, with 5-fold CV, and writes results into `results/FS/`:

```bash
python src/Main_FS.py --config src/configs/config.py
```

Outputs typically include:

- Per-dataset, per-algorithm performance metrics (e.g., AUC, accuracy).
- Runtime summaries (FS + model training).
- Aggregated tables that relate performance to dataset hardness.

---

### 2. Generate a “Hard” dataset via peeling

Use the peeling procedure to convert an easy dataset into a smaller but harder one, while preserving labels and ground-truth features:

```bash
python "src/main_Include Convert Data 2 Hard.py"     --input "data/UCI/example.csv"     --output "data/Peeling dataset/example_hard.csv"
```

You will obtain:

- A **harder** subset of the original dataset.
- A **ground-truth feature set** for evaluating FS methods.
- A report / logs demonstrating where FS methods fail on the new dataset.

---

### 3. Add a new FS algorithm

1. Implement your algorithm under:

   ```text
   src/fs_algorithms/MyNewFS.py
   ```

2. Expose it via a simple interface (see existing files for patterns).
3. Update `src/configs/config.py` to include it in the list of FS methods.
4. Re-run:

   ```bash
   python src/Main_FS.py --config src/configs/config.py
   ```

Your new method will be benchmarked across all selected datasets and hardness categories.

---

### 4. Add a new dataset

1. Place the dataset under `data/<SourceName>/your_dataset.csv`.
2. Register the dataset in `config.py` (name, path, task type, etc.).
3. (Optional) Run the hardness characterization / peeling scripts to:
   - compute complexity measures, and  
   - tag it as Easy / Medium / Hard / Fragile.

---

## Explore Results

Use the notebooks under `notebooks/` to:

- Visualize FS vs. No-FS performance by **dataset hardness**.
- Compare algorithms on a specific dataset (e.g., ranking FS methods).
- Inspect the effect of **peeling** on algorithm performance.
- Generate figures similar to those in the *Knowledge-Based Systems* paper.

---

## Citation

If you use this repository, pipeline, or hardness framework in your research, please cite:

**Plain text**

> Elmakias, I., & Vilenchik, D. (2025). *Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking*. Knowledge-Based Systems (to appear). Preprint available at SSRN: https://ssrn.com/abstract=5132451.

**BibTeX**

```bibtex
@article{ElmakiasVilenchik2025DatasetHardnessFS,
  author  = {Itamar Elmakias and Dan Vilenchik},
  title   = {Choosing the Right Dataset: Hardness Criteria for Feature Selection Benchmarking},
  journal = {Knowledge-Based Systems},
  year    = {2025},
  note    = {To appear. Ms. Ref. No. KNOSYS-D-25-08690R2. Preprint available at SSRN 5132451.}
}
```

---

## Contribution

Contributions are very welcome. You can:

- 🧬 Add new datasets under `data/`
- 🧠 Implement new FS algorithms in `src/fs_algorithms/`
- 🛠 Improve performance, documentation, or tests
- 🗣 Open issues for bugs, questions, or feature requests

Please open an issue or a pull request. For major changes, let’s discuss them first via GitHub issues.

---

## License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for full details.

© 2025 Itamar Elmakias and collaborators.
