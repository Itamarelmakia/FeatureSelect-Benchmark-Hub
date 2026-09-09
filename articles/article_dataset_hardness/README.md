# Dataset Hardness Analysis

This directory is a navigation pointer to the published Dataset Hardness / Article 1 material.

The Dataset Hardness study evaluates the difficulty of the 102 real-world benchmark datasets
used in this hub, providing a difficulty score per dataset that contextualizes the feature
selection benchmark results.

## Where to find the material

The Dataset Hardness pipeline is part of the main hub benchmark:

- **Core pipeline:** [`src/Main_FS.py`](../../src/Main_FS.py) and [`src/utilities.py`](../../src/utilities.py)
- **Config:** [`src/configs/config.py`](../../src/configs/config.py)
- **Data:** the 102 `.mat` datasets across multiple source repositories are **not** stored in
  this public git repository. See [`data/README.md`](../../data/README.md) for the Google Drive
  archive link and instructions for populating a local `data/` directory.

The benchmark includes a High-Dimensional (HD) experiment branch (`process_fs_algorithm` with
`'FS'` mode in `Main_FS.py`) that computes dataset hardness alongside feature selection results
for ETree and AdaBoost.

## Article reference

> **Itamar Elmakias, Dan Vilenchik**  
> *Choosing the right dataset: Hardness criteria for feature selection benchmarking*  
> Knowledge-Based Systems, Volume 334, 2026, Article 115022.  
> DOI: [10.1016/j.knosys.2025.115022](https://doi.org/10.1016/j.knosys.2025.115022)

See the root [`README.md`](../../README.md#publications) for the full citation and BibTeX entry.
